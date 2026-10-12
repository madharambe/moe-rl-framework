"""Drive-resilient save/load helpers (B-49).
The external SSD has dropped mid-run several times (user unplugs/replugs). A
torch.save hitting a dropped drive raises OSError and kills the trainer. These
helpers retry saves for a short window (the drive often comes back within
seconds); if it stays down, the caller's retry-wrapper restarts from the last
good checkpoint (which safe_save never corrupts — it saves atomically).
"""
import time
import shutil
from pathlib import Path

RETRIES = 6
DELAY = 8.0


def _stat_target(path):
    """stat the nearest EXISTING ancestor (the pre-flight often runs BEFORE
    the target file exists — disk_usage on a missing path raises OSError and
    crashed the final merges; the merge precedes the save by definition)."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def ensure_disk(path, need_gb, context=""):
    """Golden-rule #3 pre-flight: refuse to start a big write when the target
    filesystem lacks headroom. safe_save's ENOSPC handling only fires mid-write
    (partial tmp + retry loops); callers writing bundles/corpora/clones should
    check FIRST so a 96%-full disk is a loud, early stop, not a surprise."""
    try:
        free = shutil.disk_usage(str(_stat_target(path))).free / 1e9
    except OSError as e:
        raise SystemExit(f"[storage] cannot stat disk for {path}: {e}")
    if free < need_gb:
        raise SystemExit(
            f"[storage] {context or path}: only {free:.1f}GB free, need "
            f"{need_gb}GB — aborting (golden rule #3)")
    return free


def safe_save(obj, path, retries: int = RETRIES, delay: float = DELAY):
    """torch.save with a retry window for transient drive drops (B-49).
    Atomic-ish: writes to <path>.tmp then renames (same-device); on OSError
    retries; raises if the drive stays down so the retry-wrapper can resume.
    ENOSPC (disk full) is special-cased (12th audit P0-1): it is NOT transient —
    retrying just wastes time and leaves a partial <path>.tmp hogging the free
    space that a retry needs. Delete the partial tmp and raise immediately so
    the caller's error path can free space / surface a clear message instead of
    silently eating the reserve."""
    import os as _os
    import errno as _errno
    import torch
    from pathlib import Path

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    last = OSError("save failed")
    for attempt in range(1, retries + 1):
        try:
            torch.save(obj, tmp)
            # B-113: fsync before rename — on ntfs-3g the rename is journaled
            # but file DATA pages flush lazily; a mid-write unplug could lose
            # the checkpoint even after this function returned successfully
            import os
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
            tmp.replace(path)
            # 18th-audit: fsync the PARENT DIR so the rename itself is durable
            # (a power loss right after rename could otherwise lose the new
            # dirent; fsync dir is cheap and done at save-boundary frequency).
            try:
                _dfd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(_dfd)
                finally:
                    os.close(_dfd)
            except OSError:
                pass  # parent fsync unsupported (e.g. some FUSE) — best-effort
            return
        except (OSError, RuntimeError, ValueError) as e:
            # drive drop surfaces as OSError; a partial/garbage tmp read can
            # also raise pickle/zip (ValueError) or torch RuntimeError (B-211)
            if isinstance(e, OSError) and getattr(e, "errno", None) in (
                    _errno.ENOSPC, _errno.EDQUOT):
                # disk full — not transient; free the partial tmp and bail
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                print(f"[storage] DISK FULL saving {path.name} (errno={e.errno}) — "
                      f"removed partial tmp; run is stopping", flush=True)
                raise OSError(f"disk full writing {path.name}: {e.errno}")
            last = e
            # ROUND-204g-audit: the generic failure must also free the partial
            # tmp (only ENOSPC did before — a deterministic write-fail leaked
            # up to ~1G of <name>.pt.tmp until the startup >10min pruner ran)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            print(f"[storage] save failed ({e}) — drive drop? retry "
                  f"{attempt}/{retries} in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise last


def safe_load(path, **kw):
    """torch.load with a retry window for transient drive drops (B-49)."""
    import torch

    kw.setdefault("weights_only", True)  # B-79: never unpickle untrusted ckpts
    last = OSError("load failed")
    for attempt in range(1, RETRIES + 1):
        try:
            return torch.load(path, **kw)
        except (OSError, RuntimeError, ValueError) as e:
            last = e
            print(f"[storage] load failed ({e}) — drive drop? retry "
                  f"{attempt}/{RETRIES} in {DELAY:.0f}s", flush=True)
            time.sleep(DELAY)
    raise last