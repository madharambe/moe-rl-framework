"""B-64: watchdog for the physics VLA RL runs. PyBullet can hang in a native
busy-loop (no SIGINT response, log goes stale, GPU 0%). This monitors the
log's mtime and kill -9 + relaunches from the B-59 periodic ckpt when stale.

B-107 lifecycle hardening (the Aug-21 crash lessons):
  - try/finally: the watchdog's own death kills the trainer (no orphans)
  - SIGTERM handler: clean shutdown on service stop
  - restart budget + exponential backoff: a fast-crashing trainer no longer
    storms CUDA init every 60s forever
  - tail-read of the log instead of whole-file read_text each poll

Usage: python3 src/watch_vla_rl.py <log> <ckpt> <extra args...>
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
STALE_SEC = 360      # 6 min: physics pace is ~100 steps/3min at the slowest
MAX_RESTARTS = 20    # B-107: give up after this many relaunches
BACKOFF_BASE = 60    # seconds; doubles per consecutive failure

log_path = Path(sys.argv[1])
ckpt = Path(sys.argv[2])
extra = sys.argv[3:]

cmd = [
    ".venv/bin/python3", "-u", "src/train_vla_rl.py",
    "--backbone", "ambatron", "--lora_backbone", "--use_pretrained_siglip",
    "--vla_ckpt", str(ckpt), "--steps", "12000",
    "--out", f"out/vla_rl/{ckpt.stem}_w.pt", "--aug", "shift", "--use_physics",
] + extra
next_ckpt = BASE / f"out/vla_rl/{ckpt.stem}_w.pt"

proc = None


def mtime(path):
    return path.stat().st_mtime if path.exists() else 0.0


def log_tail_has_done():
    """Tail-read: only the last 4KB — the old read_text() loaded the whole
    unbounded log every poll."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            return b"[VLA-RL] done." in f.read()
    except OSError:
        return False


def launch():
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # chain from the latest periodic save if it exists (fresh run otherwise)
    chain = next_ckpt if next_ckpt.exists() else ckpt
    base_cmd = cmd[:]
    base_cmd[base_cmd.index("--vla_ckpt") + 1] = str(chain)
    lf = open(log_path, "a")
    # start_new_session=True + killpg: the trainer may spawn children; kill
    # the whole group so nothing survives a hang-kill (B-107)
    p = subprocess.Popen(base_cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, env=env, cwd=BASE,
                         start_new_session=True)
    lf.close()  # the child holds its own dup of the fd
    return p


def shutdown(signum, frame):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

restarts = 0
consecutive_failures = 0
try:
    proc = launch()
    print(f"[watchdog] launched pid {proc.pid} -> {log_path.name}", flush=True)
    while True:
        time.sleep(60)
        if proc.poll() is not None:
            if log_tail_has_done():
                print("[watchdog] run completed cleanly — exiting", flush=True)
                break
            restarts += 1
            if restarts > MAX_RESTARTS:
                print(f"[watchdog] restart budget exhausted "
                      f"({MAX_RESTARTS}) — giving up", flush=True)
                break
            backoff = BACKOFF_BASE * (2 ** min(consecutive_failures, 4))
            consecutive_failures += 1
            print(f"[watchdog] trainer exited rc={proc.returncode}; relaunch "
                  f"{restarts}/{MAX_RESTARTS} in {backoff}s", flush=True)
            time.sleep(backoff)
            proc = launch()
            continue
        # the trainer is alive -> reset the failure streak
        consecutive_failures = 0
        age = time.time() - mtime(log_path)
        if age > STALE_SEC:
            print(f"[watchdog] log stale {age:.0f}s — native hang; kill -9 "
                  f"+ relaunch", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            restarts += 1
            if restarts > MAX_RESTARTS:
                print(f"[watchdog] restart budget exhausted "
                      f"({MAX_RESTARTS}) — giving up", flush=True)
                break
            proc = launch()
finally:
    if proc is not None and proc.poll() is None:
        print("[watchdog] watchdog dying — killing the trainer (no orphans)",
              flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
