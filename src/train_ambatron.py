"""
Train Ambatron-430M — IMU-1 inspired dense transformer with QK-Norm,
value residuals, per-head gating, LN scaling, Muon optimizer, WSD schedule.
"""
# B-118: was INSIDE the docstring (never executed) — see train_tool_sft.py
# ROUND-181 (B6): full allocator hardening here — the laptop .bats inherit it
# (shared config for the 6GB/8GB cards; the old code set only expandable_segments
# on Windows-pythonw runs, which is exactly where OOM geometry bites hardest).
import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,throw_on_cudamalloc_oom:True,max_split_size_mb:128,"
    "garbage_collection_threshold:0.8",
)
import sys, math, time, json, subprocess, signal, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ROUND-181 (B8): pythonw hands us None stdout/stderr (no handles) — guard the
# reconfigure so a bare `pythonw train_ambatron.py` doesn't crash at import.
if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)
if sys.stderr is not None:
    sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
# ROUND-182c: WDDM (Windows shared-GPU) hardening — the desktop/Chrome shave
# CUDA-visible VRAM at random moments (esp. during backward); the expandable
# allocator then OOMs with "exceed per_process_memory_fraction" even when the
# trainer's own usage peaks far below the card. Cap the fraction explicitly so
# the trainer never stretches into the desktop's headroom (B's 8GB card:
# RSI_MEM_FRACTION=0.7 -> ~5.7GB cap; batch 1 then fits no matter the blip).
if os.environ.get("RSI_MEM_FRACTION"):
    try:
        _frac = float(os.environ["RSI_MEM_FRACTION"])
        if 0.2 <= _frac <= 0.95:
            import torch as _t
            _t.cuda.set_per_process_memory_fraction(_frac)
    except ValueError:
        pass
from storage import safe_save
from pathlib import Path
from moe_model import (MoEGPT, create_ambatron_430m_config,
                       create_ambatron_mini_config, create_ambatron_plus_config,
                       create_ambatron_v2_config, create_ambatron_v3_config,
                       create_ambatron_xl_config)
from muon import Muon, muon_param_groups
from gpu_accel import resolve_device


def _best_state_file_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)["model_state"]
    except Exception:
        return None


def _model_sd(model):
    """Strip tied lm_head + return state_dict on CPU (avoids CUDA OOM on save).

    state_dict() creates CUDA copies of all parameters (~860MB for 430M bf16).
    On a 6GB GPU at 5.6GB used, this OOMs. Fix: move model to CPU first (frees
    the CUDA space), create state_dict on CPU, then move back. Drop lm_head
    (always tied to token_embed with tie_weights=True) to shave 37.7MB.

    Called at save boundaries only (~5000 iters), not in the hot loop."""
    dev = next(model.parameters()).device
    model.to("cpu")
    sd = model.state_dict()
    sd.pop("lm_head.weight", None)  # always tied; load uses strict=False
    # ROUND-5 (Bug 6): on a CPU-only run `to("cpu")` returns SELF (no copy) —
    # sd would alias the live params and the periodic EMA update would mutate
    # the live weights. Deep-copy per-tensor so snapshots are real snapshots.
    if dev.type == "cpu":
        sd = {k: v.detach().clone() for k, v in sd.items()}
    model.to(dev)
    return sd


class GPUMonitor:
    """Periodic GPU utilization polling (nvidia-smi, ~0.2 ms overhead)."""
    def __init__(self):
        self._last_query = 0.0
        self._util = 0.0
        self._mem = 0.0
        self._temp = 0.0

    def poll(self):
        now = time.time()
        if now - self._last_query < 2.0:
            return
        self._last_query = now
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            parts = out.stdout.strip().split(', ')
            if len(parts) >= 4:
                self._util = float(parts[0])
                self._mem = float(parts[1])
                self._temp = float(parts[3])
        except Exception:
            pass

    def __str__(self):
        return f"GPU {self._util:.0f}% {self._mem:.0f}MB {self._temp:.0f}C"


class _CatPacks:
    """Row view over several token packs (205au).

    205bg: the memmap put the whole 3.4GB corpus into the trainer's RSS as
    never-reused page-cache — the box swapped the user's desktop and the
    GUI stuttered. Now: per-pack file descriptors + os.pread per row
    (2KB per read) => the corpus costs ZERO resident memory; the OS page
    cache still serves repeats without any process RSS.
    """
    def __init__(self, parts, pack_paths=None):
        import numpy as _np
        self._np = _np
        self.parts = parts
        self._lens = [int(p.shape[0]) for p in parts]
        self.shape = (sum(self._lens), int(parts[0].shape[1]))
        self.dtype = parts[0].dtype
        self._offs = _np.cumsum([0] + self._lens)
        # pread-backed parts: replace each memmap with an open fd
        self._fds = []
        self._row_bytes = int(parts[0].shape[1]) * 2  # int16
        if pack_paths:
            import os as _os
            for _p in pack_paths:
                self._fds.append(_os.open(_p, _os.O_RDONLY))
        else:
            self._fds = None

    def _pread_row(self, gidx, ridx):
        import os as _os
        import torch as _t
        if not hasattr(_os, "pread"):
            # Windows: os.pread is POSIX-only — fall back to the memmap rows
            return self.parts[gidx][ridx]
        buf = _os.pread(self._fds[gidx], self._row_bytes, ridx * self._row_bytes)
        return _t.from_numpy(self._np.frombuffer(buf, dtype=self._np.int16))

    def __len__(self):
        return int(self.shape[0])

    def __getitem__(self, idx):
        _np = self._np
        import torch as _t
        if isinstance(idx, int):
            i = int(_np.searchsorted(self._offs, idx, side="right") - 1)
            if self._fds is not None:
                return self._pread_row(i, idx - int(self._offs[i]))
            return self.parts[i][idx - int(self._offs[i])]
        if isinstance(idx, tuple):
            # 205aw: the val path does tokens[rows, :ctx] — tuple indexing
            # (row index, column slice) must work on the pack chain.
            rows, cols = idx
            n = len(rows)
            if isinstance(cols, slice):
                start = 0 if cols.start is None else int(cols.start)
                stop = self.shape[1] if cols.stop is None else int(cols.stop)
                w = max(0, stop - start)
                out = _t.empty((n, w), dtype=self.dtype)
                for j in range(n):
                    out[j] = self[int(rows[j])][start:stop]
                return out
            out = _t.empty((n, len(cols)), dtype=self.dtype)
            for j in range(n):
                out[j] = self[int(rows[j])][cols]
            return out
        out = _t.empty((len(idx), self.shape[1]), dtype=self.dtype)
        ii = idx.numpy() if hasattr(idx, "numpy") else idx
        for j, v in enumerate(ii):
            out[j] = self[int(v)]
        return out


def load_code_corpus(data_dir=None):
    # M5 (18th-audit): anchor to the repo root — the old `data/processed`
    # relative default silently looked in the CWD (broken from any other
    # working dir / laptop with a different out-symlink layout).
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data" / "processed")
    # B-37: CORPUS_FILE env override (e.g. code_corpus_curated.pt) — the
    # curated, balanced corpus built by curate_corpus.py
    name = os.environ.get("CORPUS_FILE", "code_corpus_bpe.pt")
    pt_path = Path(data_dir) / name
    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found. Run build_bpe_tokenizer.py / "
              f"curate_corpus.py first (CORPUS_FILE={name}).")
        sys.exit(1)
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    tokens = data["tokens"]
    lengths = data["lengths"]
    vocab_size = data["vocab_size"]
    # 205au: CORPUS_PACKS="a.pack.meta.pt,b.pt" chains packs as ONE virtual
    # corpus (zero-copy mmap row views; the 3.2B-token packs never enter RAM).
    _chain = os.environ.get("CORPUS_PACKS", "")
    if _chain:
        import numpy as _np
        _parts, _rows_ctx, _pack_paths = [], None, []
        for _spec in _chain.split(","):
            _mp = _spec.strip()
            if _mp.endswith(".meta.pt"):
                _meta = torch.load(_mp, map_location="cpu", weights_only=True)
                _mm = _np.memmap(_meta["pack_path"], dtype=_np.int16, mode="r",
                                 shape=(_meta["rows"], _meta["row"]))
                _parts.append(torch.from_numpy(_mm))
                _pack_paths.append(_meta["pack_path"])   # 205bk: every branch must register its path (the fds list was short -> IndexError)
                _rows_ctx = _meta["row"]
            elif _mp.endswith(".meta.json"):
                import json as _j
                _meta = _j.load(open(_mp))  # 205bj: the torch-free fusion writes json metas
                _mm = _np.memmap(_meta["pack_path"], dtype=_np.int16, mode="r",
                                 shape=(_meta["rows"], _meta["row"]))
                _parts.append(torch.from_numpy(_mm))
                _pack_paths.append(_meta["pack_path"])
                _rows_ctx = _meta["row"]
            else:
                _d = torch.load(_mp, map_location="cpu", weights_only=True)
                _parts.append(_d["tokens"])
                _rows_ctx = _d["tokens"].shape[1]
                # 205bg: int32 .pt packs (the 600M harvest) -> pread-backed
                # temp int16 copies would cost disk; keep the torch view but
                # release the memmap wrapper after load for the .pt case
                _pack_paths.append(None)
        if len(_parts) == 1:
            tokens = _parts[0]
        else:
            _cat = _CatPacks(_parts)
            if all(p is not None for p in _pack_paths):
                # 205bg: pread fds (zero-RSS reads)
                import os as _os
                _cat._fds = [_os.open(p, _os.O_RDONLY) for p in _pack_paths]
            tokens = _cat
        print(f"[Data] pack chain: {len(_parts)} packs, ctx={_rows_ctx}")
    print(f"[Data] {len(tokens)} samples, {tokens.shape[1]} ctx, vocab={vocab_size} "
          f"(corpus: {name})")
    return tokens, lengths, vocab_size


def get_lr_multiplier(iteration, warmup_iters, total_iters, decay_frac=0.1):
    """Warmup-Stable-Decay multiplier [0, 1].
    Phase 1 (warmup): linear 0 → 1 over warmup_iters
    Phase 2 (stable): constant 1.0
    Phase 3 (decay): cosine 1 → 0 over final decay_frac
    """
    if iteration < warmup_iters:
        return iteration / max(warmup_iters, 1)

    decay_start = int(total_iters * (1 - decay_frac))
    if iteration >= decay_start:
        t = (iteration - decay_start) / max(total_iters - decay_start, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return 1.0


def estimate_vram(model, batch_size=1, seq_len=1024):
    """Estimate VRAM for training."""
    n_params = sum(p.numel() for p in model.parameters())
    fp16_bytes = n_params * 2
    # ROUND-170: Muon square momentum is fp32 (Bug-16) — count 4B there, not 2B.
    muon_momentum = sum(p.numel() for _, p in model.named_parameters()
                        if p.dim() >= 2 and 'embed' not in _ and 'norm' not in _ and 'head' not in _) * 4
    adamw_1d = sum(p.numel() for _, p in model.named_parameters()
                   if p.dim() < 2 or 'embed' in _ or 'norm' in _ or 'head' in _)
    adamw_states = adamw_1d * 4 * 2
    # 18th-audit: the old constants were hardcoded to the 430M config (1152*30),
    # wildly overstating estimates for mini/plus/xl. Derive from the real config
    # so the printed VRAM is meaningful for every size.
    cfg = getattr(model, "config", None)
    n_embd = getattr(cfg, "n_embd", 1152)
    n_layer = getattr(cfg, "n_layer", 30)
    activation_est = batch_size * seq_len * n_embd * n_layer * 2 * 4

    total = fp16_bytes + muon_momentum + adamw_states + activation_est
    print(f"[VRAM Est] weights={fp16_bytes/1e9:.2f}GB | muon_mom={muon_momentum/1e9:.2f}GB | "
          f"adamw_1d={adamw_states/1e9:.2f}GB | activations~{activation_est/1e9:.2f}GB | "
          f"total~{total/1e9:.2f}GB")
    return total


def compute_accuracy(logits, targets, pad_mask=None):
    preds = logits.argmax(dim=-1)
    if pad_mask is None:
        return (preds == targets).float().mean().item()
    correct = ((preds == targets) & pad_mask).sum().item()
    return correct / max(pad_mask.sum().item(), 1)


# IMU-1 paper absolute learning rates (Muon base, AdamW = ratio × Muon)
MUON_BASE_LR = 0.0235
# Canonical Muon recipe (KellerJordan/Muon README; verified Aug 2 research):
# muon lr 0.02 for 2D hidden weights; AdamW lr 3e-4 for embed/head/gains/biases.
# The old ratio (0.007/0.0235 ~ 0.298) was ~20x too hot for the Adam groups
# (embeddings/head/norms) — a likely contributor to the loss spikes in the
# earlier run logs.
ADAMW_LR_RATIO = 3e-4 / 0.02  # ~0.015


def train(config, all_tokens, lengths, mode="full", resume=False, compile_model=False,
          dtype="fp16", resume_dir=None, resume_no_opt=False, fresh_best=False,
          min_early_iter=5000, cpu_corpus=False):
    device = resolve_device()
    model = MoEGPT(config)
    model.to(device)
    if device.type == "cuda" and dtype != "fp32":
        model = model.to(dtype=torch.float16 if dtype == "fp16" else torch.bfloat16)
        print(f"[Model] Cast to {dtype} ({'2 bytes/param' if dtype != 'fp32' else '4 bytes/param'})")
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Ambatron-{config.size}: {n_params:,} total params ({n_params/1e6:.1f}M)")

    estimate_vram(model, config.batch_size, config.block_size)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # H1 (18th-audit): stale *.tmp from a previous stop-time SIGKILL can leak
    # forever (systemd TimeoutStopSec kills mid-save, leaving <name>.pt.tmp;
    # neither watchdog nor backup prunes them). Clean any tmp older than 10 min
    # so a 95%-full disk never accumulates them.
    import time as _t
    _now = _t.time()
    for _tmp in out_dir.glob("*.pt.tmp"):
        try:
            if _now - _tmp.stat().st_mtime > 600:
                _tmp.unlink(missing_ok=True)
                print(f"[Startup] removed stale tmp {_tmp}")
        except OSError:
            pass

    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            print("[Compile] Model compiled")
        except Exception as e:
            print(f"[Compile] Failed: {e}")

    # Hold out 5% for validation — Bug 48 fix: seeded shuffle split, not the
    # contiguous tail (the tail shared source files with the train set, and the
    # model memorized the corpus -> val_loss 0.0 / acc 1.0 was meaningless).
    n_samples = len(all_tokens)
    n_val = max(1, int(n_samples * 0.05))
    rng = torch.Generator().manual_seed(1337)
    perm = torch.randperm(n_samples, generator=rng)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    # B-41: dropped pin_memory — page-locked RAM is unswappable and this
    # machine runs with <1 GB free (earlyoom); the one-shot copy at load is
    # cheap at this scale.
    # B-58: --cpu_corpus keeps the corpus on CPU (index CPU, move each batch to
    # the GPU) — frees ~0.9 GB VRAM for the 444M model whose fp32 optimizer +
    # full-corpus-on-GPU footprint OOM'd the 5.64 GB card (22 wrapper retries).
    if cpu_corpus:
        # ROUND-204 (OOM-kill): the fancy-index slices DOUBLE-HOLD 2 x 1.35G in
        # host RAM (base + slices) — the val/save boundary collapsed the ~2.6G
        # headroom and the kernel OOM-killed the python (no traceback). Use the
        # base tensor with INDEX ARITHMETIC instead: identical batch semantics,
        # -2.7G resident.
        _tl = lengths if lengths is not None else None
        train_tokens = all_tokens
        val_tokens = all_tokens
        train_idx_b = train_idx
        val_idx_b = val_idx
        train_lengths = _tl  # the loop indexes by train_idx_b anyway
        val_lengths = _tl
    else:
        train_tokens = all_tokens[train_idx].to(device, non_blocking=True)
        val_tokens = all_tokens[val_idx].to(device, non_blocking=True)
        val_lengths = lengths[val_idx].to(device) if lengths is not None else None
        train_lengths = lengths[train_idx].to(device) if lengths is not None else None
    # Sep-5 (audit): all_tokens is a ~2GB host copy retained alongside
    # train_tokens (another ~2GB) — drop the original now (only the indexed
    # views and lengths are used from here on) to keep host RSS lean
    # (MemoryMax=7500M + earlyoom headroom for Minecraft/opencode sessions).
    del all_tokens
    if lengths is not None:
        del lengths
    print(f"[Data] {len(train_idx)} train + {n_val} val samples (seeded shuffle)")

    # Fixed held-out eval batches (B-22, refs: LLM-stack checklist / ZeroEntropy
    # "noisy validation trap"): re-sampling val batches every check injects
    # sampling noise into the early-stop signal; pre-sample ONCE with a fixed
    # seed so every check (and every resume) sees the same batches.
    # B-36: 4 batches x batch_size = 16 sequences was far below the reference's
    # "a few thousand" guidance and measured ~0.23 high vs a 128-seq sample
    # (hard-outlier fixed seqs, e.g. 3.87) — enlarged to 128 sequences.
    n_val_batches = 32
    val_gen = torch.Generator().manual_seed(2025)
    val_batches = [torch.randint(0, n_val, (config.batch_size,), generator=val_gen)
                   for _ in range(n_val_batches)]

    start_iter = 0
    resumed_without_opt = False  # Bug-18: was UnboundLocalError on fresh runs
    best_loss = float("inf")
    total_tokens = 0
    total_time = 0.0
    nan_count = 0
    current_loss = float("inf")
    best_val_ema = float("inf")
    best_val_raw = float("inf")
    val_patience_count = 0
    best_val_state = None
    best_val_improved = False  # B-33b: gating for ckpt_best.pt overwrite
    best_val_iter = 0          # ROUND-170: iteration at which best_val_state was captured
    ema_state = None
    nan_aborted = False
    early_stopped = False
    stop_reason = ""

    # Resume BEFORE add_lora: wrapping renames weights (W -> base.W), which would
    # make load_state_dict miss every key. Optimizer state loads after Muon creation
    # and only for full mode (LoRA uses a fresh optimizer).
    loaded_ckpt = None
    if resume:
        resume_dir_path = Path(resume_dir) if resume_dir else out_dir
        latest = [p for p in resume_dir_path.glob("ckpt_*.pt")
                  if p.stem.split("_")[-1].isdigit()]
        latest = sorted(latest, key=lambda p: int(p.stem.split("_")[-1]))
        # 205aw: the frozen finals (ckpt_best.pt.final / ckpt_ema.pt.final)
        # were INVISIBLE to the resume (the glob wants ckpt_<digits>.pt) —
        # every "continue the v3 base" run silently trained from scratch.
        _finals = sorted([p for p in resume_dir_path.glob("ckpt_*.pt.final")
                          if not p.with_suffix("").exists()])
        for _f in _finals:
            # 205aw: only adopt a .final when it is NOT a stale post-TRAIN copy
            # of a live periodic (the .final is a completion artifact; a
            # ckpt_best.pt in the same dir is the tracked twin)
            if _f.with_suffix("").exists():
                continue
            latest.insert(0, _f)
        # ROUND-183d (audit B2): ALL candidates are always considered — the
        # signal trap writes ckpt_resume/ckpt_latest on graceful stops; the old
        # optimizer-resume branch excluded them and quietly lost up to a full
        # save_interval of progress. Priority: ckpt_resume (signal-trap) >
        # ckpt_latest (heartbeat) > ckpt.pt > newest periodics > ckpt_best.
        # The no-optimizer path (elif start_iter > 0, re-warm) handles a
        # heartbeat-only candidate safely on optimizer resume.
        latest = ([resume_dir_path / "ckpt_resume.pt",
                   resume_dir_path / "ckpt_latest.pt",
                   resume_dir_path / "ckpt.pt"] + latest
                  + [resume_dir_path / "ckpt_best.pt"])
        latest = [p for p in latest if p.exists()]
        latest_with_existed = bool(latest)
        # B-211: load resumable ckpts via storage.safe_load (retry window for
        # the external-SSD drop + weights_only allowlist, B-79). The old raw
        # torch.load weights_only=False bypassed both hardening layers.
        import torch.serialization as _ser
        from moe_model import MoEConfig as _AMCfg
        _ser.add_safe_globals([_AMCfg])
        from storage import safe_load as _safe_load
        # 11th-audit: the candidate LIST must be in highest-iteration-first order,
        # not fixed name-priority. Otherwise a STALE ckpt_resume.pt (signal-trap
        # from a previous stop) always wins over a just-written ckpt_20000.pt
        # periodic, losing up to 5k iters of progress. ckpt_best stays LAST as the
        # fallback (it may be iter 13100 but is the protected best-guard). We read
        # only the 'iteration' key of each candidate (cheap, no weight load) to sort.
        def _ckpt_iter(f):
            import time as _it_time
            for _it_at in range(2):
                try:
                    import torch.serialization as _ser2
                    from moe_model import MoEConfig as _AMC2
                    _ser2.add_safe_globals([_AMC2])
                    # map_location='meta' loads only shapes, not weights — fast and
                    # low-memory, perfect for reading just the iteration.
                    # 11th-audit fix: torch.load RETURNS a dict (NOT a file object),
                    # so the old `with torch.load(...)` raised TypeError on every
                    # call -> _ckpt_iter always returned None -> the iteration-sort
                    # silently no-op'd and the resume fell back to name-priority
                    # (stale ckpt_resume.pt won). There is no `with` here.
                    # 19th-audit B3: bounded 2x retry on a flaky drive — one
                    # transient meta-read failure must NOT demote the newest
                    # candidate to unscored (stale-wins class).
                    _d = torch.load(f, map_location="meta", weights_only=True)
                    return _d.get("iteration") if isinstance(_d, dict) else None
                except Exception:
                    if _it_at == 0:
                        _it_time.sleep(3)
                    else:
                        return None
            return None
        _scored = []
        _unscored = []
        for _f in list(latest):
            it = _ckpt_iter(_f)
            if it is not None:
                _scored.append((it, _f))
            else:
                _unscored.append(_f)
        # ROUND-183d (audit B3): at EQUAL iterations prefer the optimizer-
        # carrying snapshot (ckpt.pt > ckpt_resume > ckpt_latest > periodic >
        # best) so an early-stop boundary never silently downgrades to a
        # model-only heartbeat + 500-iter re-warm.
        def _name_prio(p):
            n = p.name
            if n == "ckpt.pt": return 4
            if n == "ckpt_resume.pt": return 3
            if n == "ckpt_latest.pt": return 2
            if n == "ckpt_best.pt": return 0
            return 1
        _scored.sort(key=lambda t: (t[0], _name_prio(t[1])), reverse=True)
        # M7 (18th-audit): the old line re-called _ckpt_iter(f) a SECOND time
        # (a full meta-deserialize per file) just to find the un-scored ones —
        # cache them from the single pass above instead.
        # 19th-audit B3: UNSCORED-FIRST. A candidate whose meta-read failed
        # during a drive flap is usually the NEWEST — it gets a fresh full
        # load attempt before any older scored candidate can win.
        latest = _unscored + [_f for _it, _f in _scored]
        loaded = False
        import zipfile as _zf  # Sep-5: BadZipFile escapes the UnpicklingError handler
        for ckpt_file in latest:
            try:
                # ROUND-204: load the resume into CPU RAM, not the GPU — a
                # map_location=device materializes the WHOLE state dict on the
                # card (1.5G for a v2) stacking over the fresh model (3G+ live,
                # and the allocator keeps the freed dict reserved → the first
                # optimizer step's allocation refuses at the wall). The
                # load_state_dict below streams CPU→GPU into the model's OWN
                # tensors (same numerics, ~1.5G streamed over PCIe at load).
                import time as _time
                ckpt = None
                _last_err = None
                # 19th-audit B4: ONE _safe_load (its own 6x8s retry already
                # covers the flap classes) — the old outer 3x nested 48s of
                # retries PER CANDIDATE, stacking to ~25 min of dead air
                # across 8 candidates and tripping stall-killers mid-resume.
                print(f"[Resume] scanning candidate {ckpt_file.name}", flush=True)
                try:
                    ckpt = _safe_load(ckpt_file, map_location="cpu")
                except (torch.serialization.pickle.UnpicklingError, EOFError, _zf.BadZipFile) as _e:
                    _last_err = _e
                if ckpt is None:
                    # M3: re-raise the LAST ORIGINAL exception so the handler
                    # keeps its structural-deletion decision (a UnpicklingError
                    # allowlist-miss is STRUCTURALLY FINE — must never delete).
                    if _last_err is None:
                        raise RuntimeError(f"resume read of {ckpt_file} returned None without an exception")
                    raise _last_err
            except (torch.serialization.pickle.UnpicklingError, EOFError, _zf.BadZipFile) as e:
                # M3 (18th-audit): an UnpicklingError can also mean a NEW
                # un-allowlisted type was added (e.g. a config/schema change) —
                # NOT corruption. Never delete on allowlist-miss; only delete on
                # structural corruption (truncated zip / EOF / not-a-zip).
                msg = str(e)
                is_structural = isinstance(e, (EOFError, _zf.BadZipFile)) \
                    or "truncated" in msg or "EOF" in msg or "unexpected end" in msg \
                    or "not a zip" in msg or "invalid load key" in msg
                if is_structural:
                    # 19th-audit B3: QUARANTINE (rename .corrupt), never unlink
                    # — a one-time drive flake reading a GOOD file must not
                    # destroy the resume anchor permanently.
                    _q = ckpt_file.with_name(ckpt_file.name + ".corrupt")
                    try:
                        ckpt_file.rename(_q)
                        print(f"[Resume] Corrupt checkpoint {ckpt_file}: {e} — quarantined to {_q.name}")
                    except OSError:
                        print(f"[Resume] Corrupt checkpoint {ckpt_file}: {e} — quarantine rename failed; keeping")
                else:
                    print(f"[Resume] Unpickle/allowlist issue {ckpt_file}: {e} — KEEPING file")
                continue
            except OSError as e:
                # Storage error (e.g. external drive I/O) — NEVER delete the
                # file; a transient error would otherwise destroy the run.
                print(f"[Resume] Storage error reading {ckpt_file}: {e} — KEEPING file")
                continue
            except Exception as e:
                print(f"[Resume] UNHANDLED error reading {ckpt_file}: {type(e).__name__}: {e}")
                continue
            if "model_state" in ckpt:
                # H3 18th-audit: guard BOTH directions of arch mismatch. F4
                # (Sep-5): score BEFORE load_state_dict so a rejected ckpt never
                # partially overwrites the model (the old code loaded then
                # checked — a reject left the model half-mismatched and
                # 'starting from scratch' actually trained on garbage weights).
                _loaded_ms = ckpt["model_state"]
                _ms = model.state_dict()
                # ROUND-5 CRITICAL: saved ckpts POP the tied lm_head.weight
                # (_model_sd), but model.state_dict() still contains it — the
                # old _frac denominator counted the tied head against every
                # valid ckpt: mini 0.837 / plus 0.860 -> ALL mini/plus resumes
                # were rejected as "arch mismatch" and silently trained from
                # scratch. Mirror the save format before scoring.
                _ms.pop("lm_head.weight", None)
                _loaded_params = sum(v.numel() for k, v in _loaded_ms.items() if k in _ms)
                _cur_params = sum(v.numel() for v in _ms.values())
                _frac = (_loaded_params / max(_cur_params, 1)) if _cur_params else 0.0
                _covered_keys = sum(1 for k in _ms if k in _loaded_ms)
                _key_frac = _covered_keys / max(len(_ms), 1)
                _ckpt_params = sum(v.numel() for v in _loaded_ms.values())
                _rev_frac = (_loaded_params / max(_ckpt_params, 1)) if _ckpt_params else 0.0
                _shape_ok = all(
                    k not in _loaded_ms or _loaded_ms[k].shape == _ms[k].shape
                    for k in _ms)
                if _frac < 0.9 or _key_frac < 0.9 or _rev_frac < 0.9 or not _shape_ok:
                    print(f"[Resume] arch mismatch: params {_frac:.1%}, keys "
                          f"{_key_frac:.1%}, reverse {_rev_frac:.1%}, shapes "
                          f"{'OK' if _shape_ok else 'MISMATCH'} "
                          f"— SKIPPING {ckpt_file} (do not mix model sizes)")
                    del _ms, _loaded_ms
                    continue
                del _ms, _loaded_ms  # freed BEFORE the load (never retained)
                model.load_state_dict(ckpt["model_state"], strict=False)
                print(f"[Resume] arch check OK: {_frac:.1%} params, "
                      f"{_key_frac:.1%} keys matched")
            if "iteration" in ckpt:
                start_iter = ckpt["iteration"] + 1
            if "total_tokens" in ckpt:
                total_tokens = ckpt["total_tokens"]
            if "best_loss" in ckpt:
                best_loss = ckpt["best_loss"]
            if "ema_state" in ckpt and ckpt["ema_state"] is not None:
                try:
                    cur = model.state_dict()
                    # F3 (Sep-5): ckpt was loaded with map_location=device (CUDA)
                    # so ema_state tensors are CUDA — the periodic-save EMA mix
                    # (ema*b + sd_cpu*(1-b)) would RuntimeError on device mismatch
                    # at the FIRST periodic after any resume. Move to CPU here.
                    ema_state = {k: v.cpu() for k, v in ckpt["ema_state"].items()
                                 if k in cur and cur[k].shape == v.shape}
                    del cur  # F3-adjacent: drop the fresh CUDA state_dict (~0.9GB)
                    print(f"[Resume] Restored post-hoc EMA ({len(ema_state)} tensors)")
                except Exception as e:
                    ema_state = None
                    print(f"[Resume] EMA state incompatible, rebuilding: {e}")
            if "best_val_loss" in ckpt and not fresh_best:
                # B-35: after corpus growth the seeded val split CHANGES — the
                # stored best-val was measured on the old split and is not
                # comparable; --fresh_best keeps tracking from inf so the
                # early-stop patience starts clean on the new split (and
                # ckpt_best.pt stays protected by B-33b either way).
                best_val_ema = ckpt["best_val_loss"]
                # ROUND-171: persist best_val_raw alongside the EMA so the
                # divergence tripwire (val > 1.5x raw) isn't tighter-than-
                # intended right after resume (old: raw=EMA → the EMA is ≥ the
                # true min raw, so a mild post-resume spike could false-trip).
                best_val_raw = ckpt.get("best_val_raw", best_val_ema)
            print(f"[Resume] Loaded {ckpt_file}: iter {start_iter - 1}, best_loss={best_loss:.4f}, best_val={best_val_ema:.4f}")
            # F7 (Sep-5): a stale-era checkpoint can carry iteration >= max_iters
            # (e.g. an old MAX_ITERS=143000 signal trap when the run is now
            # 50000) — the final-iter guard then exits 0 as "completed" WITHOUT
            # saving, and the watchdog treats it as success (no relaunch, no
            # checkpoint). Clamp to max_iters-1 so the run re-trains at least
            # to the configuration's own ceiling.
            if start_iter >= config.max_iters:
                print(f"[Resume] WARN: ckpt iteration {start_iter - 1} >= max_iters "
                      f"{config.max_iters} — clamping to {config.max_iters - 1} (stale-era ckpt)")
                start_iter = config.max_iters - 1
            loaded = True
            loaded_ckpt = ckpt
            break
        if not loaded:
            # ROUND-204-audit (BUG-2): if resume candidates EXISTED but none
            # loaded (wedged SSD), a silent fresh-start re-trains from zero and
            # the first heartbeat CLOBBERS ckpt_latest (the last good weights).
            # Refuse loudly instead — require an explicit --fresh_start.
            if latest_with_existed and not args.fresh_start:
                print("[Resume] CRITICAL: checkpoint candidates existed but none "
                      "loaded (drive wedge?) — REFUSING to start fresh; "
                      "ckpt_latest.pt preserved. Use --fresh_start to override.",
                      flush=True)
                raise SystemExit(3)
            print("[Resume] No readable checkpoint found — starting from scratch")

    # ROUND-204: RSI_SAVE_INTERVAL overrides the periodic cadence (for
    # low-space tail runs: saves only the val-best + the final).
    try:
        _si = int(os.environ.get("RSI_SAVE_INTERVAL", "0"))
        if _si > 0:
            config.save_interval = _si
    except ValueError:
        pass

    if mode == "lora":
        from lora_adapter import add_lora, freeze_base
        add_lora(model, r=16, alpha=32)
        freeze_base(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LoRA] Trainable: {trainable:,} params ({trainable/n_params*100:.2f}%)")

    param_groups = muon_param_groups(model, lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    optimizer = Muon(param_groups, lr=config.learning_rate,
                     weight_decay=config.weight_decay)

    if loaded_ckpt is not None and mode != "lora" and "optimizer" in loaded_ckpt \
            and not resume_no_opt:
        optimizer.load_state_dict(loaded_ckpt["optimizer"])
        # 18th-audit H1: torch's load_state_dict casts EVERY fp tensor state to
        # the parameter dtype via _process_value_according_to_param_policy, so on
        # a bf16 model the fp32 Muon/AdamW moments (the Bug-16 precision design)
        # silently degrade to bf16 on EVERY optimizer-resume. Restore fp32 unless
        # the run explicitly opted into bf16 moments.
        if os.environ.get("RSI_MUON_BF16", "0") != "1":
            from muon import _moment_dtype  # noqa
            if _moment_dtype() == torch.float32:
                for grp in optimizer.param_groups:
                    for p in grp["params"]:
                        st = optimizer.state.get(p)
                        if not st:
                            continue
                        for k, v in list(st.items()):
                            if isinstance(v, torch.Tensor) and v.dtype in (
                                    torch.float16, torch.bfloat16, torch.float32):
                                st[k] = v.float()
        # B-32: load_state_dict restores the CHECKPOINT's group hyperparams —
        # the 10K-iter ckpt.pt carries weight_decay=0.0 on the muon groups
        # (pre-B-24). Without this, B-24 (canonical Muon wd, the norm-growth
        # damper) would be silently defeated on every resume. momentum and the
        # lr fields are identical anyway (lr gets re-set each step in the loop).
        for pg in optimizer.param_groups:
            if pg.get("mode") == "adamw":
                pg["weight_decay"] = config.weight_decay
            else:
                pg["weight_decay"] = min(config.weight_decay, 0.01)
        # Sep-5 (audit): loaded_ckpt retains the CUDA checkpoint dict (weights
        # ~0.9GB + optimizer state ~1-2.4GB) for the WHOLE run — the hidden
        # VRAM hog that pushed today's big-batch runs past the 6GB wall. We are
        # done with it (load_state_dict already consumed it): drop + empty the
        # allocator cache.
        loaded_ckpt = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[Resume] Optimizer state loaded (weight_decay re-applied per B-32)")
    elif start_iter > 0:
        # Research (LLM Stack, WSD chapters): resuming with FRESH optimizer
        # state at full LR is the classic pitfall (momentum/v rebuild from
        # zero) — re-warm the LR for the first 500 iters after resume.
        resumed_without_opt = True
        print("[Resume] No optimizer state — re-warming LR for 500 iters")
        # Sep-5 (audit): model-only resume — loaded_ckpt (CUDA copy, ~0.9GB)
        # was only used by the weights/iteration extraction above. Drop it so
        # the run starts with full VRAM headroom, not a hidden +0.9GB keep-alive.
        loaded_ckpt = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log_file = out_dir / "log.jsonl"
    # GradScaler only works with FP32 master weights (fp32 grads).
    # FP16/BF16 weights train without a scaler (BF16 has FP32 range, no underflow).
    use_fp16 = dtype == "fp16" and device.type == "cuda"
    if use_fp16:
        # B-142: true -- BF16 has fp32 range; FP16 does NOT. Pure-fp16
        # weights/grads without a scaler underflow small gradients.
        print("[WARN] dtype=fp16 has no GradScaler and no fp32 master "
              "weights: gradient underflow likely. Prefer bf16.", flush=True)
    use_bf16 = dtype == "bf16" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (dtype == "fp32" and device.type == "cuda") else None
    accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
    gpu_mon = GPUMonitor() if device.type == "cuda" else None

    # 8th audit: interruption-safe checkpointing (stop-on-the-spot / resume-on-the-spot)
    # On lid-close/reboot (SIGTERM/SIGINT → graceful) or power-loss (hard kill),
    # the training should resume from THE SPOT, not from ckpt_best potentially
    # thousands of iters behind. Two mechanisms: (a) a 100-iter heartbeat that
    # writes ckpt_latest.pt (covers ALL stop types including SIGKILL/power-loss)
    # and (b) a signal trap that writes ckpt_resume.pt on graceful exit (covers
    # systemctl stop / Ctrl+C). Resume priority: resume > latest > periodic > best.
    _resume_saved = False  # one-shot guard for signal trap
    _ckpt_latest_path = out_dir / "ckpt_latest.pt"
    _ckpt_resume_path = out_dir / "ckpt_resume.pt"
    # ROUND-173: debug instrumentation — the training-health JSONL (one line
    # per [Diag] cadence) + imports used by the diag block.
    import json as _json_dbg
    import shutil as _shutil_dbg
    cfg_dbg = out_dir / "train_health.jsonl"

    def _save_ckpt_latest():
        """Rolling heartbeat save: overwrites ckpt_latest.pt every 100 iters.
        NOTE (Sep-5 storage bomb fix + ROUND-169 comment correction): ema_state
        (an 848MB twin of model_state) is persisted ONLY at the FINAL save
        (ckpt.pt / ckpt_ema.pt), NOT here and NOT at periodic/best boundaries —
        periodics are deliberately model-only (storage pressure: a 1.7GB
        periodic would break the 4GB disk floor with keep=2). Any resume
        rebuilds the EMA from its tail, which is bounded and correct."""
        _sd = _model_sd(model)
        # 19th-audit B1: the heartbeat is NOT the run's reward — a wedged-drive
        # save must never kill the trainer. WARN + continue (the FINAL save is
        # the only fail-loud site).
        print(f"[Save] start heartbeat ckpt_latest iter {iteration}", flush=True)
        try:
            safe_save({"model_state": _sd, "iteration": iteration,
                        "best_loss": best_loss, "total_tokens": total_tokens,
                        "best_val_loss": best_val_ema, "best_val_raw": best_val_raw},
                      _ckpt_latest_path)
        except Exception as e:
            print(f"[WARN][Save] heartbeat ckpt_latest FAILED: {e} — continuing", flush=True)

    def _signal_exit(signum=None, frame=None):
        """Graceful exit trap: save ckpt_resume.pt then exit cleanly."""
        nonlocal _resume_saved
        if _resume_saved:
            return
        _resume_saved = True
        try:
            # 18th-audit: ONE _model_sd (single CPU round-trip), reused for both
            # latest + resume — the old handler called _model_sd twice, doing two
            # ~0.5s CPU<->GPU round-trips inside a signal handler. Same storage
            # note as _save_ckpt_latest: NO ema_state in high-frequency saves.
            _sd = _model_sd(model)
            safe_save({"model_state": _sd, "iteration": iteration,
                        "best_loss": best_loss, "total_tokens": total_tokens,
                        "best_val_loss": best_val_ema, "best_val_raw": best_val_raw},
                      _ckpt_latest_path)
            safe_save({"model_state": _sd, "iteration": iteration,
                        "best_loss": best_loss, "total_tokens": total_tokens,
                        "best_val_loss": best_val_ema, "best_val_raw": best_val_raw},
                      _ckpt_resume_path)
            print(f"\n[Signal] Caught signal {signum}, saved resume checkpoint at iter {iteration}", flush=True)
        except Exception as e:
            print(f"\n[Signal] Caught signal {signum}, resume save failed: {e}", flush=True)
        # ROUND-170: exit 143 (128+SIGTERM), NOT 0 — a clean exit 0 makes
        # run_ambatron.sh write 'training completed', which the watchdog's F2
        # halt reads to permanently stop the run. An earlyoom/manual SIGTERM
        # directly on the child must look like a signal death, not completion.
        sys.exit(143)

    signal.signal(signal.SIGTERM, _signal_exit)
    signal.signal(signal.SIGINT, _signal_exit)

    # 18th-audit: if resuming from a checkpoint already at/above max_iters, the
    # loop never runs and `iteration` is undefined — the old ternary saved a
    # garbage ckpt.pt (current_loss=inf, total_tokens=0). Guard and exit cleanly.
    if start_iter >= config.max_iters:
        print(f"[Done] Already at max_iters ({start_iter} >= {config.max_iters}) — nothing to do; not re-saving")
        return model

    # 18th-audit: track successfully-backwarded micro-steps, not the iteration
    # counter, for the accumulation boundary. A NaN mid-cycle zeroes the partial
    # grads and resets accum_count, so the next optimizer step fires only after
    # a FULL accumulation window of good micro-steps — no LR/step-count drift
    # from the old `(iteration+1) % accumulation_steps` alignment.
    accum_count = 0
    _patience_gate_open = False  # ROUND-170: reset patience when the gate first opens
    for iteration in range(start_iter, config.max_iters):
        iter_start = time.time()
        grad_norm = None  # B-22: grad-norm monitoring (LLM-stack stability checklist)

        mult = get_lr_multiplier(iteration, config.warmup_iters, config.max_iters, decay_frac=0.1)
        # Resume re-warm: when resuming WITHOUT optimizer state (model-only),
        # momentum rebuilds from zero, so a brief low-LR ramp stabilizes the
        # first ~500 iters. Ramp 10% -> 100% over the first 500 iters, capped
        # at 1.0. NEVER let LR exceed the schedule peak (a prior unbounded
        # (gap+1)/500 hit 4.7x peak after a large resume gap -> loss exploded).
        _resume_gap = iteration - start_iter
        # 11th-audit P2: only re-warm in the STABLE phase (mult == 1.0). In the
        # warmup or decay tail (mult < 1.0) the schedule LR is already the
        # correct cold-momentum starting point; applying a 0.1->1.0 ramp there
        # would inject a non-monotonic LR bump into the cosine anneal.
        if resumed_without_opt and mult >= 1.0:
            # ROUND-181 (B1): ramp from gap 0 — the OLD code gated on
            # _resume_gap >= 200, so the first 200 cold-momentum iters ran at
            # FULL LR then dove to 10% (inverted non-monotonic bump). Ramp
            # 10% -> 100% over the first 500 iters instead.
            ramp = 0.1 + 0.9 * min(1.0, _resume_gap / 501)
            mult *= ramp
        muon_base = config.learning_rate if config.learning_rate > 0 else MUON_BASE_LR
        muon_lr = muon_base * max(mult, 0.01)  # floor at 1% of base
        adamw_lr = muon_base * ADAMW_LR_RATIO * max(mult, 0.01)
        for pg in optimizer.param_groups:
            if pg["mode"] == "adamw":
                pg["adamw_lr"] = adamw_lr
            else:
                pg["muon_lr"] = muon_lr

        batch_indices = torch.randint(0, len(train_idx), (config.batch_size,),
                                      device="cpu" if cpu_corpus else device)
        # ROUND-204 (zero-copy): with train_tokens = the BASE tensor, map the
        # random index through the shuffle permutation (all_tokens[train_idx]
        # as the equivalent gather).
        if cpu_corpus:
            batch_indices = train_idx[batch_indices]
        batch_tokens = train_tokens[batch_indices]  # B-58: CPU corpus -> move the batch
        if cpu_corpus:
            batch_tokens = batch_tokens.to(device, non_blocking=True)
        ctx = min(config.block_size, batch_tokens.shape[1])
        batch_tokens = batch_tokens[:, :ctx]

        x = batch_tokens[:, :-1].contiguous().long()  # corpus is int16; Embedding/CE need Long
        y = batch_tokens[:, 1:].contiguous().long()
        pad_mask = (x != 0) & (y != 0)  # <PAD> id = 0

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, loss, aux = model(x, targets=y, use_grad_ckpt=True, pad_mask=pad_mask)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count > 10:
                    print(f"[NaN] Too many NaN losses ({nan_count}), aborting")
                    nan_aborted = True
                    break
                print(f"[NaN] Loss is NaN/Inf at iter {iteration}, skipping (count={nan_count})")
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                continue
            scaler.scale(loss / accumulation_steps).backward()
            accum_count += 1
            if accum_count == accumulation_steps:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
        else:
            logits, loss, aux = model(x, targets=y, use_grad_ckpt=True, pad_mask=pad_mask)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count > 10:
                    print(f"[NaN] Too many NaN losses ({nan_count}), aborting")
                    nan_aborted = True
                    break
                print(f"[NaN] Loss is NaN/Inf at iter {iteration}, skipping (count={nan_count})")
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                continue
            (loss / accumulation_steps).backward()
            accum_count += 1
            if accum_count == accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

        total_tokens += int(pad_mask.sum())  # real tokens, not pad
        iter_time = time.time() - iter_start
        total_time += iter_time
        current_loss = loss.item()
        # 18th-audit: decay nan_count on clean iterations so a single bad batch
        # (corrupt sample) doesn't permanently consume one of the 10 abort slots
        # for the entire run. 1 clean iter gives back 1 slot (clamped >= 0).
        if nan_count > 0:
            nan_count = max(0, nan_count - 1)
        best_loss = min(best_loss, current_loss)
        acc = compute_accuracy(logits, y, pad_mask)

        # 9th audit: 100-iter heartbeat — ckpt_latest.pt gives ~2.5 min resume
        # granularity on this 6GB laptop. BUT skip when a periodic save fires
        # at the same iteration (e.g. iter 20000 is 5k-boundary AND 100-boundary)
        # to avoid two state_dict() calls in one iteration — the CUDA alloc
        # transient pushes a 6GB card to OOM (observed: iter 20000 OOM crash).
        if iteration % 100 == 0 and iteration > 0 \
                and iteration % config.save_interval != 0:
            _save_ckpt_latest()

        if iteration % config.log_interval == 0:
            tokens_per_sec = x.numel() / max(iter_time, 1e-8)
            if gpu_mon:
                gpu_mon.poll()
            gpu_str = str(gpu_mon) if gpu_mon else ""
            gn_str = (f"GradNorm {grad_norm.item():.2f} | " if grad_norm is not None else "")
            print(f"Iter {iteration:6d}/{config.max_iters} | Loss {current_loss:.4f} | "
                  f"Acc {acc:.3f} | LR {muon_lr:.2e} | {gpu_str} | {gn_str}"
                  f"Tok/s {tokens_per_sec:.0f} | Total {total_tokens:,}", flush=True)

        # ROUND-173 (debug instrumentation): compact one-line diagnostics on a
        # slower cadence (every 2x log_interval) + a machine-readable health
        # JSONL (one row per diag) for post-hoc analysis. The numbers that make
        # the OOM geometry and drift visible:
        #   vram_alloc = live tensors (memory_allocated)
        #   vram_res   = ALLOCATOR-reserved (memory_reserved — what nvidia-smi shows)
        #   vram_free  = free per the driver (mem_get_info)  -> alloc vs res vs
        #                free exposes cached-but-reclaimable + fragmentation
        #   ram_avail  = host MemAvailable (earlyoom budget)
        #   disk_free  = out_dir fs free (save/ENOSPC budget)
        #   it_ms      = this iteration wall time (perf regressions)
        #   eta_min    = projected completion at the current tok/s
        if iteration % max(2 * config.log_interval, 200) == 0 and cfg_dbg is not None:
            try:
                if device.type == "cuda":
                    _alloc = torch.cuda.memory_allocated() // (1024 * 1024)
                    _res = torch.cuda.memory_reserved() // (1024 * 1024)
                    _free, _tot = torch.cuda.mem_get_info()
                    _free //= 1024 * 1024
                else:
                    _alloc = _res = _free = 0
                _ram_mb = -1
                try:
                    for _l in open("/proc/meminfo"):
                        if _l.startswith("MemAvailable:"):
                            _ram_mb = int(_l.split()[1]) // 1024
                            break
                except OSError:
                    pass
                _disk_mb = -1
                try:
                    _disk_mb = _shutil_dbg.disk_usage(out_dir).free // (1024 * 1024)
                except OSError:
                    pass
                _eta_min = ((config.max_iters - iteration)
                            / max(tokens_per_sec, 1.0) * x.numel() / 60.0)
                _diag = (f"[Diag] iter {iteration}/{config.max_iters} | "
                         f"loss {current_loss:.4f} | acc {acc:.3f} | lr {muon_lr:.2e} | "
                         f"gradnorm {grad_norm.item():.2f} | " if grad_norm is not None else "\n"
                         f"vram alloc {_alloc}MB res {_res}MB drv-free {_free}MB | "
                         f"ram-avail {_ram_mb}MB | disk-free {_disk_mb}MB | "
                         f"it {iter_time * 1000:.0f}ms | toks/s {tokens_per_sec:.0f} | "
                         f"eta {_eta_min:.0f}min")
                print(_diag, flush=True)
                try:
                    with open(cfg_dbg, "a", encoding="utf-8") as _df:
                        _df.write(_json_dbg.dumps({
                            "iter": iteration, "loss": current_loss, "acc": acc,
                            "lr": muon_lr, "gradnorm": grad_norm.item() if grad_norm is not None else None,
                            "vram_alloc_mb": _alloc, "vram_res_mb": _res,
                            "vram_drv_free_mb": _free, "ram_avail_mb": _ram_mb,
                            "disk_free_mb": _disk_mb, "iter_ms": iter_time * 1000,
                            "toks_per_s": tokens_per_sec, "eta_min": _eta_min,
                        }) + "\n")
                except OSError:
                    pass
            except Exception as _de:
                print(f"[Diag] skipped ({_de})", flush=True)

            # Validation every 100 iters (average over VAL_SAMPLES random batches)
            if iteration % 100 == 0 and iteration > 0:
                model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for vb in val_batches:
                        _vb_idx = val_idx[vb.cpu()] if cpu_corpus else vb
                        val_batch = val_tokens[_vb_idx, :ctx]
                        if cpu_corpus:
                            val_batch = val_batch.to(device, non_blocking=True)
                        vx = val_batch[:, :-1].contiguous().long()  # corpus is int16; Embedding/CE need Long
                        vy = val_batch[:, 1:].contiguous().long()
                        vmask = (vx != 0) & (vy != 0)
                        if scaler is not None:
                            with torch.amp.autocast("cuda", dtype=amp_dtype):
                                _, vloss, _ = model(vx, targets=vy, pad_mask=vmask)
                        else:
                            _, vloss, _ = model(vx, targets=vy, pad_mask=vmask)
                        val_loss_sum += vloss.item()
                    val_loss = val_loss_sum / len(val_batches)
                model.train()
                gpu_str = f" {gpu_mon}" if gpu_mon else ""
                print(f"         Val loss {val_loss:.4f} | Best {best_loss:.4f} | "
                      f"BestVal {best_val_ema:.4f} | Patience {val_patience_count}{gpu_str}", flush=True)

                # 11th-audit P4: split the gate. Tracking val EMA / best / patience
                # runs ALWAYS (so ckpt_best.pt refreshes on real improvement and
                # monitor output is truthful) — the old code put the whole block
                # under `iteration > max(warmup, min_early_iter)`, which with
                # MIN_EARLY=150000 > max_iters froze best-val tracking for the
                # entire run and never updated ckpt_best.pt even when val improved
                # below the stored best (measured: val 2.2651 vs stored 2.2859).
                ema = val_loss if best_val_ema == float("inf") else 0.9 * best_val_ema + 0.1 * val_loss
                best_val_raw = min(best_val_raw, val_loss)
                if ema < best_val_ema - 1e-4:
                    best_val_ema = ema
                    val_patience_count = 0
                    best_val_improved = True
                    # ROUND-204g-audit: ONE capture serves the cache save (the
                    # old double _model_sd doubled the CPU transient at the exact
                    # OOM-kill moment)
                    best_val_file = out_dir / "best_state_cache.pt"
                    # 19th-audit B1: cache save wrapped — never fatal
                    print(f"[Save] start best_state_cache iter {iteration}", flush=True)
                    try:
                        safe_save({"model_state": _model_sd(model),
                                   "iteration": iteration,
                                   "best_val_raw": best_val_ema}, best_val_file)
                    except Exception as e:
                        print(f"[WARN][Save] best_state_cache FAILED: {e} — continuing", flush=True)
                    best_val_state = "CACHED"
                    best_val_iter = iteration  # ROUND-170: label the ckpt with the ITER it was captured (the old code stored save-time iter = a lie)
                else:
                    val_patience_count += 1
                # Early stop: only after warmup completes (LR stable) AND after a
                # minimum training window (B-47: a fresh/resumed model plateaus in
                # its first few thousand iters then keeps improving — the patience
                # tripwire was firing at ~0.08 tok/param, nonsense for a 121M model
                # that needs ~40K+ iters).
# ROUND-170 (F2): the patience counter incremented from iter 100 even
                # though the tripwire is gated to open later — a plateau that
                # spanned the gate boundary could trip instantly. Reset the
                # counter the first time the gate opens (or when crossing it).
                if (config.early_stop and not _patience_gate_open
                        and iteration >= max(config.warmup_iters, min_early_iter)):
                    val_patience_count = 0
                    _patience_gate_open = True
                if config.early_stop and iteration > max(config.warmup_iters,
                                                          min_early_iter):
                    diverged = val_loss > 1.5 * best_val_raw and val_patience_count >= 3
                    if val_patience_count >= config.val_patience:
                        early_stopped = True
                        stop_reason = (f"val loss no improvement for {val_patience_count} checks "
                                       f"(patience={config.val_patience}, best_val={best_val_ema:.4f})")
                    elif diverged:
                        early_stopped = True
                        stop_reason = (f"val loss diverged ({val_loss:.4f} > 1.5x best "
                                       f"{best_val_raw:.4f})")
                    if early_stopped:
                        print(f"[EarlyStop] {stop_reason} at iter {iteration}", flush=True)
                        break
            else:
                val_loss = None  # no validation this iteration; log as null

            with open(log_file, "a") as f:
                # ROUND-204ac-b5 (B crash): aux["ce_loss"] is a Tensor
                # (F.cross_entropy in moe_model) — json.dumps died on it after
                # iter 0 on EVERY v3 launch (A 22:04 + B both left log.jsonl
                # empty). Coerce all fields tensor-safe with _js.
                def _js(v):
                    return v.item() if isinstance(v, torch.Tensor) else v
                f.write(json.dumps({
                    "iter": iteration, "loss": _js(current_loss), "val_loss": _js(val_loss),
                    "acc": _js(acc), "ce_loss": _js(aux["ce_loss"]),
                    "muon_lr": _js(muon_lr), "adamw_lr": _js(adamw_lr),
                    "mult": _js(mult), "grad_norm": _js(grad_norm),
                    "tokens_per_sec": _js(tokens_per_sec), "total_tokens": _js(total_tokens),
                    "best_loss": _js(best_loss),
                }) + "\n")

        if iteration % config.save_interval == 0 and iteration > 0:
            # 9th audit: free CUDA cache before the save — state_dict() copies
            # ~860MB to CPU via _model_sd, but torch's allocator fragments and
            # can't reclaim 128MB contiguous for the next forward pass (observed
            # OOM at iter 20000). gc+empty_cache defrag before the allocation.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ckpt_path = out_dir / f"ckpt_{iteration}.pt"
            # 18th-audit: ONE _model_sd call for the periodic save AND the EMA
            # update below (they snapshot the same unchanged model). The old code
            # called _model_sd twice -> two CPU round-trips + two 860MB copies.
            sd_periodic = _model_sd(model)
            state = {
                "model_state": sd_periodic,
                "iteration": iteration,
                "loss": current_loss,
                "best_loss": best_loss,
                "best_val_loss": best_val_ema,
                "best_val_raw": best_val_raw,
                "total_tokens": total_tokens,
                "total_time": total_time,
            }
            # 19th-audit B1: periodic save wrapped — the drive flap class must
            # never kill the trainer; the FINAL save is the fail-loud site.
            print(f"[Save] start periodic {ckpt_path}", flush=True)
            try:
                safe_save(state, ckpt_path)
                print(f"[Save] {ckpt_path}")
            except Exception as e:
                print(f"[WARN][Save] periodic {ckpt_path} FAILED: {e} — continuing", flush=True)
            # B-48: also persist ckpt_best.pt at every periodic save when val has
            # improved THIS run (not just at run-end) — a mid-run kill otherwise
            # loses the best weights (they only live in the periodic snapshots).
            _bst = best_val_state
            if _bst == "CACHED":
                _bst = _best_state_file_load(best_val_file)
            if _bst is not None and best_val_improved:
                best_path = out_dir / ("ckpt_best_fresh.pt" if fresh_best
                                       else "ckpt_best.pt")
                # 11th-audit P3: include total_tokens/total_time so a model-only
                # resume anchored on ckpt_best keeps cumulative accounting (was
                # dropped -> every resume reset its token/tput account).
                # 19th-audit B1: wrapped — secondary save, never fatal.
                print(f"[Save] start ckpt_best iter {best_val_iter}", flush=True)
                try:
                    safe_save({"model_state": _bst, "iteration": best_val_iter,
                                "best_val_loss": best_val_ema,
                                "best_loss": best_loss, "config": config,
                                "total_tokens": total_tokens, "total_time": total_time},
                              best_path)
                    print(f"[Save] {best_path}")
                except Exception as e:
                    print(f"[WARN][Save] ckpt_best FAILED: {e} — continuing", flush=True)
            # Post-hoc weight EMA (IMU-1 recipe): beta-weighted average of
            # checkpoint snapshots, saved as ckpt_ema.pt at the end.
            if config.ema_beta > 0:
                if ema_state is None:
                    ema_state = sd_periodic
                else:
                    b = config.ema_beta
                    # ROUND-5 (Bug 5): a resumed EMA may lack keys added by a
                    # mid-run arch change (guard allows >=90% match) — the old
                    # dict-comp KeyError'd at the FIRST periodic after resume.
                    ema_state = {k: (ema_state.get(k, sd_periodic[k]) * b
                                     + sd_periodic[k] * (1 - b))
                                 for k in sd_periodic}
            # Prune old checkpoints to avoid filling the disk (0.9 GB each now:
            # periodic ckpts are model-only — no optimizer state, per user choice)
            keep = 2
            old = sorted((p for p in out_dir.glob("ckpt_*.pt")
                          if p.stem.split("_")[-1].isdigit()),
                         key=lambda p: int(p.stem.split("_")[-1]))
            for stale in old[:-keep]:
                stale.unlink(missing_ok=True)
                print(f"[Prune] removed {stale}")

    final_iter = iteration if (early_stopped or nan_aborted) else config.max_iters - 1
    # H4 (18th-audit): on a NaN-abort the current weights may be corrupted (a
    # finite-loss/NaN-grad path can poison params without tripping the loss
    # guard). Writing them to ckpt.pt makes the poisoned weights the highest-
    # iteration resume anchor, and the wrapper restarts into them -> repeated
    # NaN-aborts -> give-up. Save to ckpt_nan.pt instead so resume naturally
    # anchors on the last pre-NaN checkpoint, and exit non-zero.
    final_path = out_dir / ("ckpt_nan.pt" if nan_aborted else "ckpt.pt")
    if nan_aborted:
        print("[NaN] Aborting — saving diagnostic ckpt_nan.pt (NOT overwriting ckpt.pt) and exiting non-zero", flush=True)
    # 11th-audit P1: the old code saved raw model.state_dict() here — the one
    # save site that bypassed the _model_sd CPU-move done everywhere else.
    # On CUDA, model.state_dict() copies ~860MB to GPU which OOMs the 6GB card
    # at the end of a run (peak ~5.6GB used). _model_sd moves to CPU first.
    # ROUND-174 (Sep-6): the final archive (model 848MB + optimizer ~1.4GB +
    # ema) is ~2.7GB — the actual 143,000 finish ENOSPC'd mid-write (disk 2MB
    # after the burst; ckpt.pt truncated; the watchdog's relaunch re-finished
    # it). Pre-flight the disk: if < 3GB free, fall back to a MODEL-ONLY final
    # save (weights + config, no optimizer) so the run's reward is never lost,
    # and say so loudly.
    _final_state = {"model_state": _model_sd(model),
                    "iteration": final_iter,
                    "loss": current_loss,
                    "best_loss": best_loss,
                    "best_val_loss": best_val_ema,
                    "best_val_raw": best_val_raw if "best_val_raw" in dir() else None,
                    "total_tokens": total_tokens,
                    "total_time": total_time,
                    "early_stopped": early_stopped,
                    "stop_reason": stop_reason,
                    "ema_state": ema_state,   # 18th-audit: persist EMA in the final save too
                    "config": config,
                    "optimizer_full": True}
    try:
        _free_mb = _shutil_dbg.disk_usage(out_dir).free // (1024 * 1024)
    except OSError:
        _free_mb = -1
    # ROUND-183d (audit B5): the full archive needs ~4.2GB (main ~2.7GB +
    # ema ~0.85GB + best ~0.45GB) — the old 3GB gate let the cascade ENOSPC
    # at the ema/best stage. Budget the WHOLE cascade; _free_mb==-1 (usage
    # unreadable) must take the model-only branch, never the full one.
    if _free_mb < 4608:
        print(f"[FinalSave] WARN: only {_free_mb}MB free — saving a MODEL-ONLY "
              f"final (no optimizer state; weights+config preserved). The "
              f"gradient state is the only loss; re-run can complete it when "
              f"disk frees up.", flush=True)
        _final_state.pop("optimizer_full", None)
        _final_state["model_only"] = True
    else:
        _final_state["optimizer"] = optimizer.state_dict()
    safe_save(_final_state, final_path)
    # B-33b: NEVER clobber an existing ckpt_best.pt without an improvement.
    # Previously an early-stopped resume with no val improvement overwrote the
    # stored best (e.g. val 0.95) with the current worse weights. Save best only
    # when (a) val improved this run, or (b) the run completed normally.
    # B-35b: with --fresh_best the FIRST val check (from inf) would count as an
    # "improvement" and still clobber the original — route fresh_best runs to
    # ckpt_best_fresh.pt so the canonical ckpt_best.pt is never touched.
    _bst = best_val_state
    if _bst == "CACHED":
        _bst = _best_state_file_load(best_val_file)
    if _bst is not None and (best_val_improved
                             or (not early_stopped and not nan_aborted)):
        best_path = out_dir / ("ckpt_best_fresh.pt" if fresh_best else "ckpt_best.pt")
        try:
            safe_save({
                "model_state": _bst,
                "iteration": best_val_iter,
                "best_val_loss": best_val_ema,
                "best_loss": best_loss,
                "config": config,
                "total_tokens": total_tokens,  # P3: cumulative account
                "total_time": total_time,
            }, best_path)
            print(f"[Save] Best-val checkpoint -> {best_path}")
        except Exception as e:
            # ROUND-174 (N3): the ema/best cascade can still ENOSPC after the
            # main final save landed — FAIL LOUD, but NEVER let a secondary
            # save failure mask the already-complete ckpt.pt (the run's reward).
            print(f"[Save] WARN: best checkpoint save failed ({e}) — "
                  f"ckpt.pt already saved; not fatal", flush=True)
    if ema_state is not None:
        ema_path = out_dir / "ckpt_ema.pt"
        try:
            safe_save({
                "model_state": ema_state,
                "iteration": final_iter,
                "best_val_loss": best_val_ema,
                "best_loss": best_loss,
                "config": config,
            }, ema_path)
            print(f"[Save] EMA checkpoint (beta={config.ema_beta}) -> {ema_path}")
        except Exception as e:
            # ROUND-174 (N3): secondary save — never mask the completed ckpt.pt
            print(f"[Save] WARN: EMA checkpoint save failed ({e}) — ckpt.pt already saved; not fatal", flush=True)
    if nan_aborted:
        print("[Done] NaN-aborted — exiting with code 1 so the wrapper falls back")
        import sys as _sys
        _sys.exit(1)
    # 19th-audit (two-volume rule, materialized): when RSI_MIRROR_DIR is set,
    # copy the final archive + the best ckpt to the SECOND volume (root NVMe
    # when the run lives on the flaky SSD). Never fatal; skip loudly on space.
    _mirror_dir = os.environ.get("RSI_MIRROR_DIR", "").strip()
    if _mirror_dir:
        try:
            _md = Path(_mirror_dir)
            _md.mkdir(parents=True, exist_ok=True)
            for _src in (final_path, out_dir / "ckpt_best.pt"):
                if not Path(_src).exists():
                    continue
                try:
                    _need = Path(_src).stat().st_size
                    _free = _shutil_dbg.disk_usage(_md).free
                    if _free > _need + 512 * 1024 * 1024:
                        _shutil_dbg.copy2(_src, _md / Path(_src).name)
                        print(f"[Mirror] {Path(_src).name} -> {_md}", flush=True)
                    else:
                        print(f"[WARN][Mirror] {Path(_src).name}: only {_free//2**20}MB free "
                              f"on {_md} (needs {_need//2**20}MB) — skipping", flush=True)
                except OSError as e:
                    print(f"[WARN][Mirror] {Path(_src).name} copy failed: {e}", flush=True)
        except OSError as e:
            print(f"[WARN][Mirror] mirror dir {_mirror_dir} unusable: {e}", flush=True)
    print(f"\n[Done] Best loss: {best_loss:.4f} | Best val: {best_val_ema:.4f} | "
          f"Total tokens: {total_tokens:,} | Total time: {total_time:.1f}s | "
          f"Saved to {final_path}"
          + (f"\n[EarlyStop] {stop_reason}" if early_stopped else ""))
    model.eval()
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Ambatron-430M")
    parser.add_argument("--mode", choices=["full", "lora"], default="full",
                        help="full: train all params, lora: train only adapters")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_opt_resume", action="store_true",
                        help="Resume MODEL-ONLY (no optimizer state): ~1.9 GB less "
                             "VRAM, 500-iter LR re-warmup. For OOM resilience when "
                             "the GPU is partially occupied (B-33).")
    parser.add_argument("--cpu_corpus", action="store_true",
                        help="B-58: keep the corpus on CPU (frees ~0.9 GB VRAM — "
                             "lets the 444M fit the 5.64 GB card with its fp32 "
                             "optimizer; batches move per-step)")
    parser.add_argument("--fresh_best", action="store_true",
                        help="Start val best-tracking from inf on resume (B-35): "
                             "the seeded val split changes after corpus growth, so "
                             "the stored best-val is not comparable.")
    parser.add_argument("--fresh_start", action="store_true",
                        help="ROUND-204: allow fresh even when resume candidates "
                             "existed but all failed (drive-wedge protection)")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--size", choices=["430m", "mini", "plus", "xl", "v2", "v3"], default="430m",
                        help="430m: 444M (needs ~8.9B tokens — overfits our corpus). "
                             "mini: 43M, data-matched (B-38). plus: ~121M (B-44), the "
                             "phi-1-small-class step for the ~200M-token corpus. "
                             "v3: 587M 4E/2A MoE, 2048-ctx, 32K vocab (ROUND-204ac).")
    parser.add_argument("--curric", choices=["stable", "region_expand", "recovery"], default="",
                        help="ROUND-204ac LongRoPE-class staging for the v3 generation: "
                             "stable = the 1024-ctx base pack; region_expand = the 2048-ctx "
                             "region pool (CORPUS_FILE=v3_region2048.pt); recovery = the "
                             "short-context readjust leg back at 1024. Usage/block are "
                             "paired per leg (CORPUS_FILE + --block_size).")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16",
                        help="Model weight dtype (bf16 recommended on Ada; fp16 no scaler; fp32 uses GradScaler)")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=0,
                        help="Context length (default 1024; reduce to 512 for low VRAM)")
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--max_iters", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=0,
                        help="Checkpoint frequency (default 2000)")
    parser.add_argument("--resume_dir", type=str, default="",
                        help="Directory to load the resumed checkpoint from (defaults to out_dir)")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--val_patience", type=int, default=10,
                        help="Early stop after N val checks without EMA improvement (after warmup)")
    parser.add_argument("--min_early_iter", type=int, default=5000,
                        help="B-47: do not allow early-stop before this iteration "
                             "(fresh/resumed models plateau then improve; prevents "
                             "stops at ~0.1 tok/param)")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable convergence auto-stop")
    parser.add_argument("--ema_beta", type=float, default=0.8,
                        help="Post-hoc checkpoint EMA weight (0 disables; IMU-1 uses 0.8)")
    args = parser.parse_args()

    config = (create_ambatron_xl_config() if args.size == "xl" else
              create_ambatron_v2_config() if args.size == "v2" else
              create_ambatron_v3_config() if args.size == "v3" else
              create_ambatron_plus_config() if args.size == "plus" else
              create_ambatron_mini_config() if args.size == "mini" else
              create_ambatron_430m_config())
    if args.batch_size > 0:
        config.batch_size = args.batch_size
    if args.block_size > 0:
        config.block_size = args.block_size
    if args.lr > 0:
        config.learning_rate = args.lr  # Sets Muon base LR
    if args.max_iters > 0:
        config.max_iters = args.max_iters
    if args.save_interval > 0:
        config.save_interval = args.save_interval
    if args.out_dir:
        config.out_dir = args.out_dir
    config.gradient_accumulation_steps = args.grad_accum
    config.val_patience = args.val_patience
    config.early_stop = not args.no_early_stop
    config.ema_beta = args.ema_beta

    print(f"=== Ambatron-430M Training ===")
    print(f"Mode: {args.mode} | Batch: {config.batch_size} | Muon LR: {config.learning_rate} | "
          f"AdamW LR: {config.learning_rate * ADAMW_LR_RATIO:.6f} | Iters: {config.max_iters} | Accum: {args.grad_accum}")
    print(f"WSD: warmup={config.warmup_iters} stable={config.max_iters - config.warmup_iters - int(config.max_iters*0.1)} decay={int(config.max_iters*0.1)}")
    print(f"Early stop: {'ON' if config.early_stop else 'OFF'} | val patience: {config.val_patience} checks")

    all_tokens, lengths, vocab_size = load_code_corpus()
    config.vocab_size = vocab_size

    train(config, all_tokens, lengths, mode=args.mode, resume=args.resume,
          compile_model=not args.no_compile, dtype=args.dtype,
          resume_dir=args.resume_dir, resume_no_opt=args.no_opt_resume,
          fresh_best=args.fresh_best, min_early_iter=args.min_early_iter,
          cpu_corpus=args.cpu_corpus)


if __name__ == "__main__":
    main()
