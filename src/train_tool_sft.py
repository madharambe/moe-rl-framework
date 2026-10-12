"""
LoRA fine-tune Ambatron for tool use (function calling).
Loss is masked to ASSISTANT segments only (research: system/user/tool-result
tokens must not contribute); 20% corpus replay buffer prevents forgetting.

Usage:
  python3 src/train_tool_sft.py --base out/ambatron_430m_v2/ckpt_best.pt
                                --steps 600 --out out/ambatron_tool/ckpt.pt
"""
# B-118: this MUST be module-level executable code — it sat INSIDE the
# docstring for the entire project (before the closing quotes), so bare
# `python3 src/train_tool_sft.py` never enabled expandable_segments.
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from moe_model import (create_ambatron_430m_config, create_ambatron_mini_config,
                       create_ambatron_plus_config, create_ambatron_xl_config,
                       create_ambatron_v3_config, MoEGPT)
from lora_adapter import add_lora, freeze_base
from muon import muon_param_groups, Muon
from gpu_accel import resolve_device
from storage import safe_save

BASE = Path(__file__).resolve().parent.parent


def load_sft(path) -> list:
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def _tokenize(text, size=None):
    from tokenizers import Tokenizer
    # ROUND-204ac-b5.13: v3 SFT uses the 32768-vocab v3 tokenizer; everything
    # else keeps the 16K corpus tokenizer.
    if size == "v3":
        path = BASE / "v3-bank" / "tokenizer_v3.json"
    else:
        path = BASE / "data" / "processed" / "bpe_tokenizer" / "tokenizer.json"
    tok = Tokenizer.from_file(str(path))
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="430m", choices=["430m", "mini", "plus", "xl", "v3"],
                    help="base model size: 430m (444M), mini (43M), plus (~121M), "
                         "or v3 (587M 32K-vocab, ROUND-204ac; uses the v3 "
                         "tokenizer + the E1 pool by default)")
    ap.add_argument("--base", type=str, default="",  # "" -> auto-pick best
                    help="base checkpoint; defaults to ckpt_best.pt (best-val), "
                         "falling back to ckpt.pt, then ckpt_ema.pt (B-22: the "
                         "plain ckpt.pt is the early-stop artifact and can be "
                         "the diverged snapshot)")
    # 8th-wave B: the default must be the ROOT-side pool (the data/processed
    # path = the dead SSD symlink — FileNotFoundError on a bare launch).
    ap.add_argument("--sft_data", type=str, default="v2-bank/tool_sft_v2pool.jsonl")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mask_weight", type=float, default=1.0,
                    help="ROUND-204p: per-token CE weighting for the "
                         "assistant-role positions. The measured format-"
                         "collapse (4 cycles, 0%-emission at every final) is "
                         "consistent with the format-token gradient drowning "
                         "in the generic-token average — the masked CE gives "
                         "the call/answer tokens Wx the gradient. 3.0 = the "
                         "recommended first value.")
    ap.add_argument("--lr_decay", type=int, default=1,
                    help="19th-audit B8: linear LR decay to 10% over the run — "
                         "the flat 1e-4 drove the documented loss-0.1 overfit "
                         "cliff (0% tool-call emission) on earlier runs")
    ap.add_argument("--replay_frac", type=float, default=0.2,
                    help="fraction of batches drawn from the code corpus (replay)")
    ap.add_argument("--replay_tail_frac", type=float, default=0.5,
                    help="ROUND-204k: the collapse-fix. The measured cycles both"
                         " collapsed the acquired tool-format in the LR-decay TAIL"
                         " (the 2k->4k emission peak -> 0% at 6k). The prime"
                         " suspect: the 20%% pure-code replay injection at the"
                         " format-critical tail. Replay is DISABLED after this"
                         " fraction of the run (0.5 = replay for the first half"
                         " only). 0 accepts the old always-on replay.")
    ap.add_argument("--tail_data", type=str, default="",
                    help="ROUND-204m: FORMAT-REHEARSAL tail. The freeze alone "
                         "did NOT stop the collapse (measured: 6k-final still "
                         "28/29 no-call; a top_k-400 window doesn't recover the "
                         "mass). The drift survives the pool's own mixing — the "
                         "tail phase must sample ONLY the format exemplars "
                         "(the exec-verified records), so the model cannot "
                         "overwrite the call-distribution it keeps seeing.")
    ap.add_argument("--anchor_weight", type=float, default=10.0,
                    help="ROUND-174: sampling weight for tool-anchor/agentic records"
                         " (the flywheel's failure-harvested data). Anchors dilute to"
                         " ~0.25%% of the 5.6k-record pool at weight 1; 10x makes the"
                         " wrong-args curriculum actually reach the model.")
    ap.add_argument("--out", type=str, default="out/ambatron_tool/ckpt.pt")
    ap.add_argument("--resume", action="store_true",
                    help="B-49: resume from <out>.partial.pt (periodic LoRA ckpt) "
                         "if present — survives SSD unplugs via the retry wrapper")
    ap.add_argument("--lora_rank", type=int, default=16,
                    help="R411: LoRA rank (16 default; 64 = 4x capacity, still fits")
    ap.add_argument("--full_ft", action="store_true",
                    help="R410: FULL fine-tune (no LoRA): train every param with Muon/AdamW"
                         "\n instead of a 16-rank adapter. Needs ~2GB more VRAM than LoRA.")
    ap.add_argument("--ckpt_every", type=int, default=200,
                    help="periodic LoRA checkpoint interval (steps)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--deploy_align", action="store_true",
                    help="ROUND-204ac-b9.12: prepend the EXACT deployed preamble "
                         "(agentic tools section + tool-use instruction) to every "
                         "stream and anchor long-record windows on the LAST "
                         "assistant 'use tool:' segment — removes the train/serve "
                         "skew (honest probes: 0% tool-emission on every arm).")
    args = ap.parse_args()
    # ROUND-204ac-b5.13: v3-size E1 defaults (pool + tokenizer) — the caller can
    # still override --sft_data.
    if args.size == "v3" and args.sft_data == "v2-bank/tool_sft_v2pool.jsonl":
        args.sft_data = "v3-bank/e1_tool_sft.jsonl"

    import torch, os
    # B-210: hard per-process VRAM fraction BEFORE any CUDA context. This makes a
    # replay/oom-candidate batch FAIL with a Python OOM error instead of blowing
    # past the budget and either (a) tripping the flywheel's VRAM floor -> the
    # candidate is killed, or (b) worse, OOM-ing the display GPU -> GSP crash ->
    # hard reboot. The trainer is constrained to ~85% of the card; on the 6GB
    # 4050 that is ~4.9GB, leaving the desktop + safety always resident.
    try:
        _cap = float(os.environ.get("RSI_VRAM_FRACTION", "0.85"))
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(min(max(_cap, 0.2), 0.95))
    except Exception:
        pass

    device = resolve_device(args.device if args.device in ("cuda", "cpu") else "cuda")
    torch.manual_seed(0)

    tok = _tokenize("", args.size)
    if not args.base:
        # Same preference order as LLMPlanner: best-val > improved > last > EMA
        for cand in ("out/ambatron_430m_v2/ckpt_best.pt",
                     "out/ambatron_430m_v2/ckpt_improved.pt",
                     "out/ambatron_430m_v2/ckpt.pt",
                     "out/ambatron_430m_v2/ckpt_ema.pt"):
            if Path(cand).exists():
                args.base = cand
                print(f"[ToolSFT] auto-picked base: {args.base}")
                break
    if not (args.base and Path(args.base).exists()):
        import sys as _sys
        print(f"[ToolSFT] FATAL: --base {args.base or '(auto)'} not found — refusing to "
              f"train from random init", file=sys.stderr)
        _sys.exit(1)
    import torch.serialization as _ser
    from moe_model import MoEConfig as _Cfg
    _ser.add_safe_globals([_Cfg])
    ck = torch.load(args.base, map_location="cpu", weights_only=True)
    # B-XXX: build the arch from the CHECKPOINT's own config (not a hardcoded
    # factory). Otherwise a non-430m champion is silently randinit'd via the
    # strict=False below — the tool/spatial/chat "from-champion" training loop
    # wastes the whole cycle dropping its gains.
    cfg = ck.get("config")
    config = (cfg if cfg is not None else
              (create_ambatron_v3_config() if args.size == "v3" else
               create_ambatron_xl_config() if args.size == "xl" else
               create_ambatron_plus_config() if args.size == "plus" else
               create_ambatron_mini_config() if args.size == "mini" else
               create_ambatron_430m_config()))
    model = MoEGPT(config).to(device)
    state = ck.get("model_state", ck.get("model", {}))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ROUND-174: _model_sd pops the TIED lm_head.weight from every save —
    # MoEGPT re-uses the embed as the head at init, so it's never in the ckpt.
    # The old check treated that BY-DESIGN absence as a config mismatch and
    # rejected every final archive (ckpt.pt / ckpt_best of the full run).
    _real_missing = [k for k in missing if k != "lm_head.weight"]
    if _real_missing:
        import sys as _sys
        print(f"[ToolSFT] FATAL: base {args.base} missing arch keys "
              f"(config mismatch): {_real_missing}", file=_sys.stderr)
        _sys.exit(1)
    print(f"[ToolSFT] base loaded: {len(missing)} missing (incl. tied head), "
          f"{len(unexpected)} unexpected")

    # B-213: the base-ckpt dict (~900MB bf16 CPU) is only needed for
    # load_state_dict; load_state_dict COPIES into the CUDA tensors, so the CPU
    # source stays resident the whole run (at-home this meant the trainer RSS
    # pinned host RAM into the flywheel's abort zone). Drop it now.
    del ck, state
    import gc as _gc
    _gc.collect()

    # B-215: DoRA (arXiv:2402.09353, ICML'24) — weight-decomposed LoRA beats plain
    # LoRA on every rank and ~all tasks (magnitude + direction decomposed; W' =
    # m·(W0+ΔW)/||W0+ΔW||_c). Default ON; the flywheel's +1 honest gate still
    # rejects any candidate that doesn't actually improve, so a regression is
    # self-correcting. Flip RSI_DORA=0 to go back to plain LoRA.
    use_dora = os.environ.get("RSI_DORA", "1").strip() not in ("0", "false", "no")
    if args.full_ft:
        # R410: FULL fine-tune mode - no adapters, no frozen base; every
        # param trains with Muon (2D) / AdamW (1D). The save path skips the
        # LoRA merge (there is none) and writes plain model_state.
        # R411 MEASURED: the optimizer states alone (~2.8GB) + weights/grads
        # (2.4GB) sit at this card ceiling - full-ft fails at any seq length,
        # so this mode is for a larger GPU.
        print("[ToolSFT] FULL fine-tune mode (no LoRA, all params train)")
    else:
        # R411: rank is now configurable - r16 measured flat for chat;
        # r64 is the executable middle path (4x adapter capacity, still
        # ~300MB of states, fits the card).
        add_lora(model, r=args.lora_rank, alpha=max(32, args.lora_rank * 2),
                 dora=use_dora)
        freeze_base(model)
    if use_dora and not args.full_ft:
        print(f"[ToolSFT] DoRA adapters (m·norm-decomposed) active")
    if device.type == "cuda":
        from lora_adapter import LoRALinear
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                mod.base.to(dtype=torch.bfloat16)  # A/B stay fp32 masters
            elif isinstance(mod, torch.nn.Linear) and mod is not getattr(model, "lm_head", None):
                mod.to(dtype=torch.bfloat16)
        if hasattr(model, "token_embed"):
            model.token_embed.to(dtype=torch.bfloat16)
        if hasattr(model, "norm"):
            model.norm.to(dtype=torch.bfloat16)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ToolSFT] LoRA trainable: {n_trainable:,} params")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    # 8th-wave B: the startup disk gate (the golden rule "storage-check-before
    # training" materialized — the fail-closed gate lived only in auto_optimize)
    import shutil as _shdisk
    _out_parent = Path(args.out).resolve().parent
    # 204t: the gate must CREATE the out-parent first (the trainer only mkdirs
    # at the final — the earlier statvfs of a not-yet-existing path fatally
    # crashed the launch on the "flaky-SSD" red herring).
    _out_parent.mkdir(parents=True, exist_ok=True)
    _disk_free = _shdisk.disk_usage(_out_parent).free / (1024 ** 3)
    if _disk_free < 3.0:
        print(f"[ToolSFT] FATAL: only {_disk_free:.1f}G free on {_out_parent} "
              f"(need >= 3.0G for the partial + final + snapshots) — refusing "
              f"to launch into the ENOSPC class")
        import sys as _sd
        _sd.exit(2)
    records = [r for r in load_sft(args.sft_data) if r.get("segments")]
    if len(records) != sum(1 for _ in open(args.sft_data)):
        print(f"[ToolSFT] WARN: {sum(1 for _ in open(args.sft_data)) - len(records)} "
              f"segment-less records skipped (the 6th-wave P1 crash-class)")
    # ROUND-174 (anchor upsampling): weight records by data quality — the
    # flywheel's tool-anchor / agentic records (execution-verified) get
    # args.anchor_weight; ordinary ReST/compiled records get 1. Prevents the
    # ~0.25%% anchor dilution from starving the wrong-args curriculum.
    _anchor_markers = ("tool-anchor", "agentic")
    # 19th-audit B-8: the v2 bank (auto_tool_train_v2.jsonl) carries
    # exec-verified records tagged "verified": True — they have no "mode"
    # field, so anchor_weight was inert on the highest-quality data. Treat
    # the verification marker as an anchor too. (mode is deliberately not
    # added to v2 records: their schema is the scale_tool_sft emission.)
    rec_weights = torch.tensor(
        [args.anchor_weight
         if (any(m in (r.get("mode") or "") for m in _anchor_markers)
             or bool(r.get("verified")))
         else 1.0 for r in records], dtype=torch.float64)
    # corpus replay — M3: don't eagerly pin the whole 0.43GB corpus in host RAM
    # on every training run (the spatial/chat candidates barely use replay).
    # Load lazily + mmap so pages stay in the SSD-backed page cache, not process
    # RAM (which starves Minecraft on this 14GB box).
    corpus_path = BASE / "data" / "processed" / "code_corpus_bpe.pt"
    corpus = None
    _corpus_done = False
    def get_corpus():
        nonlocal corpus, _corpus_done
        if _corpus_done:
            return corpus
        _corpus_done = True
        try:
            ck = torch.load(corpus_path, map_location="cpu", weights_only=True,
                            mmap=True)
            corpus = ck["tokens"]
        except Exception:
            corpus = None
        return corpus
    if args.replay_frac > 0 and corpus_path.exists():
        get_corpus()
    # 204m: the format-rehearsal tail set + its weights (verified=10x)
    tail_records = None
    _tail_weights = None
    if args.tail_data and Path(args.tail_data).exists():
        tail_records = [r for r in load_sft(args.tail_data) if r.get("segments")]
        _tail_weights = torch.tensor(
            [args.anchor_weight if (any(m in (r.get("mode") or "") for m in _anchor_markers)
                                    or bool(r.get("verified"))) else 1.0
             for r in tail_records], dtype=torch.float64)
        print(f"[ToolSFT] format-rehearsal tail: {len(tail_records)} records")
    print(f"[ToolSFT] {len(records)} conversations; replay corpus: "
          f"{'yes' if corpus is not None else 'NO (missing/skipped)'}")

    preamble_ids = None
    if args.deploy_align:
        from agentic_tools import agentic_registry
        from tool_use import TOOL_CALL_START
        _sec = agentic_registry().get_tools_section(compiled=False)
        _pre = (f"{_sec}\n\nTo use a tool, respond with:\n"
                f"{TOOL_CALL_START}{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n\n")
        preamble_ids = tok.encode(_pre).ids
        print(f"[ToolSFT] deploy-aligned preamble: {len(preamble_ids)} tokens")

    def encode_conversation(rec, block_size, preamble_ids=None):
        """Token stream + assistant mask for NEXT-TOKEN prediction (1 = predict
        the token at THIS position's target). Targets are shifted: x=toks[:-1],
        y=toks[1:], and the mask shifts with y (loss on positions whose TARGET
        is assistant content). Bug-11 fix: y=x.clone() taught a copy task.
        b9.11: marker prepend is IDEMPOTENT (v3 E1 pools already carry markers).
        b9.12: deploy alignment — optional EXACT deployed preamble prepended +
        tool-call-aware window: the blind tail cut retained only 25-40% of
        'use tool:' calls (trajectories 56-146 segs); anchor the window on the
        LAST assistant tool-call segment when the tail would drop every call."""
        toks = list(preamble_ids) if preamble_ids is not None else []
        mask = [0] * len(toks)
        seg_starts = []
        last_tc = None
        first_user_end = 0
        for role, text in rec["segments"]:
            if role == "user":
                if not text.startswith("User:") and not text.startswith("user:"):
                    text = "User: " + text
                text += "\n"
            elif role == "assistant":
                if not text.startswith("Assistant:") and not text.startswith("assistant:"):
                    text = "Assistant: " + text
                text += "\n"
            ids = tok.encode(text).ids
            seg_starts.append(len(toks))
            toks.extend(ids)
            mask.extend([1 if role == "assistant" else 0] * len(ids))
            if role == "user" and first_user_end == 0:
                first_user_end = len(toks)
            if role == "assistant" and "use tool:" in text:
                last_tc = seg_starts[-1]
        if len(toks) < 3:
            return None, None
        if len(toks) <= block_size:
            return toks, mask
        # b16.7 (audit fix): BOTH long-record branches must keep the deployed
        # preamble + the first user query (the old window dropped them — the
        # model saw only [call, result, answer], a conditioning that never
        # occurs at serving; measured on the E1 pool: 111 anchor + 1596 tail
        # records). head = preamble + first user query, capped at block-512;
        # the tail = the newest tokens / the tool-call-anchored slice.
        head_lim = block_size - 512
        head = toks[:min(first_user_end, head_lim)] if first_user_end > 0 else []
        mask_head = mask[:len(head)]
        if not head:
            return toks[-block_size:], mask[-block_size:]
        tail_len = block_size - len(head)
        tail_off = len(toks) - tail_len
        if last_tc is not None and last_tc >= tail_off:
            return head + toks[tail_off:], mask_head + mask[tail_off:]
        if last_tc is not None:
            return (head + toks[last_tc:last_tc + tail_len],
                    mask_head + mask[last_tc:last_tc + tail_len])
        return head + toks[tail_off:], mask_head + mask[tail_off:]

    block = config.block_size
    partial_path = Path(args.out).with_name(Path(args.out).name + ".partial.pt")
    start_step = 0
    # 19th-audit B16: the resume partial now carries a POOL FINGERPRINT — a
    # mutated training pool under a --resume silently changes the curriculum
    # (the flywheel's live harvesters rewrite e1_tool_sft*.jsonl continuously).
    def _pool_fp() -> str:
        import hashlib, os as _os
        h = hashlib.sha256()
        for p in (args.sft_data, args.tail_data):
            try:
                st = _os.stat(p)
                h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
            except OSError:
                h.update(f"{p}:missing".encode())
        return h.hexdigest()[:16]
    _cur_fp = _pool_fp()
    if args.resume and partial_path.exists():
        try:
            # 205a-fix: weights_only=True can NEVER unpickle our partials
            # (config/args class refs) -> every --resume silently fell back to
            # fresh. Partial is written by safe_save on this same machine; trust it.
            pk = torch.load(partial_path, map_location="cpu", weights_only=False)
            model.load_state_dict(pk["model_state"], strict=False)
            optimizer.load_state_dict(pk["optimizer"])
            start_step = pk.get("step", 0) + 1
            # 19th-audit B7: restore the RNG so the resumed draws differ
            if "rng_state" in pk and pk["rng_state"] is not None:
                torch.set_rng_state(pk["rng_state"].cpu())
            print(f"[ToolSFT] resumed from {partial_path} at step {start_step}")
            _old_fp = pk.get("pool_fp")
            if _old_fp and _old_fp != _cur_fp:
                print(f"[ToolSFT] WARN: pool fingerprint CHANGED since the "
                      f"partial ({_old_fp} -> {_cur_fp}) — the training pool "
                      f"mutated under the resume; the curriculum shifted")
        except Exception as e:
            print(f"[ToolSFT] resume failed ({e}) — starting fresh")
    if start_step == 0:
        torch.manual_seed(0)

    for step in range(start_step, args.steps):
        # 19th-audit B8: monotone LR decay (no re-warm on resume — the
        # schedule is step-anchored; a resume continues the decay).
        if args.lr_decay and args.steps > 1:
            optimizer.param_groups[0]["lr"] = args.lr * max(
                0.1, 1.0 - step / args.steps)
        # corpus replay every 5 steps (~20% of batches when replay_frac=0.2)
        # ROUND-204k: freeze the replay in the LR-decay tail (format-critical
        # window). The pure-code batches at low LR drift the fresh tool-format
        # out — the measured 4k-peak -> 6k-collapse curve in BOTH cycles.
        _in_tail = step / max(args.steps, 1) > args.replay_tail_frac
        use_replay = corpus is not None and not _in_tail \
            and torch.rand(1).item() < args.replay_frac
        batch_t, batch_m = [], []
        if _in_tail and tail_records:
            # 204m: FORMAT-REHEARSAL — the tail draws only the format
            # exemplars (the mixed-pool drift measured at the 6k, and the
            # top_k-window test rules out a sampling artifact).
            use_replay = False
        if _in_tail and tail_records:
            idx = torch.multinomial(_tail_weights, args.batch_size, replacement=True)
            for _i in idx.tolist():
                rec = tail_records[_i]
                toks, mask = encode_conversation(rec, block, preamble_ids)
                if toks is None:
                    continue
                batch_t.append(toks)
                batch_m.append(mask)
        elif use_replay:
            idx = torch.randint(0, corpus.shape[0], (args.batch_size,))
            for i in idx.tolist():
                row = corpus[i]
                nnz = int((row != 0).sum())
                if nnz < 8:
                    continue
                toks = row[:nnz].tolist()
                # B-210: cap replay rows to 512 tokens (half the model's 1024
                # block) — full-block batch 4 corpus rows spike VRAM ~2x (the
                # observed near-floor blip in auto_optimize training) and their
                # long tails add little replay signal. Conversation batches are
                # naturally shorter; only corpus replay hits full-block length.
                _cap = 384 if (args.lora_rank > 16 or args.full_ft) else 512
                batch_t.append(toks[:_cap])
                batch_m.append([1] * min(len(toks[:_cap]), _cap))
        else:
            # ROUND-174: weighted sampling (anchors/agentic up-weighted) —
            # uniform randint drowned the failure-harvested data.
            idx = torch.multinomial(rec_weights, args.batch_size, replacement=True)
            for _i in idx.tolist():
                rec = records[_i]
                toks, mask = encode_conversation(rec, block, preamble_ids)
                if toks is None:
                    continue
                batch_t.append(toks)
                batch_m.append(mask)
        if not batch_t:
            continue
        max_len = max(len(t) for t in batch_t)
        x = torch.zeros(len(batch_t), max_len, dtype=torch.long, device=device)
        pm = torch.zeros(len(batch_t), max_len, dtype=torch.bool, device=device)
        for i, (t, m) in enumerate(zip(batch_t, batch_m)):
            x[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            pm[i, :len(t)] = torch.tensor([bool(v) for v in m], dtype=torch.bool)
        # shifted targets: predict the NEXT token; mask aligns with y
        y = x[:, 1:].clone()
        x = x[:, :-1]
        pm = pm[:, 1:]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # ROUND-204p: the WEIGHTED masked CE — the model's internal loss
        # averages every non-pad token equally; the assistant-role positions
        # (the call JSON + the answers) get mask_weight x on the gradient.
        import torch.nn.functional as _F
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, aux = model(x, targets=y, use_grad_ckpt=True, pad_mask=pm)
        else:
            logits, _, aux = model(x, targets=y, use_grad_ckpt=True, pad_mask=pm)
        _ce = _F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        _m = pm.reshape(-1).float()
        _w = torch.where(_m > 0, torch.tensor(args.mask_weight, device=_m.device),
                         torch.ones_like(_m))
        loss = (_ce * _m * _w).sum() / _m.sum().clamp_min(1.0)
        # 7th-wave B3 + 8th-wave P0: the MoE regularization terms re-attached
        # with their config coefficients (a dict-sum of keys crashed the 204r
        # runs; the aux floats were graph-detached — now raw tensors).
        if isinstance(aux, dict) and aux.get("balance_loss") is not None:
            _bl = aux.get("balance_loss", 0.0)
            _zl = aux.get("z_loss", 0.0)
            if getattr(_bl, "requires_grad", False) and getattr(_zl, "requires_grad", False):
                loss = (loss
                        + config.load_balance_coef * _bl
                        + config.z_loss_coef * _zl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"[ToolSFT] step {step} | loss {loss.item():.4f} | "
                  f"replay={use_replay} | n_tok {int(pm.sum())}")
        if args.ckpt_every > 0 and step % args.ckpt_every == 0 and step > start_step:
            # B-49: periodic LoRA checkpoint so an SSD unplug / kill resumes here
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            # 19th-audit B7: persist the RNG state so a --resume replays a
            # DIFFERENT draw sequence (the old seed-0 replay compounded the
            # pool's replication — the same ordering every resume).
            _rng_state = torch.get_rng_state().clone()
            safe_save({"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "step": step, "config": config, "args": vars(args),
                       "rng_state": _rng_state, "pool_fp": _cur_fp}, partial_path)
            # 7th-wave B1-fix: the 204m CPU-merge produced GARBAGE milestones
            # (the _flat held LoRA-wrapped keys; the plain type(model)(config)
            # matched ZERO linears; merge/remove no-oped -> fresh-init weights,
            # probed std 0.02 vs the base's 0.00096). The milestone now saves
            # the SMALL LoRA adapter snapshot (~150MB vs 3.04G): the merge
            # happens ON DEMAND at the election (tools/merge_lora_ckpt.py),
            # and the SSD budget breathes.
            _mp = Path(args.out).with_name(
                Path(args.out).name + f".step{step}.lora.pt")
            safe_save({"lora_state": model.state_dict(),
                       "config": config, "args": vars(args)}, _mp)
            print(f"[ToolSFT] milestone LoRA snapshot -> {_mp.name}", flush=True)

    if not args.full_ft:
        from lora_adapter import merge_lora, remove_lora
        merge_lora(model)   # fuse A/B into base
        remove_lora(model)  # strip the LoRALinear wrappers -> plain MoEGPT
    # R410: full_ft skips the merge - the model IS the full state already.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from storage import ensure_disk as _ed
    _ed(out_path, 3.0, context="sft final merge")
    safe_save({"model_state": model.state_dict(), "config": config,
               "args": vars(args)}, out_path)
    partial_path.unlink(missing_ok=True)  # success — drop the partial
    print(f"[ToolSFT] done (LoRA merged) -> {out_path}")


if __name__ == "__main__":
    main()
