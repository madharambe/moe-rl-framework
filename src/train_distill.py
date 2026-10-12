"""
Distillation trainer — Ambatron-370M-XL student distilled from a frozen
430M teacher via soft-target KL + hard-target cross-entropy (Hinton et al.
2015 / classic KD). Mirrors train_ambatron.py's WSD schedule, Muon+AdamW
optimizer, checkpointing, and signal-trap patterns.
"""
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, math, time, json, signal, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
import torch.serialization as _ser
from pathlib import Path
from moe_model import MoEGPT, MoEConfig, create_ambatron_xl_config, create_ambatron_430m_config
from muon import Muon, muon_param_groups
from gpu_accel import resolve_device
from storage import safe_save, safe_load


def _model_sd(model):
    dev = next(model.parameters()).device
    model.to("cpu")
    sd = model.state_dict()
    sd.pop("lm_head.weight", None)
    model.to(dev)
    return sd


def load_code_corpus(data_dir="data/processed", corpus_file=None, cpu_corpus=False):
    name = corpus_file or os.environ.get("CORPUS_FILE", "code_corpus_curated.pt")
    pt_path = Path(data_dir) / name
    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found. Run build_bpe_tokenizer.py / "
              f"curate_corpus.py first (CORPUS_FILE={name}).")
        sys.exit(1)
    data = safe_load(pt_path, map_location="cpu")
    tokens = data["tokens"]
    lengths = data["lengths"]
    vocab_size = data["vocab_size"]
    print(f"[Data] {len(tokens)} samples, {tokens.shape[1]} ctx, vocab={vocab_size} "
          f"(corpus: {name})")
    return tokens, lengths, vocab_size


def get_lr_multiplier(iteration, warmup_iters, total_iters, decay_frac=0.1):
    if iteration < warmup_iters:
        return iteration / max(warmup_iters, 1)
    decay_start = int(total_iters * (1 - decay_frac))
    if iteration >= decay_start:
        t = (iteration - decay_start) / max(total_iters - decay_start, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return 1.0


def compute_accuracy(logits, targets, pad_mask=None):
    preds = logits.argmax(dim=-1)
    if pad_mask is None:
        return (preds == targets).float().mean().item()
    correct = ((preds == targets) & pad_mask).sum().item()
    return correct / max(pad_mask.sum().item(), 1)


MUON_BASE_LR = 0.0235
ADAMW_LR_RATIO = 3e-4 / 0.02


def distill_loss(student_logits, teacher_logits, targets, pad_mask, T, alpha, vocab_size):
    """alpha * KL(student/T || teacher/T) * T^2 + (1-alpha) * CE(student, targets).
    KL computed over the PAD-masked tokens only (non-pad)."""
    B, S = targets.shape
    if pad_mask is None:
        pad_mask = torch.ones_like(targets, dtype=torch.bool)

    flat_mask = pad_mask.reshape(-1).to(student_logits.dtype)
    denom = flat_mask.sum().clamp_min(1)

    s_logits = student_logits.reshape(-1, vocab_size)
    t_logits = teacher_logits.reshape(-1, vocab_size)

    s_logs = F.log_softmax(s_logits / T, dim=-1)
    t_probs = F.softmax(t_logits / T, dim=-1)

    kl_per_token = F.kl_div(s_logs, t_probs, reduction="none").sum(dim=-1)
    kl_loss = (kl_per_token * flat_mask).sum() / denom * (T * T)

    ce_loss = F.cross_entropy(s_logits, targets.reshape(-1), reduction="none")
    ce_loss = (ce_loss * flat_mask).sum() / denom

    loss = alpha * kl_loss + (1.0 - alpha) * ce_loss
    return loss, kl_loss, ce_loss


def train(config, teacher_config, teacher, all_tokens, lengths, teacher_ckpt,
          T=2.0, alpha=0.5, resume=False, compile_model=False, resume_no_opt=False,
          cpu_corpus=False):
    device = resolve_device()
    model = MoEGPT(config)
    model.to(device)
    model = model.to(dtype=torch.bfloat16)
    model.train()
    teacher.to(device)
    teacher = teacher.to(dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    n_t_params = sum(p.numel() for p in teacher.parameters())
    print(f"[Model] Student 370M-XL: {n_params:,} total params ({n_params/1e6:.1f}M)")
    print(f"[Model] Teacher 430M:   {n_t_params:,} total params ({n_t_params/1e6:.1f}M, frozen)")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            print("[Compile] Student compiled")
        except Exception as e:
            print(f"[Compile] Failed: {e}")

    n_samples = len(all_tokens)
    n_val = max(1, int(n_samples * 0.05))
    rng = torch.Generator().manual_seed(1337)
    perm = torch.randperm(n_samples, generator=rng)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    if cpu_corpus:
        train_tokens = all_tokens[train_idx]
        val_tokens = all_tokens[val_idx]
        val_lengths = lengths[val_idx] if lengths is not None else None
        train_lengths = lengths[train_idx] if lengths is not None else None
    else:
        train_tokens = all_tokens[train_idx].to(device, non_blocking=True)
        val_tokens = all_tokens[val_idx].to(device, non_blocking=True)
        val_lengths = lengths[val_idx].to(device) if lengths is not None else None
        train_lengths = lengths[train_idx].to(device) if lengths is not None else None
    print(f"[Data] {len(train_idx)} train + {n_val} val samples (seeded shuffle)")

    n_val_batches = 32
    val_gen = torch.Generator().manual_seed(2025)
    val_batches = [torch.randint(0, n_val, (config.batch_size,), generator=val_gen)
                   for _ in range(n_val_batches)]

    start_iter = 0
    resumed_without_opt = False
    best_loss = float("inf")
    total_tokens = 0
    total_time = 0.0
    nan_count = 0
    current_loss = float("inf")
    best_val_ema = float("inf")
    best_val_raw = float("inf")
    val_patience_count = 0
    best_val_state = None
    best_val_improved = False
    nan_aborted = False
    early_stopped = False
    stop_reason = ""

    cfg_for_resume = None
    loaded_ckpt = None
    if resume:
        latest = [p for p in out_dir.glob("ckpt_*.pt")
                  if p.stem.split("_")[-1].isdigit()]
        latest = sorted(latest, key=lambda p: int(p.stem.split("_")[-1]))
        if resume_no_opt:
            latest += [out_dir / "ckpt.pt"]
            latest = [p for p in latest if p.exists()]
            latest.reverse()
            latest = ([out_dir / "ckpt_resume.pt",
                       out_dir / "ckpt_latest.pt"] + latest
                      + [out_dir / "ckpt_best.pt"])
        else:
            latest += [out_dir / "ckpt.pt"]
            latest = [p for p in latest if p.exists()]
            latest.reverse()
        latest = [p for p in latest if p.exists()]

        _ser.add_safe_globals([MoEConfig])
        loaded = False
        for ckpt_file in latest:
            try:
                ckpt = safe_load(ckpt_file, map_location=device)
            except (torch.serialization.pickle.UnpicklingError, EOFError) as e:
                print(f"[Resume] Corrupt checkpoint {ckpt_file}: {e} — skipping")
                ckpt_file.unlink(missing_ok=True)
                continue
            except OSError as e:
                print(f"[Resume] Storage error reading {ckpt_file}: {e} — KEEPING file")
                continue
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"], strict=False)
            if "iteration" in ckpt:
                start_iter = ckpt["iteration"] + 1
            if "total_tokens" in ckpt:
                total_tokens = ckpt["total_tokens"]
            if "best_loss" in ckpt:
                best_loss = ckpt["best_loss"]
            if "best_val_loss" in ckpt:
                best_val_ema = ckpt["best_val_loss"]
                best_val_raw = best_val_ema
            print(f"[Resume] Loaded {ckpt_file}: iter {start_iter - 1}, best_loss={best_loss:.4f}, best_val={best_val_ema:.4f}")
            loaded = True
            loaded_ckpt = ckpt
            break
        if not loaded:
            print("[Resume] No readable checkpoint found — starting from scratch")

    param_groups = muon_param_groups(model, lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    optimizer = Muon(param_groups, lr=config.learning_rate,
                     weight_decay=config.weight_decay)

    if loaded_ckpt is not None and "optimizer" in loaded_ckpt and not resume_no_opt:
        optimizer.load_state_dict(loaded_ckpt["optimizer"])
        for pg in optimizer.param_groups:
            if pg.get("mode") == "adamw":
                pg["weight_decay"] = config.weight_decay
            else:
                pg["weight_decay"] = min(config.weight_decay, 0.01)
        print("[Resume] Optimizer state loaded (weight_decay re-applied)")
    elif start_iter > 0:
        resumed_without_opt = True
        print("[Resume] No optimizer state — re-warming LR for 500 iters")

    log_file = out_dir / "log.jsonl"
    accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

    _resume_saved = False
    _ckpt_latest_path = out_dir / "ckpt_latest.pt"
    _ckpt_resume_path = out_dir / "ckpt_resume.pt"

    def _save_ckpt_latest():
        safe_save({"model_state": _model_sd(model), "iteration": iteration,
                    "best_loss": best_loss, "total_tokens": total_tokens,
                    "best_val_loss": best_val_ema}, _ckpt_latest_path)

    def _signal_exit(signum=None, frame=None):
        nonlocal _resume_saved
        if _resume_saved:
            return
        _resume_saved = True
        try:
            _save_ckpt_latest()
            safe_save({"model_state": _model_sd(model), "iteration": iteration,
                        "best_loss": best_loss, "total_tokens": total_tokens,
                        "best_val_loss": best_val_ema}, _ckpt_resume_path)
            print(f"\n[Signal] Caught signal {signum}, saved resume checkpoint at iter {iteration}", flush=True)
        except Exception as e:
            print(f"\n[Signal] Caught signal {signum}, resume save failed: {e}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_exit)
    signal.signal(signal.SIGINT, _signal_exit)

    for iteration in range(start_iter, config.max_iters):
        iter_start = time.time()
        grad_norm = None

        mult = get_lr_multiplier(iteration, config.warmup_iters, config.max_iters, decay_frac=0.1)
        _resume_gap = iteration - start_iter
        if resumed_without_opt and _resume_gap >= 200 and mult >= 1.0:
            ramp = 0.1 + 0.9 * min(1.0, (_resume_gap - 199) / 501)
            mult *= ramp
        muon_base = config.learning_rate if config.learning_rate > 0 else MUON_BASE_LR
        muon_lr = muon_base * max(mult, 0.01)
        adamw_lr = muon_base * ADAMW_LR_RATIO * max(mult, 0.01)
        for pg in optimizer.param_groups:
            if pg["mode"] == "adamw":
                pg["adamw_lr"] = adamw_lr
            else:
                pg["muon_lr"] = muon_lr

        batch_indices = torch.randint(0, len(train_idx), (config.batch_size,),
                                      device="cpu" if cpu_corpus else device)
        batch_tokens = train_tokens[batch_indices]
        if cpu_corpus:
            batch_tokens = batch_tokens.to(device, non_blocking=True)
        ctx = min(config.block_size, batch_tokens.shape[1])
        batch_tokens = batch_tokens[:, :ctx]

        x = batch_tokens[:, :-1].contiguous().long()
        y = batch_tokens[:, 1:].contiguous().long()
        pad_mask = (x != 0) & (y != 0)

        def _forward_both():
            s_logits, _, _ = model(x, use_grad_ckpt=True)
            with torch.no_grad():
                t_logits, _, _ = teacher(x, use_grad_ckpt=True)
            return s_logits, t_logits

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_logits, t_logits = _forward_both()
            loss, kl_loss, ce_loss = distill_loss(
                s_logits.float(), t_logits.float(), y, pad_mask,
                T=T, alpha=alpha, vocab_size=config.vocab_size)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count > 10:
                print(f"[NaN] Too many NaN losses ({nan_count}), aborting")
                nan_aborted = True
                break
            print(f"[NaN] Loss is NaN/Inf at iter {iteration}, skipping (count={nan_count})")
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / accumulation_steps).backward()
        if (iteration + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_tokens += int(pad_mask.sum())
        iter_time = time.time() - iter_start
        total_time += iter_time
        current_loss = loss.item()
        best_loss = min(best_loss, current_loss)
        acc = compute_accuracy(s_logits, y, pad_mask)

        if iteration % 100 == 0 and iteration > 0 \
                and iteration % config.save_interval != 0:
            _save_ckpt_latest()

        if iteration % config.log_interval == 0:
            tokens_per_sec = x.numel() / max(iter_time, 1e-8)
            gn_str = (f"GradNorm {grad_norm.item():.2f} | " if grad_norm is not None else "")
            print(f"Iter {iteration:6d}/{config.max_iters} | Loss {current_loss:.4f} | "
                  f"KL {kl_loss.item():.4f} | CE {ce_loss.item():.4f} | Acc {acc:.3f} | "
                  f"LR {muon_lr:.2e} | {gn_str}Tok/s {tokens_per_sec:.0f} | Total {total_tokens:,}", flush=True)

            if iteration % 100 == 0 and iteration > 0:
                model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for vb in val_batches:
                        val_batch = val_tokens[vb.cpu() if cpu_corpus else vb, :ctx]
                        if cpu_corpus:
                            val_batch = val_batch.to(device, non_blocking=True)
                        vx = val_batch[:, :-1].contiguous().long()
                        vy = val_batch[:, 1:].contiguous().long()
                        vmask = (vx != 0) & (vy != 0)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            vsl, _, _ = model(vx, use_grad_ckpt=True)
                            _, vce, _ = distill_loss(
                                vsl.float(), vsl.float(), vy, vmask,
                                T=1.0, alpha=0.0, vocab_size=config.vocab_size)
                        val_loss_sum += vce.item()
                    val_loss = val_loss_sum / len(val_batches)
                model.train()
                print(f"         Val loss {val_loss:.4f} | Best {best_loss:.4f} | "
                      f"BestVal {best_val_ema:.4f} | Patience {val_patience_count}", flush=True)

                # 11th-audit P4: track EMA/best/patience ALWAYS (not just under the
                # early-stop gate) so ckpt_best refreshes on real improvement.
                ema = val_loss if best_val_ema == float("inf") else 0.9 * best_val_ema + 0.1 * val_loss
                best_val_raw = min(best_val_raw, val_loss)
                if ema < best_val_ema - 1e-4:
                    best_val_ema = ema
                    val_patience_count = 0
                    best_val_improved = True
                    best_val_state = _model_sd(model)  # CPU, tied-head stripped (P1/P4)
                else:
                    val_patience_count += 1
                if config.early_stop and iteration > max(config.warmup_iters, 5000):
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
                val_loss = None

            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "iter": iteration, "loss": current_loss, "val_loss": val_loss,
                    "acc": acc, "kl_loss": kl_loss.item(), "ce_loss": ce_loss.item(),
                    "muon_lr": muon_lr, "adamw_lr": adamw_lr,
                    "mult": mult, "grad_norm": grad_norm.item() if grad_norm is not None else None,
                    "tokens_per_sec": tokens_per_sec, "total_tokens": total_tokens,
                    "best_loss": best_loss,
                }) + "\n")

        if iteration % config.save_interval == 0 and iteration > 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ckpt_path = out_dir / f"ckpt_{iteration}.pt"
            state = {
                "model_state": _model_sd(model),
                "iteration": iteration,
                "loss": current_loss,
                "best_loss": best_loss,
                "best_val_loss": best_val_ema,
                "total_tokens": total_tokens,
                "total_time": total_time,
            }
            safe_save(state, ckpt_path)
            print(f"[Save] {ckpt_path}")
            if best_val_state is not None and best_val_improved:
                best_path = out_dir / "ckpt_best.pt"
                safe_save({"model_state": best_val_state, "iteration": iteration,
                            "best_val_loss": best_val_ema,
                            "best_loss": best_loss, "config": config,
                            "total_tokens": total_tokens,  # P3 cumulative
                            "total_time": total_time}, best_path)
            keep = 2
            old = sorted((p for p in out_dir.glob("ckpt_*.pt")
                          if p.stem.split("_")[-1].isdigit()),
                         key=lambda p: int(p.stem.split("_")[-1]))
            for stale in old[:-keep]:
                stale.unlink(missing_ok=True)
                print(f"[Prune] removed {stale}")

    final_iter = iteration if (early_stopped or nan_aborted) else config.max_iters - 1
    final_path = out_dir / "ckpt.pt"
    # 11th-audit P1: _model_sd moves to CPU first (raw state_dict OOMs 6GB).
    safe_save({
        "model_state": _model_sd(model),
        "optimizer": optimizer.state_dict(),
        "iteration": final_iter,
        "loss": current_loss,
        "best_loss": best_loss,
        "best_val_loss": best_val_ema,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "config": config,
    }, final_path)
    if best_val_state is not None and (best_val_improved
                                       or (not early_stopped and not nan_aborted)):
        best_path = out_dir / "ckpt_best.pt"
        safe_save({
            "model_state": best_val_state,
            "iteration": final_iter,
            "best_val_loss": best_val_ema,
            "best_loss": best_loss,
            "config": config,
        }, best_path)
        print(f"[Save] Best-val checkpoint -> {best_path}")
    if nan_aborted:
        sys.exit(1)
    print(f"\n[Done] Best loss: {best_loss:.4f} | Best val: {best_val_ema:.4f} | "
          f"Total tokens: {total_tokens:,} | Total time: {total_time:.1f}s | "
          f"Saved to {final_path}"
          + (f"\n[EarlyStop] {stop_reason}" if early_stopped else ""))
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distill 430M teacher -> 370M-XL student")
    parser.add_argument("--teacher_ckpt", required=True,
                        help="Path to the frozen teacher checkpoint (must contain 'config')")
    parser.add_argument("--out_dir", type=str, default="out/ambatron_distill")
    parser.add_argument("--corpus_file", type=str, default="",
                        help="Corpus .pt filename in data/processed (default code_corpus_curated.pt)")
    parser.add_argument("--size", choices=["xl"], default="xl",
                        help="Student size (default xl = 370M)")
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--max_iters", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_opt_resume", action="store_true",
                        help="Resume MODEL-ONLY (no optimizer state)")
    parser.add_argument("--cpu_corpus", action="store_true",
                        help="Keep the corpus on CPU to save VRAM")
    parser.add_argument("--T", type=float, default=2.0, dest="T",
                        help="KL temperature (default 2.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD weight on soft KL loss (default 0.5)")
    args = parser.parse_args()

    teacher_ckpt_path = Path(args.teacher_ckpt)
    if not teacher_ckpt_path.exists():
        print(f"ERROR: teacher checkpoint {teacher_ckpt_path} not found.")
        sys.exit(1)

    _ser.add_safe_globals([MoEConfig])
    teacher_ckpt = safe_load(teacher_ckpt_path, map_location="cpu")
    if "config" not in teacher_ckpt:
        print("ERROR: teacher checkpoint has no 'config' — cannot reconstruct teacher arch.")
        sys.exit(1)

    tc = teacher_ckpt["config"]
    if isinstance(tc, dict):
        teacher_config = create_ambatron_430m_config()
        for k, v in tc.items():
            if hasattr(teacher_config, k):
                setattr(teacher_config, k, v)
    else:
        teacher_config = tc
    teacher_vocab = teacher_config.vocab_size

    teacher = MoEGPT(teacher_config)
    teacher.load_state_dict(teacher_ckpt["model_state"], strict=False)
    print(f"[Teacher] Loaded from {teacher_ckpt_path} (vocab={teacher_vocab})")

    config = create_ambatron_xl_config()
    if args.batch_size > 0:
        config.batch_size = args.batch_size
    if args.block_size > 0:
        config.block_size = args.block_size
    if args.lr > 0:
        config.learning_rate = args.lr
    if args.max_iters > 0:
        config.max_iters = args.max_iters
    if args.save_interval > 0:
        config.save_interval = args.save_interval
    config.out_dir = args.out_dir
    config.gradient_accumulation_steps = args.grad_accum

    all_tokens, lengths, vocab_size = load_code_corpus(corpus_file=args.corpus_file or None)
    config.vocab_size = vocab_size
    # D3 (18th-audit): the old code set config.vocab_size = teacher_vocab here
    # and it was IMMEDIATELY clobbered by the corpus vocab above (dead block).
    # If teacher_vocab != corpus vocab, distill_loss reshapes teacher logits to
    # the student/corpus vocab -> shape crash or silent truncated-KL. Guard it:
    if teacher_vocab != vocab_size:
        print(f"[WARN] Teacher vocab ({teacher_vocab}) != corpus vocab "
              f"({vocab_size}) — soft-target alignment relies on matching "
              f"vocabularies; verify the teacher shares this tokenizer.")

    print(f"=== Distillation Training ===")
    print(f"Student: {config.n_layer} layers, d={config.n_embd} | "
          f"Batch: {config.batch_size} | Muon LR: {config.learning_rate} | "
          f"Iters: {config.max_iters} | Accum: {args.grad_accum}")
    print(f"KD: T={args.T} alpha={args.alpha} | Teacher: frozen")

    train(config, teacher_config, teacher, all_tokens, lengths, args.teacher_ckpt,
          T=args.T, alpha=args.alpha, resume=args.resume,
          compile_model=not args.no_compile, resume_no_opt=args.no_opt_resume,
          cpu_corpus=args.cpu_corpus)


if __name__ == "__main__":
    main()
