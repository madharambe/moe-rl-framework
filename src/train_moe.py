"""
Train MoE LLM with BPE tokenizer + GPU-optimized pipeline.
- Pre-tokenized data loaded directly as GPU tensors
- torch.compile for 2-3x training speedup
- Progressive context window (curriculum learning)
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from moe_model import MoEConfig, MoEGPT
from gpu_accel import resolve_device


def load_tokenized_corpus(data_dir="data/processed"):
    pt_path = Path(data_dir) / "code_corpus_bpe.pt"
    if not pt_path.exists():
        print(f"Tokenized corpus not found at {pt_path}")
        print("Run build_bpe_tokenizer.py first or run train_moe without --bpe")
        sys.exit(1)
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    tokens = data["tokens"]
    lengths = data["lengths"]
    vocab_size = data["vocab_size"]
    print(f"Loaded {len(tokens)} samples, max_len={tokens.shape[1]}, vocab={vocab_size}")
    return tokens, lengths, vocab_size


def get_lr(iteration, config):
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / max(config.warmup_iters, 1)
    if iteration > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iteration - config.warmup_iters) / max(config.lr_decay_iters - config.warmup_iters, 1)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr_floor_ratio = 0.15
    return config.min_lr + (config.learning_rate - config.min_lr) * max(cos_decay, lr_floor_ratio)


def get_curr_ctx(iteration, config, context_stages=None):
    if context_stages:
        stage_idx = min(iteration // (config.max_iters // len(context_stages)), len(context_stages) - 1)
        return context_stages[stage_idx]
    return config.block_size


def train(config, model, all_tokens, lengths, resume=False, compile_model=True):
    device = resolve_device()
    model.to(device)
    model.train()

    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            print(f"[Compile] Model compiled with reduce-overhead")
        except Exception as e:
            print(f"[Compile] Failed: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_size = config.block_size
    context_stages = getattr(config, "context_stages", None) or None
    batch_size = config.batch_size
    total_tokens = 0
    start_iter = 0
    best_loss = float("inf")
    n_samples = len(all_tokens)

    if resume:
        latest = sorted(out_dir.glob("ckpt_*.pt")) + [out_dir / "ckpt.pt"]
        latest = [p for p in latest if p.exists()]
        if latest:
            ckpt = torch.load(latest[-1], map_location=device, weights_only=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "iteration" in ckpt:
                start_iter = ckpt["iteration"] + 1
            if "total_tokens" in ckpt:
                total_tokens = ckpt["total_tokens"]
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            if "best_loss" in ckpt:
                best_loss = ckpt["best_loss"]
            print(f"[RESUME] Iter {start_iter}, best_loss={best_loss:.4f}")

    all_tokens_gpu = all_tokens.to(device, non_blocking=True)
    lengths_gpu = lengths.to(device, non_blocking=True)

    for iteration in range(start_iter, config.max_iters):
        lr = get_lr(iteration, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        curr_ctx = get_curr_ctx(iteration, config, context_stages)
        curr_ctx = min(curr_ctx, block_size)

        batch_indices = torch.randint(0, n_samples, (batch_size,), device=device)
        batch_tokens = all_tokens_gpu[batch_indices]
        batch_lens = lengths_gpu[batch_indices]

        ctx = min(curr_ctx, batch_tokens.shape[1])
        batch_tokens = batch_tokens[:, :ctx]
        batch_lens = batch_lens.clamp(max=ctx)

        x = batch_tokens[:, :-1].contiguous()
        y = batch_tokens[:, 1:].contiguous()

        logits, loss, aux = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_tokens += x.numel()
        current_loss = loss.item()
        best_loss = min(best_loss, current_loss)

        if iteration % config.log_interval == 0:
            print(f"Iter {iteration:6d}/{config.max_iters} | Loss {current_loss:.4f} | "
                  f"CE {aux['ce_loss']:.4f} | Bal {aux['balance_loss']:.4f} | "
                  f"Z {aux['z_loss']:.4f} | LR {lr:.2e} | Tokens {total_tokens:,} | "
                  f"Ctx {ctx}")

        if iteration % config.save_interval == 0 and iteration > 0:
            ckpt_path = out_dir / f"ckpt_{iteration}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "config": config,
                "loss": current_loss,
                "best_loss": best_loss,
                "total_tokens": total_tokens,
            }, ckpt_path)

    final_path = out_dir / "ckpt.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": config.max_iters - 1,
        "config": config,
        "best_loss": best_loss,
        "total_tokens": total_tokens,
    }, final_path)
    print(f"Training complete. Best loss: {best_loss:.4f}. Saved to {final_path}")
    model.eval()
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train MoE LLM (GPU-optimized)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--context_stages", type=int, nargs="+", default=[128, 256, 512, 1024],
                        help="Progressive context sizes")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    config = MoEConfig()
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.max_iters = args.max_iters
    config.save_interval = args.save_interval
    config.context_stages = tuple(args.context_stages) if args.context_stages else ()

    all_tokens, lengths, vocab_size = load_tokenized_corpus()
    config.vocab_size = vocab_size

    model = MoEGPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MoE Model: {n_params:,} total params")
    print(f"Device: {resolve_device()}")
    print(f"Compile: {'yes' if not args.no_compile else 'no'}")

    train(config, model, all_tokens, lengths, resume=args.resume,
          compile_model=not args.no_compile)


if __name__ == "__main__":
    main()
