#!/usr/bin/env python3
# ROUND-205af: LoRA CODE-SCRIBE at 2048-ctx — the B-region corpus on the
# 6.1G card via a LoRA adapter (15M trainable -> optimizer trivial; the
# full-weight optimizer is the wall at 2048 ctx). Code-floor/long-ctx lift.
import argparse, json, os, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
import torch.serialization as ser

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--out", required=True)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
args = ap.parse_args()

from moe_model import MoEGPT, MoEConfig, create_ambatron_v3_config
ser.add_safe_globals([MoEConfig])
from lora_adapter import add_lora, freeze_base, merge_lora, remove_lora, LoRALinear
from storage import safe_save, ensure_disk

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.base, map_location="cpu", weights_only=True)
# 205af: the v3 config is natively 2048-block (sliding 1024) — the region
# leg must TRAIN at 2048. The base ckpt's config was a 1024-leg; the
# weights carry over (same structural shapes); the config is the 2048 one.
cfg = create_ambatron_v3_config()
model = MoEGPT(cfg)
model.load_state_dict(ck["model_state"], strict=False)
add_lora(model, r=args.r, alpha=args.alpha)
freeze_base(model)
model.to(torch.bfloat16)
for mod in model.modules():
    if isinstance(mod, LoRALinear):
        mod.A.to(dtype=torch.float32)
        mod.B.to(dtype=torch.float32)
model.to(dev)
model.train()

corp = torch.load(args.corpus, map_location="cpu", weights_only=True)
tokens = corp["tokens"]
N = tokens.shape[0]
blk = min(args.ctx, int(getattr(cfg, "block_size", args.ctx)))
print(f"[scribelora] corpus {N} x {tokens.shape[1]} | ctx%{blk} | "
      f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=0.05)
torch.manual_seed(205)
loss_acc = 0.0
t0 = time.time()
for step in range(args.steps):
    if step < args.warmup:
        for g in opt.param_groups:
            g["lr"] = args.lr * (step + 1) / args.warmup
    idx = torch.randint(0, N, (args.batch_size,))
    x = tokens[idx][:, :blk].to(dev).long()
    import torch.amp
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _ = model(x)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                           x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    loss_acc += loss.item()
    if (step + 1) % 100 == 0:
        el = (time.time() - t0) / (step + 1)
        print(f"[scribelora] step {step+1} | loss {loss_acc/100:.4f} | {el:.2f}s/step", flush=True)
        loss_acc = 0.0
    if (step + 1) % 2000 == 0:
        model.eval()
        safe_save({"model_state": model.state_dict(), "config": ck["config"],
                   "args": vars(args), "step": step + 1},
                  args.out + f".leg{step+1}.pt")
        model.train()

model.eval()
merge_lora(model)
remove_lora(model)
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
ensure_disk(out, 3.0, context="scribelora final")
safe_save({"model_state": model.state_dict(), "config": ck["config"],
           "args": vars(args), "step": args.steps}, out)
print(f"[scribelora] done -> {out}")
