#!/usr/bin/env python3
# ROUND-205i: CODE-SCRIBE — pure next-token continuation on the code corpus,
# initialized from a champion. NO tool-format tokens anywhere: the retry
# mode must survive, and the code-writing floor gets its lift (the 587M's
# tool-loop writes garbage code because the base lacks the skill).
import argparse, json, os, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
import torch.serialization as ser

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--out", required=True)
ap.add_argument("--save_every", type=int, default=500)
args = ap.parse_args()

from moe_model import MoEGPT, MoEConfig
ser.add_safe_globals([MoEConfig])
from storage import safe_save

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.base, map_location="cpu", weights_only=True)
model = MoEGPT(ck["config"])
model.load_state_dict(ck["model_state"], strict=False)
model.to(torch.bfloat16).to(dev)
model.train()

corp = torch.load(args.corpus, map_location="cpu", weights_only=True)
tokens = corp["tokens"]  # (N, 1024)
N = tokens.shape[0]
print(f"[scribe] corpus {N} x {tokens.shape[1]}, model ctx={ck['config'].block_size}")

# 205i: SGD-momentum fits the 6.1G card (AdamW fp32 m+v states OOM at
# batch2x1024); schedule handled below
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.05)
steps_per_epoch = max(1, N // args.batch_size)
torch.manual_seed(205)
loss_acc = 0.0
t0 = time.time()
for step in range(args.steps):
    if step < args.warmup:
        for g in opt.param_groups:
            g["lr"] = args.lr * (step + 1) / args.warmup
    else:
        for g in opt.param_groups:
            g["lr"] = args.lr
    idx = torch.randint(0, N, (args.batch_size,))
    x = tokens[idx][:, : args.ctx].to(dev).long()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _ = model(x)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                           x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    loss_acc += loss.item()
    if (step + 1) % 100 == 0:
        el = (time.time() - t0) / (step + 1)
        print(f"[scribe] step {step+1} | loss {loss_acc/100:.4f} | {el:.2f}s/step", flush=True)
        loss_acc = 0.0
    if (step + 1) % args.save_every == 0:
        model.eval()
        safe_save({"model_state": model.state_dict(), "config": ck["config"],
                   "args": vars(args), "step": step + 1}, args.out + f".scrub{step+1}.pt")
        model.train()

model.eval()
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
from storage import ensure_disk as _ed
_ed(out, 3.0, context="scribe final save")
safe_save({"model_state": model.state_dict(), "config": ck["config"],
           "args": vars(args), "step": args.steps}, out)
print(f"[scribe] done -> {out}")
