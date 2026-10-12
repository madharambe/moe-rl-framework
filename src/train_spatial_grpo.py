#!/usr/bin/env python3
# 205an: SPATIAL-GRPO — the direct-answer RL mode. The spatial axis = 0/20
# fresh on BOTH v2 and v3 (the historical positives were overlap-inflated):
# the axis needs the same verified-RL treatment, but the reward is a plain
# word-match against the ANALYTIC generator (no tools) — the cleanest
# verifiable MDP in the whole project.
import argparse, json, random, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
import torch.serialization as ser

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--group", type=int, default=6)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--n_epochs", type=int, default=2)
ap.add_argument("--max_norm", type=float, default=1.0)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--max_new", type=int, default=24)
ap.add_argument("--queries", type=int, default=24)
ap.add_argument("--out", required=True)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
args = ap.parse_args()

from moe_model import MoEConfig, MoEGPT
ser.add_safe_globals([MoEConfig])
from build_spatial_sft import gen
from lora_adapter import add_lora, freeze_base, merge_lora, remove_lora, LoRALinear
from storage import safe_save, ensure_disk

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.base, map_location="cpu", weights_only=True)
ref = MoEGPT(ck["config"]).eval()
ref.load_state_dict(ck["model_state"], strict=False)
policy = MoEGPT(ck["config"]).eval()
policy.load_state_dict(ck["model_state"], strict=False)
add_lora(policy, r=args.r, alpha=args.alpha)
freeze_base(policy)
for m in (ref, policy):
    m.to(torch.bfloat16)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            mod.A.to(dtype=torch.float32)
            mod.B.to(dtype=torch.float32)
ref.to(dev); policy.to(dev)

_v3 = int(getattr(ck["config"], "vocab_size", 0)) >= 32768
from train_tool_sft import _tokenize
tok = _tokenize("", "v3" if _v3 else None)

def scored(text, ans):
    t = text.strip().lower().rstrip(".")
    a = str(ans).strip().lower()
    if a in t:
        return True
    return bool(re.search(r"(?<![a-z0-9_'])" + re.escape(a) + r"(?![a-z0-9_'])", t))

rng = random.Random(4321)
queries = [gen(rng, i) for i in range(args.queries)]
print(f"[sgrpo] {len(queries)} spatial queries (analytic truth)")

opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=0.01)
for rnd in range(args.rounds):
    torch.manual_seed(1234 + rnd)
    tot_r = tot_c = 0
    for q, ans in queries:
        pfx = f"User: {q}\nAssistant:"
        ids = tok.encode(pfx).ids[-900:]
        comps, rewards = [], []
        for _ in range(args.group):
            with torch.no_grad():
                out = policy.generate(torch.tensor([ids], dtype=torch.long, device=dev),
                                     max_new_tokens=args.max_new, temperature=args.temperature)
            comp = tok.decode(out[0, len(ids):].tolist()).strip()
            r = 1.0 if scored(comp, ans) else 0.0
            comps.append(comp); rewards.append(r)
            tot_c += int(r > 0)
        tot_r += sum(rewards)
        rmean = sum(rewards) / len(rewards)
        # 205aj Dr. GRPO: mean-only advantages (drop the std normalizer)
        for comp in comps:
            n_prompt = len(ids)
            full = tok.encode(pfx + comp).ids
            x = torch.tensor([full[-policy.config.block_size:]], dtype=torch.long, device=dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = policy(x)
            logp = F.log_softmax(logits.float(), dim=-1)
            with torch.no_grad():
                xref = torch.tensor([full[-policy.config.block_size:]], dtype=torch.long, device=dev)
                rlogits, _, _ = ref(xref)
                rlogp = F.log_softmax(rlogits.float(), dim=-1)
            adv = r - rmean
            n_new = len(full) - n_prompt
            if n_new <= 0:
                continue
            if args.n_epochs == 1:
                lp = logp[0, -n_new-1:-1, :].gather(1, x[0, -n_new:].unsqueeze(1)).squeeze(1)
                rlp = rlogp[0, -n_new-1:-1, :].gather(1, x[0, -n_new:].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(lp - rlp)
                clipv = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                kl = torch.exp(rlp) - (rlp - lp) - 1
                loss = (-torch.min(ratio * adv, clipv * adv) + args.beta * kl).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], args.max_norm)
                opt.step()
    n = len(queries) * args.group
    print(f"[sgrpo] round {rnd} | correct {tot_c}/{n} ({tot_c/n:.1%}) | "
          f"mean reward {tot_r/n:.3f}", flush=True)

policy.eval()
merge_lora(policy); remove_lora(policy)
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
ensure_disk(out, 3.0, context="spatial grpo final")
safe_save({"model_state": policy.state_dict(), "config": ck["config"],
           "args": vars(args)}, out)
print(f"[sgrpo] done -> {out}")
