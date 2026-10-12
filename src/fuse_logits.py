"""Round-97: DLLG-style logit fusion for the tool adapter.
The weight-space task-arithmetic (Rounds 74/96) is a seesaw — blending
checkpoints averages away the tool-skill. The output-space alternative:
at each generation step, logits = (1-a)*logits13 + a*logits16 — the
v16's spatial-skill contributes softly, the v13's tool-format dominates.
The FusedMoE wrapper is a drop-in for MoEGPT (the ToolUseMoE interface).

Usage: python3 src/fuse_logits.py --sweep (or --alpha 0.3 --eval)
"""
import argparse, sys, pathlib
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])


class FusedMoE:
    def __init__(self, m1, m2, alpha):
        self.m1, self.m2, self.alpha = m1, m2, alpha
        self.config = m1.config

    def parameters(self):
        return self.m1.parameters()

    def eval(self):
        self.m1.eval(); self.m2.eval(); return self

    def to(self, device):
        self.m1.to(device); self.m2.to(device); return self

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=50,
                 top_p=0.0, repetition_penalty=1.0, no_repeat_ngram=0):
        for _ in range(max_new_tokens):
            if idx.size(1) > 0 and idx[0, -1].item() == 2:
                break
            idx_cond = idx[:, -self.config.block_size:]
            with torch.no_grad():
                l1, _, _ = self.m1(idx_cond)
                l2, _, _ = self.m2(idx_cond)
            logits = (1.0 - self.alpha) * l1 + self.alpha * l2
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            logits_orig = logits.clone()
            if no_repeat_ngram > 1:
                seq = idx[0].tolist()
                if len(seq) >= no_repeat_ngram:
                    ngram = tuple(seq[-(no_repeat_ngram - 1):])
                    banned = set()
                    for i in range(len(seq) - no_repeat_ngram + 1):
                        if tuple(seq[i:i + no_repeat_ngram - 1]) == ngram:
                            banned.add(seq[i + no_repeat_ngram - 1])
                    banned = [t for t in banned if t < logits.shape[-1]]
                    if banned:
                        logits[0, banned] = float("-inf")
                        if not torch.isfinite(logits).any():
                            logits = logits_orig.clone()
            if repetition_penalty > 1.0:
                seen = set(idx[0].tolist())
                seen = [t for t in seen if t < logits.shape[-1]]
                if seen:
                    logits[0, seen] = torch.where(
                        logits[0, seen] > 0,
                        logits[0, seen] / repetition_penalty,
                        logits[0, seen] * repetition_penalty)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p > 0.0 and top_p < 1.0:
                sorted_l, idx_sorted = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                cutoff = (cum - torch.softmax(sorted_l, dim=-1)) > top_p
                banned = idx_sorted[cutoff]
                banned = [t for t in banned[0].tolist() if t < logits.shape[-1]]
                if banned:
                    logits[0, banned] = float("-inf")
                    if not torch.isfinite(logits).any():
                        logits = logits_orig.clone()
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt1", default="out/ambatron_tool_v13/ckpt.pt")
    ap.add_argument("--ckpt2", default="out/ambatron_tool_v16/ckpt.pt")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    import torch.serialization as ser
    from moe_model import MoEConfig, MoEGPT
    from tokenizers import Tokenizer
    from tool_use import ToolRegistry, ToolUseMoE, _execute_python, _search_knowledge
    from eval_tool_use import CASES_HELDOUT, score_one
    ser.add_safe_globals([MoEConfig])

    tok = Tokenizer.from_file(str(pathlib.Path(__file__).resolve().parent.parent / "data/processed/bpe_tokenizer/tokenizer.json"))
    models = []
    for p in (args.ckpt1, args.ckpt2):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        m = MoEGPT(ck["config"]).eval()
        m.load_state_dict(ck["model_state"], strict=False)
        m.to(torch.bfloat16).to("cuda")
        models.append(m)

    def run(alpha):
        fused = FusedMoE(models[0], models[1], alpha)
        reg = ToolRegistry()
        reg.register(_execute_python, name="execute_python",
                     description="Execute Python code and return output")
        reg.register(_search_knowledge, name="search_knowledge",
                     description="Search internal knowledge base for information")
        tu = ToolUseMoE(fused, lambda t: tok.encode(t).ids, lambda i: tok.decode(i),
                        reg, compiled_schema=False)
        n = 0
        for q, exp in CASES_HELDOUT:
            res = score_one(tu, reg, q, exp)
            n += 1
            print(f"  alpha={alpha} {res:10s} | {q[:55]}", flush=True)
        return n

    alphas = [0.0, 0.15, 0.3, 0.5] if args.sweep else [args.alpha]
    for a in alphas:
        print(f"[fuse] alpha={a} (v13 x v16 logit-blend)")
        run(a)
    print("[fuse] done")


if __name__ == "__main__":
    main()