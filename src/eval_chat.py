"""Round-92: chat-quality eval (the generation-health metrics).
The chat skill is measured by the response's GENERATION QUALITY on fresh
conversational prompts (topics NOT in the training set): useful length,
no degenerate repetition loops, no immediate dead-ends. The metrics are
heuristic but honest — they capture the failure modes measured (the
repetition loops + the too-short/too-long degeneracies).

Usage: python3 src/eval_chat.py --ckpt out/ambatron_tool_v17c/ckpt.pt
"""
import argparse, sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SYS = "You are a helpful assistant."
# fresh topics — deliberately NOT in the training/augmentation data
PROMPTS = [
    "My car's check engine light came on. What should I do first?",
    "How do I plan a road trip across two countries?",
    "My headphones only work on one side. Any fix?",
    "What's a good way to reduce waste in my kitchen?",
    "I can't sleep lately. What helps?",
    "How do I negotiate a better price on a used car?",
    "My houseplants keep dying. What am I doing wrong?",
    "What should I know before adopting a dog?",
    "How do I clean my keyboard safely?",
    "My back hurts after sitting all day. What can I do?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import torch
    import torch.serialization as ser
    from moe_model import MoEConfig, MoEGPT, create_ambatron_430m_config
    ser.add_safe_globals([MoEConfig])
    from train_tool_sft import _tokenize

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck.get("config") or create_ambatron_430m_config()
    if ck.get("config") is None:
        # 205cl: the legacy datacont finals save config=None — the shape
        # match (18L/1152/32768) = the v3 class; use the exact factory
        sd0 = ck.get("model_state", {})
        _emb = sd0.get("token_embed.weight")
        _n_layer = len(set(k.split('.')[1] for k in sd0 if k.startswith('layers.')))
        if _emb is not None and _n_layer == 18 and int(_emb.shape[1]) == 1152:
            from moe_model import create_ambatron_v3_config
            cfg = create_ambatron_v3_config()
            print('[eval] derived: the v3 class factory')
        elif _emb is not None and _n_layer:
            _ne = int(_emb.shape[1]); _vb = int(_emb.shape[0])
            cfg = MoEConfig(n_layer=_n_layer, n_embd=_ne, vocab_size=_vb, block_size=1024)
            print(f"[eval] derived cfg: {_n_layer}L/{_ne}d/vocab{_vb}")
    m = MoEGPT(cfg).eval()
    m.load_state_dict(ck.get("model_state", ck.get("model", {})), strict=False)
    m.to(args.device)
    # 205aa: vocab-derived tokenizer (the fixed 16K stream mismatched v3 champions).
    tok = _tokenize("", "v3" if int(getattr(cfg, "vocab_size", 0)) >= 32768 else None)

    too_short = looped = dead = 0
    n = len(PROMPTS)
    for q in PROMPTS:
        p = f"{SYS}\nUser: {q}\nAssistant:"
        idx = torch.tensor([tok.encode(p).ids], dtype=torch.long,
                           device=next(m.parameters()).device)
        with torch.no_grad():
            out = m.generate(idx, max_new_tokens=args.max_new, temperature=0.7,
                             top_k=30, repetition_penalty=1.2)[0]
        text = tok.decode(out[len(tok.encode(p).ids):].tolist()).strip()
        words = text.split()
        if len(words) < 4:
            dead += 1
        elif len(words) < 10:
            too_short += 1
        # repetition: any 4-gram appearing 3+ times, or a 2-gram 6+ times
        grams4 = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        grams2 = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
        if any(grams4.count(g) >= 3 for g in set(grams4)) or \
           any(grams2.count(g) >= 6 for g in set(grams2)):
            looped += 1
    healthy = n - dead - too_short - looped
    print(f"[chat-eval] {args.ckpt} | n={n}")
    print(f"  healthy   : {healthy}/{n} ({healthy/n*100:.0f}%)")
    print(f"  dead-end  : {dead}/{n} ({(dead/n)*100:.0f}%)")
    print(f"  too-short : {too_short}/{n} ({(too_short/n)*100:.0f}%)")
    print(f"  looped    : {looped}/{n} ({(looped/n)*100:.0f}%)")


if __name__ == "__main__":
    main()