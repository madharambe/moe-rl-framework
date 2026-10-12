"""Round-77: spatial-reasoning eval (leak-free).
The training set was generated with SEED=7; this eval generates FRESH questions
from SEED=1234 (the same generator + verified answers), so the model can only
score by actually reasoning spatially. Direct-answer mode (no tools).

Usage:
  python3 src/eval_spatial.py --ckpt out/ambatron_tool_v7/ckpt.pt [--n 40]
"""
import argparse, sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_spatial_sft import gen

EVAL_SEED = 1234
SYS = "You are a helpful assistant with good spatial reasoning."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bf16", action="store_true",
                    help="load the model in bfloat16 (~half VRAM) to run safely "
                         "under the flywheel's 2GB VRAM gate")
    args = ap.parse_args()

    import torch
    import torch.serialization as ser
    from moe_model import MoEConfig, MoEGPT
    ser.add_safe_globals([MoEConfig])
    from train_tool_sft import _tokenize
    from gpu_accel import resolve_device

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # 205cl: the legacy finals save config=None — the 18L/1152 class =
    # the v3 factory (the same derivation eval_chat uses)
    _cfg = ck.get("config")
    if _cfg is None:
        from moe_model import create_ambatron_v3_config
        _cfg = create_ambatron_v3_config()
        print("[spatial] derived the v3 class config (legacy ckpt)")
    model = MoEGPT(_cfg).eval()
    model.load_state_dict(ck["model_state"], strict=False)
    if args.bf16:
        model.to(torch.bfloat16)
    model.to(resolve_device(args.device))
    # 205y: vocab-derived tokenizer (the fixed 16K stream mismatched every
    # v3 32K champion — the eval_agentic 204ah-21 bug class).
    tok = _tokenize("", "v3" if int(getattr(_cfg, "vocab_size", 0)) >= 32768 else None)

    rng = random.Random(EVAL_SEED)
    torch.manual_seed(0)  # M4: seed generation so champion-vs-candidate is comparable
    questions = [gen(rng, i) for i in range(args.n)]
    # ROUND-170 (F9): the generator draws 40 from a ~20-pattern space — text
    # duplicates inflate the numerator (the same skill counted twice as /40).
    # Dedup by the full question text BEFORE the fresh filter.
    _seen_q = set()
    _dedup = []
    for _q in questions:
        _k = _q[0].strip().lower()
        if _k not in _seen_q:
            _seen_q.add(_k)
            _dedup.append(_q)
    if len(_dedup) != len(questions):
        print(f"[spatial-eval] deduped {len(questions)-len(_dedup)} repeated "
              f"questions; {len(_dedup)} unique")
    questions = _dedup
    # 10th-audit F1: the eval's generator (seed 1234) overlaps the training set
    # (seed 7) verbatim — measured 20/40. Scoring memorized items inflates the
    # champion (19/20 falsely strong). Filter to only GENUINELY-FRESH questions
    # by excluding any whose text appears in the spatial SFT training set.
    import json as _json
    try:
        _train = set()
        _sp_path = pathlib.Path(__file__).resolve().parent.parent / "data/processed/spatial_sft.jsonl"
        for _l in open(_sp_path):
            _d = _json.loads(_l)
            for _seg in _d.get("segments", []):
                _t = _seg[1] if isinstance(_seg, list) else _seg.get("content", "")
                if "User:" in _t:
                    _train.add(_t.split("User:", 1)[1].strip().lower())
        fresh = [(q, exp) for q, exp in questions if q.strip().lower() not in _train]
    except (OSError, ValueError):
        fresh = questions  # if training file is missing, don't silently drop
    if fresh != questions:
        print(f"[spatial-eval] filtered {len(questions)-len(fresh)} train-overlap "
              f"questions; scoring {len(fresh)} fresh")
    questions = fresh
    correct = 0
    for q, exp in questions:
        prompt = f"{SYS}\nUser: {q}\nAssistant:"
        idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long,
                           device=next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=24, temperature=0.3,
                                 top_k=40)[0]
        text = tok.decode(out[len(tok.encode(prompt).ids):].tolist())
        norm = text.strip().lower()
        # B-162: bare substring scored negations ("not to the left")
        # CORRECT whenever the expected word appeared anywhere. Word-
        # boundary at least kills partial-word artifacts; negation-safe
        # scoring would need a response-format contract (future work).
        import re as _re
        ok = bool(_re.search(rf"\b{_re.escape(str(exp).lower())}\b", norm))
        correct += int(ok)
    print(f"[spatial-eval] {args.ckpt} | n={len(questions)} "
          f"(filtered from {args.n}) | correct "
          f"{correct}/{len(questions)} ({correct/max(1,len(questions))*100:.0f}%)")


if __name__ == "__main__":
    main()
