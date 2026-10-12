#!/usr/bin/env python3
"""ROUND-204ac-b6.2: v3 MIX SLICE - the data-improvement axis #1 (phi-1-style
"textbook" slice + honest domain mix for the pretrain).

The v3 pretrain pack is 100% code AND 100% python (_iter_text_files collects
*.py only). This builder creates a mixed-domain slice so the recovery leg /
v3.1 continuation can interleave math + natural-language + tool dialogue into
a code base:
  A) real synthetic pools the project ALREADY verified (arith_sft, auto_chat,
     chat_aug, auto_tool, agentic_sft...) - answer-bearing dialogue, no lies,
  B) freshly GENERATED arithmetic Q&A (computed inline, so the answers are
     truthful; the phi-1 "textbook" move),
Each source is encoded with the v3 tokenizer (32768 vocab), windowed to 1024
(provenance + dedup like the pack builder), and saved as the v3 mix pack.

Usage:
  .venv/bin/python3 src/build_v3_mix.py       --out "<ssd>/data_processed/v3/v3mix_bpe.pt"       --pool v3-bank/v3mix_pool.jsonl --meta v3-bank/v3mix_build.json       [--cap 40000]
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tools"))
import torch
from tokenizers import Tokenizer

TOK_PATH = BASE / "v3-bank" / "tokenizer_v3.json"
CTX = 1024
SOURCES = [
    "data/processed/arith_sft.jsonl",
    "data/processed/auto_chat_train.jsonl",
    "data/processed/chat_aug_train.jsonl",
    "data/processed/auto_tool_train.jsonl",
    "data/processed/agentic_sft.jsonl",
]


def _stable_hash(b):
    h = 5381
    for c in b:
        h = ((h << 5) + h + c) & 0xFFFFFFFF
    return h


def _line_text(line):
    try:
        o = json.loads(line)
    except Exception:
        return str(line).strip()
    if isinstance(o, dict):
        if o.get("segments"):
            parts = []
            for s in o["segments"]:
                if not isinstance(s, (list, tuple)) or len(s) < 2:
                    continue
                role, text = s[0], s[1]
                t = str(text).strip()
                if t and not t.startswith("result:"):
                    parts.append(t)
                elif t:
                    parts.append("Result: " + t[len("result:"):].strip())
            return "\n".join(parts)
        for k in ("text", "content", "answer", "prompt"):
            if isinstance(o.get(k), str) and len(o[k]) > 40:
                return o[k].strip()
        parts = [str(o.get(k, "")).strip() for k in
                 ("query", "answer") if o.get(k)]
        if parts and sum(len(p) for p in parts) > 40:
            return "\n".join(parts)
    return str(o).strip()


def _gen_arith(n=4000, seed=0):
    rng = random.Random(seed)
    outs = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*", "//", "**2", "%"])
        a = rng.randint(2, 997)
        b = rng.randint(2, 97)
        if op == "+":
            q, ans = f"What is {a} + {b}?", a + b
        elif op == "-":
            q, ans = f"What is {max(a,b)} - {min(a,b)}?", max(a, b) - min(a, b)
        elif op == "*":
            q, ans = f"What is {a} * {b}?", a * b
        elif op == "//":
            b = max(2, b)
            q, ans = f"What is {a * b} // {b}?", a
        elif op == "**2":
            q, ans = f"What is {a} squared?", a * a
        else:
            q, ans = f"What is {a} % {b}?", a % b
        outs.append(f"User: {q}\nAssistant: The result is {ans}.")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--no-synth", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOK_PATH))
    texts = []
    for src in SOURCES:
        p = BASE / src
        if not p.exists():
            print("[v3mix] skip missing:", src)
            continue
        got = 0
        for line in open(p, encoding="utf-8", errors="replace"):
            t = _line_text(line)
            if len(t) > 40 and any(ch.isalnum() for ch in t[:80]):
                texts.append(t)
                got += 1
        print(f"[v3mix] {src}: {got} lines")
    if not args.no_synth:
        texts += _gen_arith()
        print("[v3mix] + synthetic arith Q&A")

    tokens = torch.zeros(args.cap, CTX, dtype=torch.int32)
    lengths = torch.zeros(args.cap, dtype=torch.long)
    seen = set()
    n = 0
    from array import array
    with open(args.pool, "w") as pf:
        for t in texts:
            ids = tok.encode(t).ids
            for start in range(0, len(ids) - 1, CTX):
                win = ids[start:start + CTX]
                real = len(win)
                if real < CTX:
                    win = win + [0] * (CTX - real)  # <PAD>=0 tail
                key = _stable_hash(array("I", win).tobytes())
                if key in seen:
                    continue
                seen.add(key)
                tokens[n] = torch.tensor(win, dtype=torch.int32)
                lengths[n] = real
                pf.write(json.dumps({"idx": n, "src": "mix",
                                     "start": start,
                                     "len": len(win)}) + "\n")
                n += 1
                if n >= args.cap:
                    break
            if n >= args.cap:
                print("[v3mix] cap hit at", n)
                break
    tokens, lengths = tokens[:n], lengths[:n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + ".tmp"
    torch.save({"tokens": tokens, "lengths": lengths, "vocab_size": 32768}, tmp)
    import os
    os.replace(tmp, args.out)
    meta = {"source": "mixed SFT pools + synthetic arith", "cap": args.cap,
            "rows": n, "ctx": CTX, "vocab": 32768}
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    print(f"[v3mix] DONE: {n} x {CTX} -> {args.out} | meta -> {args.meta}")


if __name__ == "__main__":
    main()
