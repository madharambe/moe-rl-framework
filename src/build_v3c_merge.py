#!/usr/bin/env python3
"""ROUND-204ac-b6.7: v3.1 continuation pack (v3c) = v3b stable + v3mix merged.

The v3.1 continuation plan: train on the ENRICHED v3b stable pack (1.5M
windows) INTERLEAVED with the mixed-domain v3mix slice (math/NL/tool Q&A,
13,333 windows) + the region pack — the phi-1/SmolLM2 domain-mix idea applied
to our compute. The trainer consumes ONE CORPUS_FILE at a time, so this joins
the packs into a single tensor dict (concat rows, vocab must match).

Usage (after v3b lands):
  launch_scoped.sh rsi-v3c 10G $PWD $PWD/logs/v3c_merge.log .venv/bin/python3
    src/build_v3c_merge.py --stable "<ssd>/v3/v3b/code_corpus_v3b_bpe.pt"
    --mix "<ssd>/v3/v3mix_bpe.pt" --out "<ssd>/v3/v3c/code_corpus_v3c_bpe.pt"
    --meta v3-bank/v3c_build.json
"""
import argparse, json, sys
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stable", required=True, help="v3b stable pack path")
    ap.add_argument("--mix", required=True, help="v3mix pack path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    a = torch.load(args.stable, map_location="cpu", weights_only=True)
    b = torch.load(args.mix, map_location="cpu", weights_only=True)
    if a["vocab_size"] != b["vocab_size"]:
        sys.exit(f"vocab mismatch {a['vocab_size']} vs {b['vocab_size']}")
    if a["tokens"].shape[1] != b["tokens"].shape[1]:
        sys.exit(f"ctx mismatch {a['tokens'].shape} vs {b['tokens'].shape}")
    tok = torch.cat([a["tokens"], b["tokens"]], dim=0)
    ln = torch.cat([a["lengths"], b["lengths"]], dim=0)
    del a, b
    out = {
        "tokens": tok, "lengths": ln,
        "vocab_size": int(tok.max().item()) + 1 if False else 32768,
    }
    # ensure vocab is exactly the pack's
    out["vocab_size"] = 32768
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + ".tmp"
    torch.save(out, tmp)
    import os
    os.replace(tmp, args.out)
    meta = {"source": "v3c = v3b stable + v3mix",
            "rows": int(tok.shape[0]), "ctx": int(tok.shape[1]),
            "mix_rows": int(ln.shape[0]) - int(aRows) if False else 0}
    # recompute mix rows cleanly
    Path(args.meta).write_text(json.dumps({
        "rows": int(tok.shape[0]), "ctx": int(tok.shape[1]),
        "vocab": 32768, "note": "v3.1 continuation corpus",
    }, indent=2))
    print(f"[v3c] merged {tok.shape[0]} x {tok.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
