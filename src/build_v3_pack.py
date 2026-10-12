#!/usr/bin/env python3
"""ROUND-204ac: the v3-on-A corpus pack (the region_expand axis, LongRoPE-class).

Builds, for the v3 generation:
  1. tokenizer_v3.json      — a 32,768-vocab BPE (the scaling-law bracket for
                              a <=600M model, arXiv 2407.13623; the 76,160 of
                              the B/C mesh contract is the 1B+ bracket).
  2. code_corpus_v3_bpe.pt  — the stable leg: 1024-ctx windows (same shape
                              as the prod corpus; int32 rows since vocab >
                              32767; <PAD>=0 semantics identical).
  3. v3_region2048.pt       — the extension leg: 2048-ctx region-expanded
                              windows (adjacent-chunk concatenation = the
                              long-context signal without scarce long text;
                              assigned <PAD>-padding for short tails).
  4. v3_pretrain_pool.jsonl — the provenance index (src file + window span).
  5. v3_build.json          — the bank metadata (vocab, counts, timings).

All outputs are EXPLICIT paths (the Sep-5 --live incident class stays
closed: nothing is ever written to the production data/processed paths).

Usage:
  .venv/bin/python3 src/build_v3_pack.py \
      --tok_out v3-bank/tokenizer_v3.json \
      --base_out <ssd>/data_processed/v3/code_corpus_v3_bpe.pt \
      --reg_out  <ssd>/data_processed/v3/v3_region2048.pt \
      --pool     v3-bank/v3_pretrain_pool.jsonl \
      --meta     v3-bank/v3_build.json
"""
import argparse, json, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tools"))

V3_VOCAB = 32768
REGION_LEN = 2048
CAP_BASE = 400000
CAP_REGION = 120000
# ROUND-204ac-b5.11 (data enrichment, axis b): the old caps captured only ~4%
# of the raw tree (22,719 of ~536k files -> 216M tokens). --cap_base/--cap_region
# raise the per-leg caps for the ENRICHED (v3b) pack; writes go ONLY to explicit
# v3b paths (the live v3 packs + the running B trainer stay untouched).


def _stable_hash(b):
    h = 5381
    for c in b:
        h = ((h << 5) + h + c) & 0xFFFFFFFF
    return h


def _staged_save(obj, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out_path.resolve()).with_suffix(".pt.tmp")
    import torch
    torch.save(obj, tmp)
    with open(tmp, "rb+") as f:
        import os
        os.fsync(f.fileno())
    import os
    os.replace(tmp, out_path)


def main():
    import torch
    import build_bpe_tokenizer as bbt
    bbt.VOCAB_SIZE = V3_VOCAB

    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_out", default="",
                    help="output tokenizer path (required unless --region_only)")
    ap.add_argument("--base_out", required=True)
    ap.add_argument("--reg_out", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--meta", required=True)
    # ROUND-204ac-b3 (B wedge): the tokenizer ALREADY EXISTS complete (32768
    # vocab, sha-verified on A/B). --tok <file> reuses it and SKIPS the BPE
    # training phase (the exact phase that wedged twice on B); the corpus
    # packs then build from the existing tokenizer. Saves ~30-60 min of BPE-
    # training + dodges the 19.4GB raw collect entirely.
    ap.add_argument("--tok", default="",
                    help="reuse an existing tokenizer.json (skips BPE training)")
    ap.add_argument("--heartbeat", default=0, type=int,
                    help="print a heartbeat every N seconds (the 10s-abort golden)")
    ap.add_argument("--region_only", action="store_true",
                    help="skip BPE-training + the stable-leg rebuild (they exist "
                         "and are verified); only (re)build the region_expand "
                         "2048 pack + provenance pool from the raw corpus")
    ap.add_argument("--cap_base", type=int, default=CAP_BASE,
                    help="stable-leg window cap (default %(default)s)")
    ap.add_argument("--cap_region", type=int, default=CAP_REGION,
                    help="region_expand window cap (default %(default)s)")
    ap.add_argument("--langs", default=".py",
                    help="comma-separated file extensions to scan (default .py; "
                         "e.g. .py,.js,.ts,.go,.rs,.sh,.c,.cpp,.md,.json)")
    ap.add_argument("--filter_junk", action="store_true",
                    help="skip junk files (lockfiles/minified/maps/vendor/...)")
    args = ap.parse_args()
    import re as _re
    if args.langs != ".py":
        bbt.TEXT_EXTS = tuple(
            (e if e.startswith(".") else "." + e).strip()
            for e in args.langs.split(",") if e.strip())
    if args.filter_junk:
        bbt.JUNK_FILE_RE = _re.compile(
            r"(?i)package-lock|yarn\.lock|pnpm-lock|npm-shrinkwrap|Pipfile\.lock|"
            r"poetry\.lock|\.min\.(js|css)$|\.map$|/vendor/|/node_modules/|"
            r"/dist/|\.git/|__pycache__|venv/")
    # assign via globals(): plain assignment would shadow the module-level
    # CAP_BASE/CAP_REGION used as argparse defaults in THIS function -> UnboundLocal.
    globals()["CAP_BASE"] = args.cap_base
    globals()["CAP_REGION"] = args.cap_region
    if not args.region_only and not args.tok_out and not args.tok:
        ap.error("--tok_out required unless --region_only or --tok (reuse)")

    t0 = time.time()
    import threading
    _hb_stop = None
    if args.heartbeat > 0:
        def _hb_loop():
            while not _hb_stop.is_set():
                _hb_stop.wait(args.heartbeat)
                if not _hb_stop.is_set():
                    print(f"[v3-pack] heartbeat @ {time.time()-t0:.0f}s — alive "
                          f"(phase: pack)", flush=True)
        _hb_stop = threading.Event()
        threading.Thread(target=_hb_loop, daemon=True).start()

    if args.tok:
        from tokenizers import Tokenizer as _Tok
        _tok_path = Path(args.tok)
        tok = _Tok.from_file(str(_tok_path))
        print(f"[v3-pack] REUSING tokenizer {_tok_path} (vocab {tok.get_vocab_size()}) "
              f"— skipping BPE training (the B-wedge phase) + raw collect")
        import gc
        gc.collect()
    else:
        print("[v3-pack] collecting texts + building the 32768-vocab BPE ...")
        texts = bbt.collect_texts()
        tok = bbt.build_tokenizer(texts, out_dir=Path(args.tok_out).parent)
        tok.save(str(args.tok_out))
        print(f"[v3-pack] tokenizer saved -> {args.tok_out} (vocab {tok.get_vocab_size()})")
        del texts
        import gc
        gc.collect()

    base_pt = Path(args.base_out)
    # ---- stable leg: 1024-ctx windows (the build_bpe_tokenizer shape) ----
    # (--region_only skips the rebuild: the stable pack already exists and is
    #  verified; golden (0) = no redundant work.)
    if not args.region_only:
        print("[v3-pack] stable leg: 1024-ctx windows ...")
        bbt.MAX_SEQS = CAP_BASE
        # BUGFIX (204ac-b3-3): build_corpus does NOT mkdir its output parent —
        # on B the pack's C:\temp\rsi_v3 dir didn't exist and the 30-min
        # stable leg crashed at the final torch.save ("Parent directory ...").
        base_pt.parent.mkdir(parents=True, exist_ok=True)
        bbt.build_corpus(tok, out_path=base_pt)
        base = {}
        d = torch.load(base_pt, map_location="cpu", weights_only=True)
        base = {"tokens": len(d["tokens"]), "ctx": d["tokens"].shape[1]}
        del d
        print(f"[v3-pack] base pack: {base['tokens']} x {base['ctx']}")
    else:
        if not args.tok:
            sys.exit("[v3-pack] --region_only requires --tok (the existing "
                     "tokenizer) — aborting")
        d = torch.load(base_pt, map_location="cpu", weights_only=True)
        base = {"tokens": len(d["tokens"]), "ctx": d["tokens"].shape[1]}
        del d
        print(f"[v3-pack] region_only: stable pack REUSED "
              f"{base['tokens']} x {base['ctx']} (not rebuilt)")

    # ---- extension leg: 2048 region-expanded windows ----
    print("[v3-pack] extension leg: 2048 region-expanded windows ...")
    dt = torch.int32
    arr_code = "I"
    tokens = torch.zeros(CAP_REGION, REGION_LEN, dtype=dt)
    lengths = torch.zeros(CAP_REGION, dtype=torch.long)
    seen = set()
    n = 0
    from array import array
    with open(args.pool, "w") as pf:
        # BUGFIX (204ac-b3-2): _iter_text_files() YIELDS CONTENT, not paths
        # (build_corpus's contract) — the old `Path(path).read_text()` fed
        # file CONTENT into Path() -> "File name too long"/garbage forever.
        # Iterate the dedup'd content directly (the dedup is in the generator).
        for txt in bbt._iter_text_files():
            ids = tok.encode(txt).ids
            for start in range(0, len(ids) - REGION_LEN, REGION_LEN):
                win = ids[start:start + REGION_LEN]
                key = _stable_hash(array(arr_code, win).tobytes())
                if key in seen:
                    continue
                seen.add(key)
                tokens[n] = torch.tensor(win, dtype=dt)
                lengths[n] = REGION_LEN
                pf.write(json.dumps({"idx": n, "src": f"txt-{hex(_stable_hash(txt[:512].encode('utf-8', 'ignore')))[2:10]}",
                                     "start": start, "len": REGION_LEN}) + "\n")
                n += 1
                if n >= CAP_REGION:
                    break
            if n >= CAP_REGION:
                print("[v3-pack] WARN: region cap hit at", n)
                break
    tokens, lengths = tokens[:n], lengths[:n]
    _staged_save({"tokens": tokens, "lengths": lengths,
                  "vocab_size": tok.get_vocab_size()}, args.reg_out)
    print(f"[v3-pack] region pack: {n} x {REGION_LEN} -> {args.reg_out}")

    meta = {"vocab": tok.get_vocab_size(),
            "base": base,
            "region": {"rows": n, "ctx": REGION_LEN},
            "seconds": round(time.time() - t0, 1),
            "note": ("32K BPE (arXiv 2407.13623 bracket for <=600M); "
                     "region_expand = adjacent 2048 windows (LongRoPE "
                     "progressive extension, arXiv 2402.13753)")}
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    print("[v3-pack] meta ->", args.meta)
    print("[v3-pack] done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()