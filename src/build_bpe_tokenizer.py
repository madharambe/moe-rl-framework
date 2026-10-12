"""
Rebuild BPE tokenizer + GPU-ready corpus from data/raw/*.py files.
Replicates the original recipe: byte-level BPE, vocab 16384,
specials <PAD>/<UNK>/<EOS>. Outputs:
  data/processed/bpe_tokenizer/tokenizer.json
  data/processed/code_corpus_bpe.pt  {tokens: [N, MAX_LEN] int64, lengths, vocab_size}
"""
import sys
import hashlib
import os
from array import array
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_RAW = BASE / "data" / "raw"
DATA_PROCESSED = BASE / "data" / "processed"
TOK_DIR = DATA_PROCESSED / "bpe_tokenizer"

VOCAB_SIZE = 16384
MAX_LEN = 1024
MIN_CHUNK = 50
# Aug 2 fix: was 20000 -> hard-capped the corpus at ~20M tokens. A 430M model
# needs billions of tokens (Chinchilla ~20 tokens/param); with more raw repos
# cloned, raise the cap to 500k sequences (~512M tokens).
# B-55 (Aug 8): the PyPI crawl tripled the harvest — the 500k cap bound the
# corpus at 439.6M tokens. 1M seqs = ~1B tokens; the encode tensor is 2 GB
# (streaming keeps the rest of RAM low).
MAX_SEQS = 1000000

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def _stable_hash(s: bytes) -> str:
    return hashlib.sha1(s).hexdigest()


def collect_texts():
    seen, texts = set(), []
    files = sorted(DATA_RAW.rglob("*.py"))
    for f in files:
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(t) < 50:
            continue
        # deterministic cross-run dedup (hash() is process-local and salted)
        h = _stable_hash(t.encode("utf-8", errors="replace"))
        if h in seen:
            continue
        seen.add(h)
        texts.append(t)
    print(f"[Corpus] {len(files)} files -> {len(texts)} unique texts ({sum(len(t) for t in texts)/1e6:.1f} MB)")
    return texts


def build_tokenizer(texts, out_dir=None):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<PAD>", "<UNK>", "<EOS>"],
    )
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    # Sep-5 INCIDENT: this function used to save to the hardcoded LIVE path —
    # an audit determinism test calling build_tokenizer() silently overwrote
    # the Aug-2 production tokenizer.json (id<->token mapping changed, text-side
    # stack affected). Now: REQUIRES an explicit out_dir; the live path is only
    # ever written via the CLI default (`--token_dir`) once the user runs the
    # real build. No default parameter here — callers must say where to save.
    if out_dir is None:
        raise ValueError(
            "build_tokenizer() refuses to guess a save location (the Sep-5 "
            "incident). Pass out_dir explicitly.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    print(f"[Tokenizer] BPE vocab={tok.get_vocab_size()} -> {out_dir / 'tokenizer.json'}")
    return tok


# ROUND-204ac-b6.3: the corpus was 100% python (_iter_text_files collected
# *.py only) — the v3c pack builder can widen the language mix + drop junk
# files by setting these module attrs BEFORE the legs run. Defaults keep the
# historical .py-only behavior (the mid-flight v3b build is unaffected).
TEXT_EXTS = (".py",)
JUNK_FILE_RE = None


def _iter_text_files():
    """Stream unique text files one at a time (never hold all texts in RAM —
    earlyoom kills at <=10% mem free on this machine)."""
    seen = set()
    junk = JUNK_FILE_RE
    for f in sorted(DATA_RAW.rglob("*")):
        if not any(str(f).endswith(e) for e in TEXT_EXTS):
            continue
        if junk and junk.search(str(f)):
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(t) < 50:
            continue
        h = _stable_hash(t.encode("utf-8", errors="replace"))
        if h in seen:
            continue
        seen.add(h)
        yield t


def build_corpus(tok, texts=None, out_path=None):
    # Stream into a preallocated tensor: keeps peak RAM ~1 GB lower than the
    # list-of-chunks approach (this machine's earlyoom SIGTERMs at <=10% mem).
    cap = MAX_SEQS
    toks_dtype = torch.int16 if tok.get_vocab_size() <= 32767 else torch.int32
    # Sep-5 (audit): the array width must match the corpus dtype — with int32
    # tokens (vocab > 32767) the old array('H') hashed HALF the row bytes, so
    # dedup silently re-appended everything (and ids > 65535 raised
    # OverflowError). Build the hash from the dtype-consistent view.
    _arr_code = 'I' if toks_dtype == torch.int32 else 'H'
    tokens = torch.zeros(cap, MAX_LEN, dtype=toks_dtype)
    lengths = torch.zeros(cap, dtype=torch.long)
    seen = set()
    n = 0
    src = texts if texts is not None else _iter_text_files()
    for t in src:
        ids = tok.encode(t).ids
        for start in range(0, len(ids), MAX_LEN):
            chunk = ids[start:start + MAX_LEN]
            if len(chunk) < MIN_CHUNK:
                continue
            key = _stable_hash(array(_arr_code, chunk).tobytes())  # dtype-matched width
            if key in seen:
                continue
            seen.add(key)
            tokens[n, :len(chunk)] = torch.tensor(chunk, dtype=toks_dtype)
            lengths[n] = len(chunk)
            n += 1
            if n >= cap:
                break
        if n >= cap:
            # Sep-5 (audit): the old silent break left an unbalanced corpus
            # with no hint — say so clearly (and count what was dropped).
            print(f"[Corpus] WARN: MAX_SEQS cap ({cap}) hit at {n} rows — "
                  f"further inputs DROPPED from this build")
            break

    tokens, lengths = tokens[:n], lengths[:n]
    if out_path is None:
        # ROUND-169 (F4): mirror build_tokenizer — an omitted out_path must
        # REFUSE, not silently write the LIVE corpus (the exact incident class
        # the Sep-5 --live gate was meant to close, reachable by any stray
        # test calling build_corpus(tok) bare).
        raise ValueError(
            "build_corpus() refuses to guess a save location (the Sep-5 "
            "incident). Pass out_path explicitly (CLI: use --live).")
    out_path = Path(out_path)
    # B-34: stage the tmp NEXT TO THE RESOLVED TARGET (the external drive) —
    # os.replace() across devices raises EXDEV; the local-data/processed tmp
    # could never replace the symlinked drive target.
    tmp_path = Path(out_path.resolve()).with_suffix(".pt.tmp")
    torch.save({
        "tokens": tokens,
        "lengths": lengths,
        "vocab_size": tok.get_vocab_size(),
    }, tmp_path)
    # B-113 parity: fsync before rename (ntfs-3g flushes pages lazily — without
    # this a power-loss/SSD-unplug could corrupt the just-saved corpus).
    with open(tmp_path, "rb+") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    tmp_path.replace(out_path.resolve())  # atomic + symlink-safe (external drive)
    print(f"[Corpus] {n} sequences (max {MAX_LEN}) -> {out_path}")


def append_to_corpus(tok, new_texts, max_seqs: int = None):
    """Incrementally append newly-encoded chunks to the existing corpus .pt.
    Used by the LLM's `clone_repo_to_corpus` tool (data self-improvement):
    reads the current corpus, appends only chunks whose sha1 is unseen, saves
    atomically. Returns (added_seqs, total_seqs)."""
    import gc
    cap = max_seqs or MAX_SEQS
    out_path = DATA_PROCESSED / "code_corpus_bpe.pt"
    toks_dtype = torch.int16 if tok.get_vocab_size() <= 32767 else torch.int32

    if out_path.exists():
        ck = torch.load(out_path, map_location="cpu", weights_only=True)
        tokens, lengths = ck["tokens"], ck["lengths"]
        del ck
    else:
        tokens = torch.zeros(0, MAX_LEN, dtype=toks_dtype)
        lengths = torch.zeros(0, dtype=torch.long)

    def _row_bytes(r):
        # B-116: the old r.tolist() materialized a ~40KB Python-int list PER
        # CORPUS ROW (661k rows -> multi-GB churn + GC pressure during the
        # dedup-set build); numpy .tobytes() is a cheap contiguous copy
        return array('H', r.numpy(force=True).tobytes()).tobytes()

    # B-116b: hash only each row's VALID prefix (lengths[i]) — the old build
    # hashed full padded rows while lookups hash unpadded chunks, so any
    # sub-MAX_LEN chunk could NEVER match and was re-appended forever
    seen = {_stable_hash(_row_bytes(tokens[i, :int(lengths[i])]))
            for i in range(len(lengths))}
    new_rows = []
    new_lens = []
    for t in new_texts:
        ids = tok.encode(t).ids
        for start in range(0, len(ids), MAX_LEN):
            chunk = ids[start:start + MAX_LEN]
            if len(chunk) < MIN_CHUNK:
                continue
            # Sep-5: dtype-matched width (mirrors build_corpus) — array('H')
            # hashed HALF an int32 row so dedup re-appended everything on a
            # vocab >32767 corpus.
            key = _stable_hash(array('I' if tokens.dtype == torch.int32 else 'H',
                                      chunk).tobytes())
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(chunk)
            new_lens.append(len(chunk))
            if len(new_rows) + len(tokens) >= cap:
                break
        if len(new_rows) + len(tokens) >= cap:
            break
    del seen
    if not new_rows:
        return 0, len(tokens)

    n_new = len(new_rows)
    new_block = torch.zeros(n_new, MAX_LEN, dtype=toks_dtype)
    for i, row in enumerate(new_rows):
        new_block[i, :len(row)] = torch.tensor(row, dtype=toks_dtype)
    merged = torch.cat([tokens, new_block])
    del tokens, new_block  # B-116: free the old copy BEFORE the big save
    new_rows.clear()
    merged_lens = torch.cat([lengths, torch.tensor(new_lens, dtype=torch.long)])
    # B-34: tmp must sit on the SAME device as the resolved (drive) target
    tmp_path = Path(out_path.resolve()).with_suffix(".pt.tmp")
    torch.save({"tokens": merged, "lengths": merged_lens,
                "vocab_size": tok.get_vocab_size()}, tmp_path)
    with open(tmp_path, "rb+") as _f:
        _f.flush()
        os.fsync(_f.fileno())
    tmp_path.replace(out_path.resolve())  # symlink-safe (external drive)
    del merged
    gc.collect()
    print(f"[Corpus+] appended {n_new} seqs -> {len(merged_lens)} total")
    return n_new, len(merged_lens)


if __name__ == "__main__":
    import gc, argparse, tempfile
    ap = argparse.ArgumentParser(description="Build the BPE tokenizer + corpus")
    ap.add_argument("--live", action="store_true",
                    help="Write to the LIVE path (data/processed/bpe_tokenizer). "
                         "DANGEROUS: the Sep-5 incident destroyed the production "
                         "tokenizer because a test ran the build bare. Default is "
                         "a TEMP dir; use --live only for the deliberate rebuild.")
    ap.add_argument("--token_out", default=None,
                    help="tokenizer.json output dir (default: a temp dir unless "
                         "--live)")
    args = ap.parse_args()
    if args.token_out is None:
        args.token_out = str(TOK_DIR) if args.live else tempfile.mkdtemp(prefix="bpe_build_")
    if args.live:
        print("[WARN] --live: writing the PRODUCTION tokenizer.json — this "
              "replaces the one used by every adapter/eval. Triple-check the "
              "source data first.", flush=True)
    texts = collect_texts()
    tok = build_tokenizer(texts, out_dir=Path(args.token_out))
    del texts  # free ~0.6 GB before the encode pass (earlyoom headroom)
    gc.collect()
    # ROUND-5 (F-10): build_corpus ALSO defaults to the LIVE corpus path — a
    # bare (non---live) run would still have destroyed code_corpus_bpe.pt.
    # Route to a temp out_path unless --live.
    if args.live:
        build_corpus(tok, out_path=DATA_PROCESSED / "code_corpus_bpe.pt")
    else:
        build_corpus(tok, out_path=Path(tempfile.mkdtemp(prefix="bpe_build_"))
                     / "code_corpus_bpe.pt")
        print("[WARN] non---live run: corpus written to a TEMP dir — nothing "
              "was touched on the live paths.")
