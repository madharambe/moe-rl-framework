#!/usr/bin/env python3
# 205as: PACK the harvested corpora with the EXISTING v3 32K tokenizer
# (the b5.11 lesson: reuse --tok, never retrain BPE) into the trainer's
# {tokens, lengths, vocab_size} format. Explicit paths only.
import argparse, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tok', default=RSI_HOME + '/v3-bank/tokenizer_v3.json')
ap.add_argument('--row', type=int, default=1024)
ap.add_argument('--max-tokens', type=int, default=650_000_000)
args = ap.parse_args()

from tokenizers import Tokenizer

# Portable root: override with RSI_HOME, otherwise use this checkout.
import os as _os_rsi
from pathlib import Path as _Path_rsi
RSI_HOME = _os_rsi.environ.get('RSI_HOME') or str(_Path_rsi(__file__).resolve().parent.parent)

tok = Tokenizer.from_file(args.tok)
vocab = tok.get_vocab_size()
print(f'[pack] v3 tokenizer vocab={vocab}', flush=True)

cap_rows = args.max_tokens // args.row
rows = np.zeros((cap_rows, args.row), dtype=np.int32)
row_n = 0
carry = []
t0 = time.time()
n_files = 0

def flush_into(ids):
    global row_n
    n = len(ids)
    full = n // args.row
    for i in range(full):
        if row_n >= cap_rows:
            break
        rows[row_n] = ids[i*args.row:(i+1)*args.row]
        row_n += 1
    return ids[full*args.row:]

for path in args.inputs:
    if not os.path.exists(path):
        print(f'[pack] skip {path}')
        continue
    for line in open(path, 'r', encoding='utf-8', errors='ignore'):
        try:
            ex = json.loads(line)
        except Exception:
            continue
        text = ex.get('text')
        if not text:
            continue
        n_files += 1
        for enc in tok.encode_batch([text]):
            carry.extend(enc.ids)
        if len(carry) >= args.row * 256:
            carry = flush_into(carry)
            if row_n % 3000 == 0 and row_n:
                el = time.time() - t0
                print(f'[pack] rows {row_n} ({row_n*args.row/1e6:.0f}M tok) '
                      f'files {n_files} {el:.0f}s', flush=True)
            if row_n >= cap_rows:
                break
    if row_n >= cap_rows:
        break
if carry and row_n < cap_rows:
    r = flush_into(carry)
    if r and row_n < cap_rows:
        rows[row_n, :len(r)] = r
        row_n += 1

tokens = torch.from_numpy(rows[:row_n].copy())
lengths = torch.full((row_n,), args.row, dtype=torch.int32)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
tmp = args.out + '.tmp'
torch.save({'tokens': tokens, 'lengths': lengths, 'vocab_size': vocab}, tmp)
with open(tmp, 'rb+') as f:
    os.fsync(f.fileno())
os.replace(tmp, args.out)
ntok = row_n * args.row
rep = {'rows': row_n, 'tokens': ntok, 'files': n_files,
       'dn_unique_vs_587M': round(ntok / 587e6, 3),
       'vs_old_unique_13.6M': round(ntok / (13333*1024), 1)}
print('[pack] DONE', json.dumps(rep), '->', args.out)
json.dump(rep, open(args.out + '.build.json', 'w'), indent=1)
