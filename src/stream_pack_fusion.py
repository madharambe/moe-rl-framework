#!/usr/bin/env python3
# 205au: STREAM->PACK FUSION — no staging: stream permissive code corpora
# from HF, gate them (license/length/dedup), tokenize with the existing v3
# 32K BPE, and append straight into a preallocated int16 pack on the SSD
# (int16: vocab 32768 fits exactly; 2 bytes/token = 2x the disk of int32).
import argparse, hashlib, json, os, re, time
import numpy as np
_START_T = time.time()

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--source', action='append', default=[])
ap.add_argument('--jsonl-inputs', nargs='*', default=[])
ap.add_argument('--seg-tokens', type=int, default=500_000_000)
ap.add_argument('--repair', action='store_true',
                help='scan *.pack without a .meta.pt, write their metas '
                     'from the last non-zero row, then exit')
ap.add_argument('--max-tokens', type=int, default=3_200_000_000)
ap.add_argument('--row', type=int, default=1024)
ap.add_argument('--min-lines', type=int, default=12)
ap.add_argument('--max-file-kb', type=int, default=300)
ap.add_argument('--keep-ratio', type=float, default=0.9)
ap.add_argument('--fast', action='store_true')
ap.add_argument('--fast-k', type=int, default=4)
ap.add_argument('--fast-shards', type=int, default=12)
args = ap.parse_args()

BAD_LIC = re.compile(r'\b(gpl|agpl|sspl|proprietary|commercial)\b', re.I)

from tokenizers import Tokenizer

# Portable root: override with RSI_HOME, otherwise use this checkout.
import os as _os_rsi
from pathlib import Path as _Path_rsi
RSI_HOME = _os_rsi.environ.get('RSI_HOME') or str(_Path_rsi(__file__).resolve().parent.parent)

tok = Tokenizer.from_file(RSI_HOME + '/v3-bank/tokenizer_v3.json')
vocab = tok.get_vocab_size()
seg_rows = max(1, args.seg_tokens // args.row)
_state = {'seg': 0, 'row_in_seg': 0, 'mm': None, 'seg_rows_left': seg_rows}

def _open_seg():
    base, ext = os.path.splitext(args.out)
    if _state['seg'] == 0:
        path = args.out
    else:
        path = f"{base}_{_state['seg']:02d}.pack"
    _state['mm'] = np.memmap(path, dtype=np.int16, mode='w+',
                             shape=(seg_rows, args.row))
    _state['path'] = path
    print(f'[fz] segment {_state["seg"]} -> {path} ({seg_rows*args.row/1e6:.0f}M tok)', flush=True)

cap_rows = args.max_tokens // args.row

if args.repair:
    # 205ax: recover the interrupted segment packs — scan for the last
    # non-zero row (a preallocated memmap tail is all zeros; a real 1024-
    # token row of all-zero ids is not producible) and write the .meta.pt
    # the CORPUS_PACKS chain needs.
    import glob as _glob
    import numpy as _np2
    import json as _js2
    _segs = sorted(set(_glob.glob(args.out.replace('.pack', '_*.pack')))
                   | {args.out})
    for _pp in _segs:
        _meta_p = _pp + '.meta.pt'
        if os.path.exists(_meta_p):
            print(f'[repair] {_pp} already has a meta; skip')
            continue
        _rows_max = os.path.getsize(_pp) // (args.row * 2)
        if _rows_max <= 0:
            print(f'[repair] {_pp} empty; skip')
            continue
        _mm = _np2.memmap(_pp, dtype=_np2.int16, mode='r', shape=(_rows_max, args.row))
        _last = 0
        for _i in range(_rows_max - 1, -1, -1):
            if _mm[_i].any():
                _last = _i + 1
                break
        del _mm
        _js2.dump({'rows': _last, 'row': args.row, 'vocab_size': vocab,
                   'pack_path': os.path.abspath(_pp)}, open(_meta_p.replace('.meta.pt', '.meta.json'), 'w'))
        _meta_p = _meta_p.replace('.meta.pt', '.meta.json')
        print(f'[repair] {_pp}: {_last}/{_rows_max} rows valid '
              f'({_last*args.row/1e6:.0f}M tok) -> meta written', flush=True)
    raise SystemExit(0)

_open_seg()
from datasets import load_dataset

def text_of(ex):
    for k in ('content', 'text', 'code', 'raw_content'):
        if isinstance(ex.get(k), str):
            return ex[k]
    return None

def lic_of(ex):
    v = ex.get('license') or ex.get('license_type')
    if isinstance(v, (list, tuple)) and v:
        return str(v[0])
    return str(v) if isinstance(v, str) else ''

# 205cr: BOUND the dedup memory — an unbounded set over a 24B run
# reaches GBs of RSS (the RAM-pressure root). Rolling window: at the
# cap, reset the set (near-dup bursts inside the window still dedup).
DEDUP_CAP = int(os.environ.get("RSI_DEDUP_CAP", "3000000"))


def _dedup_add(sha_set, digest):
    if len(sha_set) >= DEDUP_CAP:
        sha_set.clear()
    if digest in sha_set:
        return False
    sha_set.add(digest)
    return True


seen = set()
row_n = 0
carry = []
_pend = []
pend = []
n_files = kept_bytes = 0
t0 = time.time()

def _close_seg():
    """205ax: write each segment's .meta.pt AT SEGMENT CLOSE — the old
    final-only write orphaned every interrupted segment (the metas are what
    the CORPUS_PACKS chain reads; without them the pack is invisible)."""
    _p = _state['mm'].filename
    _r = _state['row_in_seg']
    _state['mm'].flush()
    if _r > 0:
        # 205bi: JSON metas (NOT torch) — importing torch initialized a CUDA
        # context in the fusion (2x2.4GB!) which OOM'd the GPU jobs and
        # then D-stuck on the USB2 disk (204ah-108)
        import json as _js
        _js.dump({'rows': _r, 'row': args.row, 'vocab_size': vocab,
                  'pack_path': os.path.abspath(_p)}, open(_p + '.meta.json', 'w'))
        print(f'[fz] segment {_state["seg"]} closed: {_r} rows '
              f'({_r*args.row/1e6:.0f}M tok) -> {_p}.meta.json', flush=True)


def flush(ids):
    global row_n
    ids = list(ids)
    while ids and row_n < cap_rows:
        if _state['row_in_seg'] >= seg_rows:
            _close_seg()
            _state['row_in_seg'] = 0
            _state['seg'] += 1
            _open_seg()
        room = seg_rows - _state['row_in_seg']
        full = min(len(ids) // args.row, room)
        for i in range(full):
            _state['mm'][_state['row_in_seg']] = ids[i*args.row:(i+1)*args.row]
            _state['row_in_seg'] += 1
            row_n += 1
        if full:
            ids = ids[full*args.row:]
        elif len(ids) < args.row:
            break
    return ids

for _jp in args.jsonl_inputs:
    if not os.path.exists(_jp):
        continue
    for line in open(_jp, 'r', encoding='utf-8', errors='ignore'):
        try:
            _ex = json.loads(line)
        except Exception:
            continue
        _txt = _ex.get('text')
        if not _txt or chr(0) in _txt:
            continue
        _h = hashlib.sha256(_txt.encode('utf-8', 'ignore')).digest()
        if not _dedup_add(seen, _h):
            continue
        n_files += 1
        kept_bytes += len(_txt)
        # 205cd: batch the encodes (768 texts) — the per-record call
        # serializes the tokenizer's threadpool; batching engages all cores
        _pend.append(_txt)
        if len(_pend) >= 768:
            for enc in tok.encode_batch(_pend):
                carry.extend(enc.ids)
            _pend.clear()
        if len(carry) >= args.row * 256:
            carry = flush(carry)
    print(f'[fz] jsonl {_jp} folded ({n_files} files, {row_n*args.row/1e6:.0f}M tok)', flush=True)
    if row_n >= cap_rows:
        break

for spec in args.source:
    parts = spec.split(':')
    ddir = None
    if parts[0] == 'ddir':
        # 205bv: the Stack-style corpora use data_dir='data/<LANG>' (the
        # gateway passed; the per-language parquet trees)
        ddir, ds_id = parts[1], parts[2]
        config = parts[3] if len(parts) > 3 and parts[3] else None
        split = parts[4] if len(parts) > 4 else 'train'
    else:
        ds_id = parts[0]
        config = parts[1] if len(parts) > 1 and parts[1] else None
        split = parts[2] if len(parts) > 2 else 'train'
    if row_n * args.row >= args.max_tokens:
        break
    try:
        if ddir:
            ds = load_dataset(ds_id, data_dir=ddir, split=split, streaming=True)
        else:
            ds = load_dataset(ds_id, config, split=split, streaming=True)
    except Exception as e:
        print(f'[fz] SKIP {ds_id}: {str(e)[:110]}', flush=True)
        continue
    print(f'[fz] streaming {ds_id}:{config}:{split}', flush=True)
    if args.fast:
        import urllib.request as _ur
        import pyarrow.dataset as _pds
        import concurrent.futures as _cf
        from queue import Queue as _Q
        def _shard_paths(a_repo, a_dir):
            _trq = _ur.Request(
                f'https://huggingface.co/api/datasets/{a_repo}/tree/main/'
                f'{a_dir}', headers={'Authorization': 'Bearer ' + os.environ.get('HF_TOKEN', '')})
            _tree = json.load(_ur.urlopen(_trq))
            _fs = [t['path'] for t in _tree
                   if t.get('type') == 'file' and t['path'].endswith('.parquet')]
            return _fs[:args.fast_shards]
        _dir = ddir or ('sample/10BT' if not config else config)
        try:
            _paths = _shard_paths(ds_id, _dir)
            if not _paths and config and '-' in config:
                # 205ck: the HF config NAME differs from the tree dir
                # for the edu sets ('sample-10BT' cfg vs 'sample/10BT')
                _paths = _shard_paths(ds_id, config.replace('-', '/', 1))
        except Exception:
            try:
                _paths = _shard_paths(ds_id, 'sample/10BT')
                _dir = 'sample/10BT'
            except Exception:
                print(f'[fz-fast] no shards for {ds_id}', flush=True)
                continue
        print(f'[fz-fast] {ds_id} {_dir}: {len(_paths)} shards '
              f'K={args.fast_k} streams', flush=True)
        import urllib.request as _ur2
        def _run_shard(p):
            _tmp = '/tmp/' + p.replace('/', '_').replace(' ', '_')
            _u = ('https://huggingface.co/datasets/' + ds_id +
                  '/resolve/main/' + _ur2.quote(p) + '?download=true')
            try:
                _req = _ur2.Request(_u, headers={
                    'Authorization': 'Bearer ' + os.environ.get('HF_TOKEN', '')})
                with _ur2.urlopen(_req) as _r, open(_tmp, 'wb') as _f:
                    while True:
                        _b = _r.read(4 * 1024 * 1024)
                        if not _b:
                            break
                        _f.write(_b)
                return _pds.dataset(_tmp, format='parquet')
            except Exception as _e:
                print(f'[fz-fast] shard fail {p[:60]}: {str(_e)[:90]}', flush=True)
                return None
        _dq = _Q(maxsize=args.fast_k)      # datasets awaiting the scan
        _rowq = _Q(maxsize=args.fast_k * 4)  # materialized row-lists
        def _producer(p):                    # downloader: dataset -> _dq
            _tmp = '/tmp/' + p.replace('/', '_').replace(' ', '_')
            _u = ('https://huggingface.co/datasets/' + ds_id +
                  '/resolve/main/' + _ur2.quote(p) + '?download=true')
            try:
                _req = _ur2.Request(_u, headers={
                    'Authorization': 'Bearer ' + os.environ.get('HF_TOKEN', '')})
                _tmp2 = _tmp + '.dl'
                with _ur2.urlopen(_req) as _r, open(_tmp2, 'wb') as _f:
                    while True:
                        _b = _r.read(4 * 1024 * 1024)
                        if not _b:
                            break
                        _f.write(_b)
                _dq.put(_pds.dataset(_tmp2, format='parquet'))
            except Exception as _e:
                print(f'[fz-fast] shard fail {p[:60]}: {str(_e)[:90]}', flush=True)
                _dq.put(None)
        def _scanner():                      # rows: dataset <- _dq -> _rowq
            while True:
                _d = _dq.get()
                if _d is None:
                    _rowq.put(None); return
                try:
                    import pyarrow.compute as _pc
                    for _b in _d.iter_batches(batch_size=65536):
                        # 205cj: VECTORIZED GATES on the Arrow batch: the
                        # license regex, the byte-size cap and the line-count
                        # all run C-side; only the survivors cross to Python.
                        txt = _b.column(0)
                        mask = ~txt.is_null()
                        if 'license' in _b.column_names:
                            lic_ok = _b['license'].is_null() | ~_pc.match_substring_regex(
                                _b['license'].fill_null(''), BAD_LIC.pattern)
                            mask = mask & lic_ok
                        lens = _pc.utf8_length(txt.fill_null(''))
                        mask = mask & (lens >= 40) & (lens <= args.max_file_kb * 1024)
                        nls = _pc.utf8_count_substring(txt.fill_null(''), '\n')
                        mask = mask & (nls >= args.min_lines)
                        surv = txt.filter(mask)
                        if surv.num_rows:
                            _rowq.put(surv.to_pylist())
                except Exception:
                    pass
                _rowq.put(None)
        with _cf.ThreadPoolExecutor(max_workers=args.fast_k + 1) as _ex:
            for _p in _paths:
                _ex.submit(_producer, _p)
            for _i in range(args.fast_k):
                _ex.submit(_scanner)
            _done = 0
            while _done < args.fast_k:
                _rows = _rowq.get()
                if _rows is None:
                    _done += 1; continue
                if row_n >= cap_rows:
                    continue
                for ex in _rows:
                    txt = str(ex.get('text') or ex.get('content') or '')
                    if not txt or '\x00' in txt or len(txt) > args.max_file_kb * 1024:
                        continue
                    lic = str(ex.get('license') or ex.get('licence') or '')
                    if lic and BAD_LIC.search(lic):
                        continue
                    if sum(1 for ln in txt.split('\n') if ln.strip()) < args.min_lines:
                        continue
                    if (hash(txt) % 1000) / 1000.0 > args.keep_ratio:
                        continue
                    hh = hashlib.md5(txt.encode('utf-8', 'ignore')).digest()
                    if not _dedup_add(seen, hh):
                        continue
                    n_files += 1
                    kept_bytes += len(txt)
                    pend.append(txt)
                    if len(pend) >= 768:
                        for enc in tok.encode_batch(pend):
                            carry.extend(enc.ids)
                        pend.clear()
                    while len(carry) >= args.row * 256:
                        carry = flush(carry)
                        if row_n and row_n % 4000 == 0:
                            el = time.time() - _START_T
                            print(f'[fz] rows {row_n} ({row_n*args.row/1e6:.0f}M tok) '
                                  f'files {n_files} {el:.0f}s {kept_bytes/1e6/max(el,1):.1f}MB/s', flush=True)
        continue   # fast mode consumed this source
    try:
        for ex in ds:
            txt = text_of(ex)
            if not txt or '\x00' in txt or len(txt) > args.max_file_kb*1024:
                continue
            lic = lic_of(ex)
            if lic and BAD_LIC.search(lic):
                continue
            lines = txt.split('\n')
            if sum(1 for ln in lines if ln.strip()) < args.min_lines:
                continue
            if (hash(txt) % 1000) / 1000.0 > args.keep_ratio:
                continue
            h = hashlib.sha256(txt.encode('utf-8', 'ignore')).digest()
            if not _dedup_add(seen, h):
                continue
            n_files += 1
            kept_bytes += len(txt)
            pend.append(txt)
            if len(pend) >= 768:
                for enc in tok.encode_batch(pend):
                    carry.extend(enc.ids)
                pend.clear()
            if len(carry) >= args.row * 256:
                carry = flush(carry)
                if row_n and row_n % 4000 == 0:
                    el = time.time() - _START_T
                    print(f'[fz] rows {row_n} ({row_n*args.row/1e6:.0f}M tok) '
                          f'files {n_files} {el:.0f}s {kept_bytes/1e6/max(el,1):.1f}MB/s', flush=True)
            if row_n >= cap_rows:
                break
    except Exception as e:
        print(f'[fz] {ds_id} stopped: {str(e)[:130]}', flush=True)
if _state['mm'] is not None:
    _close_seg()
    _rows_done = _state['row_in_seg']

# 205bi: torch-free end-of-run bookkeeping (the torch import held a 2.4GB
# CUDA context in the fusion = the 204ah-108 double-booking).
ntok = row_n * args.row
base, _e = os.path.splitext(args.out)
segs = []
if _state['seg'] == 0:
    segs = [(_state['path'], _rows_done)]
else:
    for s_i in range(_state['seg']):
        segs.append((f"{base}_{s_i:02d}.pack", seg_rows))
    segs.append((_state['path'], _rows_done))
meta_chunks = []
import json as _jfinal
for _pp, _rr in segs:
    if _rr <= 0:
        continue
    mpath = f"{_pp}.meta.json"
    _jfinal.dump({'rows': _rr, 'row': args.row, 'vocab_size': vocab,
                  'pack_path': os.path.abspath(_pp)}, open(mpath, 'w'))
    meta_chunks.append(mpath)
_jfinal.dump({'segments': meta_chunks, 'total_tokens': int(ntok)},
             open(args.out + '.chain.json', 'w'))
rep = {'tokens': ntok, 'rows': row_n, 'files': n_files,
       'dn_unique': round(ntok / 587e6, 2)}
print('[fz] DONE', json.dumps(rep), '->', args.out)
print('[fz] + v3b 410M => combined D/N =',
      round((ntok + 410_000_000) / 587e6, 2))
