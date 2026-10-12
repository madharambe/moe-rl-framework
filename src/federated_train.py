#!/usr/bin/env python3
# 205az: FEDERATED SESSION TRAINING — the privacy-preserving distribution
# path (the FedLLM/PEFT pattern, arXiv 2503.12016): a user runs this ON
# THEIR OWN MACHINE over THEIR OWN sessions; only the LoRA ADAPTER +
# a provenance manifest ever leaves. The raw sessions never move.
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--sessions', required=True,
                help='session export: the irl_pool.jsonl format, or a plain '
                     'jsonl of {prompt, completion} / conversation turns')
ap.add_argument('--base', required=True, help='the RSI base ckpt (shared)')
ap.add_argument('--out-dir', required=True)
ap.add_argument('--steps', type=int, default=600)
ap.add_argument('--lr', type=float, default=3e-5)
ap.add_argument('--r', type=int, default=16)
ap.add_argument('--alpha', type=int, default=32)
ap.add_argument('--consent', default='opt-in-federated',
                help='consent tier recorded in the manifest')
args = ap.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

def load_records(path):
    recs = []
    for line in open(path, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        # irl_pool shape: tool steps -> a compact tool transcript
        if d.get('steps'):
            lines = []
            for s in d['steps'][:40]:
                inp = json.dumps(s.get('input', {}))[:300]
                lines.append(f"call {s.get('tool')}({inp}) -> {s.get('status')}")
            if lines:
                recs.append({'segments': [
                    ['user', f"User: continue the task using tools\n"],
                    ['assistant', "\n".join(lines) + "\n"],
                ]})
        elif isinstance(d.get('segments'), list):
            recs.append(d)
        elif d.get('prompt') and d.get('completion'):
            recs.append({'segments': [['user', f"User: {d['prompt']}\n"],
                                      ['assistant', f"Assistant: {d['completion']}"]]})
    return recs

recs = load_records(args.sessions)
print(f'[fed] {len(recs)} training records from {args.sessions}')
if not recs:
    print('[fed] nothing to train on'); sys.exit(1)

# 1) the manifest (ships with the adapter = provenance + consent)
manifest = {
    'kind': 'rsi-federated-lora',
    'base': os.path.basename(args.base),
    'base_sha': hashlib.sha256(open(args.base, 'rb').read(1 << 20)).hexdigest()[:16],
    'records': len(recs),
    'steps': args.steps,
    'lr': args.lr,
    'r': args.r, 'alpha': args.alpha,
    'consent': args.consent,
    'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'data_sha': hashlib.sha256(
        open(args.sessions, 'rb').read()).hexdigest()[:32],
    'raw_data_included': False,
}
json.dump(manifest, open(out / 'manifest.json', 'w'), indent=1)

# 2) the local LoRA fine-tune (the same trainer the base project uses)
sys.path.insert(0, RSI_HOME + '/src')
import torch

# Portable root: override with RSI_HOME, otherwise use this checkout.
import os as _os_rsi
from pathlib import Path as _Path_rsi
RSI_HOME = _os_rsi.environ.get('RSI_HOME') or str(_Path_rsi(__file__).resolve().parent.parent)

pool = out / '_local_records.jsonl'
with open(pool, 'w') as f:
    for r in recs:
        f.write(json.dumps(r) + '\n')
os.environ['FED_POOL'] = str(pool)
from train_tool_sft import _tokenize
from moe_model import MoEConfig, MoEGPT
import torch.serialization as ser
ser.add_safe_globals([MoEConfig])
from lora_adapter import add_lora, freeze_base, merge_lora, remove_lora, LoRALinear

ck = torch.load(args.base, map_location='cpu', weights_only=True)
model = MoEGPT(ck['config'])
model.load_state_dict(ck['model_state'], strict=False)
add_lora(model, r=args.r, alpha=args.alpha)
freeze_base(model)
model.to(torch.bfloat16)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(dev)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[fed] trainable LoRA: {trainable:,} params')
# NOTE: the heavy local trainer invocation is delegated to the project's
# own SFT script for the full run; this script's contract = the manifest +
# the adapter handoff, and the base project's train_tool_sft.py consumes
# FED_POOL with --sft_data for the actual optimization.
print('[fed] run: python src/train_tool_sft.py --base <base> '
      f'--sft_data {pool} --steps {args.steps} --lr {args.lr} '
      '--deploy_align   (adapter-only export; see manifest.json)')
print(f'[fed] manifest -> {out / "manifest.json"}')
