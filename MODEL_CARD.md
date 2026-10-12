---
license: apache-2.0
language:
- en
library_name: pytorch
tags:
- mixture-of-experts
- moe
- text-generation
- code-generation
- small-language-models
- from-scratch
pipeline_tag: text-generation
---

# moe-rl-framework — MoE language model family (~50M → ~587M parameters)

A from-scratch Mixture-of-Experts transformer, the language-model half of the
[moe-rl-framework](https://github.com/YOUR_GH_USERNAME/moe-rl-framework) project. It is
used there as the code-generation and self-refinement engine for an RL robotics stack,
and is small enough to train and serve on a 6 GB laptop GPU.

## Model details

| | |
|---|---|
| Parameters | a config family: `mini` ~50M · `plus` ~121M · `xl` ~225M · `430m` 444M dense · `v3` ~587M (32K vocab) |
| Architecture | decoder-only transformer, alternating dense / sparse layers (3 dense SwiGLU + 3 sparse MoE) |
| Attention | Grouped-Query Attention — 8 query heads / 4 KV heads (~50% KV-cache reduction) |
| Routing | top-2 of 8 experts + 1 always-on shared expert, per MoE layer |
| Positional | RoPE |
| Activations / norm | SwiGLU, RMSNorm (pre-norm) |
| Stability | load-balancing loss + router z-loss |
| Optimizer | Muon (implementation included in the repo) |
| Adaptation | LoRA adapters and int8 quantization included |

## Intended use
- research on small-scale MoE training dynamics and routing
- code generation / self-refinement inside an agent loop, on consumer hardware
- a readable reference implementation of MoE, GQA, Muon, LoRA and int8 in one place

## Limitations (stated plainly)
- **These models are small.** Expect useful code completion and short reasoning, not
  general assistant behaviour.
- It was trained on a code/instruction corpus collected for this project; it has **no
  safety alignment** and should not be exposed to untrusted users without one.
- **No checkpoints are published in this release.** The architecture and the training
  code are complete and runnable; the weights stay local because they are not yet good
  enough to be worth other people's disk space. Everything on this card describes what
  the code builds and how it is trained, not a downloadable artefact.

## How to use
```python
import sys; sys.path.insert(0, 'src')
import torch
from moe_model import MoE  # see src/moe_model.py for the exact constructor args

model = MoE(vocab_size=..., n_layer=6, n_head=8, n_kv_head=4, n_experts=8,
            n_experts_per_token=2, use_shared_expert=True).eval().cuda()
state = torch.load('moe.pt', map_location='cuda')
model.load_state_dict(state)
```

## Training
```bash
python src/train_moe.py --max_iters 5000 --lr 3e-4      # see the repo for flags
```
The corpus pipeline lives in `src/data_scraper.py` and `src/clone_corpus.py`; corpora
themselves are never committed (see `.gitignore`).

## Citation / references
Architecture choices were informed by the published work listed in the repository
README (LibMoE, Tiny-MoE, NanoChat-MoE, GQA), all verified against primary sources.

## Status
Training code and architecture: **complete and runnable**. Checkpoints: **not published** —
the models trained so far are an honest work in progress and are being kept local until
they are worth releasing. If you want one, run the trainers: a `mini` (~50M) pretraining
pass fits in a few GB of VRAM and about forty minutes on a 6 GB laptop GPU, and
`src/train_ambatron.py` is the entry point. The repository is the source of truth.