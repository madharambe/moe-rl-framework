# moe-rl-framework

> A GPU-accelerated, multi-algorithm **reinforcement-learning framework** for robotics,
> paired with a from-scratch **Mixture-of-Experts language model** that generates and
> refines its own solutions — designed to train and run on a **6 GB laptop GPU**.

**Scope of this release.** The framework, the model family, the training code, the
self-refinement loop and the evaluation suites are here and runnable. Two things are
deliberately **not** here. First, the **data-acquisition and corpus-curation pipeline** -
the part that took the experience to build, and it is not needed to use or evaluate
anything in this repository. Second, **no model checkpoints**: the models trained so far
are an honest work in progress and are being kept local rather than published as if they
were finished. Everything you need to train your own is in `src/`.
Where a result in `NOTICE` or `MODEL_CARD.md` was produced with those tools, it says so
rather than implying the numbers are reproducible from this tree alone.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ci](https://github.com/madharambe/moe-rl-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/madharambe/moe-rl-framework/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)](https://pytorch.org/)
![Weights: not published](https://img.shields.io/badge/Weights-not%20published%20yet-lightgrey)

**What is in here** — two things that share one self-refinement loop:

| | |
|---|---|
| `src/algorithms/` | PPO, SAC, TD3, DQN behind one abstract interface, with a GPU path |
| `src/moe_model.py` | a from-scratch transformer family, **~50M → ~587M**: GQA, top-2 routing, shared expert, RoPE, SwiGLU, RMSNorm, router z-loss |
| `src/self_improve.py` | generate → reflect → refine → consolidate, with LSE delta rewards |
| `src/{muon,lora_adapter,quantize}.py` | the small-hardware path: Muon optimizer, LoRA, int8 |

*(The project is developed under the internal name RSI — Reinforcement Learning System
for Intelligent Robotics. The repository is named for what it contains.)*

<!-- No demo GIF yet. Rather than a placeholder, here is the one-command
     reproduction of the two runs that produce the interesting pictures. -->

## Demo

```bash
python src/train.py --algo ppo --episodes 50        # training curves
python src/self_improve.py --moe --iterations 50    # the self-improvement loop
```

## What this is

RSI is two things in one repo:

1. **A reinforcement learning framework** — PPO, SAC, TD3 and DQN implemented from
   scratch under a single abstract interface, with GPU acceleration and a robotics
   environment built on PyBullet.
2. **A from-scratch language-model family** — `mini` (~50M), `plus` (~121M),
   `xl` (~225M), `430m` (444M dense, IMU-1 recipe) and the `v3` generation
   (~587M, 32K vocab) that the trainer here is configured for, built on modern
   used as the code-generation and self-refinement engine.

The goal is an agent that both **acts** (RL for robot control) and **improves its own
solutions** (LLM-driven generate → reflect → refine → consolidate).

## Highlights

### Reinforcement learning
- **4 algorithms** — PPO, SAC, TD3, DQN — unified under one abstract base class
- **GPU acceleration** — automatic device selection (CUDA / MPS / CPU), AMP,
  gradient clipping
- **FlashSAC-inspired** SAC — policy delay, norm-bounded critics, noise repetition,
  action rescaling
- **AE-PPO** — adaptive clipping and entropy scheduling for stable training
- **~5.3× training speedup** over the naive baseline, via flat-array replay buffers,
  single-CNN-pass updates and float32 casting

### Mixture-of-Experts language model
- **A config family, not one size** — `mini` ~50M · `plus` ~121M · `xl` ~225M ·
  `430m` 444M dense · `v3` ~587M (32K vocab), all behind one `MoEConfig`
- **Grouped-Query Attention** — 8 query heads / 4 KV heads (~50% KV-cache reduction)
- **Top-2 routing** over 8 experts + 1 shared expert per MoE layer
- **RoPE** positional embeddings, **SwiGLU** activations, **RMSNorm** pre-norm
- **Load-balancing loss** + **router z-loss** for stable expert utilisation
- Alternating dense / sparse layers (3 dense SwiGLU + 3 sparse MoE)
- **Muon optimizer** implementation, LoRA adapters, and quantization for
  small-device deployment

### Self-improvement loop
- **Reflection–consolidation cycle** — generate → reflect → refine → consolidate
- **LSE delta rewards** — cleaner learning signal from score deltas
- **Tree-guided search** — explore K refinement candidates, backtrack on failure
- **Progressive difficulty** — ramps from easy to hard
- **Cross-episode ReflectionMemory** — similarity-based retrieval with gating

## Architecture

```
src/
├── algorithms/              # RL algorithms behind one interface
│   ├── base.py              #   Abstract RLAlgorithm
│   ├── ppo.py               #   PPO — GAE, adaptive clip/entropy
│   ├── sac.py               #   SAC — twin-Q, auto-alpha, norm-bounded critics
│   ├── td3.py               #   TD3 — delayed policy updates
│   └── dqn.py               #   DQN — epsilon-greedy, target network
├── moe_model.py             # MoE transformer (GQA, top-2 routing, SwiGLU)
├── llm_agent.py             # LLMPlanner + NanoGPT/MoE backends
├── self_improve.py          # Self-improvement loop
├── muon.py                  # Muon optimizer
├── lora_adapter.py          # LoRA fine-tuning
├── quantize.py              # Quantization for small-device inference
├── gpu_accel.py             # Device resolution, torch.compile, AMP, memory budget
├── policy_network.py        # Multimodal policy (CNN visual encoder + MLP)
├── sim_environment.py       # RoboticsEnv (Gymnasium)
├── physics_engine.py        # PyBullet wrapper
├── state_representation.py  # AgentState, VisualState, TextState
├── data_scraper.py          # Code-corpus collection
├── train.py                 # Unified training entry point
└── train_moe.py             # MoE language-model training
```

## Quick start

```bash
# Train a PPO agent
python src/train.py --algo ppo --episodes 50

# Train SAC with FlashSAC features
python src/train.py --algo sac --total_steps 100000

# Train the MoE language model
python src/train_moe.py --max_iters 5000 --lr 3e-4

# Run the self-improvement loop
python src/self_improve.py --moe --iterations 50

# With PyBullet physics and rendering
python src/train.py --algo sac --use_physics --render
```

## Related work — and what this repo is not

Implementing these algorithms and architectures is a well-established exercise. **This
repo is not a novel contribution** — it is a documented, integrated implementation.
Worth knowing before you read further:

| Area | Existing work |
|---|---|
| Single-file RL implementations | [CleanRL](https://github.com/vwxyzjn/cleanrl) — the reference this design follows |
| RL algorithm library | Stable Baselines3 |
| GPT from scratch | [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat), [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| MoE from scratch | [Sayed-Husain/modern-llm-from-scratch](https://github.com/Sayed-Husain/modern-llm-from-scratch) (169M params, hand-written routing), [makeMoE](https://github.com/AviSoori1x/makeMoE) |
| Self-improving agents | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) — a published system that modifies its own code and validates changes on benchmarks |

**What this repo does add:** an RL framework and a language model in one codebase, sharing
a single self-refinement loop, with the small-hardware constraints (MoE sparsity, LoRA,
quantization, Muon) treated as design requirements rather than afterthoughts. If you want
the cleanest PPO implementation, use CleanRL. If you want the fastest GPT speedrun, use
modded-nanogpt. If you want to see how the two fit together on a 6 GB GPU, this is that.

## Design notes

- **CleanRL-inspired** — algorithms are readable single-file implementations rather
  than deeply nested class hierarchies.
- **LeanRL-style acceleration** — `torch.compile`, AMP and memory budgeting are
  first-class, not afterthoughts.
- **Small-device first** — MoE sparsity, LoRA and
  quantization are deliberate choices to keep training and inference feasible on
  consumer hardware.

## Status

| Component | Status |
|---|---|
| PPO / SAC / TD3 / DQN | ✅ Working |
| MoE language model | ✅ Training |
| GPU acceleration path | ✅ Working |
| Self-improvement loop | 🔄 In progress |
| Muon optimizer | ✅ Implemented |
| LoRA / quantization | ✅ Implemented |

## References

| Work | What it informed |
|------|------------------|
| FlashSAC (Kim et al., ICLR 2026) | Norm-bounded critics, policy delay, noise repetition |
| AE-PPO (MDPI 2026) | Adaptive clipping and entropy for PPO |
| LibMoE (arXiv 2411.00918) | Router initialisation and z-loss for MoE stability |
| NanoChat-MoE (Duan et al., 2026) | Full-stack MoE training pipeline |
| Tiny-MoE (Ebied, 2026) | Shared expert architecture, top-2 routing |
| GQA (Ainslie et al., 2023) | Grouped-query attention |
| CleanRL (Huang et al.) | Single-file RL implementations |
| ERL (Shi et al., 2026) | Experiential reflection–consolidation |
| LSE (Chen et al., 2026) | Learning to self-evolve with delta rewards |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

You are free to use, modify and distribute this software, including commercially,
**provided you retain the copyright notice and attribution**. See `NOTICE` for
attribution requirements.
