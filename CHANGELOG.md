# Changelog

## 0.1.0 — initial release

- Multi-algorithm RL: DQN, PPO, SAC, TD3, behavioural cloning and DAgger, sharing one
  environment/agent interface.
- From-scratch Mixture-of-Experts language model family behind a single `MoEConfig`:
  `mini` (~50M), `plus` (~121M), `xl` (~225M), `430m` (444M dense) and `v3` (~587M, 32K
  vocab), with GQA, top-2 routing, a shared expert, RoPE, SwiGLU, RMSNorm and router
  z-loss.
- Trainers for pretraining, tool-use SFT, LoRA/DoRA fine-tuning, GRPO-style tool RL,
  distillation and a self-refinement loop that turns its own rollouts into SFT data.
- Evaluation suites for tool use, chat, code generation, spatial reasoning and VLA
  physics, plus a regression battery.
- Quantisation utilities (int8 / int4), model merging, and an OpenAI-compatible server
  with axis routing.

**Not included in this release:** the data-acquisition and corpus-curation pipeline,
and trained checkpoints for every configuration. See the scope note in `README.md`.