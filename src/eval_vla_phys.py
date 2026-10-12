"""B-62: proper physics VLA eval — FULL 500-step horizon (the old eval capped
at 150, only 30% of the env's episode, possibly truncating goal-reaching
trajectories) + optional small inference noise (ESPD: "maintain a small
exploration factor when evaluating for bias correction").

Usage:
  python src/eval_vla_phys.py --ckpt out/vla_rl/vla_rl_phys5.pt \
      --eps 8 --horizon 500 [--noise 0.0]
"""
import argparse
import sys, pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch.serialization as ser  # noqa: E402

from moe_model import MoEConfig  # noqa: E402

ser.add_safe_globals([MoEConfig])

from vla_model import VLAPolicy  # noqa: E402
from lora_adapter import add_lora  # noqa: E402
from sim_environment import RoboticsEnv  # noqa: E402

GOALS = [
    "Navigate to the goal and pick up the red block.",
    "Go to the target area and pick up the object.",
    "Move to the goal location.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="inference-time Gaussian noise std on actions (ESPD bias correction)")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--min_spawn", type=float, default=None,
                    help="B-122: override env.min_spawn (default 1.2) — "
                         "measures goal-reach vs start-distance so a model "
                         "trained at closer spawns isn\'t condemned silently")
    ap.add_argument("--goal_min", type=float, default=-1.0,
                    help="B-172: arena goal range lower bound")
    ap.add_argument("--goal_max", type=float, default=1.0,
                    help="B-172: arena goal range upper bound")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    vla = VLAPolicy(backbone_config=ck["config"], action_dim=4, chunk_len=16,
                    freeze_backbone=True, use_pretrained_siglip=False)
    add_lora(vla.backbone, r=16)
    # D5 (18th-audit): strict=False + task-filter like every other VLA consumer —
    # strict=True crashes on pre-B-124 checkpoints (task_proj/task_mlp/task_head
    # keys added later); the model still trains without them in older ckpts.
    missing, unexpected = vla.load_state_dict(ck["policy"], strict=False)
    # ROUND-172 (N7): a wrong-ARCH ckpt loads silently (missing/unexpected have
    # tolerated vintages) -> random task pathway at eval. Same 50% key-match
    # guard as robot_core: refuse a mostly-unmatched checkpoint.
    if unexpected and len(unexpected) / max(len(ck["policy"]), 1) > 0.5:
        raise SystemExit(f"[eval_vla_phys] ckpt {args.ckpt} matches <50% of keys "
                         f"({len(unexpected)} unexpected) — wrong architecture? "
                         f"Refusing a silently-random policy eval")
    if missing or unexpected:
        print(f"[eval] load_state_dict non-strict: missing={missing[:5]}… unexpected={unexpected[:5]}…")
    vla.eval()
    vla.to("cuda")

    env = RoboticsEnv(use_physics=True, goal_range=(args.goal_min, args.goal_max))
    if args.min_spawn is not None:
        # B-122: match the training spawn distribution when requested
        env.min_spawn = args.min_spawn
    INSTR = GOALS[0]
    # B-110: seed the noise RNG — the old global np.random draws made even
    # repeated identical runs non-reproducible, so eval comparisons were noise
    noise_rng = np.random.default_rng(args.seed0 + 777)
    goals, rewards, term_steps = 0, [], []
    for ep in range(args.eps):
        obs, _ = env.reset(seed=args.seed0 + ep)
        ep_r, hit = 0.0, False
        step = trunc = 0
        for step in range(args.horizon):
            im = obs.visual.get_tensor()  # float32 in [0,1] already
            # B-85: the old /255 on the already-[0,1] floats fed ~black images
            # to SigLIP — every physics eval measured a corrupted vision channel
            if im.dtype != np.uint8 and im.max() <= 1.0:  # ROUND-170: numpy compare (torch.uint8 never equals np dtype)
                img = torch.from_numpy(im).float().to("cuda").unsqueeze(0)
            else:
                img = torch.from_numpy(im).float().to("cuda").unsqueeze(0) / 255.0
            st = torch.from_numpy(obs.get_combined_state()[150528:].astype(np.float32)).to("cuda").unsqueeze(0)
            # B-124: the task-proprioception token (matches gen_demos v3 collection)
            tv = torch.from_numpy(env.task_vector()).to("cuda").unsqueeze(0)
            with torch.no_grad():
                act = vla.deterministic_action(img, [INSTR], st, task=tv)[0].cpu().numpy()
            if args.noise > 0.0:
                act = act + noise_rng.standard_normal(4).astype(np.float32) * args.noise
                act = np.clip(act, -1, 1)
            obs, r, term, trunc, _ = env.step(act)
            ep_r += r
            if term:
                goals += 1
                hit = True
                term_steps.append(step)
                print(f"  ep {ep}: GOAL at step {step} (reward {ep_r:.1f})")
                break
        rewards.append(ep_r)
        end = step if hit or trunc else args.horizon - 1
        if not hit:
            print(f"  ep {ep}: no goal (reward {ep_r:.1f}, {'timeout' if end == args.horizon - 1 else 'trunc'})")
    print(f"\n[phys-eval] goal-reach {goals}/{args.eps} | mean reward "
          f"{np.mean(rewards):.1f} | noise {args.noise}")


if __name__ == "__main__":
    main()