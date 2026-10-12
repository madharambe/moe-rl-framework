"""B-65: generate expert demonstrations (steer-toward-goal) in the physics env
and BC-finetune the VLA on them. The RL runs never produced goal-reaching
episodes (the policy wanders away; noise-luck is ~1/10 eps), so the approach
primitive is learned from expert demos first, then RL (RWR/TD3+BC) refines.

Usage: gen_demos.py --n_eps 40 --min_spawn 0.6 --bc_updates 1200 --out ...
"""
import argparse
import math
import random
import sys, pathlib
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch.serialization as ser  # noqa: E402
from moe_model import MoEConfig  # noqa: E402
from vla_model import VLAPolicy  # noqa: E402
from lora_adapter import add_lora  # noqa: E402
from sim_environment import RoboticsEnv  # noqa: E402
from train_vla_rl import _to_uint8  # noqa: E402

ser.add_safe_globals([MoEConfig])

GOAL_INSTR = "Navigate to the goal and pick up the red block."


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def expert_action(env, noise=0.05):
    """Proportional controller: steer yaw to the goal, drive forward.
    action = [linear, angular, 0, 0] clipped to [-1, 1].
    B-110: the old a[0]=1.0 drove through the goal circle at 4 m/s without
    braking — demos rarely showed arrival. Now: brake proportionally to
    distance but keep a floor ABOVE zero so the robot actually CROSSES the
    0.2m termination threshold instead of stopping outside/at it."""
    pos = env.physics.get_end_effector_position()[:2].astype(np.float32)
    _, q = env.physics.pb.getBasePositionAndOrientation(env.physics.robot_id)
    yaw = yaw_from_quat(q)
    goal_ang = math.atan2(env._goal[1] - pos[1], env._goal[0] - pos[0])
    err = (goal_ang - yaw + math.pi) % (2 * math.pi) - math.pi
    dist = float(np.linalg.norm(env._goal - pos))
    # 1.5 (m/s)/m of distance, floored at 0.36 m/s: brakes smoothly yet
    # always reaches; 1 action unit = wheel_vel_scale 4 m/s
    lin = float(np.clip(0.375 * dist, 0.09, 1.0))
    a = np.array([lin, float(np.clip(2.5 * err, -1, 1)), 0.0, 0.0], dtype=np.float32)
    a[:2] += np.random.randn(2).astype(np.float32) * noise
    return np.clip(a, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eps", type=int, default=40)
    ap.add_argument("--min_spawn", type=float, default=0.6)
    ap.add_argument("--min_spawn_low", type=float, default=None,
                    help="alternate every other episode with this closer spawn (final-approach demos)")
    ap.add_argument("--behind_eps", type=int, default=0,
                    help="re-place the first N episodes' goals directly behind the robot "
                         "(turnaround demos)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vla_ckpt", default="out/vla_rl/vla_bc_v13wide2.pt")
    ap.add_argument("--bc_updates", type=int, default=3000)
    ap.add_argument("--bc_batch", type=int, default=32)
    ap.add_argument("--bc_lr", type=float, default=1e-3)
    ap.add_argument("--demos", default="out/demos_v2yaw.pt",
                    help="OUT .pt BUNDLE path (NOT the bc_only .npy DIR — the "
                         "old default collided with bc_only's --demos dir "
                         "semantics: writing a bundle to that path either hit "
                         "IsADirectoryError or produced a file the dir-check "
                         "rejected. Run split_demos.py after this to make the "
                         ".npy DIR for bc_only.)")
    ap.add_argument("--out", default="out/vla_rl/gen_demos_out.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = RoboticsEnv(use_physics=True)
    env.min_spawn = args.min_spawn

    demos = []
    t0 = time.time()
    n_reached = 0
    for ep in range(args.n_eps):
        if args.min_spawn_low is not None and ep % 2 == 1:
            env.min_spawn = args.min_spawn_low
        else:
            env.min_spawn = args.min_spawn
        obs, _ = env.reset(seed=None if args.seed is None else args.seed + ep)
        if ep < args.behind_eps:
            pos = env.physics.get_end_effector_position()[:2].astype(np.float32)
            _, q = env.physics.pb.getBasePositionAndOrientation(env.physics.robot_id)
            yaw = yaw_from_quat(q)
            d = float(env._rng.uniform(1.2, 1.7))
            ang = yaw + math.pi + float(env._rng.uniform(-0.6, 0.6))
            env._goal = (pos + d * np.array([math.cos(ang), math.sin(ang)])).astype(np.float32)
            env._prev_distance = np.linalg.norm(pos - env._goal)
            # Sep-5 (audit): after moving the goal, the physics MARKER still
            # shows the reset-time goal — the camera sees marker A while the
            # graph/proprioception say goal B (vision-vs-graph mismatch that
            # trains conflicting supervision). Update the marker like the
            # normal resample path does (sim_environment.py update_goal_marker).
            env.physics.update_goal_marker(np.append(env._goal, 0.0))
            obs = env._get_observation()
        ep_t = []
        reached = False
        for _ in range(500):
            a = expert_action(env, noise=0.05)
            # B-124: capture the task-proprioception vector BEFORE stepping
            # B-149: use the canonical 5D body-frame producer -- this inline
            # copy was a stale 4D WORLD-frame variant; any future consumer of
            # gen_demos-written bundles would have inherited the mismatch.
            task = env.task_vector().astype(np.float32)
            next_obs, reward, term, trunc, _ = env.step(a)
            img = _to_uint8(obs.visual.get_tensor())
            state = obs.get_combined_state()[150528:].astype(np.float32)
            ep_t.append((img, state, a.astype(np.float32), task))
            obs = next_obs
            if term or trunc:
                reached = reached or term
                break
        n_reached += reached
        demos.extend(ep_t)
        if (ep + 1) % 10 == 0:
            print(f"[demos] ep {ep + 1}/{args.n_eps} | transitions {len(demos)} "
                  f"| reached {n_reached}/{ep + 1} | {time.time() - t0:.0f}s", flush=True)
    print(f"[demos] DONE: {len(demos)} transitions, {n_reached}/{args.n_eps} reached "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not demos:
        raise SystemExit("[demos] ZERO transitions collected (n_eps=0 or no "
                         "steps per episode) — refusing to stack an empty list")

    imgs = np.stack([d[0] for d in demos])  # keep uint8 (75KB/img not 602KB) — RAM cap
    states = np.stack([d[1] for d in demos]).astype(np.float32)
    actions = np.stack([d[2] for d in demos]).astype(np.float32)
    tasks = np.stack([d[3] for d in demos]).astype(np.float32)
    del demos
    import gc
    gc.collect()
    # B-211: gen_demos saves straight to the external SSD — atomic save so a
    # mid-write drive drop can't corrupt the demo bundle.
    from storage import safe_save
    from storage import ensure_disk
    ensure_disk(pathlib.Path(__file__).resolve().parent.parent / "out", 2.0,
                "gen_demos demo bundle")
    safe_save({"imgs": imgs, "states": states, "actions": actions,
               "tasks": tasks}, args.demos)
    print(f"[demos] saved {args.demos} ({imgs.shape})", flush=True)

    from pathlib import Path as _P
    if not _P(args.vla_ckpt).exists():
        raise SystemExit(f"[gen_demos] --vla_ckpt {args.vla_ckpt} missing; "
                         f"pass a real BC checkpoint (e.g. out/vla_rl/vla_bc_v13wide2.pt)")
    ck = torch.load(args.vla_ckpt, map_location="cpu", weights_only=True)
    vla = VLAPolicy(backbone_config=ck["config"], action_dim=4, chunk_len=16,
                    freeze_backbone=True, use_pretrained_siglip=True)
    add_lora(vla.backbone, r=16)
    # B-124/B-125: strict=False — older ckpts predate task_proj/task_mlp
    # (fresh trainable modules, intentionally initialized random)
    missing, unexpected = vla.load_state_dict(ck["policy"], strict=False)
    real_missing = [k for k in missing
                    # ROUND-5 (L4): task_head was missing from the filter —
                    # pre-B-129 ckpts (which lack task_head) assert-failed.
                    if not (k.startswith("task_proj") or "task_mlp" in k
                            or "task_head" in k)]
    assert not real_missing and not unexpected, \
        f"ckpt mismatch: missing={real_missing[:4]} unexpected={unexpected[:4]}"
    cfg = ck["config"]
    del ck
    vla.to("cuda")
    for p in vla.parameters():
        if not p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    vla.train()

    opt = torch.optim.Adam([p for p in vla.parameters() if p.requires_grad], lr=args.bc_lr)
    # B-126: TWO-PHASE training. The measured failure: joint optimization
    # collapsed the fresh task_mlp to a local optimum (layer2 frozen at init,
    # output = dataset-mean action) because mean-prediction was already
    # "good enough". Phase 1 trains ONLY the task pathway — the data is
    # near-linearly separable (corr(dist,lin)=0.967), so it must fit it
    # alone; phase 2 then jointly fine-tunes everything. ROUND-170 (F4):
    # warm BOTH task_mlp AND task_head (bc_only does; the head is the final
    # projection the policy reads out — warming mlp-only left the head at
    # random init for every gen_demos checkpoint).
    task_params = [p for n, p in vla.named_parameters()
                   if "task_mlp" in n or "task_head" in n]
    task_ids = {id(p) for p in task_params}
    rest_params = [p for p in vla.parameters()
                   if p.requires_grad and id(p) not in task_ids]
    opt_task = torch.optim.Adam(task_params, lr=args.bc_lr * 5)
    WARMUP = max(1, args.bc_updates // 4)
    n = len(imgs)
    instr = [GOAL_INSTR] * args.bc_batch
    loss = torch.tensor(0.0)
    t1 = time.time()
    for step in range(args.bc_updates):
        idx = rng.integers(0, n, size=args.bc_batch)
        batch_img = torch.from_numpy(imgs[idx]).to("cuda").float() / 255.0
        batch_st = torch.from_numpy(states[idx]).to("cuda")
        batch_a = torch.from_numpy(actions[idx]).to("cuda")
        batch_task = torch.from_numpy(tasks[idx]).to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cond = vla._cond_features(batch_img, instr, batch_st)
        # B-65 fix: the flow head runs in FP32 — bf16 autocast backward was
        # underflowing the tiny action-head gradients (loss stuck at ~1.42 =
        # "outputs nothing"). Cond (bf16 backbone) is cast up; the LoRA still
        # receives gradients through the autocast'd cond backward.
        B = cond.shape[0]
        x0 = torch.zeros(B, vla.chunk_len, vla.action_dim, device=cond.device,
                         dtype=torch.float32)
        t0 = torch.zeros(B, device=cond.device, dtype=torch.float32)
        v = vla.expert(x0, t0, cond.float(), task_vec=batch_task)
        if vla.expert.tanh_output:
            v = torch.tanh(v)
        a_det = v[:, 0].float()
        loss = (a_det - batch_a).pow(2).mean()
        # B-126 two-phase: warmup steps train ONLY the task pathway
        if step < WARMUP:
            vla.zero_grad(set_to_none=True)  # ROUND-171: zero ALL params (task+non-task) — the old opt_task.zero_grad only cleared task grads, so non-task grads accumulated stale grads through the whole warmup
            opt_task.zero_grad(set_to_none=True)
            loss.backward()
            opt_task.step()
        else:
            if step == WARMUP:
                print(f"[bc] phase 2: joint fine-tune (task pathway warmed up)",
                      flush=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if step % 500 == 0:
            print(f"[bc] step {step}/{args.bc_updates} | loss {loss.item():.4f} "
                  f"| {time.time() - t1:.0f}s", flush=True)
    vla.eval()

    from storage import safe_save
    safe_save({"policy": vla.state_dict(), "config": cfg, "args": vars(args)},
              args.out, retries=2, delay=3)
    print(f"[bc] done -> {args.out} (loss {loss.item():.4f}, {time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
