"""
Train the RSI VLA (SigLIP + MoEGPT backbone + flow-matching action expert).
Stage 1: behavior cloning via conditional flow matching on trajectories.
Stage 2 (future): RL fine-tuning with algorithms/td3.py (deterministic policy).

Data sources:
  --data synth   random images + random-walk action chunks (pipeline smoke test)
  --data env     scripted waypoint expert in RoboticsEnv (real integration path)
"""
import sys, os, time, random, argparse, contextlib
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from vla_model import VLAPolicy, create_vla_small_config
from moe_model import create_ambatron_430m_config

INSTRUCTIONS = [
    "Navigate to the goal and pick up the red block.",
    "Move the robot arm to the target position.",
    "Follow the waypoint and avoid obstacles.",
    "Approach the block and grasp it.",
    "Drive to the goal location.",
]


def synth_batch(batch: int, chunk: int, act_dim: int, img_size: int, device):
    images = torch.rand(batch, img_size, img_size, 3, device=device)
    instructions = [random.choice(INSTRUCTIONS) for _ in range(batch)]
    states = torch.randn(batch, 1872, device=device) * 0.1  # non-visual modalities only
    # random-walk action chunks (temporally smooth, like robot trajectories)
    base = torch.randn(batch, 1, act_dim, device=device) * 0.3
    steps = torch.randn(batch, chunk, act_dim, device=device) * 0.05
    actions = (base + torch.cumsum(steps, dim=1)).tanh()
    return images, instructions, states, actions


def run_env_expert(env, chunk: int, steps: int = 200):
    """Roll out a waypoint expert in RoboticsEnv, collecting (img, state, action) frames."""
    frames = []
    obs, _ = env.reset()
    done = False
    step = 0
    while not done and step < steps:
        # scripted expert: steer toward the goal position
        if env.use_physics and env.physics is not None:
            action = np.zeros(4, dtype=np.float32)
        else:
            # Sep-5 (audit): old code emitted world-frame [dx,dy,0,0] but the
            # B-141 dummy step interprets [forward v, turn w] — the robot
            # spiraled away (empirically 0/3 reached). Body-frame controller:
            delta = env._goal[:2] - np.array(env._position, dtype=np.float32)
            heading = env._current_yaw()
            dist = np.linalg.norm(delta)
            bearing = np.arctan2(delta[1], delta[0]) - heading
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
            v = max(1.5 * dist, 0.36)
            w = float(np.clip(6.0 * bearing, -9.0, 9.0))
            action = np.array([v / 4.0, w / 9.0, 0.0, 0.0], dtype=np.float32)
        frames.append({
            "image": obs.visual.get_tensor(),
            "state": obs.get_combined_state()[150528:],  # non-visual slice, matches physics format
            "action": action,
            "instruction": random.choice(INSTRUCTIONS),
        })
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    return frames


def build_env_dataset(num_episodes: int, chunk: int):
    from sim_environment import RoboticsEnv
    env = RoboticsEnv(use_physics=False)
    data = []
    for _ in range(num_episodes):
        frames = run_env_expert(env, chunk)
        if len(frames) < chunk + 1:
            continue
        for i in range(0, len(frames) - chunk, chunk // 2):
            ep = frames[i:i + chunk + 1]
            img = (np.clip(np.array([f["image"] for f in ep[:-1]], dtype=np.float32), 0, 1)
                   * 255).astype(np.uint8)  # same uint8 convention as physics data
            st = np.array([f["state"] for f in ep[:-1]], dtype=np.float32)
            # ROUND-170 (F3): action_k is the action executed FROM obs_k —
            # training (obs_i, a_i) requires act = ep[:-1], NOT ep[1:] (the old
            # slice trained a one-transition-delayed mapping).
            act = np.array([f["action"] for f in ep[:-1]], dtype=np.float32)
            data.append((img, ep[0]["instruction"], st, act))
    return data


def build_physics_dataset(data_path: str, chunk: int):
    """Load collected episodes (Phase B npz) into (img, instr, state, action) windows.
    Images stay uint8 (memory); scaled to [0,1] at batch time."""
    data = np.load(data_path, allow_pickle=True)["episodes"]
    windows = []
    for ep in data:
        T = len(ep["images"])
        for i in range(0, T - chunk, chunk // 2):
            img = ep["images"][i:i + chunk + 1]              # uint8
            st = ep["states"][i:i + chunk + 1]
            # ROUND-170 (F3): same a_i convention as collect_demos/gen_demos —
            # action_k was executed from obs_k, so the window target is
            # actions[i:i+chunk], not [i+1:i+chunk+1] (which lagged one step).
            act = ep["actions"][i:i + chunk]
            if len(act) < chunk:
                continue
            windows.append((img, ep["instructions"][i], st, act))
    return windows


def main():
    ap = argparse.ArgumentParser(description="Train RSI VLA (flow matching BC)")
    ap.add_argument("--data", choices=["synth", "env", "physics"], default="synth",
                    help="physics: load collected trajectories npz (Phase B)")
    ap.add_argument("--backbone", choices=["small", "ambatron"], default="small",
                    help="small: 6-layer MoEGPT (smoke); ambatron: 444M backbone")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chunk_len", type=int, default=16)
    ap.add_argument("--act_dim", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_backbone", action="store_true", default=True)
    ap.add_argument("--unfreeze_backbone", action="store_true",
                    help="train backbone too (LoRA recommended instead)")
    ap.add_argument("--lora_backbone", action="store_true",
                    help="LoRA-adapt the backbone (rank 16) alongside projectors+expert")
    ap.add_argument("--use_pretrained_siglip", action="store_true",
                    help="download SigLIP weights (default: random-init for smoke)")
    ap.add_argument("--backbone_ckpt", type=str, default="",
                    help="Ambatron pretrain ckpt to load as the VLA backbone "
                         "before LoRA (round-11: the 'real backbone' was random)")
    ap.add_argument("--episodes", type=int, default=20,
                    help="episodes to generate for --data env")
    ap.add_argument("--data_dir", type=str, default="out/vla_data/trajectories.npz",
                    help="dataset for --data physics")
    ap.add_argument("--out_dir", type=str, default="out/vla")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.backbone == "ambatron":
        backbone_cfg = create_ambatron_430m_config()
    else:
        backbone_cfg = create_vla_small_config()

    policy = VLAPolicy(
        backbone_config=backbone_cfg,
        action_dim=args.act_dim,
        chunk_len=args.chunk_len,
        freeze_backbone=not args.unfreeze_backbone,
        freeze_vision=True,
        use_pretrained_siglip=args.use_pretrained_siglip,
        state_dim=1872,
    ).to(device)
    policy.train()

    # FP32 master weights + autocast bf16 (recommended mixed-precision pattern;
    # manual model.to(bf16) degrades descent quality — NeMo/PyTorch guidance)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else contextlib.nullcontext())

    if args.backbone_ckpt and Path(args.backbone_ckpt).exists():
        import torch.serialization as _ser
        from moe_model import MoEConfig as _Cfg
        _ser.add_safe_globals([_Cfg])
        ck = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=True)
        state = ck.get("model_state", ck.get("model", {}))
        missing, unexpected = policy.backbone.load_state_dict(state, strict=False)
        print(f"[VLA] backbone loaded from {args.backbone_ckpt}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
    elif args.backbone == "ambatron":
        print(f"[VLA] WARNING: --backbone {args.backbone} without a loaded "
              f"pretrain ckpt — the backbone is RANDOM INIT (pass --backbone_ckpt "
              f"out/ambatron_430m_v2/ckpt.pt to use the pretrained 430M)")

    if args.lora_backbone:
        from lora_adapter import add_lora, freeze_base
        add_lora(policy.backbone, r=16)
        freeze_base(policy.backbone)
        print(f"[VLA] backbone LoRA adapters added "
              f"({sum(p.numel() for p in policy.backbone.parameters() if p.requires_grad)/1e6:.2f}M trainable)")

    n_backbone = sum(p.numel() for p in policy.backbone.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[VLA] total {n_total/1e6:.1f}M | trainable {n_train/1e6:.2f}M "
          f"| backbone {n_backbone/1e6:.1f}M | device {device}")
    print(f"[VLA] feature_layer {policy.feature_layer}/{policy.n_layer} | "
          f"vis tokens {policy.vision.n_tokens} | expert dim {policy.expert.dim}")

    env_data = None
    if args.data == "env":
        print(f"[VLA] generating {args.episodes} episodes from RoboticsEnv expert...")
        env_data = build_env_dataset(args.episodes, args.chunk_len)
        print(f"[VLA] {len(env_data)} trajectory windows")
    elif args.data == "physics":
        env_data = build_physics_dataset(args.data_dir, args.chunk_len)
        print(f"[VLA] {len(env_data)} windows from {args.data_dir}")

    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        if env_data is not None:
            batch = random.sample(env_data, min(args.batch_size, len(env_data)))
            images = torch.from_numpy(np.array([b[0][0] for b in batch])).to(device)
            images = images.float().div_(255.0)
            instructions = [b[1] for b in batch]
            states = torch.from_numpy(np.array([b[2][0] for b in batch])).float().to(device)
            actions = torch.from_numpy(np.array([b[3] for b in batch])).float().to(device)
        else:
            images, instructions, states, actions = synth_batch(
                args.batch_size, args.chunk_len, args.act_dim, 224, device)

        with autocast_ctx:
            loss = policy.flow_loss(images, instructions, states, actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                chunk = policy.expert.sample(
                    policy.feature_proj(
                        policy._backbone_features_from_emb(
                            policy._build_sequence(images, instructions, states)
                        ).detach()),
                    args.chunk_len)
            act_stats = (float(chunk.mean()), float(chunk.std()))
            print(f"[VLA] step {step:5d} | flow_loss {loss.item():.4f} | "
                  f"sampled action mean {act_stats[0]:+.3f} std {act_stats[1]:.3f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    from storage import safe_save  # B-151: atomic SSD-drop-safe save
    safe_save({
        "policy": policy.state_dict(),
        "config": backbone_cfg,
        "args": vars(args),
    }, f"{args.out_dir}/vla_ckpt.pt")
    print(f"[VLA] saved to {args.out_dir}/vla_ckpt.pt")


if __name__ == "__main__":
    main()
