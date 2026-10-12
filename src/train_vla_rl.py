"""
Phase C: RL fine-tuning of the VLA with TD3 (deterministic actor).
The actor is the VLA's deterministic action head (velocity at x_0=0, t=0);
critics learn Q(cond_features, action) on top of the frozen VLA features.
Grounding: RL post-training of flow policies is normally intractable (FPO/Z-1);
a deterministic head sidesteps importance sampling entirely (TD3 needs no log-prob).
"""
import sys, os, random, argparse, contextlib
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from vla_model import VLAPolicy, create_vla_small_config
from moe_model import create_ambatron_430m_config
from sim_environment import RoboticsEnv
from gpu_accel import resolve_device

GOALS = [
    "Navigate to the goal and pick up the red block.",
    "Move the robot to the target position.",
    "Follow the waypoint toward the destination.",
    "Approach the red block and grasp it.",
    "Drive to the goal location and stop.",
]


class TwinCritic(nn.Module):
    def __init__(self, cond_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(cond_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = mlp(), mlp()

    def forward(self, cond, action):
        x = torch.cat([cond, action], dim=-1)
        return self.q1(x), self.q2(x)


class VLABuffer:
    """Replay buffer. B-61: entries carry their EPISODE RETURN (ep_ret) for
    RWR weighting, and retention is reward-prioritized (evict the LOWEST
    ep_ret transition when full) so the rare goal-reaching transitions
    (+43 episodes) survive instead of being FIFO-evicted — that lost signal
    was why the deterministic policy never learned goal-reaching."""

    def __init__(self, capacity: int = 4000):  # ~1.3 GB RAM vs ~6.3 GB at 20000 (earlyoom)
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def add(self, img, instr, state, action, reward, ep_ret,
            nimg, ninstr, nstate, term, trunc):
        entry = (img, instr, state, action, reward, ep_ret,
                 nimg, ninstr, nstate, term, trunc)
        if len(self.buf) < self.capacity:
            self.buf.append(entry)
            return
        # B-65 fix: index-based replacement — deque.remove() compares tuples with
        # == which elementwise-compares the numpy arrays inside -> ambiguous truth.
        worst_i = min(range(len(self.buf)), key=lambda i: self.buf[i][5])
        if ep_ret > self.buf[worst_i][5]:
            self.buf[worst_i] = entry

    def sample(self, batch: int, device):
        b = random.sample(self.buf, min(batch, len(self.buf)))
        # Bug 41 fix: buffer stores uint8 images (75 KB vs 602 KB each);
        # converted back to [0,1] floats here (BC convention: *255 at add, /255 at batch).
        images = torch.from_numpy(np.array([x[0] for x in b], dtype=np.float32)).to(device) / 255.0
        instrs = [x[1] for x in b]
        states = torch.from_numpy(np.array([x[2] for x in b], dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array([x[3] for x in b], dtype=np.float32)).to(device)
        rewards = torch.from_numpy(np.array([x[4] for x in b], dtype=np.float32)).to(device).unsqueeze(1)
        ep_rets = torch.from_numpy(np.array([x[5] for x in b], dtype=np.float32)).to(device).unsqueeze(1)
        nimages = torch.from_numpy(np.array([x[6] for x in b], dtype=np.float32)).to(device) / 255.0
        ninstrs = [x[7] for x in b]
        nstates = torch.from_numpy(np.array([x[8] for x in b], dtype=np.float32)).to(device)
        terms = torch.from_numpy(np.array([x[9] for x in b], dtype=np.float32)).to(device).unsqueeze(1)
        return (images, instrs, states, actions, rewards, ep_rets,
                nimages, ninstrs, nstates, terms)

    def __len__(self):
        return len(self.buf)


def _to_uint8(t: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8 (0-255); passthrough if already uint8. Clip to [0,1]
    like collect_vla_data does (camera artifacts can exceed the range)."""
    return (np.clip(t, 0.0, 1.0) * 255).astype(np.uint8) if t.dtype != np.uint8 else t


def _augment_images(images: torch.Tensor) -> torch.Tensor:
    """RAD-style pixel augmentation (Laskin et al., NeurIPS 2020, arXiv
    2004.14990): random translate (replicate pad + shifted crop) + random
    amplitude scale. Applied to the TRAINING batch only (the update's own
    images/target-images) — rollouts stay clean. Boosts data efficiency and
    generalization on a small pixel buffer (this run learned inside 4k frames)."""
    B, H, W, C = images.shape
    if B < 2 or H < 16 or W < 16:
        return images
    s = max(1, min(6, H // 40))                    # shift range (4px @ 224)
    d = images.device
    p = torch.nn.functional.pad(images.permute(0, 3, 1, 2),
                                (s, s, s, s), mode="replicate")   # [B,C,H+2s,W+2s]
    y0 = torch.randint(0, 2 * s + 1, (B, 1), device=d)
    x0 = torch.randint(0, 2 * s + 1, (B, 1), device=d)
    br = torch.arange(B, device=d).view(B, 1, 1, 1)
    cr = torch.arange(C, device=d).view(1, C, 1, 1)
    hid = (torch.arange(H, device=d).view(1, 1, H, 1) + y0.view(B, 1, 1, 1))
    wid = (torch.arange(W, device=d).view(1, 1, 1, W) + x0.view(B, 1, 1, 1))
    out = p[br, cr, hid, wid]                                     # [B,C,H,W]
    out = out * (1.0 + 0.2 * (torch.rand(B, 1, 1, 1, device=d) - 0.5))
    return out.permute(0, 2, 3, 1).clamp_(0.0, 1.0).contiguous()


def _task_from_state(states):
    """B-141: derive the 5-dim body-frame task vector from a stored state
    slice ([B, 1872]) -- layout: graph coords flat at offset 0 (goal xy,
    robot xy, sin/cos yaw). Works identically at rollout and update time,
    so the buffer needs no schema change."""
    import numpy as np
    import torch
    if isinstance(states, np.ndarray):
        gx, gy = states[..., 0], states[..., 1]
        px, py = states[..., 3], states[..., 4]
        sy, cy_ = states[..., 6], states[..., 7]
        dx_w, dy_w = gx - px, gy - py
        dist = np.hypot(dx_w, dy_w)
        # B-141 fix: world->body rotation by -yaw so the actor/critic see the
        # SAME body-frame conditioning the BC prior was trained on
        # (sim_environment.task_vector). dx_b = cos*yaw*dx_w + sin*yaw*dy_w.
        dx_b = cy_ * dx_w + sy * dy_w
        dy_b = -sy * dx_w + cy_ * dy_w
        return np.stack([dx_b / 3.0, dy_b / 3.0, dist / 3.0, sy, cy_], axis=-1).astype(np.float32)
    gx, gy = states[..., 0].float(), states[..., 1].float()
    px, py = states[..., 3].float(), states[..., 4].float()
    sy, cy_ = states[..., 6].float(), states[..., 7].float()
    dx_w, dy_w = gx - px, gy - py
    dist = torch.hypot(dx_w, dy_w)
    dx_b = cy_ * dx_w + sy * dy_w
    dy_b = -sy * dx_w + cy_ * dy_w
    return torch.stack([dx_b / 3.0, dy_b / 3.0, dist / 3.0, sy, cy_], dim=-1)


def env_step_to_vla(obs, action, env, instruction):
    """Step the env; return a buffer.add()-compatible tuple + the next observation."""
    img = _to_uint8(obs.visual.get_tensor())
    state_nv = obs.get_combined_state()[150528:].astype(np.float32)
    next_obs, reward, term, trunc, _ = env.step(action)
    nimg = _to_uint8(next_obs.visual.get_tensor())
    nstate_nv = next_obs.get_combined_state()[150528:].astype(np.float32)
    return (img, instruction, state_nv, action.astype(np.float32), float(reward),
            nimg, instruction, nstate_nv, bool(term), bool(trunc), next_obs)


def main():
    ap = argparse.ArgumentParser(description="RL fine-tune VLA with TD3 (Phase C)")
    ap.add_argument("--backbone", choices=["small", "ambatron"], default="small")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--policy_delay", type=int, default=2)
    ap.add_argument("--target_noise", type=float, default=0.2)
    ap.add_argument("--noise_clip", type=float, default=0.5)
    ap.add_argument("--explore_noise", type=float, default=0.1)
    ap.add_argument("--rwr_beta", type=float, default=0.05,
                    help="B-61 RWR temperature: exp(beta*ep_ret) weights the BC "
                         "toward goal-reaching episodes (+43 -> e^2.15=8.6, -400 -> ~0)")
    ap.add_argument("--rwr_wmax", type=float, default=10.0)
    ap.add_argument("--td3bc_alpha", type=float, default=2.5,
                    help="B-61 TD3+BC Q-term scale, normalized by mean|Q|")
    ap.add_argument("--update_every", type=int, default=4)
    ap.add_argument("--vla_ckpt", type=str, default="out/vla/vla_ckpt.pt",
                    help="BC-pretrained VLA checkpoint to warm-start from "
                         "(train_vla.py writes out/vla/vla_ckpt.pt)")
    ap.add_argument("--lora_backbone", action="store_true",
                    help="match BC training: backbone wrapped with LoRA (rank 16)")
    ap.add_argument("--use_pretrained_siglip", action="store_true",
                    help="match BC training: pretrained SigLIP")
    ap.add_argument("--out", type=str, default="out/vla_rl/vla_rl.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--backbone_ckpt", type=str, default="",
                    help="Ambatron pretrain ckpt to load as the VLA backbone "
                         "(only used when --vla_ckpt has no backbone)")
    ap.add_argument("--chunk_len", type=int, default=16,
                    help="must match the BC chunk_len (Bug-11: was 8 vs BC 16)")
    ap.add_argument("--use_physics", action="store_true",
                    help="run rollouts on the PyBullet physics env (slower, real camera)")
    ap.add_argument("--aug", type=str, default="shift", choices=["shift", "none"],
                    help="RAD-style visual augmentation on the training batch "
                         "(2004.14990); 'none' disables (rollouts are never augmented)")
    args = ap.parse_args()

    device = resolve_device(args.device if args.device in ("cuda", "cpu") else "cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    backbone_cfg = (create_ambatron_430m_config() if args.backbone == "ambatron"
                    else create_vla_small_config())
    vla = VLAPolicy(backbone_config=backbone_cfg, action_dim=4,
                    chunk_len=args.chunk_len, freeze_backbone=True,
                    use_pretrained_siglip=args.use_pretrained_siglip).to(device)
    if args.backbone_ckpt and Path(args.backbone_ckpt).exists():
        import torch.serialization as _ser
        from moe_model import MoEConfig as _Cfg
        _ser.add_safe_globals([_Cfg])
        ck = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=True)
        state = ck.get("model_state", ck.get("model", {}))
        missing, unexpected = vla.backbone.load_state_dict(state, strict=False)
        print(f"[VLA-RL] backbone loaded from {args.backbone_ckpt}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
    # FP32 master weights + autocast bf16 (recommended pattern; see train_vla.py)
    if args.lora_backbone:
        from lora_adapter import add_lora, freeze_base
        add_lora(vla.backbone, r=16)
        freeze_base(vla.backbone)
    if not os.path.exists(args.vla_ckpt):
        print(f"[VLA-RL] WARNING: --vla_ckpt {args.vla_ckpt} not found — "
              f"training from RANDOM INIT (BC warm-start is strongly "
              f"recommended)", flush=True)
    if os.path.exists(args.vla_ckpt):
        # Hybrid precision FIRST (cast before load): frozen stacks stay bf16 —
        # no optimizer state or grads, autocast guards compute precision; only
        # trainable parts (expert, projectors) keep fp32 masters for AdamW.
        # Cuts the load-time peak (fp32 model ~3 GB -> ~1.6 GB) so earlyoom
        # (kills at <=10% mem free) doesn't SIGTERM the run.
        if device.type == "cuda":
            from lora_adapter import LoRALinear
            for m in (vla.vision,):
                m.to(dtype=torch.bfloat16)
            for mod in vla.backbone.modules():
                if isinstance(mod, LoRALinear):
                    mod.base.to(dtype=torch.bfloat16)  # A/B stay fp32 masters
                elif isinstance(mod, torch.nn.Linear):
                    mod.to(dtype=torch.bfloat16)
        import torch.serialization as _ser
        from moe_model import MoEConfig as _VLA_MOE_CFG
        _ser.add_safe_globals([_VLA_MOE_CFG])  # 13th-audit: allowlist the embedded config for weights_only
        ckpt = torch.load(args.vla_ckpt, map_location="cpu",
                          weights_only=True,  # 13th-audit: was False (arbitrary pickle RCE risk if ckpt tampered)
                          )
        vla.load_state_dict(
            # ROUND-170 (F2): plain strict=False — the old task-key FILTER
            # silently STRIPPED the trained task_mlp/task_head from every
            # post-B-129 BC ckpt, starting RL with the measured-critical task
            # pathway at random init. strict=False tolerates pre-B-124
            # vintages (missing keys) while PRESERVING present task weights.
            ckpt["policy"],
            strict=False)
        del ckpt  # free the ~3 GB fp32 checkpoint (earlyoom kills at <=10% mem free)
        print(f"[VLA-RL] loaded BC checkpoint {args.vla_ckpt}", flush=True)

    # freeze the vision/backbone base weights; keep LoRA adapters + expert +
    # projectors trainable (RL-2 fix: the old loop also froze the LoRA A/B,
    # silently killing --lora_backbone)
    for name, p in vla.vision.named_parameters():
        p.requires_grad_(False)
    for name, p in vla.backbone.named_parameters():
        if not (name.endswith(".A") or name.endswith(".B")):
            p.requires_grad_(False)

    actor_opt = torch.optim.AdamW(
        [p for p in vla.parameters() if p.requires_grad], lr=args.lr)
    critic = TwinCritic(vla.n_embd, 4).to(device)
    critic_target = TwinCritic(vla.n_embd, 4).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else contextlib.nullcontext())

    env = RoboticsEnv(use_physics=args.use_physics, seed=0)
    vla.eval()  # rollouts/targets must be deterministic (no dropout)
    buffer = VLABuffer()
    _d = os.path.dirname(args.out)
    os.makedirs(_d, exist_ok=True) if _d else None
    obs, _ = env.reset(seed=0)
    instruction = random.choice(GOALS)
    ep_reward, ep_count = 0.0, 0
    ep_trans = []  # B-61: per-episode transitions (backfilled with ep return)
    total_updates = 0
    best_reward = -float("inf")

    for step in range(args.steps):
        with torch.no_grad(), autocast_ctx:
            img_t = torch.from_numpy(_to_uint8(obs.visual.get_tensor())).to(device).unsqueeze(0).float() / 255.0
            st_t = torch.from_numpy(obs.get_combined_state()[150528:].astype(np.float32)).to(device).unsqueeze(0)
            tv_t = torch.from_numpy(_task_from_state(
                st_t.cpu().numpy())).to(st_t.device)
            action = vla.deterministic_action(img_t, [instruction], st_t,
                                              task=tv_t)[0].cpu().numpy()
            action += np.random.randn(4).astype(np.float32) * args.explore_noise
            action = np.clip(action, -1, 1)
        (img, instr, state_nv, act, reward, nimg, ninstr, nstate_nv, term, trunc, obs) = \
            env_step_to_vla(obs, action, env, instruction)
        # B-61: accumulate the episode locally, backfill its RETURN on
        # termination (RWR needs episode returns, not per-step rewards).
        ep_trans.append((img, instr, state_nv, act, reward,
                         nimg, ninstr, nstate_nv, term, trunc))
        ep_reward += reward
        if step % 100 == 0:
            print(f"[VLA-RL] step {step} | buffer {len(buffer)} | ep_reward {ep_reward:.1f}",
                  flush=True)
            # B-59: periodic best-save — an SSD unplug / kill between episodes
            # otherwise loses the current best (observed: v5's +7.3 lost).
            if best_reward > -float("inf"):
                from storage import safe_save
                safe_save({"policy": vla.state_dict(), "config": backbone_cfg,
                           "args": vars(args)}, args.out, retries=2, delay=3)
        if term or trunc:
            for (i_, n_, s_, a_, r_, ni_, nin_, ns_, t_, tr_) in ep_trans:
                buffer.add(i_, n_, s_, a_, r_, ep_reward, ni_, nin_, ns_, t_, tr_)
            ep_trans = []
            ep_count += 1
            if ep_reward > best_reward:
                best_reward = ep_reward
                # RL-3 fix: save in the standard BC format so this checkpoint
                # can be re-loaded as --vla_ckpt (raw state_dict was not loadable)
                # B-211: atomic — an SSD drop mid-save would otherwise corrupt the
                # single best checkpoint (the periodic saves already use safe_save).
                from storage import safe_save
                safe_save({"policy": vla.state_dict(), "config": backbone_cfg,
                           "args": vars(args)}, args.out)
            if ep_count % 5 == 0:
                print(f"[VLA-RL] ep {ep_count} | reward {ep_reward:.1f} | "
                      f"best {best_reward:.1f} | buffer {len(buffer)}", flush=True)
            ep_reward = 0.0
            obs, _ = env.reset(seed=step)
            instruction = random.choice(GOALS)

        if len(buffer) < args.batch_size or step % args.update_every != 0:
            continue

        (images, instrs, states, actions, rewards, ep_rets,
         nimages, ninstrs, nstates, terms) = buffer.sample(args.batch_size, device)

        if args.aug == "shift":
            # Augment the update's own observations (both the current and the
            # target-time images get independent fresh transforms — the same
            # image seen under different shifts, per RAD).
            images = _augment_images(images)
            nimages = _augment_images(nimages)

        with autocast_ctx:
            with torch.no_grad():
                ncond = vla._cond_features(nimages, ninstrs, nstates).mean(dim=1)
                nact = vla.deterministic_action(
                    nimages, ninstrs, nstates,
                    task=_task_from_state(nstates))
                noise = (torch.randn_like(nact) * args.target_noise).clamp(-args.noise_clip, args.noise_clip)
                nact = (nact + noise).clamp(-1, 1)
                tq1, tq2 = critic_target(ncond, nact)
                # RL-6 fix: bootstrap on truncation (timeout), not on termination (goal)
                target = rewards + args.gamma * (1 - terms) * torch.min(tq1, tq2)

            vla.train()  # dropout on for the gradient steps only
            # RL-5 fix: critic uses detached cond — feature_proj/state_proj are
            # owned by the actor objective only (no silent grad-wiping confusion)
            cond = vla._cond_features(images, instrs, states).mean(dim=1).detach()
            q1, q2 = critic(cond, actions)
            c_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            critic_opt.zero_grad(set_to_none=True)
            c_loss.backward()
            critic_opt.step()

            if total_updates % args.policy_delay == 0:
                # Bug 12: ONE backbone graph serves both the critic and the
                # expert (cond_full is 3-D [B, n_cond, D] for the expert; the
                # critic consumes the mean-pooled view).
                cond_full = vla._cond_features(images, instrs, states)
                cond_a = cond_full.mean(dim=1)
                B = cond_a.shape[0]
                # B-85: the flow head runs FP32 (bf16 x0/t0 underflowed the
                # tiny action-head grads — the gen_demos B-65 fix, applied to
                # the RL actor too)
                x0 = torch.zeros(B, vla.chunk_len, vla.action_dim, device=cond_a.device, dtype=torch.float32)
                t0 = torch.zeros(B, device=cond_a.device, dtype=torch.float32)
                with torch.autocast("cuda", enabled=False):
                    v = vla.expert(x0, t0, cond_full.float(),
                                   task_vec=_task_from_state(states.float()))
                a_det = torch.tanh(v) if vla.expert.tanh_output else v
                # B-61: TD3+BC (Fujimoto & Gu 2021) + RWR (Peters & Schaal;
                # DRWR per DPPO, ICLR'25). The deterministic action is BC'd
                # toward the EXECUTED (noise-perturbed) actions, weighted by
                # exp(beta * ep_ret) — the goal-reaching rollouts (ep ret +43)
                # distill into the deterministic policy (self-imitation /
                # ESPD-style). Q-term normalized by mean|Q| (TD3+BC), BC term
                # dominates when the RWR weight is high.
                a_det0 = a_det[:, 0].float()
                # D2 (18th-audit): the critic must see a DETACHED cond — the old
                # code passed cond_a directly, so the actor got gradients through
                # the STATE/COND path (TD3/DDPG actors should only get the ACTION
                # gradient). That let the feature extractor distort cond to
                # inflate Q (feature hacking). The action path through the expert
                # (a_det0) still backprops into the backbone/LoRA as intended.
                q_eval = critic(cond_a.detach(), a_det0)[0]
                w = torch.clamp(torch.exp(ep_rets.squeeze(1) * args.rwr_beta),
                                max=args.rwr_wmax).detach()
                bc_loss = (w * (a_det0 - actions).pow(2).mean(-1)).mean()
                q_scale = args.td3bc_alpha / (q_eval.abs().mean() + 1e-6)
                actor_loss = q_scale.detach() * -q_eval.mean() + bc_loss
                if total_updates % args.policy_delay == 0 and total_updates % 50 == 0:
                    print(f"  [RL-7] q_scale {q_scale.item():.4f} | bc_w_mean "
                          f"{w.mean().item():.3f} | bc {bc_loss.item():.4f} | "
                          f"Q {q_eval.mean().item():.1f}", flush=True)
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()
                for tp, sp in zip(critic_target.parameters(), critic.parameters()):
                    tp.data.copy_(args.tau * sp.data + (1 - args.tau) * tp.data)
        total_updates += 1
        vla.eval()

    _d = os.path.dirname(args.out)
    os.makedirs(_d, exist_ok=True) if _d else None
    # B-151 (5th audit): end-of-run save must be atomic like the periodic ones.
    from storage import safe_save
    safe_save({"policy": vla.state_dict(), "config": backbone_cfg, "args": vars(args)},
              args.out)
    print(f"[VLA-RL] done. best reward {best_reward:.1f} -> {args.out}")


if __name__ == "__main__":
    main()
