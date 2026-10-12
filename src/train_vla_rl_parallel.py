"""B-65: PARALLEL physics VLA RL — IMPALA-style CPU PyBullet workers + a
batched GPU inference thread + a GPU TD3+BC+RWR learner (see train_vla_rl.py
for the single-env sequential version and the algorithm reference notes).

Why: the physics env is CPU-bound (PyBullet DIRECT per env; the 12 CPU cores
sit mostly idle while the GPU idles at ~1%). Reference: gym-pybullet-drones
(arXiv:2103.02142) reports ~80x data throughput from parallel PyBullet envs;
Isaac Gym (arXiv:2108.10470) CPU-parallel scenes; IMPALA-style CPU workers.

Design:
- N worker threads, each owns RoboticsEnv(use_physics=True) (each creates its
  own PyBullet DIRECT connection/world — verified independent).
- One GPU inference thread batches pending observations and runs
  deterministic_action (shared vla under a lock with the learner).
- The learner (main thread) does the TD3+BC+RWR update loop on the shared
  buffer; B-59 periodic safe_saves; respawns hung workers (pybullet native
  hangs kill the log + ignore SIGINT — the sequential run hung once at
  step 1200, hence the liveness watch here).
"""
# 19th-audit H6: PYTORCH_CUDA_ALLOC_CONF was INSIDE the module docstring
# (line 2 of the old file) and never executed — the fragmentation-killer flag
# this parallel script depends on was silently absent. Set it BEFORE any torch
# import (must precede torch's first CUDA init).
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import contextlib
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_vla_rl import (  # noqa: E402
    GOALS, TwinCritic, VLABuffer, _augment_images, _task_from_state, _to_uint8,
    env_step_to_vla,
)
from moe_model import MoEConfig  # noqa: E402
from lora_adapter import add_lora  # noqa: E402
from sim_environment import RoboticsEnv  # noqa: E402
from vla_model import VLAPolicy  # noqa: E402

import torch.serialization as ser  # noqa: E402

ser.add_safe_globals([MoEConfig])

BASE = Path(__file__).resolve().parent.parent
STALE_SEC = 300
MAX_RESPAWNS = 3
MAX_PENDING = 32


class InferenceServer:
    """Batches pending observations and runs the shared GPU policy."""

    def __init__(self, vla, device, lock):
        self.vla = vla
        self.device = device
        self.lock = lock
        self.pending = {}      # id -> (event, img_cpu, instr, st_cpu)
        self.next_id = 0
        self.cv = threading.Condition()
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def submit(self, img_cpu, instr, st_cpu):
        with self.cv:
            while len(self.pending) >= MAX_PENDING:
                self.cv.wait(5.0)
            iid = self.next_id
            self.next_id += 1
            ev = threading.Event()
            setattr(ev, "action", None)
            self.pending[iid] = (ev, img_cpu, instr, st_cpu)
            self.cv.notify_all()
        ev.wait(30.0)
        return getattr(ev, "action")

    def _run(self):
        print("[inf] thread started", flush=True)
        while not self._stop:
            try:
                self._infer_once()
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

    def _infer_once(self):
        with self.cv:
            while not self.pending:
                if self._stop:
                    return
                self.cv.wait(0.5)
            items = list(self.pending.items())
            self.pending.clear()
            self.cv.notify_all()
        if not items:
            return
        if os.environ.get("B65_DEBUG"):
            print(f"[inf] got {len(items)} items", flush=True)
        with self.lock:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                imgs = torch.stack([it[1][0].float() / 255.0 for _, it in items]).to(self.device)
                instrs = [it[2] for _, it in items]
                states = torch.stack([it[3][0] for _, it in items]).to(self.device)
                if os.environ.get("B65_DEBUG"):
                    print(f"[inf] fwd imgs {tuple(imgs.shape)} st {tuple(states.shape)}", flush=True)
                acts = self.vla.deterministic_action(
                    imgs, instrs, states,
                    task=_task_from_state(states.float()).to(self.device)).cpu()
        for (iid, (ev, _, _, _)), a in zip(items, acts):
            setattr(ev, "action", a)
            ev.set()


class LockedBuffer:
    """Thread-safe VLABuffer wrapper (workers add, learner samples)."""

    def __init__(self, capacity: int = 3000):
        self.lock = threading.Lock()
        self.buf = VLABuffer(capacity)

    def __len__(self):
        return len(self.buf)

    def sample(self, batch, device):
        with self.lock:
            return self.buf.sample(batch, device)

    def add_all(self, ep_trans, ep_reward):
        with self.lock:
            for (i_, n_, s_, a_, r_, ni_, nin_, ns_, t_, tr_) in ep_trans:
                self.buf.add(i_, n_, s_, a_, r_, ep_reward,
                             ni_, nin_, ns_, t_, tr_)


class Worker:
    """One PyBullet env + policy-acting thread."""

    def __init__(self, wid, server, buffer, args, device, counts):
        self.wid = wid
        self.server = server
        self.buffer = buffer
        self.args = args
        self.counts = counts  # {"steps": int, "eps": int}
        self.last_progress = time.time()
        self.respawns = 0
        self.t = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.t.start()
        return self

    def act(self, obs, instruction, device):
        img_t = torch.from_numpy(_to_uint8(obs.visual.get_tensor()))
        st_t = torch.from_numpy(obs.get_combined_state()[150528:].astype(np.float32))
        a = self.server.submit(img_t.unsqueeze(0), instruction, st_t.unsqueeze(0))
        a = a.numpy()
        a = a + np.random.randn(4).astype(np.float32) * self.args.explore_noise
        return np.clip(a, -1, 1)

    def run(self):
        try:
            env = RoboticsEnv(use_physics=True, seed=self.wid)
        except Exception as e:
            print(f"[W{self.wid}] env init failed: {e}", flush=True)
            self.counts["dead"][self.wid] = True
            return
        instruction = random.choice(GOALS)
        obs, _ = env.reset(seed=self.wid)
        ep_trans, ep_reward = [], 0.0
        errs = 0
        while not self.counts["stop"]:
            try:
                action = self.act(obs, instruction, self.args.device)
                tup = env_step_to_vla(obs, action, env, instruction)
                img, instr, s, a, r, nimg, nin, ns, term, trunc, obs = tup
                ep_trans.append((img, instr, s, a, r, nimg, nin, ns, term, trunc))
                ep_reward += r
                if term or trunc:
                    # B-85: check the best BEFORE resetting ep_reward (the old
                    # order always recorded the first episode as best=0.0, so
                    # the NEW-BEST save path was dead and the true best never
                    # persisted)
                    if self.counts["best"] is None or ep_reward > self.counts["best"]:
                        self.counts["best"] = ep_reward
                    self.buffer.add_all(ep_trans, ep_reward)
                    ep_trans, ep_reward = [], 0.0
                    self.counts["eps"] += 1
                    instruction = random.choice(GOALS)
                    env.min_spawn = self.counts["min_spawn"]
                    obs, _ = env.reset(seed=self.wid * 1000003 + self.counts["eps"])
                self.counts["steps"] += 1
                self.last_progress = time.time()
                errs = 0
            except Exception as e:
                errs += 1
                print(f"[W{self.wid}] worker error ({errs}): {e}", flush=True)
                import traceback as _tb
                _tb.print_exc(file=sys.stdout)
                if errs >= 5:
                    print(f"[W{self.wid}] too many consecutive errors — dying", flush=True)
                    self.counts["dead"][self.wid] = True
                    return
                time.sleep(5.0)


def main():
    ap = argparse.ArgumentParser(description="Parallel physics VLA RL (B-65)")
    ap.add_argument("--backbone", choices=["small", "ambatron"], default="ambatron")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--policy_delay", type=int, default=2)
    ap.add_argument("--target_noise", type=float, default=0.2)
    ap.add_argument("--noise_clip", type=float, default=0.5)
    ap.add_argument("--explore_noise", type=float, default=0.1)
    ap.add_argument("--rwr_beta", type=float, default=0.05)
    ap.add_argument("--rwr_wmax", type=float, default=10.0)
    ap.add_argument("--td3bc_alpha", type=float, default=2.5)
    ap.add_argument("--update_every", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vla_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug", choices=["none", "shift"], default="shift")
    ap.add_argument("--lora_backbone", action="store_true", default=True)
    ap.add_argument("--use_pretrained_siglip", action="store_true", default=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"[B-65] parallel physics RL | workers {args.workers} | GPU {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")

    ck = torch.load(args.vla_ckpt, map_location="cpu", weights_only=True)
    vla = VLAPolicy(backbone_config=ck["config"], action_dim=4, chunk_len=16,
                    freeze_backbone=True, use_pretrained_siglip=args.use_pretrained_siglip)
    add_lora(vla.backbone, r=16)
    vla.load_state_dict(ck["policy"], strict=False)  # ROUND-170: tolerate older vintages, keep present task weights
    cfg = ck["config"]
    del ck
    import gc
    gc.collect()
    vla.to(device)
    # B-65: keep the FROZEN backbone in bf16 on the card (halves the 2.9GB fp32
    # footprint; everything runs under autocast(bf16) anyway). Trainable LoRA /
    # expert / projections stay fp32 for the Adam optimizer.
    if device == "cuda":
        for p in vla.parameters():
            if not p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
    vla.eval()

    critic = TwinCritic(cfg.n_embd, 4).to(device)
    critic_target = TwinCritic(cfg.n_embd, 4).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    # D7 (18th-audit): match train_vla_rl.py — filter to trainable params (the
    # raw vla.parameters() includes frozen backbone/LoRA-free params; harmless
    # (no grads) but wasteful).
    actor_opt = torch.optim.Adam(
        [p for p in vla.parameters() if p.requires_grad], lr=args.lr)

    buffer = LockedBuffer()
    lock = threading.RLock()
    server = InferenceServer(vla, device, lock)
    counts = {"steps": 0, "eps": 0, "stop": False, "dead": {}, "best": None,
              "min_spawn": 0.6}

    workers = [Worker(i, server, buffer, args, device, counts).start()
               for i in range(args.workers)]
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else contextlib.nullcontext())
    best_reward, total_updates = -float("inf"), 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)  # ROUND-170: bare --out x.pt previously FileNotFoundError'd

    last_save = 0
    while counts["steps"] < args.steps:
        # B-65 curriculum: goals spawn at 0.6 distance early (goal episodes ->
        # RWR weights > 0 -> distillation) and harden to 1.2 as training proceeds
        counts["min_spawn"] = 0.6 + 0.6 * min(1.0, counts["steps"] / 4000)
        time.sleep(0.2)
        # liveness: respawn workers that DIED (persistent errors) — not slow ones
        now = time.time()
        counts.setdefault("respawns", {})
        for i, w in enumerate(workers):
            # ROUND-5 (H2): the respawn counter used to live on the Worker
            # OBJECT, but the respawn branch REPLACES workers[i] with a fresh
            # Worker (respawns=0) — the cap NEVER bound, and the all-dead abort
            # below could never fire (unbounded respawn loop on crashing
            # workers). Count per-wid persistently in counts instead.
            # ROUND-169: also treat STALLED workers as dead — a pybullet native
            # hang raises nothing and updates no progress, so 'dead' was never
            # set and the worker was never respawned (infinite TD3-on-stale
            # loop; the module docstring's hang-respawn promise used STALE_SEC
            # which nothing read). A thread that is no longer alive OR hasn't
            # made progress in STALE_SEC is respawn-eligible.
            _stalled = (not w.t.is_alive()) or (now - w.last_progress > STALE_SEC)
            if _stalled:
                counts["dead"][w.wid] = True
            if counts["respawns"].get(w.wid, 0) < MAX_RESPAWNS \
                    and counts["dead"].get(w.wid) \
                    and now - w.last_progress > 60:
                print(f"[B-65] worker {w.wid} died — respawning "
                      f"({counts['respawns'].get(w.wid, 0) + 1}/{MAX_RESPAWNS})",
                      flush=True)
                counts["respawns"][w.wid] = counts["respawns"].get(w.wid, 0) + 1
                counts["dead"].pop(w.wid, None)
                w.last_progress = now
                # B-85: register the NEW worker object back into the list —
                # the old code started a thread nobody tracked, so its
                # liveness/error state was invisible and respawns leaked
                nw = Worker(w.wid, server, buffer, args, device, counts)
                nw.start()
                workers[i] = nw
        if counts["best"] is not None and counts["best"] > best_reward:
            best_reward = counts["best"]
            from storage import safe_save
            safe_save({"policy": vla.state_dict(), "config": cfg, "args": vars(args)},
                      args.out, retries=2, delay=3)
            print(f"[B-65] NEW BEST {best_reward:.1f} @ {counts['steps']} steps — saved", flush=True)
        # Sep-5 (audit) + ROUND-5 (H2): ALL workers dead after exhausting the
        # PERSISTENT respawn cap -> stop sampling the stale buffer + updating
        # forever (TD3 on dead data). The old w.respawns check could never
        # fire (the counter died with the replaced Worker object).
        _dead_all = all(
            counts["dead"].get(w.wid)
            and counts["respawns"].get(w.wid, 0) >= MAX_RESPAWNS
            for w in workers) if workers else False
        if _dead_all and counts["steps"] < args.steps * 0.5:
            print(f"[B-65] ABORT: all {len(workers)} workers dead (respawn cap "
                  f"{MAX_RESPAWNS}) with only {counts['steps']}/{args.steps} "
                  f"steps — no data flow", flush=True)
            counts["stop"] = True
            break
        if counts["steps"] - last_save >= 2000:
            last_save = counts["steps"]
            from storage import safe_save
            safe_save({"policy": vla.state_dict(), "config": cfg, "args": vars(args)},
                      args.out, retries=2, delay=3)
            print(f"[B-65] saved @ {counts['steps']} steps | eps {counts['eps']} | "
                  f"buffer {len(buffer)} | best {best_reward:.1f}", flush=True)
        if len(buffer) < args.batch_size or counts["steps"] % args.update_every != 0:
            continue
        with lock:
            (images, instrs, states, actions, rewards, ep_rets,
             nimages, ninstrs, nstates, terms) = buffer.sample(args.batch_size, device)
            if args.aug == "shift":
                images = _augment_images(images)
                nimages = _augment_images(nimages)
            with autocast_ctx:
                with torch.no_grad():
                    ncond = vla._cond_features(nimages, ninstrs, nstates).mean(dim=1)
                    nact = vla.deterministic_action(
                        nimages, ninstrs, nstates,
                        task=_task_from_state(nstates.float()).to(device))
                    noise = (torch.randn_like(nact) * args.target_noise).clamp(-args.noise_clip, args.noise_clip)
                    nact = (nact + noise).clamp(-1, 1)
                    tq1, tq2 = critic_target(ncond, nact)
                    target = rewards + args.gamma * (1 - terms) * torch.min(tq1, tq2)
                vla.train()
                cond = vla._cond_features(images, instrs, states).mean(dim=1).detach()
                q1, q2 = critic(cond, actions)
                c_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
                critic_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                critic_opt.step()
                if total_updates % args.policy_delay == 0:
                    cond_full = vla._cond_features(images, instrs, states)
                    cond_a = cond_full.mean(dim=1)
                    B = cond_a.shape[0]
                    # B-85: the flow head runs FP32 (bf16 x0/t0 underflowed the
                    # tiny action-head grads — the gen_demos B-65 fix, applied
                    # to the RL actor too)
                    x0 = torch.zeros(B, vla.chunk_len, vla.action_dim, device=cond_a.device, dtype=torch.float32)
                    t0 = torch.zeros(B, device=cond_a.device, dtype=torch.float32)
                    with torch.autocast("cuda", enabled=False):
                        v = vla.expert(x0, t0, cond_full.float(),
                                       task_vec=_task_from_state(
                                           states.float()).to(device))
                    a_det = torch.tanh(v) if vla.expert.tanh_output else v
                    a_det0 = a_det[:, 0].float()
                    q_eval = critic(cond_a.detach(), a_det0)[0]
                    w = torch.clamp(torch.exp(ep_rets.squeeze(1) * args.rwr_beta),
                                    max=args.rwr_wmax).detach()
                    bc_loss = (w * (a_det0 - actions).pow(2).mean(-1)).mean()
                    q_scale = args.td3bc_alpha / (q_eval.abs().mean() + 1e-6)
                    actor_loss = q_scale.detach() * -q_eval.mean() + bc_loss
                    if total_updates % 50 == 0:
                        print(f"  [B-65] q_scale {q_scale.item():.4f} | bc_w "
                              f"{w.mean().item():.3f} | Q {q_eval.mean().item():.1f}", flush=True)
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()
                    for tp, sp in zip(critic_target.parameters(), critic.parameters()):
                        tp.data.copy_(args.tau * sp.data + (1 - args.tau) * tp.data)
                vla.eval()
            total_updates += 1
    counts["stop"] = True
    # B-211: atomic final save (matches the periodic safe_save above).
    from storage import safe_save
    safe_save({"policy": vla.state_dict(), "config": cfg, "args": vars(args)},
              args.out)
    print(f"[B-65] done. {counts['steps']} steps, {counts['eps']} eps -> {args.out}")


if __name__ == "__main__":
    main()