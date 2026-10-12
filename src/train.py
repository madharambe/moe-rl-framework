import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from sim_environment import RoboticsEnv
from policy_network import MultimodalPolicy
from algorithms import create_algorithm
from algorithms.ppo import PPOBuffer
from algorithms.sac import ReplayBuffer
from gpu_accel import resolve_device, compile_policy, get_memory_budget, MemoryBudget
from eval_logger import EvalLogger

_GC_INTERVAL = 500
_STATE_DIM = 152400


def make_env(use_physics: bool = False, render: bool = False):
    return RoboticsEnv(
        use_physics=use_physics,
        render_mode="human" if render else None,
    )


def get_checkpoint_path(algo_name: str, tag: str = "best"):
    return f"{algo_name}_model_{tag}.pt"


def save_checkpoint(algo, algo_name: str, tag: str, episode: int, extra: dict = None):
    path = get_checkpoint_path(algo_name, tag)
    latest_path = get_checkpoint_path(algo_name, "latest")
    state = {
        "policy_state_dict": algo.policy.state_dict(),
        "step_count": algo.step_count,
        "episode": episode,
    }
    if hasattr(algo, "save"):
        algo.save(path)
        algo.save(latest_path)
    else:
        torch.save(state, path)
        torch.save(state, latest_path)
    return path


def load_checkpoint(algo, algo_name: str, device):
    path = get_checkpoint_path(algo_name, "latest")
    if Path(path).exists():
        try:
            if hasattr(algo, "load"):
                algo.load(path)
            else:
                ckpt = torch.load(path, map_location=device)
                algo.policy.load_state_dict(ckpt["policy_state_dict"])
                algo.step_count = ckpt.get("step_count", 0)
            print(f"[RESUME] Loaded checkpoint from {path}")
            return True
        except Exception as e:
            print(f"[RESUME] Failed to load checkpoint: {e}")
    return False


def train_ppo(env, policy, args, mem):
    from algorithms.ppo import PPO
    algo = PPO(
        policy,
        learning_rate=args.lr,
        gamma=args.gamma,
        n_epochs=args.ppo_epochs,
        clip_epsilon=args.ppo_clip,
        device=args.device,
        compile_mode="reduce-overhead" if args.compile else None,
        adaptive_clip=not args.no_adaptive_clip,
        adaptive_entropy=not args.no_adaptive_entropy,
    )
    if args.resume:
        load_checkpoint(algo, args.algo, algo.device)
    logger = EvalLogger(algo_name="ppo", log_dir=args.log_dir, use_tensorboard=args.tensorboard)
    pbar = tqdm(range(args.episodes), desc="PPO", unit="ep")
    best_reward = -float("inf")
    for episode in pbar:
        obs, _ = env.reset()
        buffer = PPOBuffer(capacity=args.steps_per_episode, device=algo.device)
        done = False
        step = 0
        ep_reward = 0.0
        while not done and step < args.steps_per_episode:
            action, log_prob = algo.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            with torch.no_grad():
                _, extras = algo.policy(obs)
                value = extras['value'].item()
            buffer.add(obs, action, reward, value, log_prob, terminated or truncated)
            obs = next_obs
            ep_reward += reward
            step += 1
            done = terminated or truncated
        stats = algo.update(buffer) if buffer.ptr > 0 else {}
        logger.log_episode(episode, ep_reward, step,
                           ploss=stats.get('policy_loss', 0),
                           vloss=stats.get('value_loss', 0),
                           entropy=stats.get('entropy', 0),
                           clip_eps=stats.get('clip_eps', args.ppo_clip),
                           kl=stats.get('approx_kl', 0))
        if ep_reward > best_reward:
            best_reward = ep_reward
            save_checkpoint(algo, args.algo, "best", episode,
                            {"reward": ep_reward, "policy_loss": stats.get('policy_loss', 0)})
        pbar.set_postfix(reward=f"{ep_reward:.1f}", ploss=f"{stats.get('policy_loss',0):.3f}",
                         kl=f"{stats.get('approx_kl',0):.4f}")
    pbar.close()
    logger.print_summary()
    logger.close()
    return algo


def train_sac(env, policy, args, mem):
    from algorithms.sac import SAC
    per_sample = _STATE_DIM * 4
    safe_bs = mem.safe_batch_size(per_sample)
    batch_size = min(args.sac_batch_size, safe_bs)
    algo = SAC(
        policy,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=batch_size,
        device=args.device,
        compile_mode="reduce-overhead" if args.compile else None,
        policy_delay=args.sac_policy_delay,
        grad_norm_clip=args.sac_grad_clip,
        action_scale=args.sac_action_scale,
        noise_repeat=args.sac_noise_repeat,
    )
    obs, _ = env.reset()
    logger = EvalLogger(algo_name="sac", log_dir=args.log_dir, use_tensorboard=args.tensorboard)
    if args.resume:
        load_checkpoint(algo, args.algo, algo.device)
    pbar = tqdm(range(args.total_steps), desc="SAC", unit="step", mininterval=2.0, miniters=100)
    _last_postfix = 0
    ep_reward = 0.0
    ep_step = 0
    ep_count = 0
    best_reward = -float("inf")
    for step in pbar:
        action, _ = algo.get_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        algo.buffer.add(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs
        ep_reward += reward
        ep_step += 1
        if terminated or truncated:
            logger.log_episode(ep_count, ep_reward, ep_step)
            if ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(algo, args.algo, "best", ep_count,
                                {"reward": ep_reward, "step": step})
            ep_count += 1
            ep_reward = 0.0
            ep_step = 0
            obs, _ = env.reset()
        if step % args.sac_update_interval == 0 and len(algo.buffer) > algo.batch_size:
            stats = algo.update()
            logger.log_update(step, **stats)
            if step - _last_postfix >= 100:
                pbar.set_postfix(q_loss=f"{stats['q_loss']:.2f}", alpha=f"{stats['alpha']:.3f}")
                _last_postfix = step
    pbar.close()
    logger.print_summary()
    logger.close()
    return algo


def train_td3(env, policy, args, mem):
    from algorithms.td3 import TD3
    per_sample = _STATE_DIM * 4
    batch_size = min(args.td3_batch_size, mem.safe_batch_size(per_sample))
    algo = TD3(
        policy,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=batch_size,
        device=args.device,
        compile_mode="reduce-overhead" if args.compile else None,
    )
    obs, _ = env.reset()
    logger = EvalLogger(algo_name="td3", log_dir=args.log_dir, use_tensorboard=args.tensorboard)
    if args.resume:
        load_checkpoint(algo, args.algo, algo.device)
    pbar = tqdm(range(args.total_steps), desc="TD3", unit="step", mininterval=2.0, miniters=100)
    _last_postfix = 0
    ep_reward = 0.0
    ep_step = 0
    ep_count = 0
    best_reward = -float("inf")
    for step in pbar:
        action, _ = algo.get_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        algo.buffer.add(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs
        ep_reward += reward
        ep_step += 1
        if terminated or truncated:
            logger.log_episode(ep_count, ep_reward, ep_step)
            if ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(algo, args.algo, "best", ep_count,
                                {"reward": ep_reward, "step": step})
            ep_count += 1
            ep_reward = 0.0
            ep_step = 0
            obs, _ = env.reset()
        if step % args.td3_update_interval == 0 and len(algo.buffer) > algo.batch_size:
            stats = algo.update()
            logger.log_update(step, **stats)
            if step - _last_postfix >= 100:
                pbar.set_postfix(q_loss=f"{stats['q_loss']:.2f}", aloss=f"{stats['actor_loss']:.3f}")
                _last_postfix = step
        if step % _GC_INTERVAL == 0 and step > 0:
            MemoryBudget.gc()
    pbar.close()
    logger.print_summary()
    logger.close()
    return algo


def train_dqn(env, policy, args, mem):
    from algorithms.dqn import DQN
    per_sample = _STATE_DIM * 4
    batch_size = min(args.dqn_batch_size, mem.safe_batch_size(per_sample))
    algo = DQN(
        policy,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=batch_size,
        n_actions=args.dqn_actions,
        device=args.device,
        compile_mode="reduce-overhead" if args.compile else None,
    )
    obs, _ = env.reset()
    logger = EvalLogger(algo_name="dqn", log_dir=args.log_dir, use_tensorboard=args.tensorboard)
    if args.resume:
        load_checkpoint(algo, args.algo, algo.device)
    pbar = tqdm(range(args.total_steps), desc="DQN", unit="step", mininterval=2.0, miniters=100)
    _last_postfix = 0
    ep_reward = 0.0
    ep_step = 0
    ep_count = 0
    best_reward = -float("inf")
    for step in pbar:
        action, _ = algo.get_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        algo.buffer.add(obs, action.tolist()[0] if hasattr(action, '__getitem__') else action,
                        reward, next_obs, terminated or truncated)
        obs = next_obs
        ep_reward += reward
        ep_step += 1
        if terminated or truncated:
            logger.log_episode(ep_count, ep_reward, ep_step)
            if ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(algo, args.algo, "best", ep_count,
                                {"reward": ep_reward, "step": step})
            ep_count += 1
            ep_reward = 0.0
            ep_step = 0
            obs, _ = env.reset()
        if step % args.dqn_update_interval == 0 and len(algo.buffer) > algo.batch_size:
            stats = algo.update()
            logger.log_update(step, **stats)
            if step - _last_postfix >= 100:
                pbar.set_postfix(q_loss=f"{stats['q_loss']:.2f}", eps=f"{stats['epsilon']:.3f}")
                _last_postfix = step
        if step % _GC_INTERVAL == 0 and step > 0:
            MemoryBudget.gc()
    pbar.close()
    logger.print_summary()
    logger.close()
    return algo


def main(args):
    device_name = resolve_device(args.device)
    mem = get_memory_budget(memory_fraction=args.memory_fraction)
    print(f"Device: {device_name} | Algorithm: {args.algo}")
    print(f"Improvements: SAC policy_delay={args.sac_policy_delay}, "
          f"grad_clip={args.sac_grad_clip}, action_scale={args.sac_action_scale}, "
          f"noise_repeat={args.sac_noise_repeat}")

    env = make_env(args.use_physics, args.render)
    policy = MultimodalPolicy(action_dim=4)

    trainers = {
        "ppo": train_ppo,
        "sac": train_sac,
        "td3": train_td3,
        "dqn": train_dqn,
    }

    trainer_fn = trainers.get(args.algo)
    if trainer_fn is None:
        print(f"Unknown algorithm '{args.algo}'. Choose from: {list(trainers.keys())}")
        sys.exit(1)

    algo = trainer_fn(env, policy, args, mem)
    algo.save(f"{args.algo}_model.pt")
    print(f"Model saved to {args.algo}_model.pt")
    print(f"Best checkpoint saved to {args.algo}_model_best.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RSI Unified Training v2")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3", "dqn"])
    parser.add_argument("--resume", action="store_true",
                        help="Load checkpoint and continue training")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for CSV and tensorboard logs")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable tensorboard logging")
    parser.add_argument("--use_physics", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--memory_fraction", type=float, default=0.85,
                        help="Fraction of GPU memory to use (0-1)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--steps_per_episode", type=int, default=100)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--no_adaptive_clip", action="store_true",
                        help="Disable adaptive clipping in PPO")
    parser.add_argument("--no_adaptive_entropy", action="store_true",
                        help="Disable adaptive entropy in PPO")
    parser.add_argument("--sac_batch_size", type=int, default=256)
    parser.add_argument("--sac_update_interval", type=int, default=1)
    parser.add_argument("--sac_policy_delay", type=int, default=2,
                        help="SAC: update policy every N critic updates (FlashSAC)")
    parser.add_argument("--sac_grad_clip", type=float, default=5.0,
                        help="SAC: gradient norm clipping (0=disabled)")
    parser.add_argument("--sac_action_scale", type=float, default=0.3,
                        help="SAC: rescale tanh output (<1.0 for fine control)")
    parser.add_argument("--sac_noise_repeat", type=int, default=5,
                        help="SAC: repeat action noise for temporally-correlated exploration")
    parser.add_argument("--td3_batch_size", type=int, default=256)
    parser.add_argument("--td3_update_interval", type=int, default=2)
    parser.add_argument("--dqn_batch_size", type=int, default=64)
    parser.add_argument("--dqn_update_interval", type=int, default=4)
    parser.add_argument("--dqn_actions", type=int, default=4)

    args = parser.parse_args()
    main(args)
