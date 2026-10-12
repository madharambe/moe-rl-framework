#!/usr/bin/env python3
"""
[DEPRECATED] Use src/train.py --algo ppo instead.
"""
import warnings
warnings.warn("train_ppo.py is deprecated. Use `python src/train.py --algo ppo` instead.", DeprecationWarning, stacklevel=2)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sim_environment import RoboticsEnv
from robot_core import RobotAgent
from policy_network import MultimodalPolicy
from llm_agent import LLMPlanner
from ppo_trainer import PPOTrainer, PPOBuffer

import numpy as np
import torch

def main(args):
    print("--- Starting PPO Training ---")
    
    # 1. Setup Environment and Agent
    env = RoboticsEnv(use_physics=args.use_physics)
    policy = MultimodalPolicy(action_dim=4)
    llm = None if not args.use_llm else LLMPlanner()
    
    agent = RobotAgent(env=env, policy=policy, llm_planner=llm, use_llm=args.use_llm)
    trainer = PPOTrainer(policy)
    
    # 2. Training Loop
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        buffer = PPOBuffer(capacity=args.steps_per_episode)
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < args.steps_per_episode:
            # Get action and value from policy
            action, log_prob = policy.get_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Get value estimate from policy
            with torch.no_grad():
                mean, extras = policy(obs)
                value = extras['value'].item()
            
            # Store transition in buffer
            buffer.add(obs, action, reward, value, log_prob, terminated or truncated)
            
            obs = next_obs
            total_reward += reward
            step += 1
            done = terminated or truncated
            
        print(f"Episode {episode}: Total Reward={total_reward:.2f}, Steps={step}")
        
        # Perform PPO update after collecting a full episode
        if buffer.ptr > 0:
            stats = trainer.update(buffer)
            print(f"  PPO Update: policy_loss={stats['policy_loss']:.4f}, value_loss={stats['value_loss']:.4f}")
            
    print("--- Training Finished ---")
    
    # Save trained policy
    agent.save_policy("trained_policy.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Robot Agent with PPO")
    parser.add_argument("--use_physics", action="store_true", help="Enable PyBullet physics engine")
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM planning")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--steps_per_episode", type=int, default=100, help="Max steps per episode")
    
    args = parser.parse_args()
    main(args)
