#!/usr/bin/env python3
"""
src/test_agent.py - Basic script to run and test the Robot Agent with all components.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sim_environment import RoboticsEnv
from robot_core import RobotAgent
from policy_network import MultimodalPolicy
from llm_agent import LLMPlanner
from ppo_trainer import PPOTrainer, PPOBuffer

import numpy as np

print("--- RSI Project: Multimodal RL Agent ---")
print("This is a lightweight implementation combining:")
print("  1. Multimodal Policy Network (Vision + Text/Code + State)")
print("  2. Physics Simulator (PyBullet)")
print("  3. LLM for Code Generation & Planning")
print("  4. PPO Trainer for Reinforcement Learning\n")

# --- 1. Initialize components ---
print("[*] Initializing components...")

# Option 1: Simple dummy mode (no physics, no LLM API)
print("[*] Mode: Dummy (No physics, Mock LLM)")
env = RoboticsEnv(use_physics=False)
policy = MultimodalPolicy(action_dim=4)
llm = LLMPlanner()  # Will run in mock mode without API key

agent = RobotAgent(
    env=env,
    policy=policy,
    llm_planner=llm,
    use_llm=True
)

# --- 2. Optional: Quick PPO Buffer Test ---
print("\n[*] Running a single episode to verify integration...")
observation, _ = env.reset()
done = False
step_count = 0

while not done and step_count < 20:
    # Action from policy
    action, log_prob = policy.get_action(observation, deterministic=False)
    
    # Step environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    step_count += 1
    print(f"  Step {step_count}: action={action}, reward={reward:.2f}")
    
    observation = next_obs
    done = terminated or truncated

print("\n[!] Note: Full PPO training loop requires running `train_ppo.py`.")
print("[!] To use Physics Engine, ensure `pip install pybullet` is done.")
print("[!] To use LLM, set your OPENAI_API_KEY environment variable.\n")

print("--- Initial test run completed successfully! ---")
