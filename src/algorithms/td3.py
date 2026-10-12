import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from collections import deque
import random

from algorithms.base import RLAlgorithm
from gpu_accel import resolve_device
from algorithms.sac import QNetwork, ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action


class TD3(RLAlgorithm):
    def __init__(self, policy, learning_rate: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_delay: int = 2,
                 target_noise: float = 0.2, noise_clip: float = 0.5,
                 buffer_capacity: int = 100000, batch_size: int = 256,
                 device: Optional[str] = None, compile_mode: Optional[str] = None):
        super().__init__(policy, learning_rate, gamma, device, compile_mode)

        self.actor = Actor(policy.combined_dim, policy.action_dim).to(self.device)
        self.actor_target = Actor(policy.combined_dim, policy.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.q1 = QNetwork(policy.combined_dim, policy.action_dim).to(self.device)
        self.q2 = QNetwork(policy.combined_dim, policy.action_dim).to(self.device)
        self.q1_target = QNetwork(policy.combined_dim, policy.action_dim).to(self.device)
        self.q2_target = QNetwork(policy.combined_dim, policy.action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)

        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.buffer = ReplayBuffer(buffer_capacity, self.device)
        self.batch_size = batch_size
        self._total_updates = 0

    def _encode(self, data):
        if isinstance(data, np.ndarray):
            return self.policy._split_and_encode(torch.from_numpy(data).to(self.device))
        encoded = [torch.from_numpy(s.get_combined_state() if hasattr(s, 'get_combined_state') else np.asarray(s, dtype=np.float32)).to(self.device) for s in data]
        return torch.stack(encoded)

    def get_action(self, state, deterministic: bool = False):
        with torch.no_grad():
            s = self._encode([state])
            action = self.actor(s)
            if not deterministic:
                action = action + torch.randn_like(action) * 0.1
                action = action.clamp(-1.0, 1.0)
            return action.squeeze(0).cpu().numpy(), 0.0

    def update(self, buffer=None) -> Dict[str, float]:
        buf = buffer or self.buffer
        if len(buf) < self.batch_size:
            return {"q_loss": 0.0, "actor_loss": 0.0}

        batch = buf.sample(self.batch_size)
        states = self._encode(batch['states'])
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = self._encode(batch['next_states'])
        dones = batch['dones']

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.gamma * (1 - dones) * target_q

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, target)
        q2_loss = torch.nn.functional.mse_loss(q2_pred, target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        actor_loss = torch.tensor(0.0)
        if self._total_updates % self.policy_delay == 0:
            actor_loss = -self.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target, source in [(self.actor_target, self.actor),
                                    (self.q1_target, self.q1),
                                    (self.q2_target, self.q2)]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        self._total_updates += 1
        self.step_count += 1
        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item()}

    def save(self, path: str = "td3_model.pt"):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
        }, path)

    def load(self, path: str = "td3_model.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.q1.load_state_dict(ckpt['q1_state_dict'])
        self.q2.load_state_dict(ckpt['q2_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
