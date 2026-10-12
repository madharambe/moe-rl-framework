import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from collections import deque
import random

from algorithms.base import RLAlgorithm
from gpu_accel import resolve_device


class DQNNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNReplayBuffer:
    def __init__(self, capacity: int = 100000, device=None):
        self.capacity = capacity
        self.device = device or resolve_device()
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        s_vec = state.get_combined_state() if hasattr(state, 'get_combined_state') else np.asarray(state, dtype=np.float32).flatten()
        ns_vec = next_state.get_combined_state() if hasattr(next_state, 'get_combined_state') else np.asarray(next_state, dtype=np.float32).flatten()
        self.buffer.append((s_vec, action, reward, ns_vec, done))

    def sample(self, batch_size: int) -> Dict:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return {
            'states': torch.from_numpy(states).to(self.device),
            'actions': torch.from_numpy(actions).long().to(self.device).unsqueeze(1),
            'rewards': torch.from_numpy(rewards).to(self.device).unsqueeze(1),
            'next_states': torch.from_numpy(next_states).to(self.device),
            'dones': torch.from_numpy(dones).to(self.device).unsqueeze(1),
        }

    def __len__(self):
        return len(self.buffer)


class DQN(RLAlgorithm):
    def __init__(self, policy, learning_rate: float = 3e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_capacity: int = 100000,
                 batch_size: int = 64, target_update_interval: int = 100,
                 n_actions: int = 4, device: Optional[str] = None,
                 compile_mode: Optional[str] = None):
        super().__init__(policy, learning_rate, gamma, device, compile_mode)

        self.q_net = DQNNet(policy.combined_dim, n_actions).to(self.device)
        self.target_net = DQNNet(policy.combined_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.buffer = DQNReplayBuffer(buffer_capacity, self.device)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def get_action(self, state, deterministic: bool = False):
        if not deterministic and random.random() < self.epsilon:
            return np.array([np.random.randint(self.n_actions)], dtype=np.float32), 0.0

        with torch.no_grad():
            if hasattr(state, 'get_combined_state'):
                vec = state.get_combined_state().flatten()
            else:
                vec = np.asarray(state).flatten()
            s = torch.from_numpy(vec).to(self.device).unsqueeze(0)
            encoded = self.policy._split_and_encode(s)
            q_values = self.q_net(encoded)
            action = q_values.argmax(dim=-1).item()
            return np.array([action], dtype=np.float32), 0.0

    def update(self, buffer=None) -> Dict[str, float]:
        buf = buffer or self.buffer
        if len(buf) < self.batch_size:
            return {"q_loss": 0.0}

        batch = buf.sample(self.batch_size)
        states = self.policy._split_and_encode(batch['states'])
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = self.policy._split_and_encode(batch['next_states'])
        dones = batch['dones']

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * max_next_q

        loss = torch.nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"q_loss": loss.item(), "epsilon": self.epsilon}
