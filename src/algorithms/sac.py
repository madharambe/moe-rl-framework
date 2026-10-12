import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
from collections import deque
import random

from algorithms.base import RLAlgorithm
from gpu_accel import resolve_device, to_device, get_memory_budget
from policy_network import tanh_log_prob_correction


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, device: torch.device = None):
        self.capacity = capacity
        self.device = device or resolve_device()
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        s_vec = state.get_combined_state() if hasattr(state, 'get_combined_state') else np.asarray(state, dtype=np.float32)
        ns_vec = next_state.get_combined_state() if hasattr(next_state, 'get_combined_state') else np.asarray(next_state, dtype=np.float32)
        self.buffer.append((s_vec, action, reward, ns_vec, done))

    def sample(self, batch_size: int) -> Dict:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = torch.tensor(np.array([b[4] for b in batch], dtype=np.float32), device=self.device).unsqueeze(1)
        return {'states': states, 'actions': actions, 'rewards': rewards,
                'next_states': next_states, 'dones': dones}

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class NormBoundedQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 max_weight_norm: float = 10.0, max_feature_norm: float = 100.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.max_weight_norm = max_weight_norm
        self.max_feature_norm = max_feature_norm
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def _bound_norms(self, x, layer):
        with torch.no_grad():
            w_norm = layer.weight.norm().item()
            if w_norm > self.max_weight_norm:
                layer.weight.data *= self.max_weight_norm / w_norm
        x = layer(x)
        feat_norm = x.norm(dim=-1, keepdim=True)
        if feat_norm.max().item() > self.max_feature_norm:
            x = x * (self.max_feature_norm / (feat_norm + 1e-8))
        return x

    def forward(self, state, action):
        x = self._bound_norms(torch.cat([state, action], dim=-1), self.fc1)
        x = F.relu(x)
        x = self._bound_norms(x, self.fc2)
        x = F.relu(x)
        return self.fc3(x)


class SAC(RLAlgorithm):
    def __init__(self, policy, learning_rate: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2, automatic_alpha_tuning: bool = True,
                 target_entropy: Optional[float] = None, buffer_capacity: int = 100000,
                 batch_size: int = 256, device: Optional[str] = None,
                 compile_mode: Optional[str] = None,
                 policy_delay: int = 2,
                 grad_norm_clip: float = 5.0,
                 action_scale: float = 1.0,
                 noise_repeat: int = 1,
                 use_norm_bounded_q: bool = False):
        super().__init__(policy, learning_rate, gamma, device, compile_mode)

        qnet_cls = NormBoundedQNetwork if use_norm_bounded_q else QNetwork
        self.q1 = qnet_cls(self.policy.combined_dim, policy.action_dim).to(self.device)
        self.q2 = qnet_cls(self.policy.combined_dim, policy.action_dim).to(self.device)
        self.q1_target = qnet_cls(self.policy.combined_dim, policy.action_dim).to(self.device)
        self.q2_target = qnet_cls(self.policy.combined_dim, policy.action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)
        self.tau = tau
        self.alpha = alpha
        self.automatic_alpha = automatic_alpha_tuning
        if automatic_alpha_tuning:
            self.target_entropy = target_entropy or -float(policy.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)

        self.buffer = ReplayBuffer(buffer_capacity, self.device)
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.grad_norm_clip = grad_norm_clip
        self.action_scale = max(0.1, action_scale)
        self.noise_repeat = max(1, noise_repeat)
        self._total_updates = 0
        self._noise_buffer = None
        self._noise_step = 0

    def _sample_action_noise(self, shape, device):
        if self.noise_repeat <= 1:
            return torch.randn(shape, device=device)
        if self._noise_buffer is None or self._noise_step >= self.noise_repeat:
            self._noise_buffer = torch.randn(shape, device=device)
            self._noise_step = 0
        self._noise_step += 1
        return self._noise_buffer

    def get_action(self, state, deterministic: bool = False):
        with torch.no_grad():
            mean, extras = self.policy(state)
            if deterministic:
                action = torch.tanh(mean) * self.action_scale
                log_prob = -tanh_log_prob_correction(mean).item()
            else:
                std = torch.exp(extras['log_std'])
                noise = self._sample_action_noise(mean.shape, mean.device)
                raw = mean + std * noise
                action = torch.tanh(raw) * self.action_scale
                log_prob = (torch.distributions.Normal(mean, std).log_prob(raw).sum(dim=-1)
                            - tanh_log_prob_correction(raw)).item()
            return action.squeeze(0).cpu().numpy(), log_prob

    def update(self, buffer=None) -> Dict[str, float]:
        buf = buffer or self.buffer
        if len(buf) < self.batch_size:
            return {"q_loss": 0.0, "policy_loss": 0.0, "alpha": self.alpha}

        batch = buf.sample(self.batch_size)
        raw_states = batch['states']
        raw_next_states = batch['next_states']

        with torch.no_grad():
            s_t = torch.from_numpy(raw_states).to(self.device)
            ns_t = torch.from_numpy(raw_next_states).to(self.device)
            states = self.policy._split_and_encode(s_t)
            next_states = self.policy._split_and_encode(ns_t)

        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']

        with torch.no_grad():
            mean, extras = self.policy.forward_encoded(next_states)
            std = torch.exp(extras['log_std'])
            noise = torch.randn_like(mean)
            next_raw = mean + std * noise
            next_actions = torch.tanh(next_raw) * self.action_scale
            next_log_prob = (torch.distributions.Normal(mean, std).log_prob(next_raw).sum(dim=-1, keepdim=True)
                             - tanh_log_prob_correction(next_raw).unsqueeze(-1))
            target_q = (torch.min(self.q1_target(next_states, next_actions),
                                  self.q2_target(next_states, next_actions))
                        - self.alpha * next_log_prob)
            target = rewards + self.gamma * (1 - dones) * target_q

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, target)
        q2_loss = torch.nn.functional.mse_loss(q2_pred, target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_norm_clip)
        self.q_optimizer.step()

        mean_pi, extras_pi = self.policy.forward_encoded(states)
        std_pi = torch.exp(extras_pi['log_std'])
        noise_pi = torch.randn_like(mean_pi)
        pi_raw = mean_pi + std_pi * noise_pi
        new_actions = torch.tanh(pi_raw) * self.action_scale
        new_log_probs = (torch.distributions.Normal(mean_pi, std_pi).log_prob(pi_raw).sum(dim=-1, keepdim=True)
                         - tanh_log_prob_correction(pi_raw).unsqueeze(-1))

        policy_loss = torch.tensor(0.0)
        if self._total_updates % self.policy_delay == 0:
            q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
            policy_loss = (self.alpha * new_log_probs - q_new).mean()

            self.optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            if self.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clip)
            self.optimizer.step()

        if self.automatic_alpha:
            alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        for target, source in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        self._total_updates += 1
        self.step_count += 1
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(), "alpha": self.alpha}

    def save(self, path: str = "sac_model.pt"):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'alpha': self.alpha,
        }, path)

    def load(self, path: str = "sac_model.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.q1.load_state_dict(ckpt['q1_state_dict'])
        self.q2.load_state_dict(ckpt['q2_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
        self.alpha = ckpt.get('alpha', self.alpha)
