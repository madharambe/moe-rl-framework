import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
import math

from algorithms.base import RLAlgorithm
from gpu_accel import resolve_device, to_device, AMPOptimizer
from policy_network import tanh_log_prob_correction


class PPOBuffer:
    def __init__(self, capacity: int = 2048, device: torch.device = None):
        self.capacity = capacity
        self.device = device or resolve_device()
        self.clear()

    def clear(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.log_probs: list = []
        self.dones: list = []
        self.ptr = 0

    def add(self, state, action, reward, value, log_prob, done):
        if self.ptr < self.capacity:
            s_vec = state.get_combined_state() if hasattr(state, 'get_combined_state') else np.asarray(state, dtype=np.float32).flatten()
            self.states.append(s_vec)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
            self.ptr += 1

    def get(self) -> Dict:
        states_np = np.array(self.states, dtype=np.float32)
        states_t = torch.from_numpy(states_np).to(self.device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=self.device)
        values_t = torch.tensor(np.array(self.values), dtype=torch.float32, device=self.device)
        log_probs_t = torch.tensor(self.log_probs, device=self.device)
        dones_t = torch.tensor(np.array(self.dones, dtype=np.float32), device=self.device)
        return {
            'states': states_t,
            'actions': actions_t,
            'rewards': rewards_t,
            'values': values_t,
            'log_probs': log_probs_t,
            'dones': dones_t,
        }

    def __len__(self):
        return self.ptr


class PPO(RLAlgorithm):
    def __init__(self, policy, learning_rate: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, n_epochs: int = 10,
                 device: Optional[str] = None, compile_mode: Optional[str] = None,
                 adaptive_clip: bool = True,
                 adaptive_entropy: bool = True,
                 target_kl: float = 0.02):
        super().__init__(policy, learning_rate, gamma, device, compile_mode)
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.adaptive_clip = adaptive_clip
        self.adaptive_entropy = adaptive_entropy
        self.target_kl = target_kl
        self._running_kl = 0.0
        self._entropy_decay = 1.0
        self._total_steps = 0

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
        data = buffer.get()
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        old_values = data['values']
        old_log_probs = data['log_probs']
        dones = data['dones']

        advantages, returns = self.compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._total_steps += 1

        clip_eps = self.clip_epsilon
        entropy_coef = self.entropy_coef

        encoded_states = self.policy._split_and_encode(states)

        for epoch in range(self.n_epochs):
            mean, extras = self.policy.forward_encoded(encoded_states)
            std = torch.exp(extras['log_std'])
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            values = extras['value']
            entropies = dist.entropy().sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            if self.adaptive_clip:
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                    self._running_kl = 0.9 * self._running_kl + 0.1 * approx_kl
                    kl_ratio = self._running_kl / max(self.target_kl, 1e-8)
                    clip_eps = self.clip_epsilon * max(0.5, min(2.0, 1.0 / (kl_ratio + 0.1)))
            else:
                clip_eps = self.clip_epsilon

            if self.adaptive_entropy:
                progress = min(1.0, self._total_steps / 200.0)
                self._entropy_decay = 1.0 - 0.5 * progress
                with torch.no_grad():
                    current_entropy = entropies.mean().item()
                    if current_entropy < 0.5:
                        self._entropy_decay = min(2.0, self._entropy_decay * 1.1)
                    elif current_entropy > 3.0:
                        self._entropy_decay = max(0.3, self._entropy_decay * 0.95)
                entropy_coef = self.entropy_coef * self._entropy_decay
            else:
                entropy_coef = self.entropy_coef

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_coef * F.mse_loss(values, returns)
            entropy_loss = -entropy_coef * entropies.mean()
            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.step_count += 1
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                "entropy": entropies.mean().item(), "loss": loss.item(),
                "clip_eps": clip_eps, "entropy_coef": entropy_coef,
                "approx_kl": self._running_kl}

    def get_action(self, state, deterministic: bool = False):
        with torch.no_grad():
            mean, extras = self.policy(state)
            std = torch.exp(extras['log_std'])
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                raw = mean
                action = torch.tanh(raw)
            else:
                raw = mean + std * torch.randn_like(mean)
                action = torch.tanh(raw)
            log_prob = dist.log_prob(raw).sum(dim=-1) - tanh_log_prob_correction(raw)
            return action.squeeze(0).cpu().numpy(), log_prob.item()
