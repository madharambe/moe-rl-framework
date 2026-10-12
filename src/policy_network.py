import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

def tanh_log_prob_correction(pre_tanh: torch.Tensor) -> torch.Tensor:
    return (2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1)

VISUAL_HEIGHT = 224
VISUAL_WIDTH = 224
VISUAL_CHANNELS = 3
VISUAL_DIM = VISUAL_HEIGHT * VISUAL_WIDTH * VISUAL_CHANNELS  # 150528

VISUAL_ENCODED_DIM = 512
TEXT_DIM = 512
CODE_DIM = 768
EXEC_DIM = 512


class VisualEncoder(nn.Module):
    def __init__(self, encoded_dim: int = VISUAL_ENCODED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(VISUAL_CHANNELS, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, encoded_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, VISUAL_CHANNELS, VISUAL_HEIGHT, VISUAL_WIDTH)
        return self.encoder(x)


class MultimodalPolicy(nn.Module):
    def __init__(self, state_dim: int = None, action_dim: int = 4,
                 hidden_dim: int = 256, log_std_init: float = -0.5,
                 graph_dim: int = 80):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.visual_encoder = VisualEncoder(VISUAL_ENCODED_DIM)

        self.non_visual_dim = TEXT_DIM + CODE_DIM + EXEC_DIM + graph_dim
        self.combined_dim = VISUAL_ENCODED_DIM + self.non_visual_dim

        self.network = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        self.action_scale = 1.0
        self.action_bias = 0.0

    def _state_to_tensor(self, state) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(state, list):
            vecs = []
            for s in state:
                if hasattr(s, 'get_combined_state'):
                    vecs.append(s.get_combined_state())
                else:
                    vecs.append(np.asarray(s, dtype=np.float32))
            arr = np.array(vecs, dtype=np.float32)
            return torch.from_numpy(arr).to(device)
        if hasattr(state, 'get_combined_state'):
            state_vec = state.get_combined_state()
        else:
            state_vec = state
        if isinstance(state_vec, np.ndarray):
            state_vec = torch.from_numpy(state_vec).float().to(device)
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        return state_vec

    def _split_and_encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= VISUAL_DIM:
            visual_part = x[..., :VISUAL_DIM]
            non_visual_part = x[..., VISUAL_DIM:]
        else:
            return x

        visual_feats = self.visual_encoder(visual_part)
        return torch.cat([visual_feats, non_visual_part], dim=-1)

    def encode_batch(self, state_list):
        arr = np.array([s.get_combined_state() if hasattr(s, 'get_combined_state') else np.asarray(s, dtype=np.float32) for s in state_list], dtype=np.float32)
        raw = torch.from_numpy(arr).to(next(self.parameters()).device)
        return self._split_and_encode(raw)

    def forward(self, state) -> Tuple[torch.Tensor, dict]:
        x = self._state_to_tensor(state)
        x = self._split_and_encode(x)
        return self.forward_encoded(x)

    def forward_encoded(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        h = self.network(x)
        mean = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        return mean, {'log_std': self.log_std, 'value': value}

    def get_action(self, state, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            mean, extras = self.forward(state)
            std = torch.exp(extras['log_std'])
            dist = torch.distributions.Normal(mean, std)

            if deterministic:
                raw_action = mean
                action = torch.tanh(raw_action) * self.action_scale + self.action_bias
                log_prob = dist.log_prob(raw_action).sum(dim=-1) - tanh_log_prob_correction(raw_action)
            else:
                raw_action = mean + std * torch.randn_like(mean)
                action = torch.tanh(raw_action) * self.action_scale + self.action_bias
                log_prob = dist.log_prob(raw_action).sum(dim=-1) - tanh_log_prob_correction(raw_action)

            value = extras['value'].item()
            return action.squeeze(0).cpu().numpy(), log_prob.item()

    def evaluate_actions(self, states, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, extras = self.forward(states)
        std = torch.exp(extras['log_std'])
        raw_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1) - tanh_log_prob_correction(raw_actions)
        entropy = dist.entropy().sum(dim=-1).mean()
        values = extras['value']
        return values, log_probs, entropy
