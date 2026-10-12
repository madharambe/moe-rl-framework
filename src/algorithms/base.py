import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from policy_network import MultimodalPolicy
from gpu_accel import resolve_device, to_device, compile_policy


class RLAlgorithm(ABC):
    def __init__(self, policy: MultimodalPolicy, learning_rate: float = 3e-4,
                 gamma: float = 0.99, device: Optional[str] = None,
                 compile_mode: Optional[str] = None):
        self.policy = policy
        self.device = resolve_device(device or "cuda")
        self.policy = self.policy.to(self.device)

        if compile_mode:
            try:
                self.policy = compile_policy(self.policy, compile_mode)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Compile failed: {e}")

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.step_count = 0

    @abstractmethod
    def update(self, buffer: Any) -> Dict[str, float]:
        ...

    @abstractmethod
    def get_action(self, state, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        ...

    def save(self, path: str = "model.pt"):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, path)

    def load(self, path: str = "model.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.step_count = ckpt.get('step_count', 0)
