from algorithms.base import RLAlgorithm
from algorithms.ppo import PPO
from algorithms.sac import SAC
from algorithms.td3 import TD3
from algorithms.dqn import DQN

ALGORITHM_REGISTRY = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "dqn": DQN,
}

def create_algorithm(name: str, policy, **kwargs) -> RLAlgorithm:
    name = name.lower()
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name](policy, **kwargs)
