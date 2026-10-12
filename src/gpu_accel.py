import torch
import numpy as np
from typing import Optional, Any
from functools import lru_cache

_DEVICE = None
_MEM = None


def init_gpu():
    if not torch.cuda.is_available():
        print("[GPU] No CUDA device found")
        return False
    d = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[GPU] {name} | Compute {cap} | {total/1e9:.1f} GB")
    if cap >= (7, 0):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return True


def resolve_device(prefer: str = "cuda", force: bool = False) -> torch.device:
    global _DEVICE
    # 18th-audit: the cache now respects an explicit "cpu" request AND upgrades a
    # cached cpu to cuda when a later call asks for cuda — previously a first
    # call with prefer="cpu" (e.g. building the model on CPU) permanently cached
    # cpu and a later "cuda" request silently returned cpu, breaking any code
    # path that constructs on CPU then wants to move to CUDA.
    if _DEVICE is not None and not force:
        if prefer == "cpu" and _DEVICE.type != "cpu":
            # ROUND-5 (Bug 7): explicit cpu requests lose to a cached CUDA
            # device — advertise the override so a --device cpu run that OOMs
            # isn't a silent surprise.
            print(f"[GPU] WARN: explicit device cpu requested, but cached "
                  f"device is {_DEVICE} — returning {_DEVICE}")
            return _DEVICE
        if prefer == "cuda" and _DEVICE.type == "cpu" and torch.cuda.is_available():
            _DEVICE = torch.device("cuda:0")
            init_gpu()
        return _DEVICE
    if prefer == "cuda" and torch.cuda.is_available():
        _DEVICE = torch.device("cuda:0")
        init_gpu()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
        print("[GPU] Using Apple MPS")
    else:
        _DEVICE = torch.device("cpu")
        print("[GPU] Using CPU")
    return _DEVICE


def to_device(data: Any, device: Optional[torch.device] = None) -> Any:
    if device is None:
        device = resolve_device()
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data


def compile_policy(policy: torch.nn.Module, mode: str = "reduce-overhead") -> torch.nn.Module:
    if not torch.cuda.is_available():
        return policy
    try:
        compiled = torch.compile(policy, mode=mode, dynamic=False)
        print(f"[GPU] Policy compiled (mode={mode})")
        return compiled
    except Exception as e:
        print(f"[GPU] Compile failed: {e}, using uncompiled")
        return policy


class CUDAGraphRunner:
    def __init__(self, policy: torch.nn.Module, device: torch.device):
        self.policy = policy
        self.device = device
        self._graph = None
        self._static_input = torch.empty(0)
        self._static_output = None

    def capture(self, sample_input: torch.Tensor):
        if not torch.cuda.is_available():
            return
        self._graph = torch.cuda.CUDAGraph()
        self._static_input = sample_input.clone().to(self.device)
        with torch.cuda.graph(self._graph):
            self._static_output = self.policy(self._static_input)

    def replay(self, input_data: torch.Tensor):
        if self._graph is None:
            return self.policy(input_data)
        self._static_input.copy_(input_data)
        self._graph.replay()
        return self._static_output


class AMPOptimizer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.enabled else None

    def __enter__(self):
        # B-155: the old __enter__ RETURNED an autocast without entering it,
        # so `with AMPOptimizer()` silently ran fp32. Enter it properly.
        if self.enabled:
            self._ac = torch.amp.autocast("cuda")
            self._ac.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled and getattr(self, "_ac", None) is not None:
            self._ac.__exit__(*args)

    def backward(self, loss, optimizer):
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()


_RESERVE_FRACTION = 0.05

class MemoryBudget:
    def __init__(self, memory_fraction: float = 0.85):
        self.memory_fraction = memory_fraction
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.total = props.total_memory
            self.usable = int(self.total * memory_fraction)
            self.reserve = int(self.total * _RESERVE_FRACTION)
            self.budget = self.usable - self.reserve
        else:
            self.total = 8 * 1024**3
            self.usable = self.total
            self.reserve = 0
            self.budget = self.total
        print(f"[MemBudget] {self.total/1e9:.1f}GB total -> "
              f"{self.usable/1e9:.1f}GB usable -> {self.budget/1e9:.1f}GB budget")

    def used(self) -> int:
        return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    def available(self) -> int:
        return self.budget - self.used()

    def safe_batch_size(self, per_sample: int) -> int:
        avail = self.available()
        if avail <= 0 or per_sample <= 0:
            return 0                       # cannot fit even one sample
        bs = avail // per_sample
        # 18th-audit: if per_sample > avail, the old code returned 1 (still
        # OOMs). Return the honest max instead, 0 meaning "no batch fits".
        return max(0, min(bs, 8192))

    @staticmethod
    def gc():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_memory_budget(memory_fraction: float = 0.85) -> MemoryBudget:
    global _MEM
    if _MEM is None:
        _MEM = MemoryBudget(memory_fraction)
    return _MEM


class ParallelEnvRunner:
    def __init__(self, num_envs: int = 4, env_class=None, env_kwargs=None):
        self.num_envs = num_envs
        self.envs = []
        for _ in range(num_envs):
            env = (env_class or object)(**(env_kwargs or {}))
            self.envs.append(env)

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return [env.step(actions[i]) for i, env in enumerate(self.envs)]
