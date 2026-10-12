"""
LoRA (Low-Rank Adaptation) for efficient fine-tuning.
Injects trainable rank-16 matrices into Linear layers.
Freezes base model, trains only adapters (~0.2% of total params).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class LoRALinear(nn.Module):
    """Linear layer with LoRA (or DoRA) adapter.

    LoRA forward:  y = Wx + (x @ A @ B) * alpha/r
    DoRA forward:  y = m * [(W0 + ΔW) / ||W0 + ΔW||_c] @ x, m = trainable
    per-output magnitude (DoRA, Liu et al., arXiv:2402.09353 / ICML'24:
    weight-decomposed LoRA matches full-FT learning patterns and consistently
    beats plain LoRA, incl. at half rank — used rank 16 here).
    PyTorch Linear weight shape: [out_features, in_features]
    """
    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: float = 32.0,
                 dora: bool = False):
        super().__init__()
        self.base = base_linear
        self.base.requires_grad_(False)
        out_dim, in_dim = base_linear.weight.shape

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dora = dora

        self.A = nn.Parameter(torch.randn(in_dim, r, device=base_linear.weight.device,
                                          dtype=base_linear.weight.dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_dim, device=base_linear.weight.device,
                                          dtype=base_linear.weight.dtype))
        if dora:
            # magnitude = column-norm of the base weight per OUTPUT row (dim=1
            # over the input axis), init to |W0|_c so W' == W0 at step 0
            # (DoRA Eq.5: W' = m·[W/||W||_c]); y = W'ᵀx scales output rows i by m_i.
            self.m = nn.Parameter(
                torch.linalg.vector_norm(base_linear.weight, ord=2, dim=1)
                .detach().to(dtype=torch.float32).reshape(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B-83: cast A/B to x.dtype for the compute — the masters stay fp32
        # (optimizer precision) but non-autocast forwards (e.g. GRPO log-probs,
        # eval) would otherwise crash on bf16 @ fp32.
        a = self.A.to(x.dtype)
        b = self.B.to(x.dtype)
        wd = self.base.weight + (a @ b).t() * self.scaling   # [out, in]
        if self.dora:
            # DoRA §4.3: detached column-norm reduces graph memory ~24%; the
            # per-layer matrix is small here, but the trick is free to keep.
            c = torch.linalg.vector_norm(wd, ord=2, dim=1, keepdim=True).detach()
            wd = wd / c.clamp_min(1e-8)
            wd = wd * self.m.to(x.dtype)
        # 19th-audit B6 (reproduced): the bias stayed bf16 under
        # mod.base.to(dtype=bf16) while x (and thus the weight) was fp32 in a
        # non-autocast forward — F.linear raised "self and mat2 must have the
        # same dtype". Cast the bias to x.dtype like A/B.
        _bias = self.base.bias
        if _bias is not None and _bias.dtype != x.dtype:
            _bias = _bias.to(x.dtype)
        return F.linear(x, wd, _bias)


def merge_lora(model: nn.Module) -> None:
    """Fuse LoRA adapters into their base weights in place (Bug-11: the SFT
    output with A/B keys is not loadable by a plain MoEGPT; merge first).
    DoRA: bake W' = m · (W0+ΔW)/||W0+ΔW||_c into the base weight."""
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                wd = mod.base.weight + (mod.A @ mod.B).t() * mod.scaling
                if mod.dora:
                    c = torch.linalg.vector_norm(wd, ord=2, dim=1, keepdim=True).clamp_min(1e-8)
                    wd = wd / c
                    wd = wd * mod.m.to(wd.dtype)
                mod.base.weight.copy_(wd)
                mod.A.zero_()
                mod.B.zero_()


def remove_lora(model: nn.Module) -> None:
    """Replace LoRALinear modules with their fused base Linear (call AFTER
    merge_lora) so the state dict is a plain MoEGPT that the agent can load."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], mod.base)


def _add_lora_to_module(module: nn.Module, r: int = 16, alpha: float = 32.0, dora: bool = False, _path: str = ""):
    """Replace all nn.Linear in module with LoRALinear (recursive).
    lm_head is excluded: with tie_weights it SHARES the token_embed tensor, so
    merging LoRA into it would corrupt the input embedding (round-11 finding 8)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name != "lm_head":
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dora=dora))
        else:
            _add_lora_to_module(child, r=r, alpha=alpha, dora=dora, _path=_path + "." + name)


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32.0,
             target_modules: Optional[List[str]] = None, dora: bool = False):
    """Add LoRA/DoRA adapters to model. If target_modules is None, wraps ALL Linear layers."""
    if target_modules is not None:
        for name, module in model.named_modules():
            if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dora=dora))
    else:
        _add_lora_to_module(model, r=r, alpha=alpha, dora=dora)


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA adapter parameters (A and B matrices)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        # If freeze_base not called yet, find LoRALinear A/B manually
        for module in model.modules():
            if isinstance(module, LoRALinear):
                params.append(module.A)
                params.append(module.B)
    return params


def freeze_base(model: nn.Module):
    """Freeze all base model parameters, keep LoRA adapters trainable."""
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)
            if module.dora:
                module.m.requires_grad_(True)


def save_lora(model: nn.Module, path: str):
    """Save only LoRA adapter weights. Sep-5 (audit): also saves the DoRA
    magnitude vector m — the load-side restore landed earlier but save never
    wrote m, so a DoRA save->load silently kept m at init scale."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.A"] = module.A.data
            lora_state[f"{name}.B"] = module.B.data
            lora_state[f"{name}.r"] = torch.tensor(module.r)
            lora_state[f"{name}.alpha"] = torch.tensor(module.alpha)
            if module.dora:
                lora_state[f"{name}.m"] = module.m.data
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)//4} adapters to {path}")


def load_lora(model: nn.Module, path: str):
    """Load LoRA adapter weights into model. Sep-5 (audit): also restores the
    DoRA magnitude vector m (save_lora already stores it; load dropped it, so a
    DoRA save->load silently kept m at init scale)."""
    lora_state = torch.load(path, map_location='cpu', weights_only=True)
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.A"
            b_key = f"{name}.B"
            m_key = f"{name}.m"
            if a_key in lora_state and b_key in lora_state:
                module.A.data = lora_state[a_key].to(module.A.device)
                module.B.data = lora_state[b_key].to(module.B.device)
                if module.dora and m_key in lora_state:
                    module.m.data = lora_state[m_key].to(module.m.device)
                loaded += 1
    print(f"[LoRA] Loaded {loaded}/{len(lora_state)//4} adapters from {path}")


def count_lora_params(model: nn.Module) -> int:
    """Count total LoRA trainable parameters."""
    return sum(p.numel() for p in get_lora_params(model))
