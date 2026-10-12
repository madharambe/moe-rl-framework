"""
INT8 quantization for MoE/Ambatron models.
W8A8-style: weights stored as per-channel INT8 + scale (true 2x memory savings),
dequantized to fp16 on the fly during forward. Router gates stay BF16
(critical for routing precision). Based on AMD Quark INT8 scheme + DeepSeek-V3
mixed-precision findings.

NOTE: this saves memory (INT8 weight storage), not compute — real W8A8 kernels
(torch._int_mm / cuBLAS int8) would additionally speed up matmuls.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_model import MoEGPT, RotaryEmbedding


class QuantizedLinear(nn.Module):
    """Linear layer with INT8 weight storage; dequantizes to the activation
    dtype on the fly. Interfaces like nn.Linear (in_features/out_features)."""

    def __init__(self, in_features, out_features, weight_int8, scale, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", weight_int8)  # [out, in] int8
        self.register_buffer("scale", scale)              # [out, 1] fp32
        if bias is not None:
            self.register_buffer("bias", bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, getattr(self, "bias", None))

    def dequantized_weight(self) -> torch.Tensor:
        return self.weight_int8.float() * self.scale


@torch.no_grad()
def quantize_linear_int8(linear: nn.Linear, per_channel=True):
    """Quantize Linear layer weights to INT8. Returns scale + zero_point."""
    w = linear.weight
    if per_channel:
        scale = w.abs().max(dim=1, keepdim=True)[0] / 127.0
    else:
        scale = w.abs().max() / 127.0
    # B-156: all-zero rows give scale=0 -> NaN on divide.
    scale = scale.clamp_min(1e-12)
    w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


@torch.no_grad()
def quantize_model_int8(model: MoEGPT, keep_gates_bf16=True):
    """Replace all nn.Linear weights with INT8 storage + fp16 dequant matmul.
    Router gates kept at BF16 (critical for routing precision) — the whole
    model is cast to bf16 first so dtypes stay consistent.
    Returns the model with QuantizedLinear layers."""
    if keep_gates_bf16:
        model.to(torch.bfloat16)
    # Drop stale RoPE caches: they are plain attributes (not buffers) and
    # would keep the old dtype/device after model.to() or quantization.
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module._cached_cos = None
            module._cached_sin = None
    quantized_layers = 0
    total_layers = 0
    skipped_layers = []

    # Replace innermost first: iterate over a snapshot, skip children already
    # wrapped by a replaced parent.
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        total_layers += 1
        is_gate = "router" in name
        is_head = "lm_head" in name
        is_embed = "embed" in name

        if (is_gate and keep_gates_bf16) or is_head or is_embed:
            skipped_layers.append(name)
            continue

        w_int8, scale = quantize_linear_int8(module)
        ql = QuantizedLinear(module.in_features, module.out_features,
                             w_int8, scale, module.bias)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        if isinstance(parent, nn.ModuleList):
            parent[int(parts[-1])] = ql
        elif isinstance(parent, nn.ModuleDict):
            parent[parts[-1]] = ql
        else:
            setattr(parent, parts[-1], ql)
        quantized_layers += 1

    print(f"[INT8] Quantized {quantized_layers}/{total_layers} linear layers")
    print(f"[INT8] Skipped (BF16): {len(skipped_layers)} layers (gates/head/embed)")
    return model


def estimate_size_savings(model: MoEGPT):
    """Estimate memory: actual INT8 storage vs fp16.
    B-120: QuantizedLinear stores its int8 weight as a BUFFER — named_parameters()
    no longer sees it, so the old loop silently undercounted quantized models
    (reported ~fp16 sizes). Count buffers too."""
    total_fp16_bytes = 0
    total_int8_bytes = 0
    seen_ids = set()

    def _account(name, tensor):
        nonlocal total_fp16_bytes, total_int8_bytes
        if id(tensor) in seen_ids:  # tied weights (embed == lm_head)
            return
        seen_ids.add(id(tensor))
        elems = tensor.numel()
        total_fp16_bytes += elems * 2
        if "router" in name or "lm_head" in name or "embed" in name:
            total_int8_bytes += elems * 2
        else:
            total_int8_bytes += elems * 1

    for name, p in model.named_parameters():
        _account(name, p.data)
    for name, b in model.named_buffers():
        # QuantizedLinear.weight_* are the quantized payload; count int8.
        if hasattr(b, "dtype") and b.dtype in (torch.int8, torch.uint8, torch.int16,
                                              torch.int32):
            total_fp16_bytes += b.numel() * 2
            total_int8_bytes += b.numel() * b.element_size()
        else:
            _account(f"buf:{name}", b)
    ratio = total_fp16_bytes / max(total_int8_bytes, 1)
    fp16_mb = total_fp16_bytes / (1024 ** 2)
    int8_mb = total_int8_bytes / (1024 ** 2)
    print(f"[INT8] FP16: {fp16_mb:.0f} MB -> INT8: {int8_mb:.0f} MB ({ratio:.1f}x)")
    return int8_mb
