# ROUND-205a: Q4_0-style blockwise INT4 quantization for Ambatron MoE.
# Block of 32 contiguous weights, one fp16 scale per block (llama.cpp Q4_0
# layout). Symmetric. Router gates / lm_head / embeddings stay high precision
# (mirrors the INT8 path — routing precision is critical).
import torch
import torch.nn as nn

_BLOCK = 32


class QuantizedLinearInt4(nn.Module):
    """Linear with INT4 blockwise storage; dequantizes on the fly.

    weight_packed: uint8 tensor, 2 int4 values per byte (hi nibble first).
    scales: fp16, one per block of _BLOCK rows-of-out (per OUT row: blocks
            along in).
    """

    def __init__(self, in_features, out_features, weight_packed, scales, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def dequantized_weight(self) -> torch.Tensor:
        w = dequant_blockwise(self.weight_packed, self.scales,
                              self.out_features, self.in_features)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight().to(x.dtype)
        out = torch.nn.functional.linear(x, w, self.bias)
        return out


def pack_int4(w4: torch.Tensor) -> torch.Tensor:
    """w4: int8 tensor with values in [-8, 7] -> packed uint8, 2 per byte.

    Byte j covers elements (2j, 2j+1): high nibble = element 2j (two's
    complement low nibble), low nibble = element 2j+1.
    """
    qc = (w4 & 0x0F).to(torch.uint8)          # two's-complement low nibbles
    even = qc[..., 0::2]
    odd = qc[..., 1::2]
    return ((even << 4) | odd).contiguous()


def unpack_int4(packed: torch.Tensor, out_rows: int, in_cols: int) -> torch.Tensor:
    fb = packed.view(-1).to(torch.uint8)
    hi_n = ((fb >> 4) & 0x0F).to(torch.int8)   # even slot
    lo_n = (fb & 0x0F).to(torch.int8)          # odd slot
    # sign-extend the two's-complement nibbles back to [-8, 7]
    hi_n = torch.where(hi_n >= 8, hi_n - 16, hi_n)
    lo_n = torch.where(lo_n >= 8, lo_n - 16, lo_n)
    vals = torch.stack([hi_n, lo_n], dim=-1).reshape(-1)[: out_rows * in_cols]
    return vals.view(out_rows, in_cols)


@torch.no_grad()
def quantize_linear_int4(linear: nn.Linear):
    """Return (packed, scales, shape) for a per-out-row blockwise int4 encode."""
    w = linear.weight.float()
    out_rows, in_cols = w.shape
    nblocks = (in_cols + _BLOCK - 1) // _BLOCK
    wb = torch.nn.functional.pad(w, (0, nblocks * _BLOCK - in_cols))
    blocks = wb.view(out_rows, nblocks, _BLOCK)
    scale = blocks.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / 8.0
    q = (blocks / scale).round().clamp(-8, 7).to(torch.int8)
    packed = pack_int4(q)
    scales = scale.squeeze(-1).to(torch.float16)  # (out_rows, nblocks)
    return packed.contiguous(), scales.contiguous(), (out_rows, in_cols)


def dequant_blockwise(packed, scales, out_rows, in_cols):
    w = unpack_int4(packed, out_rows, in_cols)
    nblocks = (in_cols + _BLOCK - 1) // _BLOCK
    wb = w.view(out_rows, nblocks, _BLOCK).to(torch.float16)
    w = wb * scales.unsqueeze(-1)
    return w.reshape(out_rows, nblocks * _BLOCK)[:, :in_cols]


@torch.no_grad()
def quantize_model_int4(model: nn.Module, skip_names=("router", "lm_head", "embed")):
    """Replace nn.Linear with QuantizedLinearInt4 (per the skip filter)."""
    model.to(torch.float16)
    for module in model.modules():
        if isinstance(module, nn.Module) and hasattr(module, "_cached_cos"):
            module._cached_cos = None
            module._cached_sin = None
    done = total = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        total += 1
        if any(s in name for s in skip_names):
            continue
        packed, scales, shape = quantize_linear_int4(module)
        ql = QuantizedLinearInt4(module.in_features, module.out_features,
                                 packed, scales, module.bias)
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
        done += 1
    print(f"[INT4] Quantized {done}/{total} linear layers (Q4_0 block{_BLOCK})")
    return model


def int4_bytes_for(model) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinearInt4):
            total += m.weight_packed.numel() + m.scales.numel() * 2
    return total
