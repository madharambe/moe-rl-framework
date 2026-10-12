"""
MoE LLM: Mixture-of-Experts transformer with GQA, RoPE, SwiGLU, top-2 routing.
Architecture based on Tiny-MoE, NanoChat-MoE, H64LM, and LibMoE research (2025-2026).
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, field


@dataclass
class MoEConfig:
    size: str = ""          # ROUND-187b: model-family label (v2 config)
    vocab_size: int = 16384
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 512
    n_experts: int = 8
    n_experts_per_token: int = 2
    use_shared_expert: bool = True
    moe_hidden_mult: int = 3
    dense_hidden_mult: int = 4
    moe_layers: tuple = (1, 3, 5)
    dropout: float = 0.1
    bias: bool = False
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    load_balance_coef: float = 0.01
    z_loss_coef: float = 1e-3
    capacity_factor: float = 1.25
    init_std: float = 0.02
    tie_weights: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 1e-5
    batch_size: int = 16
    max_iters: int = 5000
    out_dir: str = "out/moe_gpt"
    log_interval: int = 10
    save_interval: int = 500
    sliding_window: int = 128
    compress_rate_csa: int = 4
    compress_rate_hca: int = 128
    index_topk: int = 512
    layer_types: Optional[Tuple[str, ...]] = None
    hc_mult: int = 1
    use_expert_choice: bool = False
    context_stages: tuple = ()
    gradient_accumulation_steps: int = 1
    val_patience: int = 10
    early_stop: bool = True
    ema_beta: float = 0.8
    # IMU-1 architectural improvements
    use_qk_norm: bool = False
    use_value_residual: bool = False
    use_per_head_gate: bool = False
    use_layer_norm_scaling: bool = False
    softmax_cap: float = 0.0  # B-27 (Rybakov 2410.16682 eq.3): softmax[tanh(logit/c)*c];
    # 0 = off. QK-norm + cap gives 1.5x LR headroom vs QK-norm alone (830M study).
    # 6th-audit: when the causal mask is full (sliding_window >= block_size) and
    # softmax_cap==0, SDPA can use the FA/mem-efficient kernels (2-3x faster than
    # math backend) if we hand it is_causal=True with no explicit attn_mask and
    # dropout_p=0. Default ON; set False to force the explicit-mask+dropout path.
    attn_fast_causal: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0, max_seq_len=2048):
        # ROUND-186 (a): ROPE_THETA_SCALE — monotone theta scaling at build
        # (YaRN-lite; e.g. 2.0 -> theta 1e4->2e4 for long-context inference on
        # future 2048+ block checkpoints). Identity when absent/1.0 — trained
        # models keep byte-identical tables.
        _rt = os.environ.get("ROPE_THETA_SCALE", "")
        if _rt:
            try:
                _s = float(_rt)
                if 1.0 < _s <= 16.0:
                    theta = theta * _s
            except ValueError:
                pass
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._cached_cos = None
        self._cached_sin = None

    def forward(self, seq_len, device):
        # Bug 48: RoPE must be computed in fp32 regardless of model dtype
        # (bf16 position grid corrupts phase beyond ~256 tokens; the old cache
        # key pinned the bf16 cache for the whole run). Cast to the INPUT
        # dtype at apply time via apply_rotary's callers.
        if (self._cached_cos is not None and seq_len <= self._cached_cos.shape[0]
                and self._cached_cos.device == device):
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self._cached_cos = torch.cat((cos, cos), dim=-1)
        self._cached_sin = torch.cat((sin, sin), dim=-1)
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # cast fp32 tables to the activation dtype (autograd rejects mixed bf16*fp32)
    cos = cos[:x.shape[-2], :d].unsqueeze(0).unsqueeze(1).to(x.dtype)
    sin = sin[:x.shape[-2], :d].unsqueeze(0).unsqueeze(1).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class QKNorm(nn.Module):
    """LayerNorm on Q and K before attention (improves stability at scale).
    Input: [B, n_head, T, head_dim] — we permute to [B, T, n_head, head_dim]
    for per-token normalization across head+head_dim, then permute back.
    """
    def __init__(self, n_head, n_kv_head, head_dim):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, q, k):
        B, nh, T, hd = q.shape
        # Permute to [B, T, nh, hd], apply norm over hd, permute back
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        q = self.q_norm(q.reshape(-1, hd)).reshape(B, T, nh, hd)
        k = self.k_norm(k.reshape(-1, hd)).reshape(B, T, -1, hd)
        return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)


class ValueResidual(nn.Module):
    """Normalized value residual (IMU-1 eq. 4):
    V(l) = s · (α1·V_local + α2·V(1)) / √(α1² + α2²)
    V_local = current layer's value projection (pre-attention-mixing),
    V(1) = first layer's value projection.
    Init (s, α1, α2) = (1, 1, 0) → starts as standard attention.
    """
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))

    def forward(self, v_local, v_first):
        # v_local, v_first: [B, n_head, T, head_dim]
        denom = torch.sqrt(self.alpha1 ** 2 + self.alpha2 ** 2).clamp_min(1e-6)
        return self.s * (self.alpha1 * v_local + self.alpha2 * v_first) / denom


class PerHeadGate(nn.Module):
    """Input-conditional sigmoid gate per attention head (IMU-1 / Qiu et al. 2025).
    out_h = 2 * sigma(g_h) * Attention_h, where g = W_g x is computed from the
    hidden state (conditional, not static). Factor 2 keeps output magnitude
    matching standard attention when g_h = 0 (sigma(0) = 0.5).
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, n_head, bias=False)
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, attn_out, x):
        B, nh, T, hd = attn_out.shape
        g = self.gate_proj(x)              # [B, T, nh]
        g = g.transpose(1, 2).unsqueeze(-1)  # [B, nh, T, 1]
        return attn_out * (2.0 * torch.sigmoid(g))


class LayerNormScaler(nn.Module):
    """Depth-dependent LayerNorm scaling (IMU-1 eq. 5): LN_l(x) = (1/√l)·LayerNorm(x).
    Constant per-layer factor — rebalances deep-layer contributions ("curse of depth").
    """
    def __init__(self, layer_idx):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(layer_idx + 1)))

    def forward(self, x):
        return x * self.scale


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.sliding_window = config.sliding_window
        self.use_qk_norm = config.use_qk_norm
        self.use_value_residual = config.use_value_residual
        self.use_per_head_gate = config.use_per_head_gate
        self.attn_fast_causal = getattr(config, "attn_fast_causal", True)

        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        if self.use_qk_norm:
            self.qk_norm = QKNorm(self.n_head, self.n_kv_head, self.head_dim)
        if self.use_value_residual:
            self.value_res = ValueResidual()
        if self.use_per_head_gate:
            self.head_gate = PerHeadGate(config.n_embd, self.n_head)

        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.softmax_cap = getattr(config, "softmax_cap", 0.0)
        self.attn_dropout = nn.Dropout(config.dropout)
        # 18th-audit: lazily skip the causal mask buffer when the FA fast path is
        # guaranteed for every possible T (sliding_window >= block_size so
        # full_causal is always True, fast_causal on, softmax_cap off). The
        # buffer is block_size^2 fp32 per layer (~126MB for 30 layers on 430M)
        # and is NEVER touched by the fast path. Allocate only if a slow path
        # (T > block_size, or cap/sliding that force the manual mask) could run.
        self._needs_mask = not (self.attn_fast_causal
                                and self.softmax_cap <= 0.0
                                and self.sliding_window >= config.block_size)
        if self._needs_mask:
            self._register_sliding_mask(config.block_size)
        # else: do NOT create self.mask at all — the lazy fast path never needs
        # it, and forward() lazily registers on demand if a slow path is reached.
        self._block_size = config.block_size

    def _register_sliding_mask(self, max_len):
        # ROUND-5: construct on the model device (register_buffer does NOT
        # re-home an existing CPU tensor; lazy regrow on a CUDA model would
        # crash the slow path with a device mismatch).
        dev = self.wo.weight.device
        causal = torch.tril(torch.ones(max_len, max_len, device=dev))
        if self.sliding_window > 0:
            sliding = torch.triu(torch.ones(max_len, max_len, device=dev), diagonal=-self.sliding_window + 1)
            mask = causal * sliding
        else:
            mask = causal
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len), persistent=False)

    def forward(self, x, v_first=None, return_raw_attn=False):
        B, T, C = x.shape
        # 18th-audit: the mask is lazily allocated ONLY inside the slow branches
        # below (SDPA-with-mask and the manual math path) — NOT unconditionally at
        # the top of forward. Registering here on every first forward defeated the
        # memory saving (allocated ~126MB even for fast-path-only configs and
        # never read it). getattr: when skipped, self.mask does not exist at all
        # (setting it to None would clash with register_buffer).
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        cos, sin = self.rope(T, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        v_raw = v  # pre-repeat copy propagated to later layers as V(1)
        if self.use_value_residual:
            # IMU-1 faithful placement (eq. 4): mix the value BEFORE SDPA —
            # V(l) = s*(a1*V_local + a2*V(1))/sqrt(a1^2+a2^2) is the value the
            # attention actually reads. (Previously added post-attention.)
            v = self.value_res(v, v if v_first is None else v_first)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        use_fa = T <= 8192 and self.softmax_cap <= 0.0  # SDPA has no cap hook
        if use_fa:
            # 6th-audit FA fast-path: when the mask is the FULL causal mask
            # (sliding_window >= T — truncated to the real seq len) and
            # softmax_cap==0, pass is_causal=True with no explicit mask and
            # dropout_p=0.0 so SDPA engages the FlashAttention / mem-efficient
            # CUDA kernels (2-3x faster than the O(T^2) math backend). A non-0
            # dropout_p OR an explicit attn_mask force the slow math backend.
            #
            # NOTE (10th audit): this sets dropout_p=0.0 UNCONDITIONALLY in
            # training, while the slow path uses dropout=0.1 from config.  This
            # is a DELIBERATE regularization sacrifice for ~2x attention speed;
            # embed_dropout (line ~753) and head_gate remain active, so the
            # model still has dropout.  For pretraining on a fixed 585M-token
            # corpus the FA-speed gain (~20% throughput) is judged to outweigh
            # the dropout regularization; confirmed by the model not diverging
            # and val loss improving to a new best under this configuration.
            full_causal = (self.sliding_window <= 0) or (self.sliding_window >= T)
            if full_causal and self.attn_fast_causal:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
            else:
                # F1 (Sep-5): re-register when T grows past the current mask
                # (eager-registered at block_size; slow paths with a longer
                # context would otherwise slice out of range and crash).
                if getattr(self, "mask", None) is None or self.mask.shape[-1] < T:
                    self._register_sliding_mask(max(self._block_size, T))
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=self.mask[:, :, :T, :T].bool(),
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            if getattr(self, "mask", None) is None or self.mask.shape[-1] < T:
                self._register_sliding_mask(max(self._block_size, T))
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if self.softmax_cap > 0:
                # B-27 (Rybakov 2410.16682 eq. 3): cap BEFORE the -inf mask so
                # masked cells stay -inf (attend-nothing rows still zero out via
                # the NaN guard below in CSA/HCA; GQA's causal mask has none).
                att = torch.tanh(att / self.softmax_cap) * self.softmax_cap
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1, dtype=torch.float32).to(x.dtype)
            # B-186: NaN guard — if ALL positions are masked (sliding_window < T),
            # softmax(-inf, ..., -inf) = NaN. CSA/HCA have this guard; GQA did not.
            att = att.masked_fill(~torch.isfinite(att), 0.0)
            att = self.attn_dropout(att)
            y = att @ v

        if self.use_per_head_gate:
            y = self.head_gate(y, x)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.use_value_residual:
            return self.wo(y), v_raw
        return self.wo(y)


class KVPooler(nn.Module):
    def __init__(self, n_embd, n_kv_head, head_dim, rate):
        super().__init__()
        self.rate = rate
        self.out_dim = n_kv_head * head_dim
        self.comp_k = nn.Linear(n_embd, self.out_dim, bias=False)
        self.comp_v = nn.Linear(n_embd, self.out_dim, bias=False)
        self.gate = nn.Linear(n_embd, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.comp_k.weight)
        nn.init.xavier_uniform_(self.comp_v.weight)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x):
        B, T, C = x.shape
        rate = self.rate
        pad = (rate - T % rate) % rate
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        n_blocks = Tp // rate
        x_b = x.view(B, n_blocks, rate, C)
        gate_w = F.softmax(self.gate(x_b).squeeze(-1), dim=-1).unsqueeze(-1)
        k_comp = (gate_w * self.comp_k(x_b).view(B, n_blocks, rate, self.out_dim)).sum(dim=-2)
        v_comp = (gate_w * self.comp_v(x_b).view(B, n_blocks, rate, self.out_dim)).sum(dim=-2)
        return k_comp, v_comp


class CompressedSparseAttention(nn.Module):
    def __init__(self, config, compress_rate=4):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.rate = compress_rate
        self.sliding_window = config.sliding_window

        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.pooler = KVPooler(config.n_embd, self.n_kv_head, self.head_dim, compress_rate)
        self.softmax_cap = getattr(config, "softmax_cap", 0.0)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.attn_dropout = nn.Dropout(config.dropout)
        self._block_size = config.block_size
        self._register_masks(config.block_size)

    def _register_masks(self, max_len):
        # Block-causal: block b pools [b*rate, (b+1)*rate); query t may attend
        # block b iff its max position (b+1)*rate-1 <= t (fully past, or the
        # query is its last position). Plain tril leaks up to rate-1 futures.
        # ROUND-5 CRITICAL: construct ON THE MODEL'S DEVICE — register_buffer
        # does NOT re-home an existing (CPU) tensor, so a lazy/regrown mask was
        # CPU while att was CUDA -> RuntimeError on every forward.
        dev = self.wo.weight.device
        n_blocks = max_len // self.rate + 1
        t = torch.arange(max_len, device=dev).unsqueeze(1)
        b = torch.arange(n_blocks, device=dev).unsqueeze(0)
        mask = (b + 1) * self.rate - 1 <= t
        self.register_buffer("mask", mask.view(1, 1, max_len, n_blocks), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_comp, v_comp = self.pooler(x)

        cos, sin = self.rope(T, x.device)
        q = apply_rotary(q, cos, sin)
        q = q * (self.head_dim ** -0.5)

        Bk, n_blocks, Dk = k_comp.shape
        k_comp = k_comp.view(B, self.n_kv_head, n_blocks, self.head_dim)
        v_comp = v_comp.view(B, self.n_kv_head, n_blocks, self.head_dim)
        k_cat = k_comp.repeat_interleave(self.n_rep, dim=1)
        v_cat = v_comp.repeat_interleave(self.n_rep, dim=1)

        att = q @ k_cat.transpose(-2, -1)
        # F6 (Sep-5) + ROUND-5: regrow if the mask is too SHORT for T — the mask
        # is [1,1,max_len,n_blocks], so the query-row dim is shape[-2] (the
        # old shape[-1] compared n_blocks against T: true on EVERY forward ->
        # re-registered each step AND, pre-device-fix, crashed on GPU).
        if self.mask.shape[-2] < T:
            self._register_masks(max(self._block_size, T))
        if self.softmax_cap > 0:
            att = torch.tanh(att / self.softmax_cap) * self.softmax_cap
        att = att.masked_fill(self.mask[:, :, :T, :k_cat.shape[-2]] == 0, float("-inf"))
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(x.dtype)
        # Block-causal mask leaves the first rate-1 query positions with NO
        # valid block -> softmax(all -inf) = NaN. Those rows must attend
        # nothing (zero output), not NaN (regression from the round-6 mask fix).
        att = att.masked_fill(~torch.isfinite(att), 0.0)
        att = self.attn_dropout(att)
        y = (att @ v_cat).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class HeavilyCompressedAttention(nn.Module):
    def __init__(self, config, compress_rate=128):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.rate = compress_rate
        self.sliding_window = config.sliding_window

        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.pooler = KVPooler(config.n_embd, self.n_kv_head, self.head_dim, compress_rate)
        self.softmax_cap = getattr(config, "softmax_cap", 0.0)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.attn_dropout = nn.Dropout(config.dropout)
        self._block_size = config.block_size
        self._register_masks(config.block_size)

    def _register_masks(self, max_len):
        # Block-causal: block b pools [b*rate, (b+1)*rate); query t may attend
        # block b iff its max position (b+1)*rate-1 <= t (fully past, or the
        # query is its last position). Plain tril leaks up to rate-1 futures.
        # ROUND-5 CRITICAL: construct ON THE MODEL'S DEVICE — register_buffer
        # does NOT re-home an existing (CPU) tensor, so a lazy/regrown mask was
        # CPU while att was CUDA -> RuntimeError on every forward.
        dev = self.wo.weight.device
        n_blocks = max_len // self.rate + 1
        t = torch.arange(max_len, device=dev).unsqueeze(1)
        b = torch.arange(n_blocks, device=dev).unsqueeze(0)
        mask = (b + 1) * self.rate - 1 <= t
        self.register_buffer("mask", mask.view(1, 1, max_len, n_blocks), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_comp, v_comp = self.pooler(x)

        cos, sin = self.rope(T, x.device)
        q = apply_rotary(q, cos, sin)
        q = q * (self.head_dim ** -0.5)

        Bk, n_blocks, Dk = k_comp.shape
        k_comp = k_comp.view(B, self.n_kv_head, n_blocks, self.head_dim)
        v_comp = v_comp.view(B, self.n_kv_head, n_blocks, self.head_dim)
        k_cat = k_comp.repeat_interleave(self.n_rep, dim=1)
        v_cat = v_comp.repeat_interleave(self.n_rep, dim=1)

        att = q @ k_cat.transpose(-2, -1)
        # F6 (Sep-5) + ROUND-5: regrow if the mask is too SHORT for T — the mask
        # is [1,1,max_len,n_blocks], so the query-row dim is shape[-2] (the
        # old shape[-1] compared n_blocks against T: true on EVERY forward ->
        # re-registered each step AND, pre-device-fix, crashed on GPU).
        if self.mask.shape[-2] < T:
            self._register_masks(max(self._block_size, T))
        if self.softmax_cap > 0:
            att = torch.tanh(att / self.softmax_cap) * self.softmax_cap
        att = att.masked_fill(self.mask[:, :, :T, :k_cat.shape[-2]] == 0, float("-inf"))
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(x.dtype)
        # Block-causal mask leaves the first rate-1 query positions with NO
        # valid block -> softmax(all -inf) = NaN. Those rows must attend
        # nothing (zero output), not NaN (regression from the round-6 mask fix).
        att = att.masked_fill(~torch.isfinite(att), 0.0)
        att = self.attn_dropout(att)
        y = (att @ v_cat).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class ManifoldHyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection (mHC) from DeepSeek-V4.
    Replaces residual with 4 parallel streams mixed via doubly-stochastic
    Sinkhorn-projected matrix. Spectral norm ≤ 1 across all layers.
    """
    def __init__(self, dim, hc_mult=4, sinkhorn_iters=10):
        super().__init__()
        self.hc_mult = hc_mult
        self.pre_w = nn.Parameter(torch.randn(hc_mult, dim))
        self.post_w = nn.Parameter(torch.randn(hc_mult, dim))
        self.comb = nn.Parameter(torch.randn(hc_mult, hc_mult))
        self.sinkhorn_iters = sinkhorn_iters

    def _sinkhorn(self, x):
        for _ in range(self.sinkhorn_iters):
            x = x - x.logsumexp(dim=-2, keepdim=True)
            x = x - x.logsumexp(dim=-1, keepdim=True)
        return torch.exp(x)

    def forward(self, x, h_states):
        B, T, D = x.shape
        hc = self.hc_mult
        pre = torch.sigmoid(self.pre_w).view(1, 1, hc, D)
        post = torch.sigmoid(self.post_w).view(1, 1, hc, D)
        comb = self._sinkhorn(self.comb)

        mixed = (pre * h_states).sum(dim=-1)
        mixed = torch.einsum("bth,hn->btn", mixed, comb)
        combined = mixed.unsqueeze(-1).expand(-1, -1, -1, D)
        combined = combined + post * h_states
        return combined


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult, bias=False):
        super().__init__()
        hidden = int(dim * hidden_mult * 2 / 3)
        self.w = nn.Linear(dim, hidden, bias=bias)
        self.v = nn.Linear(dim, hidden, bias=bias)
        self.o = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.o(F.silu(self.w(x)) * self.v(x))


class ExpertFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = SwiGLU(config.n_embd, config.moe_hidden_mult, config.bias)

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.capacity_factor = config.capacity_factor
        self.load_balance_coef = config.load_balance_coef
        self.z_loss_coef = config.z_loss_coef

        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=config.init_std)

        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.n_experts)])
        self.shared_expert = ExpertFFN(config) if config.use_shared_expert else None

    def _load_balance_loss(self, router_logits, expert_indices):
        tokens_per_expert = router_logits.shape[0] // max(1, router_logits.shape[1])
        route_prob = F.softmax(router_logits, dim=-1)
        expert_usage = torch.zeros(self.n_experts, device=router_logits.device)
        for i in range(self.n_experts):
            expert_usage[i] = (expert_indices == i).sum().float()
        expert_usage = expert_usage / (expert_usage.sum() + 1e-8)
        route_frac = route_prob.mean(dim=0)
        balance_loss = self.n_experts * (expert_usage * route_frac).sum()
        return balance_loss

    def _router_z_loss(self, router_logits):
        return torch.logsumexp(router_logits, dim=-1).square().mean()

    def forward(self, x):
        B, T, D = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        capacity = min(N, int(N / self.n_experts * self.capacity_factor * self.top_k))

        outputs = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            # Bug 46 fix: weight each token by the router probability of the
            # SELECTED expert slot (topk_probs is renormalized over the k slots
            # so sum over experts == 1). The old code summed the whole top-k
            # row -> weight ~1.0 for every token -> router ignored, output was
            # an unweighted 2x sum of the top-2 experts.
            match = (topk_indices == i)
            idx = match.nonzero()  # (num_matches, 2): [token_row, slot]
            if idx.numel() == 0:
                continue
            rows = idx[:capacity, 0]
            slots = idx[:capacity, 1]
            expert_out = self.experts[i](x_flat[rows])
            weights = topk_probs[rows, slots].unsqueeze(-1)
            outputs[rows] += expert_out * weights

        if self.shared_expert is not None:
            outputs = outputs + self.shared_expert(x_flat) * 0.1

        balance_loss = self._load_balance_loss(router_logits, topk_indices)
        z_loss = self._router_z_loss(router_logits)

        return outputs.view(orig_shape), balance_loss, z_loss


class ExpertChoiceMoE(nn.Module):
    """Expert Choice routing: each expert selects top-k tokens.
    Guarantees perfect load balance without auxiliary loss.
    Reference: Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (NeurIPS 2022).
    2x faster convergence than token-choice routing.
    """
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.capacity_factor = config.capacity_factor
        self.z_loss_coef = config.z_loss_coef

        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=config.init_std)

        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.n_experts)])
        self.shared_expert = ExpertFFN(config) if config.use_shared_expert else None

    def _router_z_loss(self, router_logits):
        return torch.logsumexp(router_logits, dim=-1).square().mean()

    def forward(self, x):
        B, T, D = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        capacity = max(1, int(N / self.n_experts * self.capacity_factor))

        scores_T = router_probs.transpose(0, 1)
        expert_weights, expert_tokens = torch.topk(scores_T, capacity, dim=-1)

        outputs = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            selected = expert_tokens[i]
            w = expert_weights[i].unsqueeze(-1)
            expert_out = self.experts[i](x_flat[selected])
            outputs.index_add_(0, selected, expert_out * w)

        if self.shared_expert is not None:
            outputs = outputs + self.shared_expert(x_flat) * 0.1

        z_loss = self._router_z_loss(router_logits)
        return outputs.view(orig_shape), torch.tensor(0.0, device=x.device), z_loss


def _resolve_layer_types(config):
    if config.layer_types is not None:
        return config.layer_types
    types = []
    for i in range(config.n_layer):
        if i < 2:
            types.append("sliding")
        elif i % 2 == 0:
            types.append("csa")
        else:
            types.append("hca")
    return tuple(types)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx, use_moe=False, attn_type="sliding"):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = use_moe
        self.use_mhc = config.hc_mult > 1
        self.use_ln_scaling = config.use_layer_norm_scaling
        self.use_value_residual = config.use_value_residual and attn_type == "sliding"
        self.attn_norm = RMSNorm(config.n_embd, config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, config.layer_norm_eps)
        if self.use_ln_scaling:
            self.attn_scale = LayerNormScaler(layer_idx)
            self.ffn_scale = LayerNormScaler(layer_idx)
        if self.use_mhc:
            self.mhc_pre = ManifoldHyperConnection(config.n_embd, config.hc_mult)
            self.mhc_post = ManifoldHyperConnection(config.n_embd, config.hc_mult)

        if attn_type == "csa":
            self.attn = CompressedSparseAttention(config, compress_rate=config.compress_rate_csa)
        elif attn_type == "hca":
            self.attn = HeavilyCompressedAttention(config, compress_rate=config.compress_rate_hca)
        else:
            self.attn = GroupedQueryAttention(config)

        if use_moe:
            if config.use_expert_choice:
                self.ffn = ExpertChoiceMoE(config)
            else:
                self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLU(config.n_embd, config.dense_hidden_mult, config.bias)

    def _attn(self, normed, v_first):
        if self.use_value_residual:
            out, v_local = self.attn(normed, v_first)
            if v_first is None:
                self.v_local = v_local  # layer 0: capture first-layer values
            return out
        return self.attn(normed)

    def forward(self, x, h_states=None, v_first=None):
        if self.use_mhc:
            if h_states is None:
                h_states = x.unsqueeze(2).expand(-1, -1, self.mhc_pre.hc_mult, -1).clone()
            attn_normed = self.attn_norm(x)
            if self.use_ln_scaling:
                attn_normed = self.attn_scale(attn_normed)
            attn_out = self._attn(attn_normed, v_first)
            h_states = self.mhc_pre(x, h_states)
            x = h_states[:, :, 0, :]
            if self.use_moe:
                ffn_normed = self.ffn_norm(x)
                if self.use_ln_scaling:
                    ffn_normed = self.ffn_scale(ffn_normed)
                ffn_out, bal_loss, z_loss = self.ffn(ffn_normed)
                added = ffn_out
            else:
                ffn_normed = self.ffn_norm(x)
                if self.use_ln_scaling:
                    ffn_normed = self.ffn_scale(ffn_normed)
                added = self.ffn(ffn_normed)
                bal_loss, z_loss = 0.0, 0.0
            h_states = self.mhc_post(added, h_states)
            x = h_states[:, :, 0, :]
            if self.use_moe:
                return x, bal_loss, z_loss
            return x
        else:
            attn_normed = self.attn_norm(x)
            if self.use_ln_scaling:
                attn_normed = self.attn_scale(attn_normed)
            x = x + self._attn(attn_normed, v_first)
            if self.use_moe:
                ffn_normed = self.ffn_norm(x)
                if self.use_ln_scaling:
                    ffn_normed = self.ffn_scale(ffn_normed)
                moe_out, bal_loss, z_loss = self.ffn(ffn_normed)
                x = x + moe_out
                return x, bal_loss, z_loss
            else:
                ffn_normed = self.ffn_norm(x)
                if self.use_ln_scaling:
                    ffn_normed = self.ffn_scale(ffn_normed)
                x = x + self.ffn(ffn_normed)
                return x


class MoEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        layer_types = _resolve_layer_types(config)
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            use_moe = i in config.moe_layers
            attn_type = layer_types[i] if i < len(layer_types) else "sliding"
            self.layers.append(TransformerBlock(config, i, use_moe, attn_type))

        self.norm = RMSNorm(config.n_embd, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None, use_grad_ckpt=False, pad_mask=None):
        B, T = idx.shape
        x = self.token_embed(idx)
        x = self.drop(x)

        total_balance_loss = 0.0
        total_z_loss = 0.0

        v_first = None
        for i, layer in enumerate(self.layers):
            if i == 0 and layer.use_value_residual:
                # Layer 0 runs uncheckpointed so v_first (needed by all value
                # residuals) stays in the main autograd graph — checkpointing
                # it would sever gradients to layer 0's value projection.
                if layer.use_moe:
                    x, bl, zl = layer(x, None, None)
                    total_balance_loss += bl
                    total_z_loss += zl
                else:
                    x = layer(x, None, None)
                v_first = layer.v_local
                continue
            if layer.use_moe:
                if self.training and use_grad_ckpt:
                    x, bl, zl = torch.utils.checkpoint.checkpoint(
                        layer, x, None, v_first, use_reentrant=False
                    )
                else:
                    x, bl, zl = layer(x, None, v_first)
                total_balance_loss += bl
                total_z_loss += zl
            else:
                if self.training and use_grad_ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, None, v_first, use_reentrant=False
                    )
                else:
                    x = layer(x, None, v_first)
            if v_first is None and layer.use_value_residual and hasattr(layer, "v_local"):
                v_first = layer.v_local

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if pad_mask is not None:
                # Ignore padded positions (mask False where input/target is <PAD>)
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    reduction="none")
                pm = pad_mask.reshape(-1).to(ce_loss.dtype)
                ce_loss = (ce_loss * pm).sum() / pm.sum().clamp_min(1.0)
            else:
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + self.config.load_balance_coef * total_balance_loss + self.config.z_loss_coef * total_z_loss

        # 8th-wave P0: the aux returns the RAW TENSORS (the old .item() floats
        # were graph-detached — the router balance/z regularization could never
        # train through the trainer-side weighted loss; and a dict-sum of keys
        # crashed the 204r runs at step 0).
        return logits, loss, {
            "balance_loss": total_balance_loss if total_balance_loss != 0 else 0.0,
            "z_loss": total_z_loss if total_z_loss != 0 else 0.0,
            "ce_loss": ce_loss if targets is not None else 0.0,
        }

    @torch.no_grad()  # B-159: sampling never needs a graph -- several eval
    # callers didn't wrap this and silently built autograd graphs.
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=50,
                 top_p=0.0, repetition_penalty=1.0, no_repeat_ngram=0,
                 guided_hook=None):
        """guided_hook (205u): optional GuidedSampler with .mask(row,
        logits, toks) and .feed(row, token_id) — applied per active row
        right before sampling when the caller passes it (served tool
        calls); None = fully unchanged behavior."""
        """Greedy/truncation sampling with anti-degeneration options (B-42,
        Holtzman 1904.09751 / Keskar CTRL / HF no_repeat_ngram):
        top_k: keep top-k logits (default 50; 0 disables)
        top_p: nucleus — sample from the smallest set with cum-prob >= top_p
        repetition_penalty: >1 divides scores of already-generated tokens
        no_repeat_ngram: block any n-gram that already appeared (0 = off).
        Both bans have an all-masked fallback: if every logit got -inf the
        softmax would be NaN (CUDA assert) — restore the unmasked logits."""
        B = idx.shape[0]
        # B-217: batch-safe decode. The old loop stopped the WHOLE batch on
        # row 0's EOS and read idx[0] for ngram/repetition — so stochastic
        # sampling was force batch=1 (decode happens one [1,T] forward at a
        # time through rest_round/grpo/honest-eval: the latemcy-bound 4050 left
        # the GPU half-idle during every rollout/eval). Now every active row is
        # extended per step (greedy argmax + top_k/top_p + multinomial are
        # already vectorized over the batch); EOS/no_repeat_ngram/
        # repetition_penalty are applied per row. A finished row is frozen
        # (not extended). For B=1 the RNG consumption is identical to the old
        # loop => identical samples (determinism preserved). Callers decoding a
        # batch row must cut at the first EOS (id 2) — the row is frozen there.
        buf = torch.zeros(B, idx.shape[1] + max_new_tokens, dtype=idx.dtype,
                          device=idx.device)
        buf[:, :idx.shape[1]] = idx
        active = torch.ones(B, dtype=torch.bool, device=idx.device)
        cur = idx.shape[1]
        for _ in range(max_new_tokens):
            if not active.any():
                break
            # B-217-d (audit): the old `buf[active, cur - block:cur]` sliced the
            # negative start against the FULL buf width (W = prompt+max_new), so
            # for 2*prompt_len + max_new > block the prompt HEAD was silently
            # dropped from the context. Clamp to the current length instead.
            lo = max(0, cur - self.config.block_size)
            x = buf[active, lo:cur]
            logits, _, _ = self(x)
            if temperature <= 1e-6:
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                buf[active, cur] = nxt.squeeze(-1)
                # C1 (18th-audit): nxt is [n_active] but active is [B] — the old
                # `active &= (nxt != 2)` broadcast-crashed the moment any row hit
                # EOS in a multi-row batch. Build a full-[B] mask instead.
                _full = torch.zeros(B, dtype=torch.bool, device=active.device)
                _full[active] = nxt.squeeze(-1) != 2
                active &= _full
                cur += 1
                continue
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            logits_orig = logits.clone()
            act_idx = active.nonzero(as_tuple=False).squeeze(-1)

            if no_repeat_ngram > 1:
                for r_i, r in enumerate(act_idx.tolist()):
                    seq = buf[r, :cur].tolist()
                    if len(seq) >= no_repeat_ngram:
                        ngram = tuple(seq[-(no_repeat_ngram - 1):])
                        banned = set()
                        for i in range(len(seq) - no_repeat_ngram + 1):
                            if tuple(seq[i:i + no_repeat_ngram - 1]) == ngram:
                                banned.add(seq[i + no_repeat_ngram - 1])
                        banned = [t for t in banned if t < logits.shape[-1]]
                        if banned:
                            logits[r_i, banned] = float("-inf")
                            if not torch.isfinite(logits[r_i]).any():
                                logits[r_i] = logits_orig[r_i].clone()

            if repetition_penalty > 1.0:
                for r_i, r in enumerate(act_idx.tolist()):
                    seen = set(buf[r, :cur].tolist())
                    seen = [t for t in seen if t < logits.shape[-1]]
                    if seen:
                        logits[r_i, seen] = torch.where(
                            logits[r_i, seen] > 0,
                            logits[r_i, seen] / repetition_penalty,
                            logits[r_i, seen] * repetition_penalty)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = cum > top_p
                cutoff[..., 1:] = cutoff[..., 1:] | cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float("-inf")
                masked = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
                finite_rows = torch.isfinite(masked).any(dim=-1, keepdim=True)
                logits = torch.where(finite_rows, masked, logits_orig)

            if guided_hook is not None:
                for r_i, r in enumerate(act_idx.tolist()):
                    try:
                        logits[r_i] = guided_hook.mask(r, logits[r_i],
                                                        buf[r, :cur].tolist())
                    except Exception:
                        pass  # the guided path must never break sampling
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if guided_hook is not None:
                for r_i, r in enumerate(act_idx.tolist()):
                    try:
                        guided_hook.feed(r, int(nxt[r_i]))
                    except Exception:
                        pass
            buf[active, cur] = nxt
            # C1: nxt is [n_active] — update active only at the active rows
            # (act_idx) instead of the broadcast-crash `active &= (nxt != 2)`.
            active[act_idx] &= (nxt != 2)
            cur += 1
        return buf[:, :cur]


def create_ambatron_mini_config(**kwargs):
    """~50M model matched to the CURRENT data budget (~80M curated tokens).
    B-38 (Chinchilla sizing): 444M needs 8.9B tokens (we have 0.08B — 0.9%,
    so it memorizes: train 0.15 / val 2.3+ in every run). A ~50M model at
    1.6 tok/param is 9x better matched and generalizes instead of memorizing.
    Same arch family (QK-norm, value residual, gates, LN scaling, cap) so all
    the training/agent code and the tool-SFT/VLA paths work unchanged."""
    config = MoEConfig(
        vocab_size=16384,
        block_size=1024,
        n_layer=12,
        n_head=8,
        n_kv_head=4,
        n_embd=512,
        n_experts=1,
        n_experts_per_token=1,
        use_shared_expert=False,
        moe_hidden_mult=4,
        dense_hidden_mult=4,
        moe_layers=(),
        dropout=0.1,
        bias=False,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        load_balance_coef=0.0,
        z_loss_coef=0.0,
        init_std=0.02,
        tie_weights=True,
        learning_rate=0.02,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=2000,
        lr_decay_iters=50000,
        min_lr=0.000235,
        batch_size=8,
        max_iters=50000,
        out_dir="out/ambatron_mini",
        log_interval=10,
        save_interval=2000,
        sliding_window=1024,
        layer_types=("sliding",) * 12,
        use_qk_norm=True,
        use_value_residual=True,
        use_per_head_gate=True,
        use_layer_norm_scaling=True,
        softmax_cap=0.0,
    )
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


def create_ambatron_plus_config(**kwargs):
    """~121M model (B-44): the step between mini (43M) and 430M (444M).
    121M at the ~200M-token curated corpus = ~1.6 tok/param — same ratio the
    mini had at 79M, but a bigger model is meaningfully more capable per token
    (phi-1-small class). Same arch family (QK-norm, value-residual, gates, LN
    scaling, softmax cap) so every training/agent/VLA path works unchanged."""
    config = MoEConfig(
        vocab_size=16384,
        block_size=1024,
        n_layer=12,
        n_head=8,
        n_kv_head=4,
        n_embd=640,
        n_experts=1,
        n_experts_per_token=1,
        use_shared_expert=False,
        moe_hidden_mult=4,
        dense_hidden_mult=4,
        moe_layers=(),
        dropout=0.1,
        bias=False,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        load_balance_coef=0.0,
        z_loss_coef=0.0,
        init_std=0.02,
        tie_weights=True,
        learning_rate=0.02,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=2000,
        lr_decay_iters=50000,
        min_lr=0.000235,
        batch_size=8,
        max_iters=50000,
        out_dir="out/ambatron_plus",
        log_interval=10,
        save_interval=2000,
        sliding_window=1024,
        layer_types=("sliding",) * 12,
        use_qk_norm=True,
        use_value_residual=True,
        use_per_head_gate=True,
        use_layer_norm_scaling=True,
        softmax_cap=0.0,
    )
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


def create_ambatron_xl_config(**kwargs):
    """~225M model (B-56, 13th-audit: actual 224.6M, the old docstring said 370M): the
    FULL-GPU training target. At batch 8-12 x ctx 1024 it fills the 5.6 GB card (vs the
    plus's 48% util). On the 500M curated corpus: ~2.6 tok/param (well-matched, near
    Chinchilla for ~1B tokens). Same arch family. Running on karel's 4060-8GB at batch 8
    (~7.6k tok/s).
    """
    config = MoEConfig(
        vocab_size=16384,
        block_size=1024,
        n_layer=18,
        n_head=8,
        n_kv_head=4,
        n_embd=1024,
        n_experts=1,
        n_experts_per_token=1,
        use_shared_expert=False,
        moe_hidden_mult=4,
        dense_hidden_mult=4,
        moe_layers=(),
        dropout=0.1,
        bias=False,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        load_balance_coef=0.0,
        z_loss_coef=0.0,
        init_std=0.02,
        tie_weights=True,
        learning_rate=0.02,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=2000,
        lr_decay_iters=50000,
        min_lr=0.000235,
        batch_size=8,
        max_iters=50000,
        out_dir="out/ambatron_xl",
        log_interval=10,
        save_interval=2000,
        sliding_window=1024,
        layer_types=("sliding",) * 18,
        use_qk_norm=True,
        use_value_residual=True,
        use_per_head_gate=True,
        use_layer_norm_scaling=True,
        softmax_cap=0.0,
    )
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


def create_ambatron_430m_config(**kwargs):
    """430M-parameter dense transformer config based on IMU-1 recipe.
    Architecture: d_model=1152, 30 layers, 18 heads, GQA(6), SwiGLU, QK-Norm,
    value residuals, per-head gating, LayerNorm scaling.
    Full FP16 training ~2.5 GB VRAM. LoRA: ~1.0 GB.

    LR schedule uses IMU-1 absolute values:
      Muon base LR = 0.0235 for 2D weight matrices
      AdamW base LR = 0.007 for 1D params (embeddings, norms, head)
    """
    config = MoEConfig(
        vocab_size=16384,
        block_size=1024,
        n_layer=30,
        n_head=18,
        n_kv_head=6,
        n_embd=1152,
        n_experts=1,
        n_experts_per_token=1,
        use_shared_expert=False,
        moe_hidden_mult=4,
        dense_hidden_mult=4,
        moe_layers=(),
        dropout=0.1,
        bias=False,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        load_balance_coef=0.0,
        z_loss_coef=0.0,
        init_std=0.02,
        tie_weights=True,
        learning_rate=0.02,  # canonical Muon base LR (KellerJordan/Muon)
        weight_decay=0.1,    # applied to AdamW groups (embeddings/head/norms); Muon groups use 0
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=2000,
        lr_decay_iters=50000,
        min_lr=0.000235,  # 1% of Muon base (IMU-1 style)
        batch_size=8,
        max_iters=50000,
        out_dir="out/ambatron_430m",
        log_interval=10,
        save_interval=2000,
        sliding_window=1024,  # == block_size → full causal attention
        layer_types=("sliding",) * 30,  # ALL layers use full GQA
        use_qk_norm=True,
        use_value_residual=True,
        use_per_head_gate=True,
        use_layer_norm_scaling=True,
        # B-27 (Rybakov 2410.16682 QK_norm_cap): ~identity for normal logits,
        # an emergency brake when attention logits spike — the divergence mode
        # our 10K-iter run showed. NOTE (b8.2 audit): cap is OFF (0.0) for the
        # v2-era family (the 10K run predates and the family trains stable);
        # the v3 config enables cap=50 per the paper. Comment kept for history.
        softmax_cap=0.0,
    )
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config

def create_ambatron_v2_config(**kwargs):
    """ROUND-186/187: GENERATION-2 config — the ROUND-186 ladder, materialized.
    Same 430M-lineage shape (d=1152, 30 layers, GQA-6, IMU-1) but LONGER
    CONTEXT (block 2048, sliding 1024 — KV budget unchanged vs gen-1, activations
    ~2x at equal batch) and MOE ENABLED (8 experts, 2/8 per-token → ~0.25 active
    params on expert columns). Rope theta scaled 10000->20000 (ROPE_THETA_SCALE
    also available at load); the flywheel/candidates inherit by construction.
    Identity when NOT selected (nothing reads this unless --size v2)."""
    return MoEConfig(
        size="v2",
        vocab_size=16384,
        block_size=2048,
        sliding_window=1024,
        n_layer=30,
        n_head=18,
        n_kv_head=6,
        n_embd=1152,
        n_experts=4,
        n_experts_per_token=1,
        use_shared_expert=True,
        moe_hidden_mult=2.5,
        dense_hidden_mult=4,
        moe_layers=tuple(range(10, 24)),
        dropout=0.1,
        bias=False,
        rope_theta=20000.0,
        layer_norm_eps=1e-5,
        load_balance_coef=0.01,
        z_loss_coef=0.0001,
        init_std=0.02,
        tie_weights=True,
        learning_rate=0.02,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=4000,
        lr_decay_iters=100000,
        min_lr=0.000235,
        batch_size=4,
        max_iters=100000,
        out_dir="out/ambatron_v2",
        log_interval=10,
        **kwargs,
    )


def create_ambatron_v3_config(**kwargs):
    """ROUND-204ac: GENERATION-3 config — the v3-on-A research contract,
    rescaled from the B/C mesh contract (v3-bank/v3_mesh_config.json) to THIS
    laptop's ~6GB envelope (the user's explicit: no dependence on other
    laptops):
      - vocab 32768 (NOT the contract's 76160): the scaling-law bracket for
        a <=600M model (arXiv 2407.13623; 76K = the 1B+ bracket; on a 587M
        model it would burn ~16% of params in embeddings).
      - 4E/2A MoE (NOT 16E/2A): 16E x 6.64M/expert x 18L = 2.77B (the mesh
        config) vs 4E = 477M experts; measured arithmetic = 587M total,
        <=1.2GB bf16 weights, fp32-Muon momentum 2.2GB -> ~5.1-5.4GB fits
        the 6GB envelope at batch2x1024 and batch1x2048 (the ladder leg).
      - block 2048 native (the context-ladder objective 1c), sliding 1024,
        rope_theta 40000 (the v3 contract), full IMU-1 family ON + softmax
        cap 50 (Rybakov), LongRoPE-staged region_expand/recovery curriculum
        via the v3 pool + trainer flags.
    Identity when NOT selected (nothing reads this unless --size v3)."""
    return MoEConfig(
        size="v3",
        vocab_size=32768,
        block_size=2048,
        sliding_window=1024,
        n_layer=18,
        n_head=12,
        n_kv_head=6,
        n_embd=1152,
        n_experts=4,
        n_experts_per_token=2,
        use_shared_expert=True,
        moe_hidden_mult=2.0,
        dense_hidden_mult=4,
        moe_layers=tuple(range(18)),
        dropout=0.1,
        bias=False,
        rope_theta=40000.0,
        layer_norm_eps=1e-5,
        load_balance_coef=0.01,
        z_loss_coef=0.0001,
        use_qk_norm=True,
        use_value_residual=True,
        use_per_head_gate=True,
        use_layer_norm_scaling=True,
        softmax_cap=50.0,
        init_std=0.02,
        tie_weights=True,
        learning_rate=0.02,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        warmup_iters=4000,
        lr_decay_iters=100000,
        min_lr=0.000235,
        batch_size=4,
        max_iters=100000,
        out_dir="out/ambatron_v3",
        log_interval=10,
        **kwargs,
    )
