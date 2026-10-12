"""
Muon optimizer from DeepSeek-V4 (2026) + IMU-1 NorMuon extension.
Newton-Schulz orthogonalization for SQUARE 2D matrices.
Momentum + lr scaling for NON-SQUARE 2D matrices.
AdamW for 1D params (biases, norms, embeddings, lm_head).
Faster convergence than AdamW alone (~52% FLOPs to target loss).
"""
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer


def _moment_dtype():
    """B-218: bf16 optimizer moments for the VRAM-constrained base continuation.
    fp32 moments for all 444M params (~2.4G) + bf16 weights/grads + torch-2.13
    allocator overhead OOM'd the 5.88GiB usable limit at batch4/blk1024 (torch
    was 2.5.1 on the Aug-11 run that fit); bf16 moments halve the optimizer
    (~-1.8G), standard for continued pretraining on a 6GB card. Off by default
    (fp32 is the Bug-16 precision reason); opt in via RSI_MUON_BF16=1."""
    import os
    return torch.bfloat16 if os.environ.get("RSI_MUON_BF16", "0") == "1" \
        else torch.float32


@torch.no_grad()
def _newton_schulz_batch(matrices, iterations=5):
    """Newton-Schulz orthogonalization (NorMuon/IMU-1 canonical form).

    Two fixes over the original code:
    1. Matrix products (was elementwise `*`) — the quintic only converges with
       proper matmuls.
    2. Frobenius-norm normalization FIRST (spectral norm <= 1 puts singular
       values in the converging band [0, 1]); without it real momentum
       matrices (spectral norms 7-17) diverge to NaN within 5 iterations.
    """
    if not matrices:
        return
    # 19th-audit B8: gpu_accel sets allow_tf32 globally for cap>=7 — the
    # Newton-Schulz quintic (fp32, runs on the bf16-moment path) would silently
    # run under TF32 matmuls, weakening the optimizer precision contract. The
    # chain is short + self-correcting, but keep the canonical path exact.
    _old_tf32 = torch.backends.cuda.matmul.allow_tf32
    _old_mm = torch.backends.cuda.matmul.fp32_precision if hasattr(torch.backends.cuda.matmul, "fp32_precision") else None
    torch.backends.cuda.matmul.allow_tf32 = False
    for M in matrices:
        X = M / (M.norm() + 1e-7)
        # F3 (Sep-5): NaN/Inf guard like canonical KellerJordan — a poisoned
        # momentum (inf or nan from a blown gradient) would otherwise propagate
        # through the quintic INTO the weights in-place (M.data.copy_(X)) and
        # corrupt the model. Zero the matrix instead: next step rebuilds
        # momentum from the (clean) grad.
        if not torch.isfinite(X).all():
            M.data.zero_()
            continue
        for _ in range(iterations):
            A = X @ X.transpose(-2, -1)
            X = 3.4445 * X - 4.7750 * (A @ X) + 2.0315 * ((A @ A) @ X)
        M.data.copy_(X)
    torch.backends.cuda.matmul.allow_tf32 = _old_tf32


class Muon(Optimizer):
    def __init__(self, params, lr=3e-4, weight_decay=0.1, muon_decay=0.1,
                 momentum=0.95, nesterov=True, ns_iterations=5,
                 adamw_betas=(0.9, 0.95), muon_lr=0.0235, adamw_lr=0.007):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_iterations=ns_iterations,
                        muon_lr=muon_lr, adamw_lr=adamw_lr)
        super().__init__(params, defaults)
        self.muon_decay = muon_decay
        self.adamw_betas = adamw_betas

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not globals().get("_ambatron_init_logged", False):
            globals()["_ambatron_init_logged"] = True
            # 19th-audit B7: RSI_CPU_MOMENTS is DEAD (nothing consumes it —
            # the flag was removed from the guard launch env in ROUND-204g;
            # keep the print honest by dropping the misleading signal).
            print(f"[Muon] env: BF16={bool(os.environ.get('RSI_MUON_BF16'))} "
                  f"moment_dtype={_moment_dtype()}", flush=True)
        for group in self.param_groups:
            mode = group.get("mode", "adamw")
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if os.environ.get("RSI_DUMP_MEM_ON_OOM") and not globals().get("_oom_dumped", False):
                    globals()["_oom_dumped"] = True
                    try:
                        _free, _tot = torch.cuda.mem_get_info()
                        with open("/tmp/oom_dump.txt", "w") as _f:
                            _f.write(f"free={_free/2**30:.2f}G total={_tot/2**30:.2f}G "
                                     f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G "
                                     f"allocated={torch.cuda.memory_allocated()/2**30:.2f}G "
                                     f"param={p.numel()}\"n")
                    except Exception as _e:
                        with open("/tmp/oom_dump.txt", "w") as _f:
                            _f.write(f"dump-exc: {_e}\n")
                if mode == "muon_square":
                    # Newton-Schulz orthogonalized momentum for square weight matrices
                    state = self.state[p]
                    if "momentum" not in state:
                        # Bug-16: fp32 accumulator (bf16 momentum drifts over
                        # 50K+ steps — mixed-precision guidance is unanimous
                        # that optimizer accumulators stay fp32)
                        state["momentum"] = torch.zeros_like(g, dtype=_moment_dtype())  # ROUND-183d (B4): square group honors RSI_MUON_BF16
                        state["step"] = 0
                    mom = state["momentum"]
                    state["step"] += 1
                    lr = group["muon_lr"]

                    if wd != 0:
                        g = g + wd * p

                    g32 = g.float()
                    mom.mul_(group["momentum"]).add_(g32)
                    # ROUND-169 (F4) + ROUND-170: guard the ACCUMULATOR AND the
                    # current-step grad. If g32 itself is NaN (a blown grad),
                    # `update = g32 + β·mom` is NaN even after zeroing mom —
                    # writing it poisons the WEIGHT in-place (measured: rect
                    # 32/32 NaN, adamw 8/8). Zero mom, then SKIP this param's
                    # update entirely: momentum rebuilds from the next clean grad.
                    if not torch.isfinite(mom).all():
                        mom.zero_()
                    if not torch.isfinite(g32).all():
                        continue
                    if group["nesterov"]:
                        update = g32 + group["momentum"] * mom
                    else:
                        update = mom

                    _newton_schulz_batch([update], group["ns_iterations"])
                    p.add_((update * -lr).to(p.dtype))  # descend; cast for bf16 in-place

                elif mode == "muon_rect":
                    # Momentum for non-square 2D matrices (no orthogonalization)
                    # NOTE (B-117): the canonical KellerJordan/Muon applies NS
                    # orthogonalization to ALL hidden 2D weights including
                    # rectangular ones; this momentum-only split is an IMU-1-
                    # style documented choice. The deployed base was TRAINED
                    # under it — do not change for LoRA/SFT runs (they use
                    # AdamW anyway); reconsider before any new pre-training.
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(g, dtype=_moment_dtype())
                        state["step"] = 0
                    mom = state["momentum"]
                    state["step"] += 1
                    lr = group["muon_lr"]

                    if wd != 0:
                        g = g + wd * p

                    g32 = g.float()
                    mom.mul_(group["momentum"]).add_(g32)
                    # ROUND-169 (F4) + ROUND-170: same two guards as square —
                    # zero NaN momentum AND skip the update on a NaN grad (a
                    # NaN g32 still poisoned the weights via `update = g32+...`).
                    if not torch.isfinite(mom).all():
                        mom.zero_()
                    if not torch.isfinite(g32).all():
                        continue
                    if group["nesterov"]:
                        update = g32 + group["momentum"] * mom
                    else:
                        update = mom

                    p.add_((update * -lr).to(p.dtype))  # descend; cast for bf16 in-place

                else:
                    # Decoupled AdamW for 1D params (biases, norms, embeddings, lm_head)
                    state = self.state[p]
                    if "exp_avg" not in state:
                        # fp32 accumulators (Bug-16: bf16 Adam moments drift)
                        state["exp_avg"] = torch.zeros_like(g, dtype=_moment_dtype())
                        state["exp_avg_sq"] = torch.zeros_like(g, dtype=_moment_dtype())
                        state["step"] = 0
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    beta1, beta2 = self.adamw_betas
                    lr = group["adamw_lr"]

                    g32 = g.float()
                    if not torch.isfinite(g32).all():
                        # ROUND-170: a NaN grad must not reach the AdamW state
                        # or update (exp_avg_sq NaN -> denom sqrt(NaN) -> the
                        # addcdiv_ writes NaN into the weight). Skip the step;
                        # states untouched => clean grad resumes cleanly.
                        continue
                    exp_avg.mul_(beta1).add_(g32, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1 - beta2)
                    bias_corr1 = 1 - beta1 ** state["step"]
                    bias_corr2 = 1 - beta2 ** state["step"]
                    # Bug: with RSI_MUON_BF16=1, exp_avg_sq is bf16 and add_(1e-8)
                    # rounds to 0 (bf16 min normal ~6e-8) -> division-by-zero inf.
                    # Compute the denom in fp32 (add eps there) before casting to
                    # p.dtype, so the epsilon is never silently zeroed.
                    denom = (exp_avg_sq.float().sqrt() / bias_corr2 ** 0.5).add_(1e-8).to(p.dtype)
                    p.addcdiv_((exp_avg / bias_corr1).to(p.dtype), denom, value=-lr)
                    if wd != 0:
                        p.mul_(1 - lr * wd)  # decoupled weight decay
        return loss


def muon_param_groups(model: nn.Module, lr=3e-4, weight_decay=0.1,
                       muon_lr=0.0235, adamw_lr=0.007):
    """Split model params:
    - muon_square: square weight matrices (Q, O projections) → Newton-Schulz
    - muon_rect: non-square weight matrices (K, V, FFN gate/up/down) → momentum
    - adamw: 1D params + embeddings + norms + lm_head
    Matches IMU-1 paper: all 2D weight matrices use Muon-style updates.
    """
    square_2d = []
    rect_2d = []
    adamw_1d = []
    seen_params = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Bug 45b: tied weights (token_embed == lm_head) are the SAME tensor —
        # dedupe by id() or they get 2x updates per step.
        if id(p) in seen_params:
            continue
        seen_params.add(id(p))
        if p.dim() >= 2:
            if p.shape[0] == p.shape[1] and 'embed' not in name and 'norm' not in name and 'head' not in name:
                square_2d.append(p)
            elif 'embed' not in name and 'norm' not in name and 'head' not in name:
                rect_2d.append(p)
            else:
                adamw_1d.append(p)
        else:
            adamw_1d.append(p)

    groups = [
        # B-24 (research: KellerJordan/Muon canonical recipe — MuonWithAuxAdam
        # uses weight_decay 0.01 on the Muon groups TOO, not zero). Coupled wd
        # on the momentum path damps matrix-norm growth, which Rybakov et al.
        # (2410.16682) identifies as THE divergence driver at high LR — our
        # 10K-iter run oscillated 1.5-2.8 at full LR with wd=0 on these groups.
        {"params": square_2d, "mode": "muon_square", "lr": lr,
         "weight_decay": min(weight_decay, 0.01), "muon_lr": muon_lr, "adamw_lr": adamw_lr},
        {"params": rect_2d, "mode": "muon_rect", "lr": lr,
         "weight_decay": min(weight_decay, 0.01), "muon_lr": muon_lr, "adamw_lr": adamw_lr},
        # Data-constrained recipe: heavy wd lands on the AdamW group (embeddings,
        # head, norms, biases) — canonical Muon applies no wd to the matrices.
        {"params": adamw_1d, "mode": "adamw", "lr": lr,
         "weight_decay": weight_decay, "muon_lr": muon_lr, "adamw_lr": adamw_lr},
    ]
    n_total = len(square_2d) + len(rect_2d) + len(adamw_1d)
    n_sq_params = sum(p.numel() for p in square_2d)
    n_rect_params = sum(p.numel() for p in rect_2d)
    n_1d_params = sum(p.numel() for p in adamw_1d)
    print(f"[Muon] square={len(square_2d)} ({n_sq_params:,} params) | "
          f"rect={len(rect_2d)} ({n_rect_params:,} params) | "
          f"adamw={len(adamw_1d)} ({n_1d_params:,} params)")
    return groups
