"""
BandPO: Probability-aware dynamic clipping for PPO.
Replaces canonical clip(ratio, 1-eps, 1+eps) with Band operator
that projects f-divergence trust regions into probability-aware bounds.
Reference: Li et al., "BandPO: Bridging Trust Regions and Ratio Clipping..." (2026)
"""
import torch
import math


def _solve_kl_bounds(p, delta=0.05, max_iter=100, tol=1e-6):
    """Solve KL-divergence Band bounds numerically.
    Given old probability p = P(a), find r_low, r_high such that
    KL(p || q) <= delta where q = r * p (r is the ratio).
    Uses parallel bisection (CUDA-compatible vectorized).
    """
    device = p.device
    dtype = p.dtype
    r_low = torch.zeros_like(p)
    r_high = torch.ones_like(p) * 5.0
    r_low_init = torch.zeros_like(p)

    for i in range(max_iter):
        r_mid = (r_low + r_high) / 2
        q = r_mid * p
        kl = p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))
        mask = kl > delta
        r_high = torch.where(mask, r_mid, r_high)
        r_low = torch.where(mask, r_low, r_mid)
        if (r_high - r_low).max().item() < tol:
            break

    r_high_final = (r_low + r_high) / 2
    r_low_final = torch.zeros_like(p) * 0.2
    r_high_temp = torch.ones_like(p) * 5.0
    r_low_temp = torch.zeros_like(p) * 0.001

    for i in range(max_iter):
        r_mid = (r_low_temp + r_high_temp) / 2
        q = r_mid * p
        kl = p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))
        mask = kl > delta
        r_high_temp = torch.where(mask, r_mid, r_high_temp)
        r_low_temp = torch.where(mask, r_low_temp, r_mid)
        if (r_high_temp - r_low_temp).max().item() < tol:
            break

    r_low_final = (r_low_temp + r_high_temp) / 2
    return r_low_final, r_high_final


@torch.no_grad()
def band_clip(ratio, old_log_probs, delta=0.05):
    """Apply Band clipping to probability ratio.
    Args:
        ratio: probability ratio pi_new / pi_old
        old_log_probs: log probabilities from old policy
        delta: trust region radius (default 0.05 per paper)
    Returns:
        clipped_ratio: probability-aware clipped ratio
    """
    p = torch.exp(old_log_probs).clamp(1e-6, 1.0)
    r_low, r_high = _solve_kl_bounds(p, delta)
    return torch.clamp(ratio, r_low, r_high)


def bandpo_objective(ratio, advantages, old_log_probs, delta=0.05):
    """BandPO surrogate objective.
    Replaces min(ratio * A, clip(ratio) * A) with
    min(ratio * A, band_clip(ratio) * A).
    """
    clipped = band_clip(ratio, old_log_probs, delta)
    return torch.min(ratio * advantages, clipped * advantages)
