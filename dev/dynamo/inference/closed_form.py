"""Closed-form solvers for per-embryo inference.

All solvers are batched and differentiable — gradients flow through into
network parameters via torch autograd.

Model spec references: §4.1 (c_e solve), §4.2 (R_e solve).
"""

from __future__ import annotations

import torch
from torch import Tensor


def solve_rate(
    displacements: Tensor,
    drift_direction: Tensor,
    dt: Tensor,
    mask: Tensor,
    clamp_min: float = 1e-6,
) -> Tensor:
    """Closed-form R_e solve via scalar projection (spec §4.2).

    Given observed displacements and predicted drift directions (before R_e
    scaling), finds the optimal scalar R_e that projects displacements onto
    the drift:

        R_e* = sum_t(delta_t^T f_hat_t dt) / sum_t(||f_hat_t||^2 dt^2)

    Fully differentiable — gradients flow through drift_direction into the
    potential network weights.

    Args:
        displacements: (B, T, d) observed z_{t+1} - z_t.
        drift_direction: (B, T, d) predicted drift (before R_e scaling).
        dt: (B, T) time step per transition.
        mask: (B, T) boolean mask for valid transitions.
        clamp_min: Floor on R_e to prevent division issues.

    Returns:
        (B,) inferred R_e per embryo.
    """
    mask_f = mask.float()

    # Dot product per transition: sum over d dimension
    dot = (displacements * drift_direction).sum(dim=-1)  # (B, T)

    # Squared norm of drift direction per transition
    f_sq = (drift_direction ** 2).sum(dim=-1)  # (B, T)

    # Weighted sums over time dimension
    numerator = (dot * dt * mask_f).sum(dim=1)        # (B,)
    denominator = (f_sq * dt ** 2 * mask_f).sum(dim=1)  # (B,)

    # Scalar projection with numerical safety
    R_e = numerator / denominator.clamp(min=1e-10)
    R_e = R_e.clamp(min=clamp_min)

    return R_e
