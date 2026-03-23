"""Prediction interface and dummy predictors for eval pipeline testing.

Defines the common prediction result format and protocol that all models
(kernel baseline, phi0-only, full model) must implement. Includes simple
predictors for end-to-end pipeline validation before any model is trained.

Model spec references: §11 (evaluation), §15.2 (build sequence step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..data.dataset import FragmentBatch


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Standardized output from any predictor.

    All tensors are on the same device as the input batch.

    Attributes:
        predicted_mean: (B, D) predicted target location.
        predicted_cov_diag: (B, D) diagonal covariance (variance per dim).
        forward_samples: (B, n_samples, D) optional forward-simulated trajectories.
        horizon_k: (B,) integer horizon used for each sample (in frame units).
        diffusion_D: Global diffusion coefficient (scalar), if applicable.

        Mode diagnostics (populated only by models with modes):
        mode_loadings: (B, M) inferred c_e per embryo.
        local_correction_norm: (B,) ||v_e|| per embryo.
        residual_norm: (B,) ||residual|| after closed-form solve.
        rate: (B,) inferred R_e per embryo.
    """
    predicted_mean: Tensor               # (B, D)
    predicted_cov_diag: Tensor           # (B, D) diagonal variance
    forward_samples: Optional[Tensor] = None  # (B, n_samples, D)
    horizon_k: Optional[Tensor] = None   # (B,) int
    diffusion_D: Optional[float] = None

    # Mode diagnostics (optional)
    mode_loadings: Optional[Tensor] = None       # (B, M)
    local_correction_norm: Optional[Tensor] = None  # (B,)
    residual_norm: Optional[Tensor] = None       # (B,)
    rate: Optional[Tensor] = None                # (B,)


# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Predictor(Protocol):
    """Interface that all models/baselines must implement."""

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        """Produce predictions for a batch of fragments.

        Args:
            batch: A FragmentBatch from the data pipeline.

        Returns:
            PredictionResult with at least predicted_mean and predicted_cov_diag.
        """
        ...


# ---------------------------------------------------------------------------
# Dummy predictors for pipeline testing
# ---------------------------------------------------------------------------

class PersistencePredictor:
    """Predict that the target equals the last observed context frame.

    This is a trivial baseline: the "prediction" is simply that the embryo
    stays where it was last seen. Variance is set to a fixed isotropic value
    scaled by the horizon time gap.
    """

    def __init__(self, noise_scale: float = 0.1) -> None:
        self.noise_scale = noise_scale

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        B = batch.context.shape[0]
        D = batch.context.shape[2]

        # Last real context frame for each sample
        lengths = batch.context_mask.sum(dim=1).long()  # (B,)
        last_idx = lengths - 1
        last_frame = batch.context[torch.arange(B, device=batch.context.device), last_idx]  # (B, D)

        # Variance scales with horizon_dt (longer horizon → more uncertainty)
        var = (self.noise_scale ** 2) * batch.horizon_dt.unsqueeze(-1).expand(B, D)

        return PredictionResult(
            predicted_mean=last_frame,
            predicted_cov_diag=var,
        )


class LinearExtrapolationPredictor:
    """Predict by fitting a linear trend to the last N context frames.

    Fits a least-squares line through the last N valid context frames
    (in time), then extrapolates forward by horizon_dt. More robust than
    using only 2 frames when observations are noisy.
    """

    def __init__(self, noise_scale: float = 0.1, n_points: int = 5) -> None:
        self.noise_scale = noise_scale
        self.n_points = n_points

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        B = batch.context.shape[0]
        D = batch.context.shape[2]
        device = batch.context.device

        lengths = batch.context_mask.sum(dim=1).long()  # (B,)

        # Process each sample individually (variable-length contexts)
        means = []
        for b in range(B):
            L = lengths[b].item()
            n_use = min(self.n_points, L)

            if n_use < 2:
                # Fall back to persistence
                means.append(batch.context[b, L - 1])
                continue

            # Extract last n_use frames and their cumulative times
            start = L - n_use
            frames = batch.context[b, start:L]  # (n_use, D)

            # Build time axis from inter-frame deltas, relative to first used frame
            if n_use == L:
                deltas = batch.time_deltas[b, :L - 1]  # (L-1,)
            else:
                deltas = batch.time_deltas[b, start:L - 1]  # (n_use-1,)
            t = torch.zeros(n_use, device=device)
            t[1:] = torch.cumsum(deltas, dim=0)

            # Least-squares: velocity = Cov(t, z) / Var(t), intercept follows
            t_mean = t.mean()
            t_centered = t - t_mean  # (n_use,)
            var_t = (t_centered ** 2).sum().clamp(min=1e-8)

            z_mean = frames.mean(dim=0)  # (D,)
            z_centered = frames - z_mean.unsqueeze(0)  # (n_use, D)
            cov_tz = (t_centered.unsqueeze(-1) * z_centered).sum(dim=0)  # (D,)

            velocity = cov_tz / var_t  # (D,)

            # Extrapolate from last frame
            last_frame = frames[-1]  # (D,)
            mean = last_frame + velocity * batch.horizon_dt[b]
            means.append(mean)

        mean = torch.stack(means, dim=0)  # (B, D)
        var = (self.noise_scale ** 2) * batch.horizon_dt.unsqueeze(-1).abs().expand(B, D)

        return PredictionResult(
            predicted_mean=mean,
            predicted_cov_diag=var,
        )


class GaussianNoisePredictor:
    """Random Gaussian predictions centered on global data mean.

    Useful only for verifying that metrics worsen with random predictions.
    """

    def __init__(self, mean: Optional[Tensor] = None, std: float = 1.0) -> None:
        self._mean = mean  # (D,) or None
        self._std = std

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        B = batch.context.shape[0]
        D = batch.context.shape[2]
        device = batch.context.device

        if self._mean is not None:
            center = self._mean.to(device).unsqueeze(0).expand(B, D)
        else:
            # Use batch mean as fallback
            mask = batch.context_mask.unsqueeze(-1)  # (B, L, 1)
            center = (batch.context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        mean = center + torch.randn(B, D, device=device) * self._std
        var = torch.full((B, D), self._std ** 2, device=device)

        return PredictionResult(
            predicted_mean=mean,
            predicted_cov_diag=var,
        )
