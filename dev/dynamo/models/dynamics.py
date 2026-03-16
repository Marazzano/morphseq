"""Phi0-only dynamical model (Stage 1).

Combines a learned baseline potential phi_0 with global drift scale beta,
global diffusion D, and per-embryo rate R_e (closed-form). Implements the
Predictor protocol for compatibility with the eval pipeline.

SDE:  dz = R_e * (-beta * grad phi_0(z)) dt + sqrt(2D) dW

Model spec references: §3.1–3.2 (SDE/drift), §4.2 (R_e solve), §7.3 (loss),
    §7.5 (forward pass), §15.2 step 5.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..data.dataset import FragmentBatch
from ..eval.predictions import PredictionResult
from ..inference.closed_form import solve_rate
from .potential import PotentialNetwork


class Phi0OnlyModel(nn.Module):
    """Stage 1 model: learned baseline potential with no modes.

    Drift: f(z) = R_e * (-beta * grad_phi0(z))
    SDE:   dz = f(z) dt + sqrt(2D) dW

    Implements the Predictor protocol so it plugs directly into the existing
    evaluation pipeline via ``run_evaluation(model, dataset)``.

    Args:
        input_dim: Latent space dimension d.
        hidden_dim: MLP hidden layer width.
        n_hidden: Number of MLP hidden layers.
        activation: Smooth nonlinearity ("softplus" or "elu").
        init_log_beta: Initial value of log(beta).
        init_log_D: Initial value of log(D).
        n_forward_samples: Number of Euler-Maruyama samples for predict().
        rate_clamp_min: Floor on R_e from closed-form solve.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        activation: str = "softplus",
        init_log_beta: float = 0.0,
        init_log_D: float = -2.0,
        n_forward_samples: int = 50,
        rate_clamp_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.phi0 = PotentialNetwork(input_dim, hidden_dim, n_hidden, activation)
        self.log_beta = nn.Parameter(torch.tensor(init_log_beta, dtype=torch.float32))
        self.log_D = nn.Parameter(torch.tensor(init_log_D, dtype=torch.float32))
        self.n_forward_samples = n_forward_samples
        self.rate_clamp_min = rate_clamp_min

    @property
    def beta(self) -> Tensor:
        """Global drift scale (positive)."""
        return self.log_beta.exp()

    @property
    def D(self) -> Tensor:
        """Global diffusion coefficient (positive)."""
        return self.log_D.exp()

    def compute_drift_direction(self, z: Tensor) -> Tensor:
        """Compute f_hat(z) = -beta * grad_phi0(z) (before R_e scaling).

        Args:
            z: (*, d) input points.

        Returns:
            (*, d) drift direction vectors.
        """
        grad = self.phi0.gradient(z)
        return -self.beta * grad

    def _extract_context_transitions(self, batch: FragmentBatch):
        """Extract transitions, drift directions, and R_e from context.

        Returns:
            z_last: (B, d) last valid context frame per sample.
            R_e: (B,) inferred rate per sample.
        """
        B, L_max, d = batch.context.shape

        # Context transition pairs
        z_from = batch.context[:, :-1, :]  # (B, L_max-1, d)
        z_to = batch.context[:, 1:, :]     # (B, L_max-1, d)
        displacements = z_to - z_from      # (B, L_max-1, d)
        dt_ctx = batch.time_deltas         # (B, L_max-1)

        # Valid transition mask: both endpoints must be real frames
        trans_mask = batch.context_mask[:, :-1] & batch.context_mask[:, 1:]  # (B, L_max-1)

        # Gradients of phi0 at departure points
        z_flat = z_from.reshape(-1, d)              # (B*(L_max-1), d)
        grad_flat = self.phi0.gradient(z_flat)       # (B*(L_max-1), d)
        grad_phi0 = grad_flat.reshape(B, L_max - 1, d)

        # Drift direction (before R_e scaling)
        f_hat = -self.beta * grad_phi0  # (B, L_max-1, d)

        # Closed-form R_e from context transitions
        R_e = solve_rate(displacements, f_hat, dt_ctx, trans_mask,
                         clamp_min=self.rate_clamp_min)  # (B,)

        # Last valid context frame
        lengths = batch.context_mask.sum(dim=1).long()  # (B,)
        last_idx = (lengths - 1).clamp(min=0)           # (B,)
        z_last = batch.context[torch.arange(B, device=batch.context.device), last_idx]  # (B, d)

        return z_last, R_e

    def forward(self, batch: FragmentBatch) -> dict:
        """Training forward pass: compute NLL loss on target transitions.

        Infers R_e from context transitions (closed-form), then scores the
        target transition as a single-step Euler-Maruyama NLL. Gradients
        flow through the R_e solve into phi0 network weights.

        Args:
            batch: FragmentBatch from the data pipeline.

        Returns:
            Dict with keys:
                loss: scalar mean NLL over batch.
                nll: (B,) per-sample NLL.
                R_e: (B,) inferred rates.
                beta: scalar.
                D: scalar.
        """
        d = batch.context.shape[-1]
        z_last, R_e = self._extract_context_transitions(batch)

        # Drift at last context frame → predicted target
        grad_phi0_last = self.phi0.gradient(z_last)                       # (B, d)
        drift = R_e.unsqueeze(-1) * (-self.beta * grad_phi0_last)         # (B, d)
        horizon_dt = batch.horizon_dt                                     # (B,)
        predicted = z_last + drift * horizon_dt.unsqueeze(-1)             # (B, d)

        # Single-step NLL (spec §7.3):
        #   NLL = (d/2) log(4 pi D dt) + ||target - predicted||^2 / (4 D dt)
        diff = batch.target - predicted                                   # (B, d)
        D = self.D
        four_D_dt = 4.0 * D * horizon_dt                                 # (B,)
        nll = (
            0.5 * d * torch.log(math.pi * four_D_dt.clamp(min=1e-10))
            + (diff ** 2).sum(dim=-1) / four_D_dt.clamp(min=1e-10)
        )  # (B,)

        loss = nll.mean()

        return {
            "loss": loss,
            "nll": nll.detach(),
            "R_e": R_e.detach(),
            "beta": self.beta.detach(),
            "D": D.detach(),
        }

    @torch.no_grad()
    def predict(self, batch: FragmentBatch) -> PredictionResult:
        """Prediction for eval pipeline (Predictor protocol).

        Infers R_e from context, then produces predicted mean, diagonal
        covariance, and forward samples via Euler-Maruyama.

        Args:
            batch: FragmentBatch from the data pipeline.

        Returns:
            PredictionResult with predicted_mean, predicted_cov_diag,
            forward_samples, rate, and diffusion_D.
        """
        B, _, d = batch.context.shape
        z_last, R_e = self._extract_context_transitions(batch)

        # Drift at last context frame
        grad_phi0_last = self.phi0.gradient(z_last)
        drift = R_e.unsqueeze(-1) * (-self.beta * grad_phi0_last)
        horizon_dt = batch.horizon_dt

        # Mean prediction: single Euler step
        predicted_mean = z_last + drift * horizon_dt.unsqueeze(-1)  # (B, d)

        # Diagonal covariance: 2D * dt per dimension
        var = 2.0 * self.D * horizon_dt.unsqueeze(-1)              # (B, 1)
        predicted_cov_diag = var.expand(B, d)                       # (B, d)

        # Forward samples: mean + Gaussian noise
        noise_std = (2.0 * self.D * horizon_dt).sqrt()              # (B,)
        noise = torch.randn(B, self.n_forward_samples, d,
                            device=batch.context.device)
        forward_samples = (
            predicted_mean.unsqueeze(1)
            + noise * noise_std.unsqueeze(-1).unsqueeze(-1)
        )  # (B, n_samples, d)

        return PredictionResult(
            predicted_mean=predicted_mean,
            predicted_cov_diag=predicted_cov_diag,
            forward_samples=forward_samples,
            rate=R_e,
            diffusion_D=self.D.item(),
        )
