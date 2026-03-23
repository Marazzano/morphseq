"""Tests for phi0-only model (build step 5).

Tests cover:
    - PotentialNetwork: output shapes, gradient shapes, gradient correctness,
      create_graph for second-order gradients
    - solve_rate: known-R recovery, batch shape, mask handling, differentiability
    - Phi0OnlyModel: forward output shapes, loss finiteness, gradient flow,
      Predictor protocol, predict() output shapes
    - Integration: overfit on synthetic quadratic potential
    - Checkpoint round-trip
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dev.dynamo.data.loading import EmbryoTrajectory, TrajectoryDataset
from dev.dynamo.data.dataset import FragmentDataset, FragmentBatch, fragment_collate_fn
from dev.dynamo.eval.predictions import Predictor, PredictionResult
from dev.dynamo.models.potential import PotentialNetwork
from dev.dynamo.models.dynamics import Phi0OnlyModel
from dev.dynamo.inference.closed_form import solve_rate


# ---------------------------------------------------------------------------
# Fixtures: synthetic data (mirrors test_kernel.py patterns)
# ---------------------------------------------------------------------------

def _make_trajectory(
    embryo_id: str = "emb_001",
    n_frames: int = 20,
    n_dim: int = 5,
    delta_t: float = 300.0,
    temperature: float = 28.5,
    perturbation_class: str = "wildtype",
    experiment_id: str = "exp_001",
    seed: int = 42,
) -> EmbryoTrajectory:
    """Create a synthetic trajectory."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_frames, n_dim)) * 0.1
    traj = np.cumsum(steps, axis=0)
    times = np.arange(n_frames, dtype=np.float64) * delta_t
    return EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=traj,
        time_seconds=times,
        delta_t=delta_t,
        temperature=temperature,
        perturbation_class=perturbation_class,
        experiment_id=experiment_id,
    )


def _make_dataset(
    n_embryos: int = 10,
    n_frames: int = 20,
    n_dim: int = 5,
) -> TrajectoryDataset:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    classes = ["wildtype", "mutant_a", "mutant_b"]
    trajs = []
    for i in range(n_embryos):
        trajs.append(_make_trajectory(
            embryo_id=f"emb_{i:03d}",
            n_frames=n_frames,
            n_dim=n_dim,
            perturbation_class=classes[i % len(classes)],
            seed=i,
        ))

    pca = PCA(n_components=n_dim)
    pca.fit(np.eye(n_dim))
    scaler = StandardScaler()
    scaler.fit(np.zeros((2, n_dim)))

    return TrajectoryDataset(
        trajectories=trajs,
        pca=pca,
        scaler=scaler,
        z_mu_cols=[f"z_{i}" for i in range(n_dim)],
    )


def _make_batch(B: int = 8, L: int = 10, d: int = 5, M: int = 1) -> FragmentBatch:
    """Create a synthetic FragmentBatch for testing."""
    context = torch.randn(B, L, d)
    context_mask = torch.ones(B, L, dtype=torch.bool)
    targets = torch.randn(B, M, d)
    predecessors = torch.randn(B, M, d)
    time_deltas = torch.full((B, L - 1), 300.0)
    horizon_dts = torch.full((B, M), 300.0)
    context_to_target_dts = torch.full((B, M), 300.0)
    delta_t = torch.full((B,), 300.0)
    temperature = torch.full((B,), 28.5)
    class_idx = torch.zeros(B, dtype=torch.long)
    embryo_idx = torch.arange(B, dtype=torch.long)
    return FragmentBatch(
        context=context,
        context_mask=context_mask,
        targets=targets,
        predecessors=predecessors,
        time_deltas=time_deltas,
        horizon_dts=horizon_dts,
        context_to_target_dts=context_to_target_dts,
        delta_t=delta_t,
        temperature=temperature,
        class_idx=class_idx,
        embryo_idx=embryo_idx,
    )


# ===========================================================================
# PotentialNetwork tests
# ===========================================================================

class TestPotentialNetwork:
    """Tests for the PotentialNetwork MLP."""

    def test_output_shape(self):
        net = PotentialNetwork(input_dim=5, hidden_dim=16, n_hidden=2)
        z = torch.randn(8, 5)
        phi = net(z)
        assert phi.shape == (8,)

    def test_output_shape_unbatched(self):
        net = PotentialNetwork(input_dim=5, hidden_dim=16, n_hidden=2)
        z = torch.randn(5)
        phi = net(z)
        assert phi.shape == ()

    def test_gradient_shape(self):
        net = PotentialNetwork(input_dim=5, hidden_dim=16, n_hidden=2)
        z = torch.randn(8, 5)
        grad = net.gradient(z)
        assert grad.shape == (8, 5)

    def test_gradient_correctness(self):
        """Compare gradient() against manual torch.autograd.grad."""
        net = PotentialNetwork(input_dim=3, hidden_dim=8, n_hidden=1)
        z = torch.randn(4, 3)

        # Our method
        grad_ours = net.gradient(z)

        # Manual reference
        z_ref = z.detach().requires_grad_(True)
        phi_ref = net(z_ref)
        grad_ref = torch.autograd.grad(
            phi_ref, z_ref,
            grad_outputs=torch.ones_like(phi_ref),
            create_graph=False,
        )[0]

        torch.testing.assert_close(grad_ours.detach(), grad_ref, atol=1e-5, rtol=1e-5)

    def test_gradient_creates_graph(self):
        """Verify that gradient() supports second-order differentiation."""
        net = PotentialNetwork(input_dim=3, hidden_dim=8, n_hidden=1)
        z = torch.randn(4, 3)
        grad = net.gradient(z)
        assert grad.requires_grad, "gradient() must return tensors with grad_fn"

        # Should be able to backprop through the gradient
        loss = (grad ** 2).sum()
        loss.backward()
        # At least some network parameters should receive gradients
        has_any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net.parameters()
        )
        assert has_any_grad, "Gradients should flow into at least some network params"

    def test_activation_softplus(self):
        net = PotentialNetwork(input_dim=3, activation="softplus")
        assert any(isinstance(m, torch.nn.Softplus) for m in net.net.modules())

    def test_activation_elu(self):
        net = PotentialNetwork(input_dim=3, activation="elu")
        assert any(isinstance(m, torch.nn.ELU) for m in net.net.modules())

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            PotentialNetwork(input_dim=3, activation="relu")


# ===========================================================================
# solve_rate tests
# ===========================================================================

class TestSolveRate:
    """Tests for the closed-form R_e solve."""

    def test_known_rate_recovery(self):
        """Given synthetic data with known R, verify recovery."""
        torch.manual_seed(42)
        B, T, d = 4, 20, 5
        R_true = torch.tensor([1.0, 2.0, 0.5, 3.0])

        # Random drift directions
        f_hat = torch.randn(B, T, d)
        dt = torch.full((B, T), 0.1)
        mask = torch.ones(B, T, dtype=torch.bool)

        # Construct displacements = R * f_hat * dt (no noise)
        displacements = R_true[:, None, None] * f_hat * dt[:, :, None]

        R_solved = solve_rate(displacements, f_hat, dt, mask)
        torch.testing.assert_close(R_solved, R_true, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        B, T, d = 6, 10, 3
        R = solve_rate(
            torch.randn(B, T, d),
            torch.randn(B, T, d),
            torch.ones(B, T),
            torch.ones(B, T, dtype=torch.bool),
        )
        assert R.shape == (B,)

    def test_mask_handling(self):
        """Masked transitions should not affect R_e."""
        torch.manual_seed(0)
        B, T, d = 2, 10, 3
        R_true = torch.tensor([1.5, 2.5])
        f_hat = torch.randn(B, T, d)
        dt = torch.full((B, T), 0.1)
        displacements = R_true[:, None, None] * f_hat * dt[:, :, None]

        # Full mask
        mask_full = torch.ones(B, T, dtype=torch.bool)
        R_full = solve_rate(displacements, f_hat, dt, mask_full)

        # Partial mask (drop last 5)
        mask_partial = torch.ones(B, T, dtype=torch.bool)
        mask_partial[:, 5:] = False
        # Corrupt masked transitions to ensure they're ignored
        displacements_corrupt = displacements.clone()
        displacements_corrupt[:, 5:] = 999.0
        R_partial = solve_rate(displacements_corrupt, f_hat, dt, mask_partial)

        # Should recover same R from unmasked data
        torch.testing.assert_close(R_full, R_true, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(R_partial, R_true, atol=1e-4, rtol=1e-4)

    def test_differentiable(self):
        """R_e should have a grad_fn when inputs require grad."""
        B, T, d = 2, 5, 3
        f_hat = torch.randn(B, T, d, requires_grad=True)
        displacements = torch.randn(B, T, d)
        dt = torch.ones(B, T)
        mask = torch.ones(B, T, dtype=torch.bool)

        R = solve_rate(displacements, f_hat, dt, mask)
        assert R.requires_grad
        loss = R.sum()
        loss.backward()
        assert f_hat.grad is not None

    def test_clamp_min(self):
        """R_e should never go below clamp_min."""
        # Zero displacements → R_e should be clamped
        B, T, d = 2, 5, 3
        R = solve_rate(
            torch.zeros(B, T, d),
            torch.randn(B, T, d),
            torch.ones(B, T),
            torch.ones(B, T, dtype=torch.bool),
            clamp_min=0.01,
        )
        assert (R >= 0.01).all()


# ===========================================================================
# Phi0OnlyModel tests
# ===========================================================================

class TestPhi0OnlyModel:
    """Tests for the phi0-only dynamical model."""

    def test_forward_output_keys(self):
        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        batch = _make_batch(B=4, L=8, d=5)
        result = model(batch)
        assert "loss" in result
        assert "nll" in result
        assert "hessian_penalty" in result
        assert "R_e" in result
        assert "beta" in result
        assert "D" in result

    def test_forward_output_shapes(self):
        B, d = 4, 5
        model = Phi0OnlyModel(input_dim=d, hidden_dim=16, n_hidden=1)
        batch = _make_batch(B=B, L=8, d=d)
        result = model(batch)
        assert result["loss"].shape == ()
        assert result["nll"].shape == (B,)
        assert result["R_e"].shape == (B,)

    def test_loss_finite(self):
        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        batch = _make_batch(B=4, L=8, d=5)
        result = model(batch)
        assert torch.isfinite(result["loss"]).all()
        assert torch.isfinite(result["nll"]).all()

    def test_gradient_flow(self):
        """Loss.backward() should populate gradients on phi0 weights."""
        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        batch = _make_batch(B=4, L=8, d=5)
        result = model(batch)
        result["loss"].backward()

        # Check that phi0 network has gradients
        phi0_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.phi0.parameters()
        )
        assert phi0_has_grad, "Gradients must flow into phi0 network"

        # Check that log_beta and log_D have gradients
        assert model.log_beta.grad is not None
        assert model.log_D.grad is not None

    def test_predict_protocol(self):
        """Model should satisfy the Predictor protocol."""
        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        assert isinstance(model, Predictor)

    def test_predict_output_type(self):
        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        model.eval()
        batch = _make_batch(B=4, L=8, d=5)
        result = model.predict(batch)
        assert isinstance(result, PredictionResult)

    def test_predict_output_shapes(self):
        B, d, n_samples = 4, 5, 20
        model = Phi0OnlyModel(input_dim=d, hidden_dim=16, n_hidden=1,
                              n_forward_samples=n_samples)
        model.eval()
        batch = _make_batch(B=B, L=8, d=d)
        result = model.predict(batch)

        assert result.predicted_mean.shape == (B, d)
        assert result.predicted_cov_diag.shape == (B, d)
        assert result.forward_samples is not None
        assert result.forward_samples.shape == (B, n_samples, d)
        assert result.rate is not None
        assert result.rate.shape == (B,)
        assert result.diffusion_D is not None

    def test_predict_cov_positive(self):
        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        model.eval()
        batch = _make_batch(B=4, L=8, d=5)
        result = model.predict(batch)
        assert (result.predicted_cov_diag > 0).all()

    def test_beta_D_positive(self):
        model = Phi0OnlyModel(input_dim=5)
        assert model.beta.item() > 0
        assert model.D.item() > 0


# ===========================================================================
# Integration: overfit on synthetic quadratic potential
# ===========================================================================

class TestOverfitQuadratic:
    """Train on synthetic data from a known quadratic potential and verify recovery."""

    def _generate_quadratic_data(
        self, n_embryos: int = 30, n_frames: int = 30, d: int = 3,
        R_true: float = 1.0, beta_true: float = 1.0, D_true: float = 0.01,
        dt: float = 0.1, seed: int = 42,
    ) -> TrajectoryDataset:
        """Generate trajectories from phi(z) = 0.5 * ||z||^2.

        Drift: f(z) = R * (-beta * z), so dz = -R*beta*z*dt + sqrt(2D)*dW.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        trajs = []
        for i in range(n_embryos):
            z = rng.standard_normal(d) * 2.0  # start scattered
            trajectory = [z.copy()]
            for _ in range(n_frames - 1):
                drift = -R_true * beta_true * z
                noise = np.sqrt(2 * D_true * dt) * rng.standard_normal(d)
                z = z + drift * dt + noise
                trajectory.append(z.copy())
            traj = np.stack(trajectory)
            times = np.arange(n_frames, dtype=np.float64) * dt

            trajs.append(EmbryoTrajectory(
                embryo_id=f"emb_{i:03d}",
                trajectory=traj,
                time_seconds=times,
                delta_t=dt,
                temperature=28.5,
                perturbation_class="wildtype",
                experiment_id="exp_001",
            ))

        pca = PCA(n_components=d)
        pca.fit(np.eye(d))
        scaler = StandardScaler()
        scaler.fit(np.zeros((2, d)))

        return TrajectoryDataset(
            trajectories=trajs,
            pca=pca,
            scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(d)],
        )

    def test_overfit_loss_decreases(self):
        """Training loss should decrease on synthetic quadratic data.

        Uses running averages to smooth stochastic fragment sampling noise.
        """
        d = 3
        ds = self._generate_quadratic_data(n_embryos=30, n_frames=30, d=d)
        frag = FragmentDataset(ds, min_context=5, horizons=(1,), epoch_length=500)
        loader = torch.utils.data.DataLoader(
            frag, batch_size=64, collate_fn=fragment_collate_fn
        )

        model = Phi0OnlyModel(
            input_dim=d, hidden_dim=32, n_hidden=2,
            init_log_beta=0.0, init_log_D=-3.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        losses = []
        for epoch in range(50):
            epoch_loss = 0.0
            n = 0
            for batch in loader:
                result = model(batch)
                optimizer.zero_grad()
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                epoch_loss += result["loss"].item()
                n += 1
            losses.append(epoch_loss / max(n, 1))

        # Compare smoothed early vs late loss (more negative = better for NLL)
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss did not decrease: early avg={early_avg:.4f}, late avg={late_avg:.4f}"
        )

    def test_overfit_D_convergence(self):
        """After training, the learned diffusion D should converge toward true D.

        D convergence is a robust signal that the model is learning meaningful
        dynamics, even when the potential landscape takes longer to converge.
        """
        d = 3
        D_true = 0.01
        ds = self._generate_quadratic_data(
            n_embryos=30, n_frames=30, d=d,
            R_true=1.0, beta_true=1.0, D_true=D_true, dt=0.1,
        )
        frag = FragmentDataset(ds, min_context=5, horizons=(1,), epoch_length=500)
        loader = torch.utils.data.DataLoader(
            frag, batch_size=64, collate_fn=fragment_collate_fn
        )

        # Start D far from truth: exp(-1) ≈ 0.37, true D = 0.01
        model = Phi0OnlyModel(
            input_dim=d, hidden_dim=32, n_hidden=2,
            init_log_beta=0.0, init_log_D=-1.0,
        )
        D_init = model.D.item()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        for epoch in range(80):
            for batch in loader:
                result = model(batch)
                optimizer.zero_grad()
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

        D_learned = model.D.item()
        # D should have moved closer to true value
        assert abs(D_learned - D_true) < abs(D_init - D_true), (
            f"D did not converge: init={D_init:.4f}, learned={D_learned:.4f}, "
            f"true={D_true:.4f}"
        )


# ===========================================================================
# Checkpoint round-trip
# ===========================================================================

class TestCheckpoint:
    """Test save/load checkpoint round-trip."""

    def test_save_load_roundtrip(self):
        from dev.dynamo.training.trainer import Stage1Trainer, TrainConfig

        model = Phi0OnlyModel(input_dim=5, hidden_dim=16, n_hidden=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = TrainConfig(hidden_dim=16, n_hidden=1)

        # Get a prediction before saving
        batch = _make_batch(B=4, L=8, d=5)
        model.eval()
        pred_before = model.predict(batch)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            torch.save({
                "epoch": 10,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "hidden_dim": 16,
                    "n_hidden": 1,
                    "activation": "softplus",
                    "init_log_beta": 0.0,
                    "init_log_D": -2.0,
                    "n_forward_samples": 50,
                    "rate_clamp_min": 1e-6,
                    "alpha_0": 0.01,
                    "hessian_n_points": 64,
                },
                "best_nll": 5.0,
                "input_dim": 5,
            }, ckpt_path)

            # Load
            model_loaded, config_loaded = Stage1Trainer.load_checkpoint(ckpt_path)

        # Predict with loaded model
        model_loaded.eval()
        torch.manual_seed(0)
        pred_loaded = model_loaded.predict(batch)
        torch.manual_seed(0)
        pred_before2 = model.predict(batch)

        torch.testing.assert_close(
            pred_loaded.predicted_mean, pred_before2.predicted_mean,
            atol=1e-5, rtol=1e-5,
        )


# ===========================================================================
# Integration with eval pipeline
# ===========================================================================

class TestEvalIntegration:
    """Test that Phi0OnlyModel works with run_evaluation()."""

    def test_run_evaluation(self):
        from dev.dynamo.eval.evaluate import run_evaluation

        d = 5
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=d)
        frag = FragmentDataset(ds, min_context=3, horizons=(1, 2))
        model = Phi0OnlyModel(input_dim=d, hidden_dim=16, n_hidden=1)
        model.eval()

        result = run_evaluation(model, frag, n_batches=5, batch_size=8)
        assert result.n_samples > 0
        assert "nll" in result.metrics
        assert "mse" in result.metrics
        assert result.calibration >= 0.0
