"""Tests for the branching particle filter baseline (build step 4).

Tests cover the three implementation layers separately and together:

  Layer 1 -- Reference selection:
    - Only trajectories with ≥ min_points_in_radius qualify
    - Weights combine positional and directional terms
    - Trajectories moving in the opposite direction get zero weight
    - exclude_self correctly suppresses the query embryo
    - Speed ratios are positive

  Layer 2 -- Forward prediction without recruitment:
    - Particles advance by the expected frame offset
    - Dead particles (trajectory ends) are removed
    - Output positions match expected locations on straight-line data

  Layer 3 -- Recruitment:
    - Recruited particles have start_step > 0
    - Inherited weights are proportional to proximity
    - Already-active trajectories are not re-recruited
    - The active set grows when nearby trajectories exist

  Integration:
    - predict() returns correct shapes and types
    - Predictor protocol compliance
    - Fallback to persistence when no references found
    - Metric ordering: particle filter beats random on structured data
    - Branching data: predictive distribution remains bimodal
    - Comparison with SimpleKernelPredictor on parallel-line data
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from dev.dynamo.data.loading import EmbryoTrajectory, TrajectoryDataset
from dev.dynamo.data.dataset import FragmentDataset, fragment_collate_fn, FragmentBatch
from dev.dynamo.eval.predictions import Predictor, PredictionResult
from dev.dynamo.models.particle_filter import (
    BranchingParticleFilter,
    _ReferenceBank,
    _Particle,
    _build_reference_bank,
    _linear_fit,
    _interpolate_frame,
    _local_speed,
    _auto_radius,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic trajectories and datasets
# ---------------------------------------------------------------------------

def _make_trajectory(
    embryo_id: str = "emb_000",
    n_frames: int = 20,
    n_dim: int = 5,
    delta_t: float = 300.0,
    temperature: float = 28.5,
    perturbation_class: str = "wildtype",
    experiment_id: str = "exp_001",
    seed: int = 42,
    direction: np.ndarray | None = None,
    start: np.ndarray | None = None,
    noise_scale: float = 0.01,
) -> EmbryoTrajectory:
    """Create a synthetic trajectory with optional directional structure."""
    rng = np.random.default_rng(seed)
    if direction is not None and start is not None:
        t_vals = np.linspace(0, 1, n_frames)[:, None]
        traj = start[None] + t_vals * direction[None] + rng.standard_normal((n_frames, n_dim)) * noise_scale
    else:
        steps = rng.standard_normal((n_frames, n_dim)) * 0.1
        traj = np.cumsum(steps, axis=0)
    times = np.arange(n_frames, dtype=np.float64) * delta_t
    return EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=traj.astype(np.float64),
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
    classes: list[str] | None = None,
    experiments: list[str] | None = None,
    direction: np.ndarray | None = None,
    noise_scale: float = 0.01,
) -> TrajectoryDataset:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if classes is None:
        classes = ["wildtype", "mutant_a", "mutant_b"]
    if experiments is None:
        experiments = ["exp_001"]

    trajs = []
    for i in range(n_embryos):
        start = np.zeros(n_dim) + np.random.default_rng(i * 7).standard_normal(n_dim) * 0.05
        trajs.append(_make_trajectory(
            embryo_id=f"emb_{i:03d}",
            n_frames=n_frames,
            n_dim=n_dim,
            perturbation_class=classes[i % len(classes)],
            experiment_id=experiments[i % len(experiments)],
            seed=i,
            direction=direction,
            start=start if direction is not None else None,
            noise_scale=noise_scale,
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


def _make_batch(ds: TrajectoryDataset, n: int = 8, seed: int = 0) -> FragmentBatch:
    fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        # Draw deterministically for reproducibility
        samples.append(fds[0])
    return fragment_collate_fn(samples)


# ---------------------------------------------------------------------------
# Tests: geometry helpers
# ---------------------------------------------------------------------------

class TestLinearFit:
    def test_single_point(self) -> None:
        pts = np.array([[1.0, 2.0, 3.0]])
        mid, dir_ = _linear_fit(pts)
        np.testing.assert_allclose(mid, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(dir_, [0.0, 0.0, 0.0])

    def test_two_point_direction(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        mid, dir_ = _linear_fit(pts)
        # Direction should be along x-axis (up to sign).
        assert abs(abs(dir_[0]) - 1.0) < 1e-6
        assert abs(dir_[1]) < 1e-6

    def test_returns_unit_magnitude(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((10, 4))
        _, dir_ = _linear_fit(pts)
        assert abs(np.linalg.norm(dir_) - 1.0) < 1e-6

    def test_midpoint_is_centroid(self) -> None:
        pts = np.arange(12, dtype=float).reshape(4, 3)
        mid, _ = _linear_fit(pts)
        np.testing.assert_allclose(mid, pts.mean(axis=0))


class TestInterpolateFrame:
    def test_integer_frame_returns_exact(self) -> None:
        traj = np.arange(15, dtype=float).reshape(5, 3)
        for i in range(5):
            result = _interpolate_frame(traj, float(i))
            np.testing.assert_allclose(result, traj[i])

    def test_midpoint_interpolates(self) -> None:
        traj = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        result = _interpolate_frame(traj, 0.5)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_out_of_bounds_negative(self) -> None:
        traj = np.ones((5, 2))
        assert _interpolate_frame(traj, -0.1) is None

    def test_out_of_bounds_positive(self) -> None:
        traj = np.ones((5, 2))
        assert _interpolate_frame(traj, 5.0) is None

    def test_exactly_last_frame(self) -> None:
        traj = np.arange(10, dtype=float).reshape(5, 2)
        result = _interpolate_frame(traj, 4.0)
        np.testing.assert_allclose(result, traj[4])


class TestLocalSpeed:
    def test_constant_speed(self) -> None:
        # Trajectory moves 1 unit per frame; dt=1 → speed = 1.
        traj = np.stack([np.arange(10), np.zeros(10)], axis=1).astype(float)
        frames = np.arange(10, dtype=np.int64)
        spd = _local_speed(traj, frames, dt=1.0)
        assert abs(spd - 1.0) < 1e-9

    def test_positive(self) -> None:
        traj = np.random.default_rng(0).standard_normal((10, 3))
        frames = np.array([0, 1, 2, 3])
        spd = _local_speed(traj, frames, dt=300.0)
        assert spd > 0


# ---------------------------------------------------------------------------
# Tests: reference bank
# ---------------------------------------------------------------------------

class TestReferenceBank:
    def test_construction_shapes(self) -> None:
        ds = _make_dataset(n_embryos=5, n_frames=10, n_dim=3)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)
        assert bank.all_points.shape == (50, 3)
        assert len(bank.traj_id) == 50
        assert len(bank.traj_points) == 5
        assert (bank.traj_lengths == 10).all()

    def test_auto_radius_positive(self) -> None:
        ds = _make_dataset(n_embryos=5, n_frames=10, n_dim=3)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)
        r = _auto_radius(bank)
        assert r > 0

    def test_auto_radius_deterministic(self) -> None:
        ds = _make_dataset(n_embryos=5, n_frames=10, n_dim=3)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)
        r1 = _auto_radius(bank, seed=7)
        r2 = _auto_radius(bank, seed=7)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: Layer 1 — reference selection
# ---------------------------------------------------------------------------

class TestReferenceSelection:
    def test_returns_nonempty_for_nearby_data(self) -> None:
        """With enough nearby trajectories the selector should find references."""
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=D, direction=direction)
        pf = BranchingParticleFilter(ds, radius=5.0, min_points_in_radius=1, exclude_self=False)

        # Build a context that sits in the middle of the data.
        context = ds.trajectories[0].trajectory[:5]
        particles = pf.select_references(context=context, delta_t=300.0, embryo_idx=-1)
        assert len(particles) > 0

    def test_weights_sum_to_one(self) -> None:
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=D, direction=direction)
        pf = BranchingParticleFilter(ds, radius=5.0, min_points_in_radius=1, exclude_self=False)

        context = ds.trajectories[0].trajectory[:5]
        particles = pf.select_references(context=context, delta_t=300.0, embryo_idx=-1)
        if particles:
            total = sum(p.weight for p in particles)
            assert abs(total - 1.0) < 1e-9

    def test_exclude_self(self) -> None:
        """With only 1 trajectory and exclude_self=True, no references found."""
        ds = _make_dataset(n_embryos=1, n_frames=20, n_dim=3)
        pf = BranchingParticleFilter(ds, radius=100.0, min_points_in_radius=1, exclude_self=True)
        context = ds.trajectories[0].trajectory[:5]
        particles = pf.select_references(context=context, delta_t=300.0, embryo_idx=0)
        assert len(particles) == 0

    def test_opposite_direction_gets_zero_weight(self) -> None:
        """Reference moving opposite to query should contribute zero directional weight."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        D = 2
        rng = np.random.default_rng(0)

        # Query context: moves in +x direction.
        q_ctx = np.column_stack([np.linspace(0, 1, 8), np.zeros(8)])

        # Single reference trajectory: moves in -x direction.
        ref_traj = np.column_stack([np.linspace(1, 0, 20), np.zeros(20)])
        times = np.arange(20, dtype=float) * 300.0
        ref = EmbryoTrajectory(
            embryo_id="opp",
            trajectory=ref_traj,
            time_seconds=times,
            delta_t=300.0,
            temperature=28.5,
            perturbation_class="wt",
            experiment_id="exp_001",
        )
        pca = PCA(n_components=D)
        pca.fit(np.eye(D))
        scaler = StandardScaler()
        scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=[ref], pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )
        pf = BranchingParticleFilter(
            ds, radius=5.0, min_points_in_radius=1,
            directional_alpha=2.0, exclude_self=False,
        )
        particles = pf.select_references(context=q_ctx, delta_t=300.0, embryo_idx=-1)
        # All particles for the opposite-direction reference should have weight ≈ 0.
        for p in particles:
            assert p.weight < 1e-6, f"Expected near-zero weight, got {p.weight}"

    def test_speed_ratios_positive(self) -> None:
        D = 3
        direction = np.array([1.0, 0.5, 0.0])
        ds = _make_dataset(n_embryos=8, n_frames=20, n_dim=D, direction=direction)
        pf = BranchingParticleFilter(ds, radius=5.0, min_points_in_radius=1, exclude_self=False)
        context = ds.trajectories[0].trajectory[:5]
        particles = pf.select_references(context=context, delta_t=300.0, embryo_idx=-1)
        for p in particles:
            assert p.speed_ratio > 0

    def test_min_points_filter(self) -> None:
        """Trajectories with fewer than min_points_in_radius points in R are skipped."""
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=6, n_frames=20, n_dim=D, direction=direction)
        # Very small radius → almost no points qualify.
        pf_strict = BranchingParticleFilter(
            ds, radius=1e-6, min_points_in_radius=3, exclude_self=False,
        )
        context = ds.trajectories[0].trajectory[:5]
        particles = pf_strict.select_references(context=context, delta_t=300.0, embryo_idx=-1)
        assert len(particles) == 0


# ---------------------------------------------------------------------------
# Tests: Layer 2 — particle advancement (single-shot, no recruitment)
# ---------------------------------------------------------------------------

class TestParticleAdvancement:
    def _make_pf_no_recruit(self, ds: TrajectoryDataset) -> BranchingParticleFilter:
        """Particle filter with n_steps-of-advance effectively suppressed recruitment
        by using a near-zero sigma_recruit (so recruitment weight is ~0)."""
        return BranchingParticleFilter(
            ds,
            radius=5.0,
            min_points_in_radius=1,
            sigma_recruit=1e-12,  # effectively no recruitment
            exclude_self=False,
            n_max_particles=500,
        )

    def test_particle_position_at_step_zero(self) -> None:
        """At step 0 (anchor frame), position should be the anchor point."""
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=5, n_frames=20, n_dim=D, direction=direction)
        pf = self._make_pf_no_recruit(ds)

        traj = ds.trajectories[0].trajectory
        p = _Particle(traj_id=0, anchor_frame=5, speed_ratio=1.0, weight=1.0, dt_ref=300.0, start_step=0)
        pos = pf._particle_position(p, step=0, delta_t=300.0)
        np.testing.assert_allclose(pos, traj[5], atol=1e-9)

    def test_particle_position_advances(self) -> None:
        """With speed_ratio=1 and same dt, step k should advance k frames."""
        D = 2
        traj_arr = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        times = np.arange(20, dtype=float) * 300.0
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        ref = EmbryoTrajectory(
            embryo_id="e0", trajectory=traj_arr, time_seconds=times,
            delta_t=300.0, temperature=28.5, perturbation_class="wt",
            experiment_id="exp_001",
        )
        pca = PCA(n_components=D); pca.fit(np.eye(D))
        scaler = StandardScaler(); scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=[ref], pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )
        pf = BranchingParticleFilter(ds, radius=100.0, exclude_self=False, sigma_recruit=1e-12)

        p = _Particle(traj_id=0, anchor_frame=0, speed_ratio=1.0, weight=1.0, dt_ref=300.0, start_step=0)
        # Step 3 → frame offset = 1.0 * 3 * 300 / 300 = 3 → position x=3.
        pos = pf._particle_position(p, step=3, delta_t=300.0)
        assert pos is not None
        assert abs(pos[0] - 3.0) < 1e-9

    def test_dead_particle_returns_none(self) -> None:
        """A particle that runs off the end of its trajectory returns None."""
        D = 2
        short_traj = np.ones((5, D))
        times = np.arange(5, dtype=float) * 300.0
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        ref = EmbryoTrajectory(
            embryo_id="short", trajectory=short_traj, time_seconds=times,
            delta_t=300.0, temperature=28.5, perturbation_class="wt",
            experiment_id="exp_001",
        )
        pca = PCA(n_components=D); pca.fit(np.eye(D))
        scaler = StandardScaler(); scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=[ref], pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )
        pf = BranchingParticleFilter(ds, radius=100.0, exclude_self=False, sigma_recruit=1e-12)

        # Anchor at frame 0, speed_ratio=1, dt same → step 10 needs frame 10 but traj has only 5.
        p = _Particle(traj_id=0, anchor_frame=0, speed_ratio=1.0, weight=1.0, dt_ref=300.0, start_step=0)
        pos = pf._particle_position(p, step=10, delta_t=300.0)
        assert pos is None

    def test_speed_ratio_doubles_advance(self) -> None:
        """With speed_ratio=2, step k should advance 2k reference frames."""
        D = 1
        traj_arr = np.arange(40, dtype=float).reshape(40, 1)
        times = np.arange(40, dtype=float) * 300.0
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        ref = EmbryoTrajectory(
            embryo_id="e0", trajectory=traj_arr, time_seconds=times,
            delta_t=300.0, temperature=28.5, perturbation_class="wt",
            experiment_id="exp_001",
        )
        pca = PCA(n_components=D); pca.fit(np.eye(D))
        scaler = StandardScaler(); scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=[ref], pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )
        pf = BranchingParticleFilter(ds, radius=100.0, exclude_self=False, sigma_recruit=1e-12)

        p = _Particle(traj_id=0, anchor_frame=0, speed_ratio=2.0, weight=1.0, dt_ref=300.0, start_step=0)
        # step=3 → frame offset = 2 * 3 * 300/300 = 6 → value = 6.
        pos = pf._particle_position(p, step=3, delta_t=300.0)
        assert pos is not None
        assert abs(pos[0] - 6.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Layer 3 — recruitment
# ---------------------------------------------------------------------------

class TestRecruitment:
    def test_recruited_particles_have_nonzero_start_step(self) -> None:
        """After running forward prediction, recruited particles have start_step > 0."""
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=12, n_frames=30, n_dim=D, direction=direction, noise_scale=0.001)
        pf = BranchingParticleFilter(
            ds, radius=2.0, min_points_in_radius=1,
            sigma_recruit=1.0, n_max_particles=500,
            exclude_self=False,
        )

        context = ds.trajectories[0].trajectory[:6]
        particles = pf.select_references(context=context, delta_t=300.0, embryo_idx=-1)
        if not particles:
            pytest.skip("No initial references found with this data/radius")

        query_speed = float(np.linalg.norm(np.diff(context, axis=0), axis=1).mean() / 300.0)
        advanced = pf._forward_predict(
            particles=particles,
            n_steps=3,
            delta_t=300.0,
            embryo_idx=-1,
            query_speed=query_speed,
        )

        start_steps = [p.start_step for p in advanced]
        # At least some particles should have been recruited (start_step > 0)
        # when there are enough nearby trajectories.
        # (If the data is dense enough, this should fire; otherwise skip.)
        if len(advanced) > len(particles):
            assert any(s > 0 for s in start_steps)

    def test_already_active_not_re_recruited(self) -> None:
        """_recruit_from should skip trajectories in exclude_traj_ids."""
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=8, n_frames=20, n_dim=D, direction=direction)
        pf = BranchingParticleFilter(
            ds, radius=5.0, min_points_in_radius=1,
            sigma_recruit=1.0, exclude_self=False,
        )
        position = ds.trajectories[0].trajectory[5]
        exclude = {0, 1, 2, 3}
        query_speed = 0.001 / 300.0

        new_ps = pf._recruit_from(
            position=position,
            parent_weight=1.0,
            current_step=1,
            delta_t=300.0,
            embryo_idx=-1,
            exclude_traj_ids=exclude,
            query_speed=query_speed,
        )
        recruited_ids = {p.traj_id for p in new_ps}
        assert recruited_ids.isdisjoint(exclude)

    def test_recruitment_inherits_weight(self) -> None:
        """Recruited particle weights should be ≤ parent weight."""
        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=6, n_frames=20, n_dim=D, direction=direction, noise_scale=0.001)
        pf = BranchingParticleFilter(
            ds, radius=5.0, min_points_in_radius=1,
            sigma_recruit=1.0, exclude_self=False,
        )
        position = ds.trajectories[0].trajectory[5]
        new_ps = pf._recruit_from(
            position=position,
            parent_weight=0.5,
            current_step=1,
            delta_t=300.0,
            embryo_idx=-1,
            exclude_traj_ids=set(),
            query_speed=0.001 / 300.0,
        )
        for rp in new_ps:
            assert rp.weight <= 0.5 + 1e-12


# ---------------------------------------------------------------------------
# Tests: predict() output shapes and protocol
# ---------------------------------------------------------------------------

class TestPredictOutputs:
    def test_protocol_compliance(self) -> None:
        ds = _make_dataset()
        pf = BranchingParticleFilter(ds, radius=5.0, n_samples=20)
        assert isinstance(pf, Predictor)

    def test_output_shapes(self) -> None:
        D = 3
        B = 4
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=D)
        pf = BranchingParticleFilter(ds, radius=5.0, n_samples=30, exclude_self=False)
        batch = _make_batch(ds, n=B)
        result = pf.predict(batch)

        assert result.predicted_mean.shape == (B, D)
        assert result.predicted_cov_diag.shape == (B, D)
        assert result.forward_samples is not None
        assert result.forward_samples.shape == (B, 30, D)

    def test_output_dtype(self) -> None:
        ds = _make_dataset(n_embryos=6, n_frames=15, n_dim=3)
        pf = BranchingParticleFilter(ds, radius=5.0, n_samples=10, exclude_self=False)
        batch = _make_batch(ds, n=2)
        result = pf.predict(batch)
        assert result.predicted_mean.dtype == torch.float32
        assert result.predicted_cov_diag.dtype == torch.float32

    def test_cov_diag_positive(self) -> None:
        ds = _make_dataset(n_embryos=8, n_frames=20, n_dim=3)
        pf = BranchingParticleFilter(ds, radius=5.0, n_samples=10, exclude_self=False)
        batch = _make_batch(ds, n=4)
        result = pf.predict(batch)
        assert (result.predicted_cov_diag > 0).all()

    def test_fallback_when_no_references(self) -> None:
        """Single embryo + exclude_self → fallback to persistence."""
        ds = _make_dataset(n_embryos=1, n_frames=20, n_dim=3)
        pf = BranchingParticleFilter(ds, radius=100.0, n_samples=5, exclude_self=True)
        batch = _make_batch(ds, n=2)
        result = pf.predict(batch)
        # Fallback: cov_diag should be all-ones.
        assert result.predicted_mean.shape == (2, 3)
        assert (result.predicted_cov_diag == 1.0).all()

    def test_auto_radius(self) -> None:
        ds = _make_dataset(n_embryos=8, n_frames=20, n_dim=3)
        pf = BranchingParticleFilter(ds, radius=None, n_samples=10, exclude_self=False)
        assert pf.radius > 0


# ---------------------------------------------------------------------------
# Tests: prediction quality on structured data
# ---------------------------------------------------------------------------

class TestPredictionQuality:
    def test_parallel_lines_low_variance(self) -> None:
        """When all training trajectories are parallel, the predictive
        covariance should be small (they all agree on the target)."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        trajs = []
        for i in range(12):
            start = np.array([0.0, 0.0, 0.0]) + np.random.default_rng(i).standard_normal(D) * 0.002
            trajs.append(_make_trajectory(
                embryo_id=f"e{i}", n_frames=30, n_dim=D,
                direction=direction, start=start, seed=i + 100, noise_scale=0.001,
            ))
        pca = PCA(n_components=D); pca.fit(np.eye(D))
        scaler = StandardScaler(); scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=trajs, pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )
        pf = BranchingParticleFilter(
            ds, radius=1.0, min_points_in_radius=1, n_samples=50, exclude_self=False,
        )
        batch = _make_batch(ds, n=4)
        result = pf.predict(batch)
        # Variance should be small along the x-axis (direction of travel).
        assert result.predicted_cov_diag[:, 0].mean().item() < 0.5

    def test_beats_random_on_structured_data(self) -> None:
        """On smooth trajectories, particle filter should have lower MSE
        than a random Gaussian predictor."""
        from dev.dynamo.eval.predictions import GaussianNoisePredictor
        from dev.dynamo.eval.evaluate import run_evaluation

        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=15, n_frames=30, n_dim=D, direction=direction, noise_scale=0.005)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))

        pf = BranchingParticleFilter(
            ds, radius=None, n_samples=20, exclude_self=True,
        )
        random_pred = GaussianNoisePredictor(std=2.0)

        pf_result = run_evaluation(pf, fds, n_batches=8, batch_size=8)
        rand_result = run_evaluation(random_pred, fds, n_batches=8, batch_size=8)

        assert pf_result.metrics["mse"] < rand_result.metrics["mse"]


# ---------------------------------------------------------------------------
# Tests: bimodal / branching data
# ---------------------------------------------------------------------------

class TestBranchingData:
    def test_bimodal_distribution_is_multimodal(self) -> None:
        """On data with two branches, the forward_samples should span both.

        Design: both branches share a strong positive x-component so the
        directional filter keeps both when the query context moves in +x.
        Branch A diverges in +y, branch B in -y.  At the prediction horizon
        the y-coordinates should be bimodal → y_std > 0.05.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        D = 2
        n_per_branch = 10
        trajs = []

        # Both branches have positive x (aligns with +x query context).
        # They diverge in y: A goes up, B goes down.
        dir_a = np.array([1.0,  1.0])
        dir_b = np.array([1.0, -1.0])

        for i in range(n_per_branch):
            start = np.zeros(D) + np.random.default_rng(i).standard_normal(D) * 0.005
            trajs.append(_make_trajectory(
                embryo_id=f"a{i}", n_frames=30, n_dim=D,
                direction=dir_a, start=start, seed=i, noise_scale=0.002,
            ))
        for i in range(n_per_branch):
            start = np.zeros(D) + np.random.default_rng(i + 100).standard_normal(D) * 0.005
            trajs.append(_make_trajectory(
                embryo_id=f"b{i}", n_frames=30, n_dim=D,
                direction=dir_b, start=start, seed=i + 200, noise_scale=0.002,
            ))

        pca = PCA(n_components=D); pca.fit(np.eye(D))
        scaler = StandardScaler(); scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=trajs, pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )

        pf = BranchingParticleFilter(
            ds, radius=0.5, min_points_in_radius=1,
            n_samples=150, exclude_self=False,
            sigma_pos=0.25,
        )

        # Query context: moves along +x — aligns with both branches' x-component.
        context = np.column_stack([np.linspace(0, 0.08, 5), np.zeros(5)])
        delta_t = 300.0
        result = pf._predict_single(
            context=context, delta_t=delta_t,
            horizon_dt=6 * delta_t, embryo_idx=-1,
        )
        samples = result["samples"]  # (150, 2)

        # Verify both branches have non-trivial selection weight.
        particles = pf.select_references(context, delta_t, embryo_idx=-1)
        branch_a_ids = set(range(n_per_branch))
        branch_b_ids = set(range(n_per_branch, 2 * n_per_branch))
        w_a = sum(p.weight for p in particles if p.traj_id in branch_a_ids)
        w_b = sum(p.weight for p in particles if p.traj_id in branch_b_ids)
        assert w_a > 0.05, f"Branch A under-represented: w_a={w_a:.4f}"
        assert w_b > 0.05, f"Branch B under-represented: w_b={w_b:.4f}"

        # y-spread should indicate both branches are in the particle cloud.
        y_std = float(np.std(samples[:, 1]))
        assert y_std > 0.05, f"Expected bimodal y-spread, got y_std={y_std:.4f}"


# ---------------------------------------------------------------------------
# Tests: integration with eval pipeline
# ---------------------------------------------------------------------------

class TestEvalIntegration:
    def test_run_evaluation(self) -> None:
        from dev.dynamo.eval.evaluate import run_evaluation

        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))
        pf = BranchingParticleFilter(ds, radius=None, n_samples=10, exclude_self=False)

        result = run_evaluation(pf, fds, n_batches=5, batch_size=8)

        assert result.n_samples > 0
        assert "nll" in result.metrics
        assert "mse" in result.metrics
        assert 0.0 <= result.calibration <= 1.0
        assert len(result.per_horizon) >= 1

    def test_comparison_with_kernel_on_parallel_data(self) -> None:
        """On very structured parallel-line data, particle filter and kernel
        should both produce low MSE (either can win; we just check both run)."""
        from dev.dynamo.models.kernel import SimpleKernelPredictor
        from dev.dynamo.eval.evaluate import run_evaluation

        D = 3
        direction = np.array([1.0, 0.0, 0.0])
        ds = _make_dataset(n_embryos=12, n_frames=25, n_dim=D, direction=direction, noise_scale=0.005)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))

        pf = BranchingParticleFilter(ds, radius=None, n_samples=20, exclude_self=True)
        kernel = SimpleKernelPredictor(ds, bandwidth=None, exclude_self=True)

        pf_result = run_evaluation(pf, fds, n_batches=5, batch_size=8)
        kernel_result = run_evaluation(kernel, fds, n_batches=5, batch_size=8)

        # Both should produce finite metrics.
        assert np.isfinite(pf_result.metrics["mse"])
        assert np.isfinite(kernel_result.metrics["mse"])


# ---------------------------------------------------------------------------
# Local run_evaluation helper (avoids circular import at module level)
# ---------------------------------------------------------------------------

def run_evaluation(predictor, dataset, n_batches, batch_size):
    from dev.dynamo.eval.evaluate import run_evaluation as _run_eval
    return _run_eval(predictor, dataset, n_batches=n_batches, batch_size=batch_size)
