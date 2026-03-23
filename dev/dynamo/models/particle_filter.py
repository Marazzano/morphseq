"""Branching particle filter baseline (model spec §9.2).

Follows matched reference trajectories forward, dynamically recruits new
references at each step, and preserves multimodal predictions through
bifurcations. More complex than simple kernel regression but substantially
more powerful: it is direction-aware, handles rate variation, and maintains
independent branch pools so bifurcations remain bimodal rather than being
smeared.

Implemented in three layers (spec §15.2 build sequence step 4):

  Layer 1 -- Reference selection (§9.2.1):
    Spatial filter → local linear fits → direction-aware weighting →
    speed-ratio computation → anchor assignment.

  Layer 2 -- Forward prediction without recruitment (§9.2.2):
    Time-stepped particle tracking in developmental-progress units.
    Particle death when trajectory runs out.

  Layer 3 -- Recruitment (§9.2.2):
    At each step, each active particle recruits nearby trajectories.
    Weights are inherited from the recruiting particle scaled by proximity.
    Recruitment is local per-particle, never against a global centroid.
    This preserves multimodal distributions at bifurcations.

Key invariant: a recruited particle's speed ratio is always computed as
  v_query / v_ref_local
where v_query is the (fixed) average speed of the query context window and
v_ref_local is the local speed of the new reference near its anchor point.
This maintains developmental-progress matching through any chain of
recruitments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..data.dataset import FragmentBatch
from ..data.loading import EmbryoTrajectory, TrajectoryDataset
from ..eval.predictions import PredictionResult


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Particle:
    """One active particle in the forward prediction.

    Attributes:
        traj_id: Index into bank.traj_points.
        anchor_frame: Frame in that trajectory when this particle was
            "born" (at query step start_step).
        speed_ratio: rho = v_query / v_ref.  At each query step the
            reference advances rho * dt_query / dt_ref reference frames.
        weight: Current (unnormalized) particle weight.
        dt_ref: Time step of this reference trajectory (seconds).
        start_step: Query step at which this particle was created
            (0 for initial particles, k for particles recruited at step k).
    """
    traj_id: int
    anchor_frame: int
    speed_ratio: float
    weight: float
    dt_ref: float
    start_step: int = 0


@dataclass
class _ReferenceBank:
    """Pre-concatenated training data for fast vectorised lookup.

    Attributes:
        all_points: (N_total, D) all training points concatenated.
        traj_id: (N_total,) which trajectory each point belongs to.
        frame_idx: (N_total,) frame index within its trajectory.
        traj_lengths: (N_traj,) length of each trajectory.
        traj_points: List of (T_i, D) arrays per trajectory.
        traj_delta_t: (N_traj,) Δt per trajectory (seconds).
        embryo_idx: (N_traj,) original index into trajectory list.
        experiment_id: List of experiment_id strings per trajectory.
        class_idx: (N_traj,) perturbation class index per trajectory.
    """
    all_points: np.ndarray
    traj_id: np.ndarray
    frame_idx: np.ndarray
    traj_lengths: np.ndarray
    traj_points: List[np.ndarray]
    traj_delta_t: np.ndarray
    embryo_idx: np.ndarray
    experiment_id: List[str]
    class_idx: np.ndarray


# ---------------------------------------------------------------------------
# Reference bank construction
# ---------------------------------------------------------------------------

def _build_reference_bank(
    trajectories: List[EmbryoTrajectory],
    class_to_idx: Dict[str, int],
) -> _ReferenceBank:
    """Build a pre-concatenated reference bank from a trajectory list.

    Args:
        trajectories: List of EmbryoTrajectory objects.
        class_to_idx: Mapping from perturbation class name to integer index.

    Returns:
        Populated _ReferenceBank.
    """
    all_pts: List[np.ndarray] = []
    all_traj_id: List[np.ndarray] = []
    all_frame_idx: List[np.ndarray] = []
    traj_points: List[np.ndarray] = []
    traj_lengths: List[int] = []
    traj_delta_t: List[float] = []
    embryo_idx: List[int] = []
    experiment_ids: List[str] = []
    class_idxs: List[int] = []

    for i, traj in enumerate(trajectories):
        T = len(traj.trajectory)
        all_pts.append(traj.trajectory)
        all_traj_id.append(np.full(T, i, dtype=np.int64))
        all_frame_idx.append(np.arange(T, dtype=np.int64))
        traj_points.append(traj.trajectory)
        traj_lengths.append(T)
        traj_delta_t.append(traj.delta_t)
        embryo_idx.append(i)
        experiment_ids.append(traj.experiment_id)
        class_idxs.append(class_to_idx.get(traj.perturbation_class, -1))

    return _ReferenceBank(
        all_points=np.concatenate(all_pts, axis=0),
        traj_id=np.concatenate(all_traj_id),
        frame_idx=np.concatenate(all_frame_idx),
        traj_lengths=np.array(traj_lengths, dtype=np.int64),
        traj_points=traj_points,
        traj_delta_t=np.array(traj_delta_t, dtype=np.float64),
        embryo_idx=np.array(embryo_idx, dtype=np.int64),
        experiment_id=experiment_ids,
        class_idx=np.array(class_idxs, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _linear_fit(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a linear trend through a set of points via SVD.

    Args:
        points: (N, D) array; N ≥ 1.

    Returns:
        midpoint: (D,) centroid.
        direction: (D,) first right singular vector (principal direction).
            Zero vector if N < 2 or points are degenerate.
    """
    midpoint = points.mean(axis=0)
    if len(points) < 2:
        return midpoint, np.zeros(points.shape[1])
    centered = points - midpoint
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    # If the first singular value is negligible, direction is undefined.
    if s[0] < 1e-12:
        return midpoint, np.zeros(points.shape[1])
    return midpoint, Vt[0]


def _interpolate_frame(traj: np.ndarray, frame_float: float) -> Optional[np.ndarray]:
    """Linearly interpolate a trajectory at a fractional frame index.

    Args:
        traj: (T, D) trajectory array.
        frame_float: Fractional frame index (may be non-integer).

    Returns:
        Interpolated (D,) position, or None if out of bounds.
    """
    T = len(traj)
    if frame_float < 0:
        return None
    if frame_float >= T - 1:
        # Allow landing exactly on or just past the last frame.
        if frame_float <= T - 1 + 1e-9:
            return traj[T - 1].copy()
        return None
    f_lo = int(np.floor(frame_float))
    f_hi = f_lo + 1
    if f_hi >= T:
        return traj[T - 1].copy()
    alpha = frame_float - f_lo
    return (1.0 - alpha) * traj[f_lo] + alpha * traj[f_hi]


def _local_speed(
    traj: np.ndarray,
    frame_indices: np.ndarray,
    dt: float,
) -> float:
    """Compute mean per-frame speed (latent-distance / second) near given frames.

    Uses consecutive pairs from the sorted unique set of frame_indices.
    Falls back to the global trajectory average if fewer than two frames are
    available.

    Args:
        traj: (T, D) trajectory.
        frame_indices: Frame indices to consider (within-radius frames).
        dt: Trajectory time step (seconds).

    Returns:
        Mean speed in latent-distance per second (always positive).
    """
    eps = max(dt, 1e-6)
    sorted_frames = np.unique(frame_indices)
    speeds = []
    for f in sorted_frames:
        if int(f) + 1 < len(traj):
            speeds.append(float(np.linalg.norm(traj[int(f) + 1] - traj[int(f)])) / eps)
    if speeds:
        return float(np.mean(speeds))
    # Global fallback
    if len(traj) >= 2:
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).mean()) / eps
    return 1.0


# ---------------------------------------------------------------------------
# Auto-calibration helpers
# ---------------------------------------------------------------------------

def _auto_radius(
    bank: _ReferenceBank,
    n_samples: int = 500,
    seed: int = 42,
) -> float:
    """Estimate a reasonable spatial radius via median pairwise distance.

    Subsamples points and computes pairwise distances on a random subset
    of pairs, returning the median as the radius estimate.

    Args:
        bank: Pre-built reference bank.
        n_samples: Number of points to subsample.
        seed: RNG seed.

    Returns:
        Estimated radius (positive scalar).
    """
    rng = np.random.default_rng(seed)
    N = len(bank.all_points)
    n = min(n_samples, N)
    idx = rng.choice(N, size=n, replace=False)
    pts = bank.all_points[idx]
    n_pairs = min(10_000, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    dists = np.linalg.norm(pts[i_idx] - pts[j_idx], axis=1)
    return float(np.median(dists))


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class BranchingParticleFilter:
    """Branching particle filter baseline (spec §9.2).

    Maintains a weighted set of reference trajectory fragments and steps
    them forward in developmental-progress-matched time.  At each step each
    active particle can recruit new reference trajectories near its current
    position, inheriting a fraction of its weight.  Recruitment is strictly
    local (per-particle), so two branches of a bifurcation maintain
    completely independent particle pools and the final predictive
    distribution is naturally multimodal.

    Args:
        dataset: TrajectoryDataset containing training trajectories.
        radius: Spatial filter radius R (latent-space units).
            None → auto-calibrated via median pairwise distance.
        min_points_in_radius: Minimum number of reference-trajectory frames
            that must fall within R for that trajectory to qualify as a
            reference during initial selection.
        context_window: Number of most-recent context frames used when
            fitting the query direction vector.
        sigma_pos: Positional kernel bandwidth for reference weighting.
            None → radius / 2.
        directional_alpha: Exponent α on the cosine similarity term.
            Higher values penalise directional mismatch more strongly.
        sigma_recruit: Gaussian bandwidth for recruitment weighting.
            None → same as sigma_pos.
        n_max_particles: Hard cap on the active particle set; excess
            particles are pruned by weight (top-N) at each step.
        n_samples: Number of forward samples to return per query.
        exclude_self: Exclude the query embryo's own trajectory.
        renorm_every: Renormalise particle weights every this many steps
            to prevent underflow.
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        radius: Optional[float] = None,
        min_points_in_radius: int = 3,
        context_window: int = 7,
        sigma_pos: Optional[float] = None,
        directional_alpha: float = 2.0,
        sigma_recruit: Optional[float] = None,
        n_max_particles: int = 200,
        n_samples: int = 100,
        exclude_self: bool = True,
        renorm_every: int = 5,
    ) -> None:
        self.trajectories = dataset.trajectories
        self.class_to_idx = dataset.class_to_idx
        self.bank = _build_reference_bank(self.trajectories, self.class_to_idx)
        self.min_points_in_radius = min_points_in_radius
        self.context_window = context_window
        self.directional_alpha = directional_alpha
        self.n_max_particles = n_max_particles
        self.n_samples = n_samples
        self.exclude_self = exclude_self
        self.renorm_every = renorm_every
        self._rng = np.random.default_rng()

        self.radius = radius if radius is not None else _auto_radius(self.bank)
        self.sigma_pos = sigma_pos if sigma_pos is not None else self.radius / 2.0
        self.sigma_recruit = sigma_recruit if sigma_recruit is not None else self.sigma_pos

    # ------------------------------------------------------------------
    # Public interface (Predictor protocol)
    # ------------------------------------------------------------------

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        """Produce predictions for a batch of fragments.

        Args:
            batch: FragmentBatch from the data pipeline.

        Returns:
            PredictionResult with predicted_mean (B, D), predicted_cov_diag
            (B, D), and forward_samples (B, n_samples, D).
        """
        B = batch.context.shape[0]
        D = batch.context.shape[2]

        means = np.zeros((B, D), dtype=np.float64)
        cov_diags = np.zeros((B, D), dtype=np.float64)
        all_samples = np.zeros((B, self.n_samples, D), dtype=np.float64)

        for b in range(B):
            L = int(batch.context_mask[b].sum().item())
            context = batch.context[b, :L].numpy()        # (L, D)
            delta_t = float(batch.delta_t[b].item())
            horizon_dt = float(batch.horizon_dt[b].item())
            embryo_idx = int(batch.embryo_idx[b].item())

            result = self._predict_single(
                context=context,
                delta_t=delta_t,
                horizon_dt=horizon_dt,
                embryo_idx=embryo_idx,
            )
            means[b] = result["mean"]
            cov_diags[b] = result["cov_diag"]
            all_samples[b] = result["samples"]

        return PredictionResult(
            predicted_mean=torch.from_numpy(means).float(),
            predicted_cov_diag=torch.from_numpy(cov_diags).float(),
            forward_samples=torch.from_numpy(all_samples).float(),
        )

    # ------------------------------------------------------------------
    # Per-sample prediction
    # ------------------------------------------------------------------

    def _predict_single(
        self,
        context: np.ndarray,
        delta_t: float,
        horizon_dt: float,
        embryo_idx: int,
    ) -> dict:
        """Full prediction pipeline for one query.

        Args:
            context: (L, D) observed context frames (real frames only, no padding).
            delta_t: Query experiment time step (seconds).
            horizon_dt: Time gap from last context frame to target (seconds).
            embryo_idx: Index of query embryo (used for exclude_self).

        Returns:
            Dict with keys 'mean' (D,), 'cov_diag' (D,), 'samples' (n_samples, D).
        """
        D = context.shape[1]
        query_point = context[-1]  # (D,)

        # Number of query-time steps to the prediction horizon.
        n_steps = max(1, int(round(horizon_dt / max(delta_t, 1e-6))))

        # Query context speed: mean ||z_{t+1} - z_t|| / dt over context window.
        if len(context) >= 2:
            diffs = np.linalg.norm(np.diff(context, axis=0), axis=1)
            query_speed = float(diffs.mean() / max(delta_t, 1e-6))
        else:
            query_speed = 1.0

        # ---- Layer 1: reference selection --------------------------------
        particles = self._select_references(
            context=context,
            query_point=query_point,
            delta_t=delta_t,
            embryo_idx=embryo_idx,
            query_speed=query_speed,
        )
        if not particles:
            return self._fallback(query_point, D)

        # ---- Layers 2+3: forward prediction with recruitment -------------
        particles = self._forward_predict(
            particles=particles,
            n_steps=n_steps,
            delta_t=delta_t,
            embryo_idx=embryo_idx,
            query_speed=query_speed,
        )
        if not particles:
            return self._fallback(query_point, D)

        # ---- Collect final positions -------------------------------------
        positions, weights = self._collect_positions(particles, n_steps, delta_t)
        if positions is None or len(positions) == 0:
            return self._fallback(query_point, D)

        w_sum = weights.sum()
        if w_sum < 1e-12:
            return self._fallback(query_point, D)

        w_norm = weights / w_sum
        mean = w_norm @ positions                                    # (D,)
        diff = positions - mean                                      # (N, D)
        cov_diag = (w_norm[:, None] * diff ** 2).sum(axis=0)        # (D,)
        cov_diag = np.maximum(cov_diag, 1e-8)

        indices = self._rng.choice(
            len(positions), size=self.n_samples, replace=True, p=w_norm,
        )
        samples = positions[indices]                                 # (n_samples, D)

        return {"mean": mean, "cov_diag": cov_diag, "samples": samples}

    # ------------------------------------------------------------------
    # Layer 1: reference selection
    # ------------------------------------------------------------------

    def _select_references(
        self,
        context: np.ndarray,
        query_point: np.ndarray,
        delta_t: float,
        embryo_idx: int,
        query_speed: float,
    ) -> List[_Particle]:
        """Select and weight reference trajectories near the query point.

        Implements spec §9.2.1 steps 1–6.

        Args:
            context: (L, D) query context frames.
            query_point: (D,) last context frame (= context[-1]).
            delta_t: Query time step (seconds).
            embryo_idx: Query embryo index (for exclude_self).
            query_speed: Mean speed of query in latent-distance / second.

        Returns:
            List of _Particle objects with normalised weights.
            Empty list if no eligible references found.
        """
        # Fit query direction from the last context_window frames.
        W = min(self.context_window, len(context))
        q_mid, q_dir = _linear_fit(context[-W:])

        # Find all reference points within radius R of the query point.
        dists_sq = np.sum((self.bank.all_points - query_point) ** 2, axis=1)
        within = dists_sq < self.radius ** 2

        if self.exclude_self:
            within &= (self.bank.traj_id != embryo_idx)

        if not within.any():
            return []

        traj_ids_in_radius = np.unique(self.bank.traj_id[within])
        particles: List[_Particle] = []

        for tid in traj_ids_in_radius:
            tid_and_within = (self.bank.traj_id == tid) & within
            frames_in_radius = self.bank.frame_idx[tid_and_within]

            if len(frames_in_radius) < self.min_points_in_radius:
                continue

            traj = self.bank.traj_points[int(tid)]
            dt_ref = float(self.bank.traj_delta_t[int(tid)])
            pts_in_radius = traj[frames_in_radius]     # (N_r, D)

            # Local linear fit for this reference.
            ref_mid, ref_dir = _linear_fit(pts_in_radius)

            # Anchor: frame closest to query_point.
            dists_to_q = np.linalg.norm(pts_in_radius - query_point, axis=1)
            anchor_local = int(np.argmin(dists_to_q))
            anchor_frame = int(frames_in_radius[anchor_local])

            # Must have at least one future frame from the anchor.
            if anchor_frame + 1 >= len(traj):
                continue

            # Positional weight.
            pos_dist_sq = float(np.sum((ref_mid - q_mid) ** 2))
            w_pos = np.exp(-pos_dist_sq / (2.0 * max(self.sigma_pos, 1e-12) ** 2))

            # Directional weight: cosine similarity raised to alpha, clamped ≥ 0.
            q_norm = float(np.linalg.norm(q_dir))
            r_norm = float(np.linalg.norm(ref_dir))
            if q_norm > 1e-8 and r_norm > 1e-8:
                cos_sim = float(np.dot(q_dir, ref_dir) / (q_norm * r_norm))
                w_dir = float(max(0.0, cos_sim) ** self.directional_alpha)
            else:
                w_dir = 1.0  # degenerate: no direction information

            weight = w_pos * w_dir
            if weight < 1e-15:
                continue

            # Speed ratio: v_query / v_ref_local.
            ref_speed = _local_speed(traj, frames_in_radius, dt_ref)
            speed_ratio = query_speed / max(ref_speed, 1e-8)

            particles.append(_Particle(
                traj_id=int(tid),
                anchor_frame=anchor_frame,
                speed_ratio=speed_ratio,
                weight=weight,
                dt_ref=dt_ref,
                start_step=0,
            ))

        # Normalise initial weights.
        if particles:
            total_w = sum(p.weight for p in particles)
            if total_w > 1e-12:
                for p in particles:
                    p.weight /= total_w

        return particles

    # ------------------------------------------------------------------
    # Layers 2 + 3: forward prediction
    # ------------------------------------------------------------------

    def _forward_predict(
        self,
        particles: List[_Particle],
        n_steps: int,
        delta_t: float,
        embryo_idx: int,
        query_speed: float,
    ) -> List[_Particle]:
        """Advance particles forward, recruiting new ones at each step.

        Implements spec §9.2.2.  The active set is stepped from k=1 to
        k=n_steps.  Particles whose trajectory ends before reaching the
        current step are removed (particle death).  Surviving particles
        recruit new candidates from nearby trajectories (particle
        recruitment).  The set is capped at n_max_particles at each step.

        Args:
            particles: Initial particle set from _select_references.
            n_steps: Total number of query-time steps to advance.
            delta_t: Query time step (seconds).
            embryo_idx: Query embryo index (for exclude_self).
            query_speed: Query speed for computing new speed ratios.

        Returns:
            Final active particle list (positions at step n_steps are
            recoverable via _particle_position).
        """
        active = list(particles)

        for k in range(1, n_steps + 1):
            # --- Advance and filter dead particles ---
            alive: List[_Particle] = []
            alive_positions: List[np.ndarray] = []

            for p in active:
                pos = self._particle_position(p, k, delta_t)
                if pos is not None:
                    alive.append(p)
                    alive_positions.append(pos)

            if not alive:
                break

            # --- Recruitment (skip at the final step, nothing to follow after) ---
            if k < n_steps:
                active_traj_ids: Set[int] = {p.traj_id for p in alive}
                # Accumulate recruited weights per trajectory across all recruiting particles.
                recruited_w: Dict[int, float] = {}
                recruited_meta: Dict[int, Tuple[int, float, float]] = {}  # tid → (anchor, speed_ratio, dt_ref)

                for pos, p in zip(alive_positions, alive):
                    new_ps = self._recruit_from(
                        position=pos,
                        parent_weight=p.weight,
                        current_step=k,
                        delta_t=delta_t,
                        embryo_idx=embryo_idx,
                        exclude_traj_ids=active_traj_ids,
                        query_speed=query_speed,
                    )
                    for rp in new_ps:
                        if rp.traj_id in recruited_w:
                            recruited_w[rp.traj_id] += rp.weight
                        else:
                            recruited_w[rp.traj_id] = rp.weight
                            recruited_meta[rp.traj_id] = (rp.anchor_frame, rp.speed_ratio, rp.dt_ref)

                for tid, w in recruited_w.items():
                    anchor, sr, dtr = recruited_meta[tid]
                    alive.append(_Particle(
                        traj_id=tid,
                        anchor_frame=anchor,
                        speed_ratio=sr,
                        weight=w,
                        dt_ref=dtr,
                        start_step=k,
                    ))

            # --- Renormalise ---
            if k % self.renorm_every == 0 or k == n_steps:
                total_w = sum(p.weight for p in alive)
                if total_w > 1e-12:
                    for p in alive:
                        p.weight /= total_w

            # --- Cap ---
            if len(alive) > self.n_max_particles:
                alive.sort(key=lambda p: p.weight, reverse=True)
                alive = alive[:self.n_max_particles]
                total_w = sum(p.weight for p in alive)
                if total_w > 1e-12:
                    for p in alive:
                        p.weight /= total_w

            active = alive

        return active

    def _particle_position(
        self,
        p: _Particle,
        step: int,
        delta_t: float,
    ) -> Optional[np.ndarray]:
        """Return the position of particle p at query step `step`.

        At step `step`, the particle has been alive for
        (step - p.start_step) query-time steps since its anchor.
        Frame offset in the reference timeline:
            Δframe = ρ · (step - start_step) · dt_query / dt_ref

        Args:
            p: Particle to evaluate.
            step: Current query step.
            delta_t: Query time step (seconds).

        Returns:
            Interpolated (D,) position, or None if out of bounds.
        """
        elapsed = step - p.start_step
        frame_offset = p.speed_ratio * elapsed * delta_t / max(p.dt_ref, 1e-6)
        frame_float = p.anchor_frame + frame_offset
        return _interpolate_frame(self.bank.traj_points[p.traj_id], frame_float)

    # ------------------------------------------------------------------
    # Layer 3: recruitment helper
    # ------------------------------------------------------------------

    def _recruit_from(
        self,
        position: np.ndarray,
        parent_weight: float,
        current_step: int,
        delta_t: float,
        embryo_idx: int,
        exclude_traj_ids: Set[int],
        query_speed: float,
    ) -> List[_Particle]:
        """Find candidate trajectories near `position` and create new particles.

        Implements spec §9.2.2 particle recruitment.  Each eligible
        trajectory j creates at most one new particle, with inherited weight:
            w_j = parent_weight · exp(-||z_j - position||² / (2σ_recruit²))

        Speed ratio for j is v_query / v_j_local (not relative to parent),
        which keeps developmental-progress matching relative to the fixed
        query timeline.

        Args:
            position: (D,) current position of the recruiting particle.
            parent_weight: Weight of the recruiting particle.
            current_step: Query step at which recruitment happens (= start_step
                for the new particles).
            delta_t: Query time step (seconds).
            embryo_idx: Query embryo index.
            exclude_traj_ids: Trajectory IDs already active (skip them).
            query_speed: Global query speed for computing speed ratios.

        Returns:
            List of new _Particle objects (weights unnormalised).
        """
        dists_sq = np.sum((self.bank.all_points - position) ** 2, axis=1)
        within = dists_sq < self.radius ** 2

        if self.exclude_self:
            within &= (self.bank.traj_id != embryo_idx)

        if not within.any():
            return []

        new_particles: List[_Particle] = []
        traj_ids = np.unique(self.bank.traj_id[within])

        for tid in traj_ids:
            if int(tid) in exclude_traj_ids:
                continue

            tid_and_within = (self.bank.traj_id == tid) & within
            frames_in_radius = self.bank.frame_idx[tid_and_within]
            if len(frames_in_radius) == 0:
                continue

            traj = self.bank.traj_points[int(tid)]
            dt_ref = float(self.bank.traj_delta_t[int(tid)])
            pts = traj[frames_in_radius]

            # Anchor: closest frame to position.
            d2 = np.sum((pts - position) ** 2, axis=1)
            anchor_local = int(np.argmin(d2))
            anchor_frame = int(frames_in_radius[anchor_local])

            # Must extend forward at least one frame.
            if anchor_frame + 1 >= len(traj):
                continue

            anchor_pos = traj[anchor_frame]
            dist_sq = float(np.sum((anchor_pos - position) ** 2))
            sigma_sq = max(self.sigma_recruit, 1e-12) ** 2
            w = parent_weight * np.exp(-dist_sq / (2.0 * sigma_sq))
            if w < 1e-15:
                continue

            ref_speed = _local_speed(traj, frames_in_radius, dt_ref)
            speed_ratio = query_speed / max(ref_speed, 1e-8)

            new_particles.append(_Particle(
                traj_id=int(tid),
                anchor_frame=anchor_frame,
                speed_ratio=speed_ratio,
                weight=w,
                dt_ref=dt_ref,
                start_step=current_step,
            ))

        return new_particles

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _collect_positions(
        self,
        particles: List[_Particle],
        n_steps: int,
        delta_t: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Collect particle positions and weights at the final step.

        Args:
            particles: Final active particle list.
            n_steps: Total steps taken.
            delta_t: Query time step.

        Returns:
            Tuple of (positions (N, D), weights (N,)).  Both None if no
            valid particle has a position at n_steps.
        """
        positions = []
        weights = []
        for p in particles:
            pos = self._particle_position(p, n_steps, delta_t)
            if pos is not None:
                positions.append(pos)
                weights.append(p.weight)
        if not positions:
            return None, None
        pos_arr = np.array(positions, dtype=np.float64)      # (N, D)
        w_arr = np.array(weights, dtype=np.float64)           # (N,)
        return pos_arr, w_arr

    def _fallback(self, query_point: np.ndarray, D: int) -> dict:
        """Persistence fallback when no valid references or particles exist."""
        return {
            "mean": query_point.copy(),
            "cov_diag": np.ones(D, dtype=np.float64),
            "samples": np.tile(query_point, (self.n_samples, 1)),
        }

    # ------------------------------------------------------------------
    # Diagnostic helpers (used by notebook / tests)
    # ------------------------------------------------------------------

    def select_references(
        self,
        context: np.ndarray,
        delta_t: float,
        embryo_idx: int = -1,
    ) -> List[_Particle]:
        """Public wrapper around Layer 1 for inspection and testing.

        Args:
            context: (L, D) observed context frames.
            delta_t: Query time step (seconds).
            embryo_idx: Query embryo index (-1 = no exclusion).

        Returns:
            List of initial _Particle objects with normalised weights.
        """
        if len(context) >= 2:
            diffs = np.linalg.norm(np.diff(context, axis=0), axis=1)
            query_speed = float(diffs.mean() / max(delta_t, 1e-6))
        else:
            query_speed = 1.0
        return self._select_references(
            context=context,
            query_point=context[-1],
            delta_t=delta_t,
            embryo_idx=embryo_idx,
            query_speed=query_speed,
        )
