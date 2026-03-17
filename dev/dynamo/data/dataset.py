"""PyTorch Dataset implementing fragment sampling (model spec §7.2).

Each sample is a randomly drawn contiguous fragment from one embryo's
trajectory, paired with M target observations at randomly sampled
prediction horizons (teacher-forced: each target is scored as a single-step
transition from its observed predecessor).

Returned tensors:
    context      : (L, D)   PC-space trajectory fragment (the "observed" part)
    targets      : (M, D)   target PC-space vectors at M sampled horizons
    predecessors : (M, D)   observed predecessor for each target (teacher forcing)
    time_deltas  : (L-1,)   inter-frame Δt values within the context (seconds)
    horizon_dts  : (M,)     time gap from predecessor to target for each target
    delta_t      : scalar   experiment-level median Δt (seconds)
    temperature  : scalar   incubation temperature (°C, may be NaN)
    class_idx    : int      integer index for perturbation class
    embryo_idx   : int      index into the trajectory list

The custom collate function `fragment_collate_fn` pads variable-length
context fragments and returns a boolean mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .loading import TrajectoryDataset


class FragmentDataset(Dataset):
    """Dataset that samples random fragments + prediction horizons from embryo trajectories.

    Args:
        trajectory_dataset: Loaded TrajectoryDataset from loading.py.
        min_context: Minimum number of context frames (≥2 for at least one transition).
        max_context: Maximum context frames. None = use all available (minus horizon).
        horizons: Prediction horizons to sample from (in units of frames).
        epoch_length: Virtual epoch size. Since sampling is stochastic, this controls
            how many samples constitute one "epoch" for the DataLoader.
        gamma: Class-balanced sampling strength (spec §7.2). 0 = proportional to
            class frequency (no rebalancing), 1 = uniform across classes.
            Default 0.5 (square-root weighting).
        n_targets: Number of target transitions to sample per fragment (M in spec
            §7.3). Each target is teacher-forced: scored as a single-step transition
            from its observed predecessor. Default 1.
    """

    def __init__(
        self,
        trajectory_dataset: TrajectoryDataset,
        min_context: int = 2,
        max_context: Optional[int] = None,
        horizons: Sequence[int] = (1, 2, 3, 4),
        epoch_length: Optional[int] = None,
        gamma: float = 0.5,
        n_targets: int = 1,
    ) -> None:
        self.trajs = trajectory_dataset.trajectories
        self.class_to_idx = trajectory_dataset.class_to_idx
        self.n_components = trajectory_dataset.n_components
        self.min_context = min_context
        self.max_context = max_context
        self.horizons = list(horizons)
        self.max_horizon = max(self.horizons)

        # Pre-filter to trajectories long enough for at least min_context + 1 horizon
        min_len = self.min_context + 1
        self.valid_indices = [
            i for i, t in enumerate(self.trajs)
            if len(t.trajectory) >= min_len
        ]
        if not self.valid_indices:
            raise ValueError(
                f"No trajectories with ≥{min_len} frames "
                f"(min_context={min_context}, min horizon=1)"
            )
        self._epoch_length = epoch_length or len(self.valid_indices)
        self._rng = np.random.default_rng()
        self.n_targets = n_targets

        # Class-balanced sampling weights (spec §7.2)
        self._sampling_weights = self._compute_class_weights(gamma)

    def _compute_class_weights(self, gamma: float) -> np.ndarray:
        """Compute per-trajectory sampling weights from class frequencies.

        w_p ∝ (N_p / N_total)^(1 - gamma), then each trajectory in class p
        gets weight w_p / N_p so the per-trajectory probabilities sum to 1.

        Args:
            gamma: Balancing strength. 0 = natural frequencies, 1 = uniform classes.

        Returns:
            (len(valid_indices),) normalized probability array.
        """
        # Count class frequencies among valid trajectories
        class_counts: Dict[str, int] = {}
        for i in self.valid_indices:
            cls = self.trajs[i].perturbation_class
            class_counts[cls] = class_counts.get(cls, 0) + 1
        n_total = len(self.valid_indices)

        # Per-class weight: (N_p / N_total)^(1-gamma)
        class_weight = {
            cls: (count / n_total) ** (1.0 - gamma)
            for cls, count in class_counts.items()
        }

        # Per-trajectory weight: class_weight / N_p (uniform within class)
        weights = np.array([
            class_weight[self.trajs[i].perturbation_class] / class_counts[self.trajs[i].perturbation_class]
            for i in self.valid_indices
        ], dtype=np.float64)

        # Normalize to probability distribution
        weights /= weights.sum()
        return weights

    def __len__(self) -> int:
        return self._epoch_length

    def seed_worker(self, seed: int) -> None:
        """Re-seed the RNG (call from DataLoader worker_init_fn)."""
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample a random fragment + M target transitions from a random trajectory.

        The ``idx`` argument is used only to drive the DataLoader iteration
        count; the actual embryo and fragment are sampled randomly.

        Each of the M targets is teacher-forced: we return both the target
        point and its observed predecessor, so the model scores a single-step
        transition from real data (never chaining its own predictions).
        """
        rng = self._rng

        # Pick a random trajectory (class-balanced)
        sel = rng.choice(len(self.valid_indices), p=self._sampling_weights)
        traj_idx = self.valid_indices[sel]
        traj = self.trajs[traj_idx]
        T = len(traj.trajectory)

        # Maximum horizon that fits
        feasible_horizons = [k for k in self.horizons if T >= self.min_context + k]
        if not feasible_horizons:
            feasible_horizons = [1]
        k_max_feasible = max(feasible_horizons)

        # Determine context length bounds (leave room for largest feasible horizon)
        max_ctx = T - k_max_feasible
        if self.max_context is not None:
            max_ctx = min(max_ctx, self.max_context)
        max_ctx = max(max_ctx, self.min_context)
        ctx_len = int(rng.integers(self.min_context, max_ctx + 1))

        # Pick random start position
        latest_start = T - ctx_len - 1  # need at least 1 future frame
        start = int(rng.integers(0, latest_start + 1))
        context_end = start + ctx_len  # exclusive; this is t_n

        # Target pool: indices from t_n+1 to min(T-1, t_n + K_max)
        target_pool_end = min(T, context_end + self.max_horizon)
        target_pool = list(range(context_end, target_pool_end))

        # Sample M targets (with replacement if pool < M)
        M = self.n_targets
        if len(target_pool) >= M:
            target_indices = rng.choice(target_pool, size=M, replace=False)
        else:
            target_indices = rng.choice(target_pool, size=M, replace=True)
        target_indices = np.sort(target_indices)

        # Extract arrays
        context = traj.trajectory[start:context_end]                     # (L, D)
        targets = traj.trajectory[target_indices]                         # (M, D)
        predecessors = traj.trajectory[target_indices - 1]                # (M, D)
        time_ctx = traj.time_seconds[start:context_end]                   # (L,)

        # Inter-frame deltas within context
        time_deltas = np.diff(time_ctx)                                   # (L-1,)

        # Per-target: time from predecessor to target (single-step dt for teacher forcing)
        horizon_dts = (traj.time_seconds[target_indices]
                       - traj.time_seconds[target_indices - 1])           # (M,)

        # Time from last context frame to each target (for eval/baselines)
        context_to_target_dts = (traj.time_seconds[target_indices]
                                 - time_ctx[-1])                          # (M,)

        class_idx = self.class_to_idx.get(traj.perturbation_class, -1)

        return {
            "context": torch.from_numpy(context).float(),                # (L, D)
            "targets": torch.from_numpy(targets).float(),                # (M, D)
            "predecessors": torch.from_numpy(predecessors).float(),      # (M, D)
            "time_deltas": torch.from_numpy(time_deltas).float(),        # (L-1,)
            "horizon_dts": torch.from_numpy(horizon_dts).float(),        # (M,)
            "context_to_target_dts": torch.from_numpy(
                context_to_target_dts).float(),                          # (M,)
            "delta_t": torch.tensor(traj.delta_t, dtype=torch.float32),
            "temperature": torch.tensor(traj.temperature, dtype=torch.float32),
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
            "embryo_idx": torch.tensor(traj_idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

@dataclass
class FragmentBatch:
    """Padded batch of fragments with M teacher-forced targets.

    Attributes:
        context: (B, L_max, D) padded context trajectories.
        context_mask: (B, L_max) boolean mask — True for real frames.
        targets: (B, M, D) target vectors.
        predecessors: (B, M, D) observed predecessor for each target (teacher forcing).
        time_deltas: (B, L_max-1) padded inter-frame Δt values.
        horizon_dts: (B, M) single-step dt from predecessor to target.
        context_to_target_dts: (B, M) time from last context frame to each target.
        delta_t: (B,) experiment-level median Δt.
        temperature: (B,) incubation temperatures.
        class_idx: (B,) perturbation class indices.
        embryo_idx: (B,) trajectory indices.
    """
    context: torch.Tensor
    context_mask: torch.Tensor
    targets: torch.Tensor
    predecessors: torch.Tensor
    time_deltas: torch.Tensor
    horizon_dts: torch.Tensor
    context_to_target_dts: torch.Tensor
    delta_t: torch.Tensor
    temperature: torch.Tensor
    class_idx: torch.Tensor
    embryo_idx: torch.Tensor

    @property
    def target(self) -> torch.Tensor:
        """First target vector (B, D). Backward-compat for single-target code."""
        return self.targets[:, 0]

    @property
    def horizon_dt(self) -> torch.Tensor:
        """Time from last context frame to first target (B,). Backward-compat."""
        return self.context_to_target_dts[:, 0]


def fragment_collate_fn(samples: List[Dict[str, torch.Tensor]]) -> FragmentBatch:
    """Collate variable-length fragments into a padded batch."""
    B = len(samples)
    D = samples[0]["context"].shape[-1]
    lengths = [s["context"].shape[0] for s in samples]
    L_max = max(lengths)

    context = torch.zeros(B, L_max, D)
    context_mask = torch.zeros(B, L_max, dtype=torch.bool)
    time_deltas = torch.zeros(B, L_max - 1)

    for i, s in enumerate(samples):
        L = lengths[i]
        context[i, :L] = s["context"]
        context_mask[i, :L] = True
        td = s["time_deltas"]
        time_deltas[i, :len(td)] = td

    return FragmentBatch(
        context=context,
        context_mask=context_mask,
        targets=torch.stack([s["targets"] for s in samples]),
        predecessors=torch.stack([s["predecessors"] for s in samples]),
        time_deltas=time_deltas,
        horizon_dts=torch.stack([s["horizon_dts"] for s in samples]),
        context_to_target_dts=torch.stack([s["context_to_target_dts"] for s in samples]),
        delta_t=torch.stack([s["delta_t"] for s in samples]),
        temperature=torch.stack([s["temperature"] for s in samples]),
        class_idx=torch.stack([s["class_idx"] for s in samples]),
        embryo_idx=torch.stack([s["embryo_idx"] for s in samples]),
    )


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's RNG independently."""
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.seed_worker(torch.initial_seed() % (2**32) + worker_id)
