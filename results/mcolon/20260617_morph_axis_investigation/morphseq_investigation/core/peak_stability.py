"""Peak-count stability vote objects (COMPOSE_single_path_plan.md Sec 1.5, 1.5d).

Stage 2a scope only (`ResolutionStrategy.MODE_VOTE_FULL_DATA` +
`BootstrapRetention.SUMMARY_ONLY`, plan Part 4). This module holds:

- `PeakCountVote`       raw bootstrap-draw evidence (sparse peak-count histogram)
- `PeakCountStabilityPolicy` the swappable "is this vote stable" threshold
- `PeakCountStability`  the derived interpretation of a vote under a policy
- `PeakSeed` / `PeakSeedSet`   the target-count consensus locations built from
                               per-draw candidate centers (the vote gives a
                               COUNT, not locations -- plan Sec 1.5d)

Deliberately NOT built here (later stages per plan Part 4): `BootstrapPeakCandidate`
persistence with `(draw, candidate)` keys (Stage 2b), `PeakStabilityGraph` /
`PeakNode` / `PeakStabilityGraphSpec` (Stage 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PeakCountVote:
    """Raw bootstrap-draw evidence: a sparse histogram of observed peak
    counts across valid draws. Only OBSERVED counts appear as keys -- no
    zero-frequency entries (plan Sec 1.5)."""

    peak_count_frequencies: Mapping[int, int]
    n_draws_requested: int
    n_draws_valid: int
    sample_fraction: float

    def __post_init__(self) -> None:
        frequencies = {int(k): int(v) for k, v in dict(self.peak_count_frequencies).items()}
        object.__setattr__(self, "peak_count_frequencies", frequencies)
        if self.n_draws_requested < 0:
            raise ValueError("n_draws_requested must be non-negative.")
        if self.n_draws_valid < 0 or self.n_draws_valid > self.n_draws_requested:
            raise ValueError("n_draws_valid must be in [0, n_draws_requested].")
        if sum(frequencies.values()) != self.n_draws_valid:
            raise ValueError(
                "peak_count_frequencies must sum to n_draws_valid "
                f"(got {sum(frequencies.values())}, expected {self.n_draws_valid})."
            )

    def frequency_for(self, count: int) -> int:
        return int(self.peak_count_frequencies.get(int(count), 0))

    def probability_for(self, count: int) -> float:
        if self.n_draws_valid <= 0:
            return float("nan")
        return float(self.frequency_for(count)) / float(self.n_draws_valid)


@dataclass(frozen=True)
class PeakCountStabilityPolicy:
    """Swappable "is the vote stable" threshold, defined in one place (plan
    Sec 1.5). Ontology default is 0.50; callers pin the actual threshold used
    (e.g. `valley_visualization.py` pins 0.80 to preserve current figure
    behavior -- see that module for the deviation note)."""

    min_mode_frequency: float = 0.50


def _vote_mean_and_variance(vote: PeakCountVote) -> tuple[float, float]:
    if vote.n_draws_valid <= 0:
        return float("nan"), float("nan")
    counts = np.asarray(list(vote.peak_count_frequencies.keys()), dtype=float)
    freqs = np.asarray(list(vote.peak_count_frequencies.values()), dtype=float)
    total = float(np.sum(freqs))
    if total <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(counts * freqs) / total)
    variance = float(np.sum(freqs * (counts - mean) ** 2) / total)
    return mean, variance


def compute_peak_count_stability(
    vote: PeakCountVote, policy: PeakCountStabilityPolicy
) -> "PeakCountStability":
    """Derive the interpretation of a raw vote under a stability policy
    (plan Sec 1.5: mean/variance are pure functions of the vote over VALID
    draws; an invalid draw is excluded, not a vote for "0 modes")."""
    if vote.n_draws_valid <= 0 or not vote.peak_count_frequencies:
        return PeakCountStability(
            vote=vote,
            mode_peak_count=None,
            mode_frequency=float("nan"),
            mean_peak_count=float("nan"),
            peak_count_variance=float("nan"),
            valid_draw_fraction=(
                float(vote.n_draws_valid) / float(vote.n_draws_requested)
                if vote.n_draws_requested > 0
                else float("nan")
            ),
            count_is_stable=False,
        )

    items = sorted(vote.peak_count_frequencies.items())
    best_count, best_freq_count = max(items, key=lambda item: (item[1], -item[0]))
    mode_frequency = float(best_freq_count) / float(vote.n_draws_valid)
    mean_peak_count, peak_count_variance = _vote_mean_and_variance(vote)
    valid_draw_fraction = (
        float(vote.n_draws_valid) / float(vote.n_draws_requested)
        if vote.n_draws_requested > 0
        else float("nan")
    )
    count_is_stable = bool(mode_frequency >= float(policy.min_mode_frequency))

    return PeakCountStability(
        vote=vote,
        mode_peak_count=int(best_count),
        mode_frequency=mode_frequency,
        mean_peak_count=mean_peak_count,
        peak_count_variance=peak_count_variance,
        valid_draw_fraction=valid_draw_fraction,
        count_is_stable=count_is_stable,
    )


@dataclass(frozen=True)
class PeakCountStability:
    """The DERIVED interpretation of a `PeakCountVote` under a
    `PeakCountStabilityPolicy` (plan Sec 1.5). `count_is_stable` is ONLY the
    vote check; whether the final full-data basins validate
    (`resolution_succeeded`) is a separate, orthogonal check -- `is_reliable`
    on `ResolvedPeakDistribution` is the AND of both."""

    vote: PeakCountVote
    mode_peak_count: int | None
    mode_frequency: float
    mean_peak_count: float
    peak_count_variance: float
    valid_draw_fraction: float
    count_is_stable: bool


@dataclass(frozen=True)
class PeakSeed:
    """One consensus peak location built from per-draw candidate centers
    (plan Sec 1.5d: the vote gives a COUNT, locations come from a separate
    operation over the transiently-collected per-draw candidates)."""

    seed_index: int
    center: tuple[float, float]
    supporting_draw_fraction: float


@dataclass(frozen=True)
class PeakSeedSet:
    """`target_peak_count` consensus locations, built by `construction_method`
    (plan Sec 1.5d). MVP resolve (MODE_VOTE_FULL_DATA) uses these as fixed
    seeds to carve the ONE honest full-data density -- nothing is re-fit."""

    target_peak_count: int
    seeds: tuple[PeakSeed, ...]
    construction_method: str

    def __post_init__(self) -> None:
        if len(self.seeds) != int(self.target_peak_count):
            raise ValueError(
                "PeakSeedSet.seeds must contain exactly target_peak_count seeds "
                f"(got {len(self.seeds)}, expected {self.target_peak_count})."
            )

    @property
    def centers(self) -> tuple[tuple[float, float], ...]:
        return tuple(seed.center for seed in self.seeds)


def build_consensus_seed_set(
    *,
    target_peak_count: int,
    draw_centers: Sequence[Sequence[tuple[float, float]]],
    n_draws_at_target: int,
) -> PeakSeedSet:
    """Build a `PeakSeedSet` of `target_peak_count` consensus locations from
    the per-draw candidate centers of draws whose count equals the vote's
    mode (plan Sec 1.5d).

    Construction method (`"bootstrap_consensus_centroid"`): seed the
    `target_peak_count` clusters greedily from the draw with the most
    "typical" spread of centers (here: simply the first qualifying draw,
    since all qualifying draws already have exactly `target_peak_count`
    candidates), then assign every other qualifying draw's candidates to
    their nearest running seed by mutual nearest-neighbor greedy matching,
    and centroid each cluster across draws. This keeps cross-draw identity
    a proximity-based match (not an inherited index), matching the plan's
    "cross-draw identity is CONSTRUCTED by matching, never inherited"
    invariant for candidate-level structures (Sec 1.5b) applied to the
    simpler MODE_VOTE seed-construction case.

    `n_draws_at_target` is the count of draws that voted for
    `target_peak_count` (used only to report `supporting_draw_fraction`).
    """
    if target_peak_count <= 0:
        return PeakSeedSet(target_peak_count=0, seeds=(), construction_method="empty")

    qualifying = [tuple(centers) for centers in draw_centers if len(centers) == int(target_peak_count)]
    if not qualifying:
        raise ValueError(
            "build_consensus_seed_set requires at least one draw whose candidate "
            "count matches target_peak_count."
        )

    # Seed clusters from the first qualifying draw's centers.
    clusters: list[list[tuple[float, float]]] = [[c] for c in qualifying[0]]

    for draw_centers_i in qualifying[1:]:
        seed_points = np.asarray([np.mean(cluster, axis=0) for cluster in clusters], dtype=float)
        draw_points = np.asarray(draw_centers_i, dtype=float)
        # Greedy nearest-neighbor matching: repeatedly pick the closest
        # (cluster, candidate) pair not yet matched, so each cluster gets at
        # most one candidate from this draw and vice versa.
        dist2 = np.sum((seed_points[:, None, :] - draw_points[None, :, :]) ** 2, axis=-1)
        used_clusters: set[int] = set()
        used_points: set[int] = set()
        n_pairs = min(len(clusters), len(draw_points))
        for _ in range(n_pairs):
            remaining = dist2.copy()
            remaining[list(used_clusters), :] = np.inf
            remaining[:, list(used_points)] = np.inf
            flat_idx = int(np.argmin(remaining))
            if not np.isfinite(remaining.flat[flat_idx]):
                break
            cluster_idx, point_idx = np.unravel_index(flat_idx, remaining.shape)
            clusters[cluster_idx].append(tuple(draw_points[point_idx]))
            used_clusters.add(int(cluster_idx))
            used_points.add(int(point_idx))

    n_qualifying = len(qualifying)
    seeds = tuple(
        PeakSeed(
            seed_index=idx,
            center=(float(np.mean([c[0] for c in cluster])), float(np.mean([c[1] for c in cluster]))),
            supporting_draw_fraction=float(len(cluster)) / float(n_qualifying) if n_qualifying > 0 else float("nan"),
        )
        for idx, cluster in enumerate(clusters)
    )
    return PeakSeedSet(
        target_peak_count=int(target_peak_count),
        seeds=seeds,
        construction_method="bootstrap_consensus_centroid",
    )


__all__ = [
    "PeakCountStability",
    "PeakCountStabilityPolicy",
    "PeakCountVote",
    "PeakSeed",
    "PeakSeedSet",
    "build_consensus_seed_set",
    "compute_peak_count_stability",
]
