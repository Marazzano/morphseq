"""Distribution record/comparison containers -- Stage 1 of
COMPOSE_single_path_plan.md (see that doc for the full ontology).

Greenfield rewrite (plan Sec 5.5 decision 1): the prior `add_*`-verb scaffold
here was never wired into the live pipeline (`valley_visualization.py` used
`resolved_peak_analysis.resolve_points_with_analysis_spec` directly). This
module now implements the plan's `compute_*` verb set, Stage-1 scoped:

- `derive_shared_grid`      -> CanonicalGrid                    (plan Part 2 (1))
- `DistributionRecord`      persistent subject: points + analysis_context
                            + resolved_peaks | None + peak_stats | None
- `compute_resolved_peaks`  -> record.resolved_peaks   (delegates to the ONE
                               engine: `resolved_peak_analysis.
                               resolve_points_with_analysis_spec`; no math
                               reimplemented here)
- `compute_peak_stats`      -> record.peak_stats        (delegates to
                               `summarize_resolved_peak_distribution`)
- `DistributionComparison`  persistent contrast (reference/target roles)
- `compute_observed_metrics` -> comparison.observed_metrics (one registry:
                               RESOLVED_PEAK_METRICS)

Density is specified ONCE: `DistributionAnalysisContext.spec`
(`ResolvedPeakAnalysisSpec`) is the sole bandwidth authority. There is no
parallel `kde=` parameter anywhere in this module -- `compute_resolved_peaks`
derives density from the spec alone, inside
`resolve_points_with_analysis_spec` (Stage 0).

Deliberately NOT built here (Stage 2+ per plan Part 4): PeakResolutionConfig
strategy/retention axes, the bootstrap vote, PeakSeedSet, mass-splitting,
PeakStabilityGraph, is_reliable / resolved_peak_count-may-be-None failure
semantics. `compute_resolved_peaks` today is a single full-data resolve
(MODE_VOTE_FULL_DATA's Stage-1 stand-in), same math the figure already uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ._resample_adapters import bootstrap_peak_vote
from .density_composition import CanonicalGrid, DensityGrid
from .peak_acceptance import PeakAcceptancePolicy
from .peak_counting import assign_cells_to_peaks, assign_points_to_peaks
from .peak_stability import (
    PeakCountRobustnessPolicy,
    PeakCountVote,
    PeakSeedSet,
    PeakVotingSpec,
    build_consensus_seed_set,
    compute_peak_count_stability,
)
from .resolved_peak_analysis import (
    ResolvedPeakAnalysisSpec,
    _compute_peak_detection_with_analysis_spec,
    resolve_density_grid_with_analysis_spec,
    resolve_points_with_analysis_spec,
)
from .resolved_peak_metrics import (
    RESOLVED_PEAK_METRICS,
    PeakGeometry,
    PeakResolutionEvidence,
    ResolvedPeak,
    ResolvedPeakDistribution,
    ResolvedPeakDistributionSummary,
    summarize_resolved_peak_distribution,
)
from .resolved_peak_metrics import _radius_and_cv_from_distances, _within_peak_r80_density


def _readonly_array(values: np.ndarray | list[Any] | tuple[Any, ...], *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _readonly_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _grid_signature(grid: CanonicalGrid) -> tuple[float, float, float, float, int]:
    return (
        float(grid.x_min),
        float(grid.x_max),
        float(grid.y_min),
        float(grid.y_max),
        int(grid.grid_size),
    )


def _assert_grid_compatibility(left: CanonicalGrid, right: CanonicalGrid) -> None:
    if _grid_signature(left) != _grid_signature(right):
        raise ValueError("DistributionComparison members must share an identical canonical grid.")


def derive_shared_grid(
    reference_points: np.ndarray,
    target_points: np.ndarray,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    grid_size: int = 61,
    margin: float = 0.1,
) -> CanonicalGrid:
    """Derive ONE `CanonicalGrid` shared by a reference/target pair (plan
    Part 2 step (1)). Pure coordinate frame -- owns no points, no density.

    If explicit bounds are not given, they are the union bounding box of both
    point sets, padded by `margin` (fraction of the box's span per axis).
    This is a Stage-1-scoped convenience for building typed
    `DistributionRecord`s outside the figure pipeline (which derives its grid
    via `plotting.modal_distribution_plotting.derive_shared_grid`'s
    support-fraction box instead -- both produce a `CanonicalGrid`, only the
    bound-selection policy differs, and that policy is not part of the
    Stage 0/1 density-authority collapse).
    """
    reference_points = np.asarray(reference_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    pooled = np.concatenate([reference_points, target_points], axis=0)

    if x_min is None or x_max is None:
        lo, hi = float(np.min(pooled[:, 0])), float(np.max(pooled[:, 0]))
        pad = (hi - lo) * float(margin) if hi > lo else 1.0
        x_min = lo - pad if x_min is None else x_min
        x_max = hi + pad if x_max is None else x_max
    if y_min is None or y_max is None:
        lo, hi = float(np.min(pooled[:, 1])), float(np.max(pooled[:, 1]))
        pad = (hi - lo) * float(margin) if hi > lo else 1.0
        y_min = lo - pad if y_min is None else y_min
        y_max = hi + pad if y_max is None else y_max

    return CanonicalGrid(
        x_min=float(x_min), x_max=float(x_max),
        y_min=float(y_min), y_max=float(y_max),
        grid_size=int(grid_size),
    )


@dataclass(frozen=True)
class DistributionAnalysisContext:
    """Grid + spec, bundled so both comparison members are guaranteed to
    share ONE frame and ONE bandwidth authority (plan "Target API" section):
    prevents a grid built with spec A being resolved with spec B."""

    grid: CanonicalGrid
    spec: ResolvedPeakAnalysisSpec
    density_grid: DensityGrid | None = None

    def __post_init__(self) -> None:
        if self.density_grid is not None:
            if self.density_grid.grid is None:
                raise ValueError("A supplied density_grid must retain its CanonicalGrid.")
            _assert_grid_compatibility(self.grid, self.density_grid.grid)


@dataclass(frozen=True)
class DistributionRecord:
    """Persistent subject. `points` is the raw empirical support; everything
    else is an enrichment computed FROM points + analysis_context by a
    `compute_*` verb and stored back onto the record (plan Part 2)."""

    distribution_id: str
    points: np.ndarray
    analysis_context: DistributionAnalysisContext
    resolved_peaks: ResolvedPeakDistribution | None = None
    peak_stats: ResolvedPeakDistributionSummary | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _readonly_array(self.points, dtype=float)
        if points.ndim != 2:
            raise ValueError("DistributionRecord.points must be a 2-D array.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def canonical_grid(self) -> CanonicalGrid:
        return self.analysis_context.grid

    @property
    def analysis_spec(self) -> ResolvedPeakAnalysisSpec:
        return self.analysis_context.spec


@dataclass(frozen=True)
class DistributionComparison:
    comparison_id: str
    members: Mapping[str, DistributionRecord]
    observed_metrics: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        members = _readonly_mapping(self.members)
        if len(members) < 2:
            raise ValueError("DistributionComparison requires at least two members.")
        grids = [member.canonical_grid for member in members.values()]
        first_grid = grids[0]
        for grid in grids[1:]:
            _assert_grid_compatibility(first_grid, grid)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "observed_metrics", _readonly_mapping(self.observed_metrics))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True)
class PeakResolutionConfig:
    """Bundles the Stage-2a MODE_VOTE_FULL_DATA / SUMMARY_ONLY knobs (plan
    "Target API" section + Part 4 Stage 2a). Defaults match
    `valley_visualization.py`'s pre-refactor `_resampled_mode_count` numbers
    exactly (`N_MODE_RESAMPLE_DRAWS=80`, `MODE_DOWNSAMPLE_FRACTION=0.80`,
    `MODE_RESAMPLE_MIN_FREQ=0.80`), so `compute_resolved_peaks`'s default
    behavior does not silently change for any other caller.

    Only the MODE_VOTE_FULL_DATA strategy + SUMMARY_ONLY retention are built
    here (Stage 2a scope); `strategy`/`retention` fields are recorded for
    forward-compatibility with Stage 2b/5 but only the one combination is
    implemented -- `compute_resolved_peaks` raises `NotImplementedError` on
    any other value.
    """

    strategy: str = "MODE_VOTE_FULL_DATA"
    retention: str = "SUMMARY_ONLY"
    n_bootstrap_draws: int = 80
    bootstrap_sample_fraction: float = 0.80
    min_bootstrap_sample_size: int = 10
    min_valid_draws: int = 1
    robustness_policy: PeakCountRobustnessPolicy = field(
        default_factory=lambda: PeakCountRobustnessPolicy(min_mode_frequency=0.80)
    )
    peak_acceptance_policy: PeakAcceptancePolicy | None = None
    seed: int = 42
    voting_spec: PeakVotingSpec | None = None

    def __post_init__(self) -> None:
        if self.voting_spec is not None:
            object.__setattr__(self, "n_bootstrap_draws", self.voting_spec.n_draws)
            object.__setattr__(self, "bootstrap_sample_fraction", self.voting_spec.sample_fraction)
            object.__setattr__(self, "min_valid_draws", self.voting_spec.min_valid_draws)
        else:
            object.__setattr__(
                self,
                "voting_spec",
                PeakVotingSpec(
                    n_draws=self.n_bootstrap_draws,
                    sample_fraction=self.bootstrap_sample_fraction,
                    min_valid_draws=self.min_valid_draws,
                ),
            )


def _acceptance_policy_for_spec(spec: ResolvedPeakAnalysisSpec) -> PeakAcceptancePolicy:
    return PeakAcceptancePolicy(
        min_sample_fraction=spec.min_sample_fraction,
        min_prominence_ratio=spec.min_prominence_ratio,
        min_component_mass_fraction=spec.min_component_mass_frac,
    )


def _resolve_draw_for_vote(canonical_grid, analysis_spec, *, sweep_steps: int | None = None):
    """Build the `resolve_draw` closure `bootstrap_peak_vote` calls per draw:
    subsample -> the canonical detection-only primitive
    (`_compute_peak_detection_with_analysis_spec`) -> `(count, centers)`.

    A vote only needs the accepted candidate count and centers, not a full
    `ResolvedPeakDistribution` -- calling the detection-only primitive skips
    sample-to-peak reassignment, per-peak R80/radius/CV geometry, and
    `ResolvedPeakDistribution.__post_init__` validation, none of which the
    vote reads (plan Sec 1.5c: retention governs PERSISTED evidence, not
    transient resolver inputs; this is the transient path).

    `sweep_steps=None` (the default) leaves `detect_peaks` at its own default
    resolution -- identical numeric behavior to the pre-optimization full
    resolve. A non-`None` value is an explicit approximation knob, not
    enabled by default (see PeakResolutionConfig.vote_sweep_steps)."""

    def _resolve_draw(sample_points: np.ndarray) -> tuple[int, tuple[tuple[float, float], ...]]:
        _density_grid, _points_array, detection_result = _compute_peak_detection_with_analysis_spec(
            sample_points, canonical_grid, analysis_spec, sweep_steps=sweep_steps,
        )
        accepted = tuple(d for d in detection_result.candidate_details if d.accepted)
        centers = tuple((float(d.peak_x), float(d.peak_y)) for d in accepted)
        return len(accepted), centers

    return _resolve_draw


def _carve_final_peaks(
    *,
    distribution_id: str,
    density_grid: DensityGrid,
    sample_points: np.ndarray,
    seed_centers: tuple[tuple[float, float], ...],
    acceptance_policy: PeakAcceptancePolicy,
) -> tuple[tuple[ResolvedPeak, ...], np.ndarray, np.ndarray, tuple[float, ...], tuple[bool, ...]]:
    """Mass-split the ONE honest full-data density into `len(seed_centers)`
    in-basins seeded at `seed_centers` (plan Sec 1.5d MVP resolve: "keep the
    KDE honest, split its mass"). Uses the public seeded-N carving primitives
    directly (`assign_cells_to_peaks` / `assign_points_to_peaks`) -- nothing
    re-fit.

    Returns `(peaks, sample_resolved_peak_ids, resolved_peak_field,
    basin_component_mass_fractions, basin_validation)`. `peaks` /
    `sample_resolved_peak_ids` / `resolved_peak_field` are EMPTY/-1-filled if
    any basin fails `validate_resolved_basins` -- the crisp all-or-nothing
    failure semantics (plan Sec 1.5d): a caller must check
    `all(basin_validation)` (equivalently: `len(peaks) > 0` when
    `seed_centers` is nonempty) to know whether resolution succeeded.
    """
    n_samples = len(sample_points)
    if not seed_centers:
        return (), np.full(n_samples, -1, dtype=int), np.zeros(density_grid.xx.shape, dtype=int), (), ()

    grid_field = assign_cells_to_peaks(density_grid, seed_centers)  # 1-based, 0=none
    sample_field = (
        assign_points_to_peaks(sample_points, seed_centers) if n_samples > 0 else np.zeros(0, dtype=int)
    )

    density = np.asarray(density_grid.density, dtype=float)
    total_mass = float(np.sum(np.where(np.isfinite(density), density, 0.0)))
    basin_mass_fractions: list[float] = []
    for seed_idx in range(1, len(seed_centers) + 1):
        basin_mass = float(np.sum(np.where(grid_field == seed_idx, density, 0.0)))
        basin_mass_fractions.append(basin_mass / total_mass if total_mass > 0 else 0.0)

    basin_validation = acceptance_policy.validate_resolved_basins(
        basin_component_mass_fractions=basin_mass_fractions
    )

    if not all(basin_validation):
        return (), np.full(n_samples, -1, dtype=int), np.zeros(density_grid.xx.shape, dtype=int), tuple(basin_mass_fractions), tuple(basin_validation)

    # All basins accepted: build the FINAL peaks. resolved_peak_id is 0-based
    # (matches the plan's peak_id convention used elsewhere in this module --
    # PeakGeometry.peak_id is an int identity, not tied to the 1-based grid
    # label; -1 remains reserved for "unassigned").
    peaks: list[ResolvedPeak] = []
    resolved_sample_ids = np.full(n_samples, -1, dtype=int)
    resolved_field = np.full(density_grid.xx.shape, -1, dtype=int)
    for seed_idx, center in enumerate(seed_centers):
        peak_id = seed_idx
        grid_mask = grid_field == (seed_idx + 1)
        resolved_field[grid_mask] = peak_id
        sample_mask = sample_field == (seed_idx + 1) if n_samples > 0 else np.zeros(0, dtype=bool)
        resolved_sample_ids[sample_mask] = peak_id

        member_points = sample_points[sample_mask] if n_samples > 0 else np.empty((0, 2))
        support_fraction = float(len(member_points) / n_samples) if n_samples > 0 else float("nan")
        if len(member_points) == 0:
            radius, cv = float("nan"), float("nan")
        else:
            distances = np.linalg.norm(member_points - np.asarray(center, dtype=float), axis=1)
            radius, cv = _radius_and_cv_from_distances(distances)
        geometry = PeakGeometry(
            peak_id=peak_id,
            center_coordinate=(float(center[0]), float(center[1])),
            total_support_fraction=support_fraction,
            radius=radius,
            cv_radius_from_center=cv,
            within_peak_r80_density=_within_peak_r80_density(support_fraction, radius),
        )
        peaks.append(
            ResolvedPeak(
                geometry=geometry,
                source_type="empirical",
                detector_detail=None,
                provenance={
                    "accepted": True,
                    "n_assigned_samples": int(len(member_points)),
                    "basin_component_mass_fraction": basin_mass_fractions[seed_idx],
                    "resolution_strategy": "MODE_VOTE_FULL_DATA",
                },
            )
        )

    return tuple(peaks), resolved_sample_ids, resolved_field, tuple(basin_mass_fractions), tuple(basin_validation)


def compute_resolved_peaks(
    record: DistributionRecord,
    resolution_config: PeakResolutionConfig | None = None,
) -> DistributionRecord:
    """The one resolve entrypoint (plan Part 2 step 3 / Part 4 Stage 2a):

      a. `resample.subsample` bootstrap vote (via `_resample_adapters.
         bootstrap_peak_vote`) -> `PeakCountVote` -> `target_peak_count`
         (mode of the vote over VALID draws)
      b. `PeakSeedSet`: `target_peak_count` consensus locations built from
         the transient per-draw candidate centers
      c. ONE honest full-data density (from the single-pass engine,
         `resolve_points_with_analysis_spec`) mass-split into `target_peak_count`
         in-basins seeded at the `PeakSeedSet` locations -- nothing re-fit
      d. `PeakAcceptancePolicy.validate_resolved_basins` on those basins ->
         `resolution_succeeded`; `count_is_stable` = vote mode_frequency >=
         policy threshold; `is_reliable` = both
      e. FOREGROUND: `peaks` + `sample_resolved_peak_ids` (`.sample_peak_ids`)
         + `resolved_peak_field` (`.empirical_basin_labels`-shaped raster) +
         `resolved_peak_count` + `is_reliable`
         BACKGROUND: `resolution_evidence` (`PeakResolutionEvidence`)

    `resolution_config=None` uses `PeakResolutionConfig()` defaults, which
    match the pre-refactor `_resampled_mode_count` numbers exactly (80 draws,
    0.80 sample fraction, 0.80 min_freq) -- default behavior is unchanged for
    any caller that does not pass a config.

    The null-permutation path (`resolved_peak_analysis.
    run_resolved_peak_permutation_draws` -> `_resolve_and_summarize` ->
    `summarize_points_with_analysis_spec` -> `resolve_points_with_analysis_spec`)
    NEVER calls this function -- it is a single-pass resolve by construction
    (plan Sec 1.10 NullResolutionMode.SINGLE_PASS), so it does not pick up
    the vote added here.
    """
    config = resolution_config if resolution_config is not None else PeakResolutionConfig()
    if config.strategy != "MODE_VOTE_FULL_DATA" or config.retention != "SUMMARY_ONLY":
        raise NotImplementedError(
            "compute_resolved_peaks only implements strategy=MODE_VOTE_FULL_DATA + "
            "retention=SUMMARY_ONLY (Stage 2a); got "
            f"strategy={config.strategy!r}, retention={config.retention!r}."
        )

    analysis_spec = record.analysis_spec
    canonical_grid = record.canonical_grid
    points = record.points

    # (c) ONE honest full-data resolve -- density + candidate detection.
    if record.analysis_context.density_grid is None:
        full_data = resolve_points_with_analysis_spec(
            distribution_id=record.distribution_id,
            points=points,
            canonical_grid=canonical_grid,
            analysis_spec=analysis_spec,
        )
    else:
        full_data = resolve_density_grid_with_analysis_spec(
            distribution_id=record.distribution_id,
            points=points,
            density_grid=record.analysis_context.density_grid,
            analysis_spec=analysis_spec,
        )

    acceptance_policy = config.peak_acceptance_policy or _acceptance_policy_for_spec(analysis_spec)

    # (a) Bootstrap vote.
    counts, centers_by_draw, n_failed = bootstrap_peak_vote(
        points=points,
        resolve_draw=_resolve_draw_for_vote(canonical_grid, analysis_spec),
        n_draws=config.n_bootstrap_draws,
        sample_fraction=config.bootstrap_sample_fraction,
        min_sample_size=config.min_bootstrap_sample_size,
        seed=config.seed,
    )
    n_draws_valid = len(counts)
    frequencies: dict[int, int] = {}
    for count in counts:
        frequencies[count] = frequencies.get(count, 0) + 1
    vote = PeakCountVote(
        peak_count_frequencies=frequencies,
        n_draws_requested=config.n_bootstrap_draws,
        n_draws_valid=n_draws_valid,
        sample_fraction=config.bootstrap_sample_fraction,
    )
    count_stability = compute_peak_count_stability(
        vote, config.robustness_policy, config.voting_spec
    )
    target_peak_count = count_stability.mode_peak_count

    if target_peak_count == 0:
        # Zero is a valid modal answer: materialize an empty resolved group
        # with every sample explicitly unassigned.
        consensus_seed_set = PeakSeedSet(target_peak_count=0, seeds=(), construction_method="modal_zero")
        resolved = dc_replace(
            full_data,
            peaks=(),
            sample_peak_ids=np.full(len(points), -1, dtype=int) if full_data.sample_peak_ids is not None else full_data.sample_peak_ids,
            empirical_basin_labels=np.zeros(full_data.density_grid.xx.shape, dtype=int),
            resolved_peak_count=0,
            is_reliable=count_stability.is_robust,
            resolution_evidence=PeakResolutionEvidence(
                target_peak_count=target_peak_count,
                resolution_succeeded=True,
                count_is_stable=count_stability.count_is_stable,
                count_stability=count_stability,
                consensus_seed_set=consensus_seed_set,
                full_data_detection=full_data.detection_result,
                basin_component_mass_fractions=(),
                basin_validation=(),
            ),
        )
        return dc_replace(record, resolved_peaks=resolved)

    # (b) Consensus seed set from per-draw candidates agreeing with the vote.
    consensus_seed_set = build_consensus_seed_set(
        target_peak_count=target_peak_count,
        draw_centers=centers_by_draw,
        n_draws_at_target=count_stability.vote.frequency_for(target_peak_count),
    )

    # (c)+(d) Mass-split the honest density at the consensus seeds; validate.
    peaks, sample_resolved_peak_ids, resolved_peak_field, basin_mass_fractions, basin_validation = _carve_final_peaks(
        distribution_id=record.distribution_id,
        density_grid=full_data.density_grid,
        sample_points=points,
        seed_centers=consensus_seed_set.centers,
        acceptance_policy=acceptance_policy,
    )
    resolution_succeeded = bool(peaks)  # crisp: nonempty iff every basin validated
    is_reliable = bool(count_stability.count_is_stable and resolution_succeeded)

    resolution_evidence = PeakResolutionEvidence(
        target_peak_count=target_peak_count,
        resolution_succeeded=resolution_succeeded,
        count_is_stable=count_stability.count_is_stable,
        count_stability=count_stability,
        consensus_seed_set=consensus_seed_set,
        full_data_detection=full_data.detection_result,
        basin_component_mass_fractions=basin_mass_fractions,
        basin_validation=basin_validation,
    )

    resolved = dc_replace(
        full_data,
        peaks=peaks,
        sample_peak_ids=sample_resolved_peak_ids,
        empirical_basin_labels=resolved_peak_field,
        resolved_peak_count=(target_peak_count if resolution_succeeded else None),
        is_reliable=is_reliable,
        resolution_evidence=resolution_evidence,
    )
    return dc_replace(record, resolved_peaks=resolved)


def compute_peak_stats(record: DistributionRecord) -> DistributionRecord:
    """Scalar summary of the FINAL resolved peaks (plan Part 2 step (5)).
    Requires `compute_resolved_peaks` to have run first."""
    if record.resolved_peaks is None:
        raise ValueError(
            "compute_peak_stats requires record.resolved_peaks; call "
            "compute_resolved_peaks(record) first."
        )
    summary = summarize_resolved_peak_distribution(record.resolved_peaks)
    return dc_replace(record, peak_stats=summary)


def compute_observed_metrics(
    comparison: DistributionComparison,
    *,
    name: str = "primary",
    reference_role: str = "reference",
    target_role: str = "target",
    metric_registry: Mapping[str, Any] = RESOLVED_PEAK_METRICS,
    replace: bool = False,
) -> DistributionComparison:
    """One registry, one table: per-metric reference/target/observed_difference
    rows computed from each member's `peak_stats` (plan Part 2 step (6))."""
    if name in comparison.observed_metrics and not replace:
        raise KeyError(f"Product {name!r} already exists; pass replace=True to overwrite it.")
    if reference_role not in comparison.members:
        raise KeyError(f"Missing comparison member role: {reference_role!r}")
    if target_role not in comparison.members:
        raise KeyError(f"Missing comparison member role: {target_role!r}")

    reference = comparison.members[reference_role]
    target = comparison.members[target_role]
    _assert_grid_compatibility(reference.canonical_grid, target.canonical_grid)

    if reference.peak_stats is None:
        raise ValueError(f"Member {reference_role!r} is missing peak_stats; call compute_peak_stats first.")
    if target.peak_stats is None:
        raise ValueError(f"Member {target_role!r} is missing peak_stats; call compute_peak_stats first.")

    reference_summary = reference.peak_stats
    target_summary = target.peak_stats

    rows: list[dict[str, Any]] = []
    for metric_name, metric_def in metric_registry.items():
        if not hasattr(reference_summary, metric_name) or not hasattr(target_summary, metric_name):
            raise KeyError(f"Unknown resolved-peak metric: {metric_name!r}")
        reference_value = float(getattr(reference_summary, metric_name))
        target_value = float(getattr(target_summary, metric_name))
        rows.append(
            {
                "comparison_id": comparison.comparison_id,
                "reference_role": reference_role,
                "target_role": target_role,
                "metric_name": metric_name,
                "minimum_peak_count": getattr(metric_def, "minimum_peak_count", None),
                "observed_reference_value": reference_value,
                "observed_target_value": target_value,
                "observed_difference": target_value - reference_value,
                "reference_distribution_id": reference.distribution_id,
                "target_distribution_id": target.distribution_id,
            }
        )

    table = pd.DataFrame(rows)
    updated = dict(comparison.observed_metrics)
    updated[name] = table
    return dc_replace(comparison, observed_metrics=_readonly_mapping(updated))


__all__ = [
    "DistributionAnalysisContext",
    "DistributionComparison",
    "DistributionRecord",
    "PeakResolutionConfig",
    "compute_observed_metrics",
    "compute_peak_stats",
    "compute_resolved_peaks",
    "derive_shared_grid",
]
