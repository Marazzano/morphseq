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

from .density_composition import CanonicalGrid
from .resolved_peak_analysis import ResolvedPeakAnalysisSpec, resolve_points_with_analysis_spec
from .resolved_peak_metrics import (
    RESOLVED_PEAK_METRICS,
    ResolvedPeakDistribution,
    ResolvedPeakDistributionSummary,
    summarize_resolved_peak_distribution,
)


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


def compute_resolved_peaks(record: DistributionRecord) -> DistributionRecord:
    """The one resolve entrypoint: density -> detect -> resolve, delegated
    whole to `resolved_peak_analysis.resolve_points_with_analysis_spec` (the
    single engine both this record path and the null-permutation path call --
    plan invariant #2). Returns a new record with `resolved_peaks` filled in;
    `sample_resolved_peak_ids` / `resolved_peak_field` are fields already
    carried on the returned `ResolvedPeakDistribution`
    (`.sample_peak_ids` / basin raster), not separate compute steps (plan
    Part 4 Stage 1)."""
    resolved = resolve_points_with_analysis_spec(
        distribution_id=record.distribution_id,
        points=record.points,
        canonical_grid=record.canonical_grid,
        analysis_spec=record.analysis_spec,
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
    "compute_observed_metrics",
    "compute_peak_stats",
    "compute_resolved_peaks",
    "derive_shared_grid",
]
