"""Orchestration layer wiring points -> KDE -> detector -> resolved-peak geometry.

`resolved_peak_metrics.py` implements the resolved-peak object model and
scalar-summary primitives but does not itself know how to build a `DensityGrid`
from raw points or how to route `PeakDetectionResult` construction. This module
is the single place that translates a fixed analysis configuration into that
wiring, so KDE/detector-argument plumbing does not get duplicated across every
runner and null draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from ._resample_adapters import run_permutation_draws
from .bandwidth_tuning import (
    bandwidth_geometry_scales,
    evaluate_isotropic_gaussian_kde_from_dist2,
    precompute_squared_distances,
)
from .density_composition import CanonicalGrid, DensityGrid
from .peak_counting import PeakDetectionResult, detect_peaks
from .resolved_peak_metrics import (
    ResolvedPeakDistribution,
    ResolvedPeakDistributionSummary,
    ResolvedPeakRunContext,
    resolve_empirical_peak_distribution,
    run_empirical_null_test,
    summarize_resolved_peak_distribution,
)


@dataclass(frozen=True)
class ResolvedPeakAnalysisSpec:
    """Fixed KDE + detector configuration for one resolved-peak analysis pass.

    Deliberately excludes null-test settings (see `EmpiricalNullSpec`) so the
    same spec can be reused across smoke tests, truth calibration, and future
    sensitivity runs without dragging bootstrap/permutation concerns along.
    """

    bandwidth_rule: str
    bandwidth_multiplier: float
    peak_detector_method: str
    assignment_rule: str = "nearest_all_candidates_mask_rejected"
    min_component_mass_frac: float = 0.10
    min_sample_fraction: float = 0.05
    min_prominence_ratio: float = 0.10
    outlier_density_floor_fraction: float | None = None


@dataclass(frozen=True)
class EmpiricalNullSpec:
    """Null-test configuration, kept separate from `ResolvedPeakAnalysisSpec`."""

    method: Literal["pooled_label_permutation"]
    n_draws: int = 500
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    min_valid_null_fraction: float = 0.80


_SUPPORTED_BANDWIDTH_RULES = (
    "median_kNN_distance",
    "longest_non_outlier_MST_edge",
)
# Both supported rules derive an isotropic sigma from point-cloud geometry and
# feed the single dense isotropic Gaussian evaluator. Alternate estimator
# backends are deliberately outside the unified resolved-peak API.
_GEOMETRY_BANDWIDTH_RULES = ("median_kNN_distance", "longest_non_outlier_MST_edge")

# Canonical V0 analysis configuration -- the one validated by the smoke test and
# the SGE array path. min_sample_fraction is raised from the detector default
# (0.05) to 0.10 because QC on one_peak_compact at n=80 showed 0.05 admits
# spurious low-support "peaks" (~7-9% support) sitting in near-empty tail
# density. Anything reusing the resolved-peak engine on real data (e.g. the
# valley-visualization reference readout) should import this rather than
# re-declaring a spec, so figures and null-test tables share one configuration.
# Bandwidth: the default is the geometry-derived `longest_non_outlier_MST_edge`
# at multiplier 0.75 — the configuration VALIDATED against synthetic truth in
# v0/validate_v0_resolved_peaks.py (one/two/three-peak fixtures recover the right
# counts). This is an isotropic sigma from the point cloud's largest non-outlier
# spanning-tree gap, NOT scipy/Scott, and it matches the rest of the framework
# (support_geometry.isotropic_geometry_kde_spec uses the same rule) so target,
# reference, and every null resample each get their own geometry-appropriate sigma
# and the whole figure stays on-method.
#
# Scott/SciPy was an earlier experimental backend. It is no longer a supported
# resolved-peak configuration: adding another estimator requires an explicit
# design/calibration change rather than a second live backend branch here.
DEFAULT_ANALYSIS_SPEC = ResolvedPeakAnalysisSpec(
    bandwidth_rule="longest_non_outlier_MST_edge",
    bandwidth_multiplier=0.75,
    peak_detector_method="kde_peak_basins_sample_support",
    min_sample_fraction=0.10,
)


def _evaluate_density_for_spec(
    points: np.ndarray,
    canonical_grid: CanonicalGrid,
    analysis_spec: ResolvedPeakAnalysisSpec,
) -> np.ndarray:
    """Evaluate the KDE density on `canonical_grid` under the spec's bandwidth rule.

    Every supported rule selects an isotropic sigma via
    ``bandwidth_geometry_scales``. ``bandwidth_multiplier`` scales that sigma;
    the resulting field is always evaluated by the same Gaussian KDE kernel.
    """
    rule = analysis_spec.bandwidth_rule
    if rule not in _SUPPORTED_BANDWIDTH_RULES:
        raise NotImplementedError(
            f"bandwidth_rule={rule!r} not yet wired; supported: {_SUPPORTED_BANDWIDTH_RULES}"
        )

    # Geometry-derived isotropic sigma (median kNN / longest non-outlier MST edge).
    # The NotImplementedError check above guarantees rule is in
    # _GEOMETRY_BANDWIDTH_RULES here, which never includes connectivity_90_radius
    # / graph_connectivity_radius -- so that O(n^2) union-find sweep is never
    # needed on this path.
    scales = bandwidth_geometry_scales(points, include_connectivity_radius=False)
    geometry_scale = scales.get(rule, float("nan"))
    if not np.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError(
            f"bandwidth_rule={rule!r} produced a non-finite/nonpositive scale "
            f"({geometry_scale}); cannot evaluate KDE for {len(points)} points."
        )
    bandwidth = float(geometry_scale) * float(analysis_spec.bandwidth_multiplier)

    grid_points = np.column_stack([canonical_grid.xx.ravel(), canonical_grid.yy.ravel()])
    dist2 = precompute_squared_distances(grid_points, np.asarray(points, dtype=float))
    cell_area = canonical_grid.cell_area if hasattr(canonical_grid, "cell_area") else None
    density_flat = evaluate_isotropic_gaussian_kde_from_dist2(
        dist2, bandwidth, cell_area=cell_area, normalize_grid=cell_area is not None,
    )
    return density_flat.reshape(canonical_grid.xx.shape)


def _compute_peak_detection_with_analysis_spec(
    points: np.ndarray,
    canonical_grid: CanonicalGrid,
    analysis_spec: ResolvedPeakAnalysisSpec,
    *,
    sweep_steps: int | None = None,
) -> tuple[DensityGrid, np.ndarray, PeakDetectionResult]:
    """KDE density -> `detect_peaks`, stopping short of full empirical resolution.

    This is the canonical density-to-detection path: `_resolve_points_single_pass`
    delegates to it for the full resolve, and callers that only need a peak count
    and candidate centers (e.g. bootstrap-vote draws) can call it directly to skip
    sample-to-peak reassignment, per-peak geometry, and `ResolvedPeakDistribution`
    construction/validation entirely.

    `sweep_steps=None` omits the kwarg to `detect_peaks`, so its own default (50,
    the full-resolution honest value) applies. Only vote-only callers should ever
    pass a non-`None` value here.
    """
    points_array = np.asarray(points, dtype=float)
    density = _evaluate_density_for_spec(points_array, canonical_grid, analysis_spec)
    density_grid = DensityGrid(xx=canonical_grid.xx, yy=canonical_grid.yy, density=density, grid=canonical_grid)

    detect_peaks_kwargs = dict(
        method=analysis_spec.peak_detector_method,
        grid=density_grid,
        sample_points=points_array,
        min_component_mass_frac=analysis_spec.min_component_mass_frac,
        min_sample_fraction=analysis_spec.min_sample_fraction,
        min_prominence_ratio=analysis_spec.min_prominence_ratio,
    )
    if sweep_steps is not None:
        detect_peaks_kwargs["sweep_steps"] = sweep_steps

    detection_result = detect_peaks(density, **detect_peaks_kwargs)
    return density_grid, points_array, detection_result


def _resolve_points_single_pass(
    *,
    distribution_id: str,
    points: np.ndarray,
    canonical_grid: CanonicalGrid,
    analysis_spec: ResolvedPeakAnalysisSpec,
) -> ResolvedPeakDistribution:
    """Build a `ResolvedPeakDistribution` from raw points under a fixed analysis spec.

    Centralizes: KDE evaluation, `DensityGrid` construction, `detect_peaks`
    argument routing, and `resolve_empirical_peak_distribution` construction.
    Always resolves at full resolution (sweep_steps left at its detect_peaks
    default) -- this is the honest, non-approximated path.
    """
    points_array = np.asarray(points, dtype=float)
    density = _evaluate_density_for_spec(points_array, canonical_grid, analysis_spec)
    density_grid = DensityGrid(
        xx=canonical_grid.xx,
        yy=canonical_grid.yy,
        density=density,
        grid=canonical_grid,
    )
    return _resolve_density_grid_single_pass(
        distribution_id=distribution_id,
        points=points_array,
        density_grid=density_grid,
        analysis_spec=analysis_spec,
    )


def _resolve_density_grid_single_pass(
    *,
    distribution_id: str,
    points: np.ndarray,
    density_grid: DensityGrid,
    analysis_spec: ResolvedPeakAnalysisSpec,
) -> ResolvedPeakDistribution:
    """Resolve an already-calculated density without fitting or evaluating a KDE.

    This is the authoritative supplied-density boundary. The analysis spec
    contributes detector and assignment configuration only; the density values
    are consumed exactly as supplied.
    """
    points_array = np.asarray(points, dtype=float)
    detect_peaks_kwargs = dict(
        method=analysis_spec.peak_detector_method,
        grid=density_grid,
        sample_points=points_array,
        min_component_mass_frac=analysis_spec.min_component_mass_frac,
        min_sample_fraction=analysis_spec.min_sample_fraction,
        min_prominence_ratio=analysis_spec.min_prominence_ratio,
    )
    detection_result = detect_peaks(
        np.asarray(density_grid.density, dtype=float), **detect_peaks_kwargs
    )

    return resolve_empirical_peak_distribution(
        distribution_id=distribution_id,
        density_grid=density_grid,
        sample_points=points_array,
        detection_result=detection_result,
        outlier_density_floor_fraction=analysis_spec.outlier_density_floor_fraction,
    )


def summarize_points_with_analysis_spec(
    *,
    distribution_id: str,
    points: np.ndarray,
    canonical_grid: CanonicalGrid,
    analysis_spec: ResolvedPeakAnalysisSpec,
) -> ResolvedPeakDistributionSummary:
    """Resolve and immediately summarize, discarding the full resolved object.

    Used by null draws (see `run_resolved_peak_permutation_comparison`), which
    only need the scalar summary and should not retain per-draw resolved
    objects in memory.
    """
    distribution = _resolve_points_single_pass(
        distribution_id=distribution_id,
        points=points,
        canonical_grid=canonical_grid,
        analysis_spec=analysis_spec,
    )
    return summarize_resolved_peak_distribution(distribution)


def resolved_peak_to_rows(distribution: ResolvedPeakDistribution) -> list[dict[str, Any]]:
    """One row per accepted resolved peak (`resolved_peak_table` schema)."""
    rows: list[dict[str, Any]] = []
    for peak in distribution.peaks:
        geometry = peak.geometry
        rows.append(
            {
                "distribution_id": distribution.distribution_id,
                "source_type": distribution.source_type,
                "candidate_peak_id": geometry.peak_id,
                "center_x": geometry.center_coordinate[0],
                "center_y": geometry.center_coordinate[1],
                "total_support_fraction": geometry.total_support_fraction,
                "radius": geometry.radius,
                "cv_radius_from_center": geometry.cv_radius_from_center,
                "within_peak_r80_density": geometry.within_peak_r80_density,
            }
        )
    return rows


def resolved_peak_summary_to_row(
    summary: ResolvedPeakDistributionSummary,
    *,
    context: ResolvedPeakRunContext,
) -> dict[str, Any]:
    """One row per resolved distribution (`resolved_peak_distribution_summary_table` schema)."""
    return {
        "analysis_id": context.analysis_id,
        "distribution_id": summary.distribution_id,
        "source_type": summary.source_type,
        "scenario_id": context.scenario_id,
        "replicate_id": context.replicate_id,
        "seed": context.seed,
        "n": context.n,
        "bandwidth_rule": context.bandwidth_rule,
        "bandwidth_multiplier": context.bandwidth_multiplier,
        "bandwidth_value": context.bandwidth_value,
        "peak_detector_method": context.peak_detector_method,
        "canonical_grid_id": context.canonical_grid_id,
        "assignment_rule": context.assignment_rule,
        "number_of_peaks": summary.number_of_peaks,
        "assigned_support_fraction": summary.assigned_support_fraction,
        "unassigned_support_fraction": summary.unassigned_support_fraction,
        "across_peak_total_support_fraction_mean": summary.across_peak_total_support_fraction_mean,
        "across_peak_total_support_fraction_skew": summary.across_peak_total_support_fraction_skew,
        "across_peak_radius_mean": summary.across_peak_radius_mean,
        "across_peak_radius_skew": summary.across_peak_radius_skew,
        "across_peak_cv_radius_from_center_mean": summary.across_peak_cv_radius_from_center_mean,
        "across_peak_distance_mean": summary.across_peak_distance_mean,
        "across_peak_r80_density_mean": summary.across_peak_r80_density_mean,
    }


_PRIMARY_NULL_TEST_METRICS = (
    "number_of_peaks",
    "assigned_support_fraction",
    "across_peak_r80_density_mean",
    "across_peak_radius_mean",
    "across_peak_cv_radius_from_center_mean",
    "across_peak_distance_mean",
)


def _metric_value(summary: ResolvedPeakDistributionSummary, metric: str) -> float:
    return float(getattr(summary, metric))


def compute_observed_delta(
    *,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    analysis_spec: ResolvedPeakAnalysisSpec,
    canonical_grid: CanonicalGrid,
    metrics: tuple[str, ...] = _PRIMARY_NULL_TEST_METRICS,
) -> tuple[dict[str, float], ResolvedPeakDistributionSummary, ResolvedPeakDistributionSummary]:
    """Observed reference/target summaries and per-metric observed_delta.

    Factored out of `run_resolved_peak_permutation_comparison` so the array-job
    path (`run_resolved_peak_permutation_draws`, one process per task) and the
    serial path compute the observed statistic identically instead of via
    copy-pasted logic.
    """
    reference_points = np.asarray(reference_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    reference_summary = summarize_points_with_analysis_spec(
        distribution_id="observed_reference", points=reference_points,
        canonical_grid=canonical_grid, analysis_spec=analysis_spec,
    )
    target_summary = summarize_points_with_analysis_spec(
        distribution_id="observed_target", points=target_points,
        canonical_grid=canonical_grid, analysis_spec=analysis_spec,
    )

    observed_delta = {
        metric: _metric_value(target_summary, metric) - _metric_value(reference_summary, metric)
        for metric in metrics
    }
    return observed_delta, reference_summary, target_summary


def run_resolved_peak_permutation_draws(
    *,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    analysis_spec: ResolvedPeakAnalysisSpec,
    canonical_grid: CanonicalGrid,
    n_draws: int,
    rng: np.random.Generator,
    metrics: tuple[str, ...] = _PRIMARY_NULL_TEST_METRICS,
) -> dict[str, np.ndarray]:
    """Run `n_draws` pooled-label permutation draws and return raw null_delta arrays.

    This is the parallelizable unit of `run_resolved_peak_permutation_comparison`:
    it has no dependency on the observed statistic, so independent processes
    (e.g. SGE array tasks, each with its own `rng` and its own slice of
    `n_draws`) can call it and have their `null_delta` arrays concatenated
    before the final `run_empirical_null_test` reduction. Reduction is
    intentionally excluded here -- it must run once, after all slices are
    pooled, not per-slice.

    Internally this now delegates draw generation to
    `analyze.utils.resampling` (via `core._resample_adapters`) instead of a
    hand-rolled permutation loop -- COMPOSE_single_path_plan Sec 1.11 /
    Stage 1. The external signature (an already-seeded `rng`, not an int
    `seed`) is preserved exactly so the SGE array-task caller does not need
    to change: a fixed-width integer seed is deterministically derived from
    `rng` so this function stays a pure function of its caller-supplied RNG
    stream, not of wall-clock or process state. Per the resampling package's
    documented SeedSequence clean break (see its README), the exact null
    values drawn here shift relative to the old hand-rolled loop for the same
    `rng` -- this is the expected, documented re-baseline, not a bug.
    """
    reference_points = np.asarray(reference_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    if n_draws <= 0:
        return {metric: np.asarray([], dtype=float) for metric in metrics}

    derived_seed = int(rng.integers(0, 2**32 - 1))

    def _resolve_and_summarize(points: np.ndarray) -> dict[str, float]:
        summary = summarize_points_with_analysis_spec(
            distribution_id="null_draw", points=points,
            canonical_grid=canonical_grid, analysis_spec=analysis_spec,
        )
        return {metric: _metric_value(summary, metric) for metric in metrics}

    return run_permutation_draws(
        reference_points=reference_points,
        target_points=target_points,
        metrics=metrics,
        resolve_and_summarize=_resolve_and_summarize,
        n_draws=n_draws,
        seed=derived_seed,
    )


def reduce_permutation_null_test(
    *,
    observed_delta: dict[str, float],
    reference_summary: ResolvedPeakDistributionSummary,
    target_summary: ResolvedPeakDistributionSummary,
    null_deltas: dict[str, np.ndarray],
    analysis_spec: ResolvedPeakAnalysisSpec,
    null_spec: EmpiricalNullSpec,
    context: ResolvedPeakRunContext,
    metrics: tuple[str, ...] = _PRIMARY_NULL_TEST_METRICS,
) -> pd.DataFrame:
    """Reduce pooled null_delta arrays (all draws, from one or many sources) into
    the `resolved_peak_null_test_table` schema. Shared by the serial and
    array-job paths so the final reduction logic exists in exactly one place.
    """
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        null_result = run_empirical_null_test(
            observed_value=observed_delta[metric],
            null_values=np.asarray(null_deltas[metric], dtype=float),
            alternative=null_spec.alternative,
            min_valid_null_fraction=null_spec.min_valid_null_fraction,
        )
        rows.append(
            {
                "analysis_id": context.analysis_id,
                "comparison_id": f"{reference_summary.distribution_id}_vs_{target_summary.distribution_id}",
                "reference_group_id": "reference",
                "target_group_id": "target",
                "metric_name": metric,
                "observed_reference_value": _metric_value(reference_summary, metric),
                "observed_target_value": _metric_value(target_summary, metric),
                "observed_difference": null_result.observed_value,
                "null_mean": null_result.null_mean,
                "null_std": null_result.null_std,
                "null_median": null_result.null_median,
                "null_q025": null_result.null_q025,
                "null_q975": null_result.null_q975,
                "empirical_p_value": null_result.empirical_p_value,
                "standardized_effect": null_result.standardized_effect,
                "n_null": null_result.n_null,
                "n_valid_null": null_result.n_valid_null,
                "valid_null_fraction": null_result.valid_null_fraction,
                "test_is_valid": null_result.test_is_valid,
                "alternative": null_spec.alternative,
                "null_generation_method": "pooled_label_permutation",
                "bandwidth_rule": analysis_spec.bandwidth_rule,
                "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
                "peak_detector_method": analysis_spec.peak_detector_method,
                "n": context.n,
            }
        )

    return pd.DataFrame(rows)


def run_resolved_peak_permutation_comparison(
    *,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    analysis_spec: ResolvedPeakAnalysisSpec,
    null_spec: EmpiricalNullSpec,
    context: ResolvedPeakRunContext,
    rng: np.random.Generator,
    canonical_grid: CanonicalGrid,
    metrics: tuple[str, ...] = _PRIMARY_NULL_TEST_METRICS,
) -> pd.DataFrame:
    """Pooled-label permutation null test for reference vs. target resolved-peak metrics.

    Null construction (two-group, statistic-preserving): pool reference and
    target points, permute labels while preserving original group sizes,
    resolve+summarize both null groups independently under the fixed
    `analysis_spec`, and compute `null_delta = null_target_metric -
    null_reference_metric` per draw. This must match the shape of the observed
    statistic (`observed_delta = target_metric - reference_metric`) -- a
    single-group null value is not comparable to a two-group observed
    difference.

    Serial reference implementation. For large `null_spec.n_draws`, see
    `run_resolved_peak_permutation_draws` (parallelizable draw generation) and
    `reduce_permutation_null_test` (shared final reduction), which this
    function is a thin wrapper around.
    """
    if null_spec.method != "pooled_label_permutation":
        raise NotImplementedError(f"null_spec.method={null_spec.method!r} not yet wired")

    observed_delta, reference_summary, target_summary = compute_observed_delta(
        reference_points=reference_points, target_points=target_points,
        analysis_spec=analysis_spec, canonical_grid=canonical_grid, metrics=metrics,
    )
    null_deltas = run_resolved_peak_permutation_draws(
        reference_points=reference_points, target_points=target_points,
        analysis_spec=analysis_spec, canonical_grid=canonical_grid,
        n_draws=null_spec.n_draws, rng=rng, metrics=metrics,
    )
    return reduce_permutation_null_test(
        observed_delta=observed_delta, reference_summary=reference_summary,
        target_summary=target_summary, null_deltas=null_deltas,
        analysis_spec=analysis_spec, null_spec=null_spec, context=context, metrics=metrics,
    )


__all__ = [
    "EmpiricalNullSpec",
    "ResolvedPeakAnalysisSpec",
    "_compute_peak_detection_with_analysis_spec",
    "compute_observed_delta",
    "reduce_permutation_null_test",
    "resolved_peak_summary_to_row",
    "resolved_peak_to_rows",
    "run_resolved_peak_permutation_comparison",
    "run_resolved_peak_permutation_draws",
    "summarize_points_with_analysis_spec",
]
