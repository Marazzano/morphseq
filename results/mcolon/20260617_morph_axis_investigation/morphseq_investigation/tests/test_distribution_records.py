"""Tests for the Stage-1 typed distribution record/comparison layer
(`compute_*` verbs; see COMPOSE_single_path_plan.md).

Mirrors the structure of the pre-refactor `add_*`-scaffold tests this file
replaces: pipeline populates fields, duplicate-safety, comparison/grid
compatibility, mismatched-grid rejection -- adapted to the new
`DistributionRecord(analysis_context=...)` + `compute_resolved_peaks` /
`compute_peak_stats` / `compute_observed_metrics` shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from morphseq_investigation.core.distribution_records import (
    DistributionAnalysisContext,
    DistributionComparison,
    DistributionRecord,
    compute_observed_metrics,
    compute_peak_stats,
    compute_resolved_peaks,
    derive_shared_grid,
)
from morphseq_investigation.core.density_composition import CanonicalGrid
from morphseq_investigation.core.resolved_peak_analysis import ResolvedPeakAnalysisSpec
from morphseq_investigation.core.resolved_peak_metrics import RESOLVED_PEAK_METRICS


def _two_cluster_points(center_offset: float = 0.0) -> np.ndarray:
    offsets = np.array(
        [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05], [0.02, 0.02], [-0.02, -0.02]]
    )
    left = np.array([-2.0 - center_offset, 0.0]) + offsets
    right = np.array([2.0 + center_offset, 0.0]) + offsets
    return np.concatenate([left, right], axis=0)


def _analysis_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="scipy_default",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
        min_sample_fraction=0.10,
    )


def _build_record(distribution_id: str, points: np.ndarray, grid: CanonicalGrid) -> DistributionRecord:
    record = DistributionRecord(
        distribution_id=distribution_id,
        points=points,
        analysis_context=DistributionAnalysisContext(grid=grid, spec=_analysis_spec()),
        metadata={"kind": "test"},
    )
    record = compute_resolved_peaks(record)
    record = compute_peak_stats(record)
    return record


def test_distribution_record_pipeline_populates_resolved_peaks_and_peak_stats():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid_size=61)

    record = _build_record("sample", points, grid)

    assert record.points.flags.writeable is False
    assert record.resolved_peaks is not None
    assert record.peak_stats is not None
    assert record.peak_stats.number_of_peaks == record.resolved_peaks.number_of_peaks
    # Density authority is sole (Stage 0): resolved_peaks carries the density
    # grid derived purely from analysis_context.spec, no separate kde= input.
    assert record.resolved_peaks.density_grid is not None


def test_compute_peak_stats_requires_resolved_peaks_first():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid_size=61)
    record = DistributionRecord(
        distribution_id="sample",
        points=points,
        analysis_context=DistributionAnalysisContext(grid=grid, spec=_analysis_spec()),
    )

    with pytest.raises(ValueError):
        compute_peak_stats(record)


def test_distribution_comparison_observed_metrics_and_grid_compatibility():
    target_points = _two_cluster_points(center_offset=0.0)
    reference_points = _two_cluster_points(center_offset=0.5)
    grid = derive_shared_grid(target_points, reference_points, grid_size=61)

    target = _build_record("target", target_points, grid)
    reference = _build_record("reference", reference_points, grid)
    comparison = DistributionComparison(
        comparison_id="cmp",
        members={"reference": reference, "target": target},
    )
    comparison = compute_observed_metrics(comparison, name="primary")
    table = comparison.observed_metrics["primary"]

    assert set(table["metric_name"]) == set(RESOLVED_PEAK_METRICS)
    assert len(table) == len(RESOLVED_PEAK_METRICS)
    assert np.all(table["comparison_id"] == "cmp")
    assert table["observed_target_value"].notna().any()


def test_compute_observed_metrics_duplicate_name_rejected_by_default():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid_size=61)
    target = _build_record("target", points, grid)
    reference = _build_record("reference", points, grid)
    comparison = DistributionComparison(
        comparison_id="cmp", members={"reference": reference, "target": target},
    )
    comparison = compute_observed_metrics(comparison, name="primary")

    with pytest.raises(KeyError):
        compute_observed_metrics(comparison, name="primary")


def test_distribution_comparison_rejects_mismatched_grids():
    points = _two_cluster_points()
    grid_a = derive_shared_grid(points, points, grid_size=61)
    grid_b = derive_shared_grid(points, points, grid_size=63)

    target = _build_record("target", points, grid_a)
    reference = _build_record("reference", points, grid_b)

    with pytest.raises(ValueError):
        DistributionComparison(comparison_id="cmp", members={"reference": reference, "target": target})


def test_derive_shared_grid_always_carries_pooled_bounds():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid_size=51)

    assert grid.grid_size == 51
    assert grid.x_min < points[:, 0].min()
    assert grid.x_max > points[:, 0].max()
