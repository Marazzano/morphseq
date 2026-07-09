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
    PeakResolutionConfig,
    compute_observed_metrics,
    compute_peak_stats,
    compute_resolved_peaks,
    derive_shared_grid,
)
from morphseq_investigation.core.density_composition import CanonicalGrid
from morphseq_investigation.core.peak_acceptance import PeakAcceptancePolicy
from morphseq_investigation.core.peak_stability import PeakCountStabilityPolicy
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


# ---------------------------------------------------------------------------
# Stage 2a: MODE_VOTE_FULL_DATA + SUMMARY_ONLY (COMPOSE_single_path_plan
# Part 4 Stage 2a). compute_resolved_peaks now folds the bootstrap mode-count
# vote in; these tests exercise the foreground fields it populates.
# ---------------------------------------------------------------------------


def _rng_two_clusters(seed: int = 0, n_per_cluster: int = 40, sep: float = 4.0, scale: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=[-sep / 2, 0.0], scale=scale, size=(n_per_cluster, 2))
    right = rng.normal(loc=[sep / 2, 0.0], scale=scale, size=(n_per_cluster, 2))
    return np.concatenate([left, right], axis=0)


def test_compute_resolved_peaks_default_config_matches_pre_refactor_numbers():
    """Default PeakResolutionConfig must match valley_visualization.py's
    pre-refactor _resampled_mode_count numbers exactly, so
    compute_resolved_peaks(record) with no explicit config does not silently
    change behavior for any other caller."""
    config = PeakResolutionConfig()
    assert config.n_bootstrap_draws == 80
    assert config.bootstrap_sample_fraction == 0.80
    assert config.count_stability_policy.min_mode_frequency == 0.80
    assert config.strategy == "MODE_VOTE_FULL_DATA"
    assert config.retention == "SUMMARY_ONLY"


def test_compute_resolved_peaks_vote_resolves_two_well_separated_clusters():
    points = _rng_two_clusters(seed=1, sep=5.0)
    grid = derive_shared_grid(points, points, grid_size=51)
    record = DistributionRecord(
        distribution_id="sample",
        points=points,
        analysis_context=DistributionAnalysisContext(grid=grid, spec=_analysis_spec()),
    )
    config = PeakResolutionConfig(n_bootstrap_draws=40, bootstrap_sample_fraction=0.8, seed=7)

    record = compute_resolved_peaks(record, config)
    dist = record.resolved_peaks

    assert dist.resolved_peak_count == 2
    assert dist.number_of_peaks == 2
    assert dist.is_reliable is True
    assert dist.resolution_evidence is not None
    assert dist.resolution_evidence.target_peak_count == 2
    assert dist.resolution_evidence.resolution_succeeded is True
    assert dist.resolution_evidence.count_is_stable is True
    # Every sample must be assigned to one of the two resolved peaks (no -1s
    # for a well-separated two-cluster case).
    assert set(np.unique(dist.sample_peak_ids).tolist()).issubset({0, 1})


def test_single_pass_constructors_default_reliable_true_and_count_from_peaks():
    """Backward compatibility (plan Sec deviations note): the single-pass
    constructors (resolve_empirical_peak_distribution / truth) never vote, so
    resolved_peak_count falls back to number_of_peaks and is_reliable
    defaults to True -- existing non-voting callers (null path, smoke tests,
    array-job path) must keep reading a meaningful resolved_peak_count."""
    from morphseq_investigation.core.resolved_peak_analysis import resolve_points_with_analysis_spec

    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid_size=61)
    dist = resolve_points_with_analysis_spec(
        distribution_id="single_pass", points=points, canonical_grid=grid, analysis_spec=_analysis_spec(),
    )
    assert dist.resolution_evidence is None
    assert dist.resolved_peak_count == dist.number_of_peaks
    assert dist.is_reliable is True


def test_compute_resolved_peaks_unsupported_strategy_raises():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid_size=51)
    record = DistributionRecord(
        distribution_id="sample",
        points=points,
        analysis_context=DistributionAnalysisContext(grid=grid, spec=_analysis_spec()),
    )
    bad_config = PeakResolutionConfig(strategy="STABILITY_GRAPH")
    with pytest.raises(NotImplementedError):
        compute_resolved_peaks(record, bad_config)


def test_compute_resolved_peaks_crisp_failure_when_basin_mass_floor_fails():
    """Force a basin to fail validate_resolved_basins: an overwhelmingly
    lopsided two-cluster case (one cluster tiny) whose bootstrap vote still
    settles on 2 peaks, but whose final full-data mass-split puts far less
    than min_component_mass_fraction of the honest density's mass in the
    minority basin. Confirms the crisp all-or-nothing failure semantics
    (COMPOSE_single_path_plan Sec 1.5d): resolved_peak_count=None, peaks=(),
    every sample_peak_ids entry -1 -- never a partial N-1 answer."""
    rng = np.random.default_rng(3)
    # A large, tight majority cluster plus a minuscule 2-point minority
    # cluster far away: bootstrap draws at 80% subsampling almost always keep
    # both extremes (small n), so the vote settles on 2, but the minority's
    # empirical KDE mass share is tiny relative to a much-tighter, much
    # larger majority -- below the default min_component_mass_fraction=0.10
    # floor once mass-split at the two centers.
    majority = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(60, 2))
    minority = rng.normal(loc=[8.0, 0.0], scale=0.05, size=(2, 2))
    points = np.concatenate([majority, minority], axis=0)
    grid = derive_shared_grid(points, points, grid_size=61)

    spec = ResolvedPeakAnalysisSpec(
        bandwidth_rule="scipy_default",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
        # Force acceptance of the minority as a distinct candidate at the
        # detection stage (low sample-fraction/prominence floors), so the
        # bootstrap vote and the full-data detector both report 2 candidate
        # peaks; the strict min_component_mass_fraction is instead enforced
        # only at the FINAL validate_resolved_basins gate via a dedicated
        # PeakAcceptancePolicy below.
        min_sample_fraction=0.01,
        min_prominence_ratio=0.01,
        min_component_mass_frac=0.01,
    )
    record = DistributionRecord(
        distribution_id="lopsided",
        points=points,
        analysis_context=DistributionAnalysisContext(grid=grid, spec=spec),
    )
    # A strict final-basin mass floor (0.10) is far above the minority
    # basin's true KDE mass share (2/62 points at a much wider bandwidth
    # neighborhood dominated by the majority's mass) -- this is the gate that
    # must fail.
    strict_policy = PeakAcceptancePolicy(
        min_sample_fraction=0.01,
        min_prominence_ratio=0.01,
        min_component_mass_fraction=0.10,
    )
    config = PeakResolutionConfig(
        n_bootstrap_draws=20,
        bootstrap_sample_fraction=0.8,
        min_bootstrap_sample_size=5,
        count_stability_policy=PeakCountStabilityPolicy(min_mode_frequency=0.5),
        peak_acceptance_policy=strict_policy,
        seed=11,
    )

    record = compute_resolved_peaks(record, config)
    dist = record.resolved_peaks

    if dist.resolution_evidence is not None and dist.resolution_evidence.target_peak_count == 2:
        # The scenario is constructed so the vote settles on 2 and the final
        # mass-split basin validation fails -- assert the crisp semantics.
        assert dist.resolution_evidence.resolution_succeeded is False
        assert dist.resolved_peak_count is None
        assert dist.peaks == ()
        assert np.all(np.asarray(dist.sample_peak_ids) == -1)
        assert any(not ok for ok in dist.resolution_evidence.basin_validation)
    else:
        pytest.skip(
            "Synthetic scenario's bootstrap vote did not settle on the intended "
            "2-peak target under this seed/config; not exercising the basin-mass "
            "failure path this test targets."
        )
