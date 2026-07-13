"""Fidelity tests for the sole resolver-to-ontology adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from morphseq_investigation.core.peak_stability import (
    PeakCountRobustnessPolicy,
    PeakCountVote,
    PeakResolutionSummary,
    PeakVotingSpec,
)
from morphseq_investigation.engine.objects import (
    DensityEstimate,
    DensityEstimateSpec,
    DensityGrid,
    Distribution,
    Grid,
    UNASSIGNED_LABEL,
)
from morphseq_investigation.engine.peak_adapter import label_group_from_resolved_peaks


def _distribution() -> Distribution:
    return Distribution(
        distribution_id="dist-a",
        sample_ids=("s0", "s1", "s2", "s3"),
        feature_names=("PC1", "PC2"),
        feature_values=np.asarray([[0, 0], [1, 1], [8, 8], [4, 4]], dtype=float),
    )


def _density() -> DensityEstimate:
    axes = (np.asarray([0.0, 8.0]), np.asarray([0.0, 8.0]))
    grid = Grid("grid-a", ("PC1", "PC2"), axes, "fixture")
    return DensityEstimate(
        "dist-a",
        ("PC1", "PC2"),
        DensityEstimateSpec(grid_params={"resolution": 2}),
        grid,
        DensityGrid("grid-a", ("PC1", "PC2"), np.ones((2, 2))),
    )


def _resolved():
    vote = PeakCountVote({2: 7, 1: 1}, 8, 8, 0.75)
    summary = PeakResolutionSummary(
        peak_count_vote=vote,
        voting_spec=PeakVotingSpec(n_draws=8, sample_fraction=0.75, min_valid_draws=6),
        robustness_policy=PeakCountRobustnessPolicy(min_mode_frequency=0.8),
        resolved_peak_count=2,
        is_robust=True,
    )
    geometry0 = SimpleNamespace(
        peak_id=0,
        center_coordinate=(0.5, 0.5),
        radius=1.25,
        total_support_fraction=0.45,
        within_peak_r80_density=0.18,
        cv_radius_from_center=0.12,
    )
    geometry1 = SimpleNamespace(
        peak_id=1,
        center_coordinate=(8.0, 8.0),
        radius=0.75,
        total_support_fraction=0.30,
        within_peak_r80_density=0.27,
        cv_radius_from_center=0.08,
    )
    return SimpleNamespace(
        distribution_id="dist-a",
        source_type="empirical",
        sample_peak_ids=np.asarray([0, 0, 1, -1]),
        peaks=(SimpleNamespace(geometry=geometry0), SimpleNamespace(geometry=geometry1)),
        resolution_evidence=SimpleNamespace(count_stability=summary),
        provenance={"resolver": "fixture", "seed": 41},
    ), summary


def test_adapter_transfers_assignments_geometry_vote_density_and_provenance_losslessly():
    resolved, summary = _resolved()
    density = _density()
    group = label_group_from_resolved_peaks(
        _distribution(), resolved, name="resolved_peak", density=density
    )

    assert group.assignments == {
        "s0": "peak_0", "s1": "peak_0", "s2": "peak_1", "s3": UNASSIGNED_LABEL
    }
    assert group.categories() == ("peak_0", "peak_1")
    assert group.peak_resolution_summary is summary
    assert group.density is density
    assert group.peak_count == group.sample_set_count == 2
    assert group.is_robust is True
    assert np.array_equal(group.sample_set_geometries["peak_0"].center, [0.5, 0.5])
    assert group.sample_set_geometries["peak_0"].support_fraction == pytest.approx(0.45)
    assert group.sample_set_geometries["peak_1"].r80_radial_concentration == pytest.approx(0.27)
    assert group.labeling_provenance.detail["resolver_provenance"] is resolved.provenance


def test_adapter_performs_no_analytical_recomputation(monkeypatch):
    resolved, _ = _resolved()

    def forbidden(*args, **kwargs):
        raise AssertionError("adapter attempted analytical recomputation")

    monkeypatch.setattr(
        "morphseq_investigation.engine.grid.evaluate_density", forbidden
    )
    monkeypatch.setattr(
        "morphseq_investigation.core.peak_counting.detect_peaks", forbidden
    )
    group = label_group_from_resolved_peaks(
        _distribution(), resolved, name="resolved_peak", density=_density()
    )
    assert group.peak_count == 2


def test_adapter_rejects_misaligned_or_unknown_assignments():
    resolved, _ = _resolved()
    resolved.sample_peak_ids = np.asarray([0, 9, 1, -1])
    with pytest.raises(ValueError, match="unknown resolved peak"):
        label_group_from_resolved_peaks(
            _distribution(), resolved, name="resolved_peak", density=_density()
        )
