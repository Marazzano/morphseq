"""TASK_B — labeler tests (genotype + peak_finding).

Both labelers run the SAME skeleton and pass the SAME ``validate_label_group``
(peer symmetry). Genotype proves the skeleton; peak_finding folds the live
``compute_resolved_peaks`` machinery into the ontology.
"""

import numpy as np
import pytest

from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.invariants import validate_label_group
from morphseq_investigation.engine.objects import (
    Distribution,
    LabelGroup,
    SampleSet,
    SampleSetGeometry,
)
from morphseq_investigation.engine.grid import build_grid
from morphseq_investigation.engine.labelers import label_genotype, label_peak_finding


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_distribution(values, sample_ids, *, scope="b9d2", time_bin=30, role="target"):
    dist_id = make_distribution_id(scope, time_bin, role)
    return Distribution(
        distribution_id=dist_id,
        scope_id=scope,
        time_bin=time_bin,
        role=role,
        sample_ids=tuple(sample_ids),
        feature_names=("PC1", "PC2"),
        feature_values=np.asarray(values, dtype=float),
    )


def _genotype_distribution():
    # 6 samples, 2 features. Labels exercise: two real categories, the literal
    # string "unknown", and one NA (None).
    values = np.arange(12, dtype=float).reshape(6, 2)
    sample_ids = [f"e{i}" for i in range(6)]
    return _make_distribution(values, sample_ids)


# --------------------------------------------------------------------------- #
# genotype labeler
# --------------------------------------------------------------------------- #
def test_genotype_category_counts_match():
    dist = _genotype_distribution()
    labels = ["wildtype", "wildtype", "homo", "homo", "homo", "wildtype"]
    lg, sample_sets = label_genotype(dist, "column", {"labels": labels, "column": "genotype"})

    by_name = {s.sample_set_name: s for s in sample_sets}
    assert set(by_name) == {"wildtype", "homo"}
    assert len(by_name["wildtype"].sample_ids) == 3
    assert len(by_name["homo"].sample_ids) == 3
    # peer shape
    assert isinstance(lg, LabelGroup)
    assert all(isinstance(s, SampleSet) for s in sample_sets)
    # provided labels carry no measured geometry
    assert all(s.geometry is None for s in sample_sets)
    # feature_names the labeler USED to form groups is empty for a provided labeler
    assert lg.provenance["labeler"]["feature_names"] == ()
    assert lg.artifacts is None


def test_genotype_unknown_is_a_real_sample_set():
    dist = _genotype_distribution()
    # a LITERAL "unknown" string (not NA) must become a real SampleSet, not abstention
    labels = ["wildtype", "unknown", "unknown", "homo", "homo", "wildtype"]
    lg, sample_sets = label_genotype(dist, "column", {"labels": labels})

    by_name = {s.sample_set_name: s for s in sample_sets}
    assert "unknown" in by_name
    unknown_set = by_name["unknown"]
    assert len(unknown_set.sample_ids) == 2
    # a literal "unknown" is a real category, NOT a missing value
    assert unknown_set.provenance["evidence"]["is_missing_value"] is False
    # and it is NOT in unassigned
    assert unknown_set.sample_ids[0] not in lg.unassigned_sample_ids


def test_genotype_na_source_marks_is_missing_value():
    dist = _genotype_distribution()
    labels = ["wildtype", None, "homo", np.nan, "homo", "wildtype"]
    lg, sample_sets = label_genotype(
        dist, "column", {"labels": labels, "missing_name": "na_bucket"}
    )
    by_name = {s.sample_set_name: s for s in sample_sets}
    assert "na_bucket" in by_name
    na_set = by_name["na_bucket"]
    # both the None and the NaN sample landed in the missing bucket
    assert len(na_set.sample_ids) == 2
    assert na_set.provenance["evidence"]["is_missing_value"] is True
    # the missing bucket is a REAL SampleSet, not abstention
    assert na_set.sample_set_id in lg.sample_set_ids


def test_genotype_coverage_invariant_holds():
    dist = _genotype_distribution()
    labels = ["wildtype", "wildtype", "homo", "homo", None, "unknown"]
    lg, sample_sets = label_genotype(dist, "column", {"labels": labels})
    # validate_label_group already ran inside the labeler; assert coverage directly too
    validate_label_group(dist, lg, sample_sets)
    covered = set(lg.sample_id_to_sample_set_id) | set(lg.unassigned_sample_ids)
    assert covered == set(dist.sample_ids)


def test_genotype_forced_abstention_goes_to_unassigned():
    dist = _genotype_distribution()
    labels = ["wildtype"] * 6
    lg, sample_sets = label_genotype(
        dist, "column", {"labels": labels, "unassigned": {"e0", "e5"}}
    )
    assert set(lg.unassigned_sample_ids) == {"e0", "e5"}
    # the abstained samples are in NO SampleSet
    for s in sample_sets:
        assert "e0" not in s.sample_ids and "e5" not in s.sample_ids
    validate_label_group(dist, lg, sample_sets)


# --------------------------------------------------------------------------- #
# peak_finding fixtures
# --------------------------------------------------------------------------- #
def _fast_config():
    # A cheaper bootstrap vote so tests run quickly (the numbers only affect the
    # vote's precision, not the mapping being tested).
    from morphseq_investigation.core.distribution_records import PeakResolutionConfig
    from morphseq_investigation.core.peak_stability import PeakCountStabilityPolicy

    return PeakResolutionConfig(
        n_bootstrap_draws=15,
        bootstrap_sample_fraction=0.80,
        min_bootstrap_sample_size=10,
        count_stability_policy=PeakCountStabilityPolicy(min_mode_frequency=0.50),
        seed=7,
    )


def _bimodal_points(seed=0, n_per=90, sep=8.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=0.6, size=(n_per, 2))
    b = rng.normal(loc=(sep, sep), scale=0.6, size=(n_per, 2))
    return np.vstack([a, b])


def _unimodal_points(seed=1, n=180):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=(0.0, 0.0), scale=0.8, size=(n, 2))


def _peak_distribution(points, *, scope="b9d2", role="target"):
    sample_ids = [f"p{i}" for i in range(len(points))]
    return _make_distribution(points, sample_ids, scope=scope, role=role)


def _grid_for(points):
    sample_ids = tuple(f"p{i}" for i in range(len(points)))
    return build_grid(
        ("PC1", "PC2"),
        np.asarray(points, dtype=float),
        sample_ids,
        "pooled_min_max",
        {"resolution": 61},
    )


# --------------------------------------------------------------------------- #
# peak_finding labeler
# --------------------------------------------------------------------------- #
def test_peak_finding_bimodal_gives_two_sample_sets_with_geometry():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    grid = _grid_for(points)
    lg, sample_sets = label_peak_finding(
        dist, "peak_finding", {"grid": grid, "resolution_config": _fast_config()}
    )

    assert len(sample_sets) == 2, "clearly bimodal fixture must resolve 2 modes"
    for s in sample_sets:
        assert s.geometry is not None
        assert isinstance(s.geometry, SampleSetGeometry)
        assert s.geometry.center.shape == (2,)
        assert np.isfinite(s.geometry.radius)
        assert s.geometry.grid_id == grid.grid_id
        # geometry carries NO run-relative scalars
        assert not hasattr(s.geometry, "support_fraction")
        assert not hasattr(s.geometry, "prominence_rank")
    # resolved_peak_count is derived = len(sample_set_ids)
    assert len(lg.sample_set_ids) == 2


def test_peak_finding_unimodal_gives_one_sample_set():
    points = _unimodal_points()
    dist = _peak_distribution(points)
    grid = _grid_for(points)
    lg, sample_sets = label_peak_finding(
        dist, "peak_finding", {"grid": grid, "resolution_config": _fast_config()}
    )
    assert len(sample_sets) == 1


def test_peak_finding_per_sample_set_metrics_populated_off_geometry():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    grid = _grid_for(points)
    lg, sample_sets = label_peak_finding(
        dist, "peak_finding", {"grid": grid, "resolution_config": _fast_config()}
    )
    # per_sample_set_metrics holds the run-relative scalars, keyed by sample_set_id
    for s in sample_sets:
        metrics = lg.per_sample_set_metrics[s.sample_set_id]
        assert "support_fraction" in metrics
        assert "prominence_rank" in metrics
        assert "height_relative_to_max" in metrics
        assert "is_dominant" in metrics
    # exactly one dominant peak
    n_dominant = sum(
        lg.per_sample_set_metrics[s.sample_set_id]["is_dominant"] == 1.0 for s in sample_sets
    )
    assert n_dominant == 1


def test_peak_finding_artifacts_carry_grid_density_basins_with_grid_id():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    grid = _grid_for(points)
    lg, sample_sets = label_peak_finding(
        dist, "peak_finding", {"grid": grid, "resolution_config": _fast_config()}
    )
    art = lg.artifacts
    assert art is not None
    assert art.grid_id == grid.grid_id
    assert art.grid is grid
    assert art.density_grid is not None
    assert art.density_grid.grid_id == grid.grid_id
    assert art.basin_labels is not None
    assert art.basin_labels.shape == art.density_grid.density.shape


def test_peak_finding_vote_lives_in_provenance():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    grid = _grid_for(points)
    lg, sample_sets = label_peak_finding(
        dist, "peak_finding", {"grid": grid, "resolution_config": _fast_config()}
    )
    assert "is_reliable" in lg.provenance
    assert "vote" in lg.provenance
    assert "mode_peak_count" in lg.provenance["vote"]
    assert "peak_count_frequencies" in lg.provenance["vote"]


def test_peak_finding_vote_collapse_no_phantom_sample_sets():
    # Two nearby Gaussian clusters that the KDE reads as ONE mode. Naive
    # per-cluster counting would say 2; the density/vote resolves 1 -> ONE
    # SampleSet, and the rejected split is NOT a phantom SampleSet.
    rng = np.random.default_rng(3)
    a = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(120, 2))
    b = rng.normal(loc=(1.2, 0.0), scale=1.0, size=(120, 2))  # heavily overlapping
    points = np.vstack([a, b])
    dist = _peak_distribution(points)
    grid = _grid_for(points)
    lg, sample_sets = label_peak_finding(
        dist, "peak_finding", {"grid": grid, "resolution_config": _fast_config()}
    )
    # one coherent mode; never a phantom set per rejected candidate
    assert len(sample_sets) == 1
    # every accepted set corresponds to a real declared sample_set_id
    assert set(s.sample_set_id for s in sample_sets) == set(lg.sample_set_ids)
