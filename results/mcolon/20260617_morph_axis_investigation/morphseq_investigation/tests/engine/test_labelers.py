"""TASK_B — labeler tests: ``Distribution.detect_peaks`` writes a LABEL GROUP.

``detect_peaks`` is the SIBLING of ``with_label`` (TASK_0): it attaches a new
label column and returns a NEW Distribution wrapped as a ``DistributionLabelGroup``
(``.distribution`` carries the column). No ``(LabelGroup, [SampleSet])`` tuple
return survives from the pre-migration peer skeleton.
"""

from __future__ import annotations

import numpy as np
import pytest

from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.objects import (
    Distribution,
    DistributionLabelGroup,
    HDR,
    SampleSetGeometry,
    UNASSIGNED_LABEL,
)
from morphseq_investigation.engine.labelers import label_column_from_series


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_distribution(values, sample_ids, *, coordinates=None):
    coordinates = coordinates if coordinates is not None else {"scope_id": "b9d2", "time_bin": 30}
    return Distribution(
        distribution_id=make_distribution_id(coordinates),
        sample_ids=tuple(sample_ids),
        feature_names=("PC1", "PC2"),
        feature_values=np.asarray(values, dtype=float),
        coordinates=coordinates,
    )


def _bimodal_points(seed=0, n_per=90, sep=8.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=0.6, size=(n_per, 2))
    b = rng.normal(loc=(sep, sep), scale=0.6, size=(n_per, 2))
    return np.vstack([a, b])


def _unimodal_points(seed=1, n=180):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=(0.0, 0.0), scale=0.8, size=(n, 2))


def _peak_distribution(points, *, coordinates=None):
    sample_ids = [f"p{i}" for i in range(len(points))]
    return _make_distribution(points, sample_ids, coordinates=coordinates)


def _fast_resolution_config():
    # A cheaper bootstrap vote so tests run quickly (the numbers only affect the
    # vote's precision, not the peak-count answer being tested).
    from morphseq_investigation.core.distribution_records import PeakResolutionConfig
    from morphseq_investigation.core.peak_stability import PeakCountStabilityPolicy

    return PeakResolutionConfig(
        n_bootstrap_draws=15,
        bootstrap_sample_fraction=0.80,
        min_bootstrap_sample_size=10,
        count_stability_policy=PeakCountStabilityPolicy(min_mode_frequency=0.50),
        seed=7,
    )


def _detect(dist, **kwargs):
    from morphseq_investigation.engine.labelers import detect_peaks as _detect_peaks

    kwargs.setdefault("resolution_config", _fast_resolution_config())
    return _detect_peaks(dist, features=("PC1", "PC2"), **kwargs)


# --------------------------------------------------------------------------- #
# detect_peaks — bimodal fixture: writes resolved_peak column, residual unassigned
# --------------------------------------------------------------------------- #
def test_detect_peaks_bimodal_writes_resolved_peak_column_with_two_categories():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = _detect(dist)

    assert isinstance(lg, DistributionLabelGroup)
    column = lg.distribution.label_column("resolved_peak")
    categories = column.categories()
    assert len(categories) == 2, "clearly bimodal fixture must resolve 2 modes"
    assert set(categories) == {"peak_0", "peak_1"}


def test_detect_peaks_residual_is_unassigned_label():
    # Residual (non-accepted-mode) support goes through the ordinary
    # UNASSIGNED_LABEL machinery, not a phantom category.
    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = _detect(dist)
    column = lg.distribution.label_column("resolved_peak")
    values = set(column.values.values())
    # UNASSIGNED_LABEL is a legal value in the column (whether or not any
    # sample actually lands there for this fixture); categories() excludes it.
    assert UNASSIGNED_LABEL not in column.categories()
    assert values <= set(column.categories()) | {UNASSIGNED_LABEL}


def test_detect_peaks_unimodal_gives_one_category():
    points = _unimodal_points()
    dist = _peak_distribution(points)
    lg = _detect(dist)
    column = lg.distribution.label_column("resolved_peak")
    assert len(column.categories()) == 1


# --------------------------------------------------------------------------- #
# eager geometry — sample_sets("resolved_peak") carries geometry (rings survive)
# --------------------------------------------------------------------------- #
def test_sample_sets_resolved_peak_carry_geometry():
    points = _bimodal_points()
    dist = _peak_distribution(points)

    # via the public Distribution.detect_peaks surface (TASK_0 stub -> TASK_B body)
    lg = dist.detect_peaks(features=("PC1", "PC2"))
    sets = lg.distribution.sample_sets("resolved_peak")
    assert len(sets) >= 1
    for s in sets:
        assert s.geometry is not None


def test_sample_sets_resolved_peak_carry_typed_geometry_and_hdr_directly():
    # Distribution.sample_sets("resolved_peak") unpacks the eager CategoryShape
    # into the SampleSet's TYPED slots -- a proper SampleSetGeometry on
    # .geometry and an HDR on .hdr -- with NO wrapper and NO separate unpack call.
    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = _detect(dist)

    sets = lg.distribution.sample_sets("resolved_peak")
    assert len(sets) == 2
    for s in sets:
        assert isinstance(s.geometry, SampleSetGeometry)
        assert s.geometry.center.shape == (2,)
        assert np.isfinite(s.geometry.radius)
        assert isinstance(s.hdr, HDR)
        assert s.hdr.mask is not None
        # geometry carries NO run-relative scalars (kept off SampleSetGeometry)
        assert not hasattr(s.geometry, "support_fraction")
        assert not hasattr(s.geometry, "prominence_rank")


# --------------------------------------------------------------------------- #
# provenance — records method + features + spec
# --------------------------------------------------------------------------- #
def test_detect_peaks_provenance_records_method_features_spec():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = _detect(dist)

    column = lg.distribution.label_column("resolved_peak")
    prov = column.provenance
    assert prov is not None
    assert prov.method == "detect_peaks"
    assert prov.features == ("PC1", "PC2")
    assert "bandwidth_rule" in prov.spec
    assert "peak_detector_method" in prov.spec
    assert "is_reliable" in prov.spec
    assert "vote" in prov.spec
    # eager per-category geometry is present for every produced category
    assert set(prov.geometry) == set(column.categories())


# --------------------------------------------------------------------------- #
# returns a NEW Distribution — original untouched
# --------------------------------------------------------------------------- #
def test_detect_peaks_returns_new_distribution_original_unchanged():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = _detect(dist)

    assert "resolved_peak" not in dist.labels
    assert "resolved_peak" in lg.distribution.labels
    assert lg.distribution is not dist


def test_stub_wiring_matches_direct_call():
    # Distribution.detect_peaks (the TASK_0 stub) must delegate to the SAME
    # engine.labelers.detect_peaks body implemented here.
    points = _unimodal_points()
    dist = _peak_distribution(points)
    via_method = dist.detect_peaks(features=("PC1", "PC2"), spec=None)
    assert isinstance(via_method, DistributionLabelGroup)
    assert "resolved_peak" in via_method.distribution.labels
    assert "resolved_peak" not in dist.labels


# --------------------------------------------------------------------------- #
# local-peak-id honesty — two independent dists may both have peak_0, no
# cross-distribution correspondence claimed anywhere.
# --------------------------------------------------------------------------- #
def test_local_peak_ids_are_not_cross_distribution_comparable():
    points_a = _bimodal_points(seed=0)
    points_b = _bimodal_points(seed=5)
    dist_a = _peak_distribution(points_a, coordinates={"scope_id": "b9d2", "time_bin": 30})
    dist_b = _peak_distribution(points_b, coordinates={"scope_id": "b9d2", "time_bin": 48})

    lg_a = _detect(dist_a)
    lg_b = _detect(dist_b)

    cats_a = set(lg_a.distribution.label_column("resolved_peak").categories())
    cats_b = set(lg_b.distribution.label_column("resolved_peak").categories())
    assert "peak_0" in cats_a
    assert "peak_0" in cats_b
    # distinct distribution_ids -- nothing unifies "peak_0" across them.
    assert lg_a.distribution.distribution_id != lg_b.distribution.distribution_id
    # No shared-id / cross-distribution matching machinery exists on the
    # label group or its provenance -- assert the absence explicitly.
    assert not hasattr(lg_a, "match_peaks")
    assert not hasattr(lg_a, "global_peak_id")
    column_a = lg_a.distribution.label_column("resolved_peak")
    assert "global_peak_id" not in column_a.provenance.spec
    assert "matched_peak_id" not in column_a.provenance.spec


# --------------------------------------------------------------------------- #
# provided-column labeling — Distribution.with_label / label_column_from_series
# --------------------------------------------------------------------------- #
def test_label_column_from_series_bucket_na_and_provided_values():
    dist = _peak_distribution(_unimodal_points(n=6))
    labels = ["wildtype", "wildtype", None, "homo", np.nan, "homo"]
    d2 = label_column_from_series(dist, "genotype", labels, missing_name="na_bucket")

    column = d2.label_column("genotype")
    assert column.values[d2.sample_ids[2]] == "na_bucket"
    assert column.values[d2.sample_ids[4]] == "na_bucket"
    assert column.values[d2.sample_ids[0]] == "wildtype"
    assert set(column.categories()) == {"wildtype", "homo", "na_bucket"}
    # returns a NEW distribution; original untouched
    assert "genotype" not in dist.labels


def test_no_tuple_return_shape_remains():
    # The old peer-contract tuple return is retired: engine.labelers no longer
    # exposes label_genotype / label_peak_finding.
    import morphseq_investigation.engine.labelers as labelers_module

    assert not hasattr(labelers_module, "label_genotype")
    assert not hasattr(labelers_module, "label_peak_finding")
    assert not hasattr(labelers_module, "_finalize")


# --------------------------------------------------------------------------- #
# regression — peak COUNT per bin matches the old engine's counts.
#
# The b9d2 worked example (v0/b9d2_worked_example.py) demonstrates the
# emergence signal on real imaging QC data (target phenotype peak count grows
# 1 -> 2 as hpf increases), but records no fixed numeric table -- it computes
# counts at runtime from a CSV that is gitignored and not present in a fresh
# checkout. This regression instead builds a b9d2-LIKE synthetic fixture
# engineered to reproduce that documented qualitative signal (14hpf-like bin:
# 1 mode; later bins: 2 modes) at a FIXED, reproducible seed, and asserts the
# NEW detect_peaks output shape reproduces the SAME counts the OLD
# (LabelGroup, [SampleSet])-returning label_peak_finding would have produced
# for identical points/grid/config (the peak algorithm is byte-for-byte reused
# -- see engine/labelers.py's detect_peaks docstring -- only the OUTPUT SHAPE
# moved).
# --------------------------------------------------------------------------- #
def _b9d2_like_bin_points(hpf: int, seed: int = 100):
    """One synthetic 'target phenotype' point cloud per design hpf.

    14hpf: a single compact cluster (phenotype not yet resolvable).
    18/24/30/48hpf: two increasingly separated clusters (CE vs HTA emerging),
    matching the worked example's documented 1 -> 2 emergence signal.
    """
    rng = np.random.default_rng(seed + hpf)
    if hpf <= 14:
        return rng.normal(loc=(0.0, 0.0), scale=1.0, size=(120, 2))
    # separation grows with hpf so later bins are unambiguously bimodal.
    sep = 4.0 + 0.3 * (hpf - 14)
    a = rng.normal(loc=(0.0, 0.0), scale=0.8, size=(60, 2))
    b = rng.normal(loc=(sep, sep), scale=0.8, size=(60, 2))
    return np.vstack([a, b])


EXPECTED_PEAK_COUNTS = {14: 1, 18: 2, 24: 2, 30: 2, 48: 2}


@pytest.mark.parametrize("hpf", sorted(EXPECTED_PEAK_COUNTS))
def test_detect_peaks_peak_count_regression_matches_old_engine_per_bin(hpf):
    points = _b9d2_like_bin_points(hpf)
    dist = _peak_distribution(points, coordinates={"scope_id": "b9d2", "time_bin": hpf})
    lg = _detect(dist, resolution=61)
    column = lg.distribution.label_column("resolved_peak")
    assert len(column.categories()) == EXPECTED_PEAK_COUNTS[hpf], (
        f"{hpf}hpf: expected {EXPECTED_PEAK_COUNTS[hpf]} peak(s) "
        f"(pre-migration engine parity), got {len(column.categories())}"
    )
