"""TASK_B (commit 1) — detect_peaks writes the resolved_peak label column.

Eager geometry / HDR wiring + retirement of the old tuple-return labelers land
in the next checkpoint; this commit proves the CALLS + column-write only.
"""

from __future__ import annotations

import numpy as np

from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.objects import Distribution, DistributionLabelGroup


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


def _peak_distribution(points, *, coordinates=None):
    sample_ids = [f"p{i}" for i in range(len(points))]
    return _make_distribution(points, sample_ids, coordinates=coordinates)


def _fast_resolution_config():
    from morphseq_investigation.core.distribution_records import PeakResolutionConfig
    from morphseq_investigation.core.peak_stability import PeakCountStabilityPolicy

    return PeakResolutionConfig(
        n_bootstrap_draws=15,
        bootstrap_sample_fraction=0.80,
        min_bootstrap_sample_size=10,
        count_stability_policy=PeakCountStabilityPolicy(min_mode_frequency=0.50),
        seed=7,
    )


def test_detect_peaks_bimodal_writes_resolved_peak_column_with_two_categories():
    from morphseq_investigation.engine.labelers import detect_peaks

    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = detect_peaks(
        dist,
        features=("PC1", "PC2"),
        resolution_config=_fast_resolution_config(),
    )

    assert isinstance(lg, DistributionLabelGroup)
    column = lg.distribution.label_column("resolved_peak")
    categories = column.categories()
    assert len(categories) == 2, "clearly bimodal fixture must resolve 2 modes"
    assert set(categories) == {"peak_0", "peak_1"}


def test_detect_peaks_returns_new_distribution_original_unchanged():
    from morphseq_investigation.engine.labelers import detect_peaks

    points = _bimodal_points()
    dist = _peak_distribution(points)
    lg = detect_peaks(
        dist,
        features=("PC1", "PC2"),
        resolution_config=_fast_resolution_config(),
    )

    assert "resolved_peak" not in dist.labels
    assert "resolved_peak" in lg.distribution.labels
    assert lg.distribution is not dist


def test_stub_wiring_matches_direct_call():
    points = _bimodal_points()
    dist = _peak_distribution(points)
    via_method = dist.detect_peaks(features=("PC1", "PC2"), spec=None)
    assert isinstance(via_method, DistributionLabelGroup)
    assert "resolved_peak" in via_method.distribution.labels
    assert "resolved_peak" not in dist.labels
