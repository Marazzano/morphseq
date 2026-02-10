"""Tests for compact OT feature extraction."""

from __future__ import annotations

import numpy as np

from analyze.utils.optimal_transport import UOTResult
from analyze.optimal_transport_morphometrics.uot_masks.feature_compaction.features import (
    dct_radial_band_energy_fractions,
    extract_pair_feature_record,
    upsert_ot_feature_matrix_parquet,
)


def _make_result() -> UOTResult:
    coupling = np.array(
        [
            [0.6, 0.4],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    vel = np.zeros((8, 8, 2), dtype=np.float32)
    vel[2:6, 2:6, 0] = 2.0
    vel[2:6, 2:6, 1] = 1.0
    mc = np.zeros((8, 8), dtype=np.float32)
    md = np.zeros((8, 8), dtype=np.float32)
    mc[2:6, 2:6] = 0.2
    md[2:6, 2:6] = 0.1

    return UOTResult(
        cost=10.0,
        coupling=coupling,
        mass_created_px=mc,
        mass_destroyed_px=md,
        velocity_px_per_frame_yx=vel,
        support_src_yx=np.array([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32),
        support_tgt_yx=np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32),
        weights_src=np.array([1.0, 1.0], dtype=np.float32),
        weights_tgt=np.array([1.0, 1.0], dtype=np.float32),
        transform_meta={},
        diagnostics={
            "metrics": {
                "total_transport_cost_um2": 10.0,
                "mean_transport_distance_um": 2.5,
                "specific_transport_cost_um2_per_mass": 5.0,
                "transported_mass_pct_src": 90.0,
                "transported_mass_pct_tgt": 85.0,
                "created_mass_pct": 4.0,
                "destroyed_mass_pct": 3.0,
                "mass_ratio_crop": 1.1,
                "mass_delta_crop": 2.0,
                "proportion_transported": 0.9,
            }
        },
    )


def test_dct_band_fractions_sum_to_one():
    field = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    bands = dct_radial_band_energy_fractions(field, n_bands=8)
    assert bands.shape == (8,)
    assert np.isclose(float(bands.sum()), 1.0, atol=1e-6)
    assert np.all(bands >= 0)


def test_extract_pair_feature_record_contains_expected_keys():
    result = _make_result()
    rec = extract_pair_feature_record(
        run_id="run_a",
        pair_id="pair_a",
        result=result,
        backend="OTT",
        n_bands=8,
    )
    assert rec["run_id"] == "run_a"
    assert rec["pair_id"] == "pair_a"
    assert rec["backend"] == "OTT"
    assert rec["n_dct_bands"] == 8
    for k in [
        "ot_total_transport_cost_um2",
        "ot_mean_transport_distance_um",
        "ot_specific_transport_cost_um2_per_mass",
        "bar_disp_mean_um",
        "bar_disp_p95_um",
        "bar_disp_max_um",
    ]:
        assert k in rec

    for prefix in ("dct_vx_band_", "dct_vy_band_", "dct_div_band_", "dct_curl_band_"):
        for i in range(8):
            assert f"{prefix}{i:02d}" in rec

    # Fractions for each field should sum to ~1
    for prefix in ("dct_vx_band_", "dct_vy_band_", "dct_div_band_", "dct_curl_band_"):
        vals = np.array([rec[f"{prefix}{i:02d}"] for i in range(8)], dtype=np.float64)
        assert np.isclose(float(vals.sum()), 1.0, atol=1e-6)


def test_upsert_ot_feature_matrix_parquet_idempotent(tmp_path):
    p = tmp_path / "ot_feature_matrix.parquet"
    row1 = {"run_id": "r1", "pair_id": "p1", "backend": "OTT", "n_dct_bands": 8, "x": 1.0}
    row2 = {"run_id": "r1", "pair_id": "p1", "backend": "OTT", "n_dct_bands": 8, "x": 2.0}
    upsert_ot_feature_matrix_parquet(p, [row1])
    out = upsert_ot_feature_matrix_parquet(p, [row2])
    assert len(out) == 1
    assert float(out["x"].iloc[0]) == 2.0
    assert str(out["run_id"].dtype) == "string"
    assert str(out["pair_id"].dtype) == "string"
    assert str(out["backend"].dtype) == "category"
