"""surface_area_qc compute + config tests.

Covers the spec "Done When": low / normal / high / missing / duplicate snip_id, stage-binned
behavior (same area flags at one stage, passes at another), and the self-documenting band
statement reflecting resolved k_lower / k_upper.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.surface_area_qc.compute import (
    compute_surface_area_qc_flags,
)
from data_pipeline.quality_control.surface_area_qc.config import band_statement, resolve_config
from data_pipeline.quality_control.surface_area_qc.contract import validate_surface_area_qc
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"
PHYS = build_physical_embryo_id(WELL, 1)

# Flat reference band: p5=100, p95=200 at every stage from 10..50 hpf.
_FLAT_REF = pd.DataFrame(
    {
        "stage_hpf": [10.0, 50.0],
        "p5": [100.0, 100.0],
        "p50": [150.0, 150.0],
        "p95": [200.0, 200.0],
        "n": [50, 50],
    }
)
# Stage-dependent band: small at 10 hpf, large at 50 hpf.
_RAMP_REF = pd.DataFrame(
    {
        "stage_hpf": [10.0, 50.0],
        "p5": [100.0, 1000.0],
        "p50": [150.0, 1500.0],
        "p95": [200.0, 2000.0],
        "n": [50, 50],
    }
)


def _snip(t):
    image_id = build_image_id(WELL, CHANNEL, t)
    embryo_id = build_embryo_id(PHYS, image_id)
    return build_snip_id(embryo_id, image_id), embryo_id, image_id


def _universe(snip_ids_with_embryo):
    rows = []
    for snip_id, embryo_id in snip_ids_with_embryo:
        rows.append(
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": PHYS,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
            }
        )
    return pd.DataFrame(rows)


def _build(areas, stages, ref=_FLAT_REF, config=None):
    """areas/stages: dict snip_index -> value. Returns the computed QC df."""
    config = config or resolve_config()
    snips = {t: _snip(t) for t in areas}
    universe = _universe([(snips[t][0], snips[t][1]) for t in areas])
    mask_geometry = pd.DataFrame(
        [{"snip_id": snips[t][0], "area_um2": areas[t]} for t in areas]
    )
    stage_df = pd.DataFrame(
        [{"snip_id": snips[t][0], "predicted_stage_hpf": stages[t]} for t in areas]
    )
    return compute_surface_area_qc_flags(mask_geometry, stage_df, universe, ref, config=config)


def test_normal_area_not_flagged():
    out = _build(areas={0: 150.0}, stages={0: 30.0})
    assert out["sa_outlier_flag"].tolist() == [False]
    validate_surface_area_qc(out)


def test_too_large_flagged():
    # area 300 > 1.4 * p95(200) = 280 -> flagged
    out = _build(areas={0: 300.0}, stages={0: 30.0})
    assert out["sa_outlier_flag"].tolist() == [True]


def test_too_small_flagged():
    # area 50 < 0.7 * p5(100) = 70 -> flagged
    out = _build(areas={0: 50.0}, stages={0: 30.0})
    assert out["sa_outlier_flag"].tolist() == [True]


def test_stage_binned_same_area_different_verdict():
    # Same area 250 against the ramp band. Tolerance bands:
    #   stage 10: [0.7*100, 1.4*200] = [70, 280]   -> 250 is INSIDE  -> not flagged
    #   stage 50: [0.7*1000, 1.4*2000] = [700, 2800] -> 250 < 700    -> flagged (too small)
    # Proves the verdict is stage-driven, not area-only.
    out_small_stage = _build(areas={0: 250.0}, stages={0: 10.0}, ref=_RAMP_REF)
    out_large_stage = _build(areas={1: 250.0}, stages={1: 50.0}, ref=_RAMP_REF)
    assert out_small_stage["sa_outlier_flag"].tolist() == [False]
    assert out_large_stage["sa_outlier_flag"].tolist() == [True]


def test_missing_stage_fails_loud():
    snip_id, embryo_id, _ = _snip(0)
    universe = _universe([(snip_id, embryo_id)])
    mask_geometry = pd.DataFrame([{"snip_id": snip_id, "area_um2": 150.0}])
    stage_df = pd.DataFrame(columns=["snip_id", "predicted_stage_hpf"])
    with pytest.raises(ValueError, match="no stage_predictions row"):
        compute_surface_area_qc_flags(mask_geometry, stage_df, universe, _FLAT_REF, config=resolve_config())


def test_missing_area_fails_loud():
    snip_id, embryo_id, _ = _snip(0)
    universe = _universe([(snip_id, embryo_id)])
    mask_geometry = pd.DataFrame(columns=["snip_id", "area_um2"])
    stage_df = pd.DataFrame([{"snip_id": snip_id, "predicted_stage_hpf": 30.0}])
    with pytest.raises(ValueError, match="no mask_geometry row"):
        compute_surface_area_qc_flags(mask_geometry, stage_df, universe, _FLAT_REF, config=resolve_config())


def test_duplicate_snip_in_universe_fails_loud():
    snip_id, embryo_id, _ = _snip(0)
    universe = _universe([(snip_id, embryo_id), (snip_id, embryo_id)])
    mask_geometry = pd.DataFrame([{"snip_id": snip_id, "area_um2": 150.0}])
    stage_df = pd.DataFrame([{"snip_id": snip_id, "predicted_stage_hpf": 30.0}])
    with pytest.raises(ValueError, match="duplicate snip_id"):
        compute_surface_area_qc_flags(mask_geometry, stage_df, universe, _FLAT_REF, config=resolve_config())


def test_output_is_full_spine_and_validates():
    out = _build(areas={0: 150.0, 1: 300.0}, stages={0: 30.0, 1: 30.0})
    for col in ("experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id"):
        assert col in out.columns
    validate_surface_area_qc(out)


def test_band_statement_reflects_resolved_k():
    stmt = band_statement(resolve_config())
    assert "k_lower(0.70)" in stmt and "k_upper(1.40)" in stmt
    assert "predicted_stage_hpf" in stmt
    assert "too small" in stmt and "too large" in stmt


def test_band_statement_reflects_override():
    stmt = band_statement(resolve_config({"k_upper": 1.5, "k_lower": 0.6}))
    assert "k_lower(0.60)" in stmt and "k_upper(1.50)" in stmt


def test_unknown_config_key_fails():
    with pytest.raises(ValueError, match="unknown config key"):
        resolve_config({"bogus": 1})
