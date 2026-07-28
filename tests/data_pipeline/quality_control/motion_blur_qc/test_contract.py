"""motion_blur_qc contract tests — spine first, then metrics + flag."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.motion_blur_qc.contract import (
    MOTION_BLUR_QC_TABLE_COLUMNS,
    validate_motion_blur_qc,
)
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


def _valid_df(n=2):
    rows = []
    for t in range(n):
        image_id = build_image_id(WELL, CHANNEL, t)
        embryo_id = build_embryo_id(PHYS, image_id)
        snip_id = build_snip_id(embryo_id, image_id)
        rows.append(
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": PHYS,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "mask_pixel_ncc_mean": 0.98,
                "mask_pixel_ncc_min": 0.95,
                "mask_pixel_ncc_p05": 0.95,
                "mask_pixel_bad_pair_frac": 0.0,
                "mask_pixel_longest_bad_run": 0,
                "n_z_planes": 3,
                "n_z_pairs": 2,
                "n_valid_z_pairs": 2,
                "n_flat_z_pairs": 0,
                "n_mask_pixels": 100,
                "motion_blur_flag": False,
            }
        )
    df = pd.DataFrame(rows, columns=MOTION_BLUR_QC_TABLE_COLUMNS)
    df["motion_blur_flag"] = df["motion_blur_flag"].astype(bool)
    return df


def test_valid_df_passes():
    validate_motion_blur_qc(_valid_df())


def test_missing_spine_column_fails():
    with pytest.raises(ValueError, match="identity-spine"):
        validate_motion_blur_qc(_valid_df().drop(columns=["physical_embryo_id"]))


def test_missing_flag_column_fails():
    with pytest.raises(ValueError, match="missing required column"):
        validate_motion_blur_qc(_valid_df().drop(columns=["motion_blur_flag"]))


def test_null_flag_fails():
    df = _valid_df()
    df["motion_blur_flag"] = pd.Series([True, None], dtype="object")
    with pytest.raises(ValueError, match="null value"):
        validate_motion_blur_qc(df)


def test_non_boolean_flag_fails():
    df = _valid_df()
    df["motion_blur_flag"] = [1, 0]
    with pytest.raises(ValueError, match="boolean dtype"):
        validate_motion_blur_qc(df)


def test_null_metric_fails():
    df = _valid_df()
    df["mask_pixel_ncc_mean"] = [0.95, None]
    with pytest.raises(ValueError, match="null value"):
        validate_motion_blur_qc(df)


def test_non_numeric_metric_fails():
    df = _valid_df()
    df["n_valid_z_pairs"] = ["a", "b"]
    with pytest.raises(ValueError, match="must be numeric"):
        validate_motion_blur_qc(df)
