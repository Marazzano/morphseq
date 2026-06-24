"""mask_quality_qc contract tests — spine first, then the three boolean flags."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.mask_quality_qc.contract import (
    MASK_QUALITY_QC_REQUIRED_COLUMNS,
    validate_mask_quality_qc,
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
_FLAGS = ("edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag")


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
                "edge_flag": False,
                "discontinuous_mask_flag": False,
                "overlapping_mask_flag": False,
            }
        )
    df = pd.DataFrame(rows, columns=MASK_QUALITY_QC_REQUIRED_COLUMNS)
    for f in _FLAGS:
        df[f] = df[f].astype(bool)
    return df


def test_valid_df_passes():
    validate_mask_quality_qc(_valid_df())


def test_missing_physical_embryo_id_fails():
    with pytest.raises(ValueError, match="identity-spine"):
        validate_mask_quality_qc(_valid_df().drop(columns=["physical_embryo_id"]))


def test_missing_flag_column_fails():
    with pytest.raises(ValueError, match="missing required column"):
        validate_mask_quality_qc(_valid_df().drop(columns=["overlapping_mask_flag"]))


def test_null_flag_fails():
    df = _valid_df()
    df["edge_flag"] = pd.Series([True, None], dtype="object")
    with pytest.raises(ValueError, match="null value"):
        validate_mask_quality_qc(df)


def test_non_boolean_flag_fails():
    df = _valid_df()
    df["discontinuous_mask_flag"] = [1, 0]
    with pytest.raises(ValueError, match="boolean dtype"):
        validate_mask_quality_qc(df)


def test_no_composite_flag_in_contract():
    assert "mask_quality_flag" not in MASK_QUALITY_QC_REQUIRED_COLUMNS
