"""focus_qc contract tests — spine first, then the metric + flag."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.focus_qc.contract import (
    FOCUS_QC_TABLE_COLUMNS,
    validate_focus_qc,
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
                "interior_strong_edge_fraction": 0.65,
                "interior_n_px": 1000,
                "focus_flag": False,
                "focus_qc_applicability": "exclusion",
            }
        )
    df = pd.DataFrame(rows, columns=FOCUS_QC_TABLE_COLUMNS)
    df["focus_flag"] = df["focus_flag"].astype(bool)
    return df


def test_valid_df_passes():
    validate_focus_qc(_valid_df())


def test_missing_physical_embryo_id_fails():
    with pytest.raises(ValueError, match="identity-spine"):
        validate_focus_qc(_valid_df().drop(columns=["physical_embryo_id"]))


def test_missing_flag_column_fails():
    with pytest.raises(ValueError, match="missing required column"):
        validate_focus_qc(_valid_df().drop(columns=["focus_flag"]))


def test_missing_metric_column_fails():
    with pytest.raises(ValueError, match="missing required column"):
        validate_focus_qc(_valid_df().drop(columns=["interior_strong_edge_fraction"]))


def test_null_flag_fails():
    df = _valid_df()
    df["focus_flag"] = pd.Series([True, None], dtype="object")
    with pytest.raises(ValueError, match="null value"):
        validate_focus_qc(df)


def test_non_boolean_flag_fails():
    df = _valid_df()
    df["focus_flag"] = [1, 0]
    with pytest.raises(ValueError, match="boolean dtype"):
        validate_focus_qc(df)


def test_null_metric_fails():
    df = _valid_df()
    df["interior_strong_edge_fraction"] = [0.5, None]
    with pytest.raises(ValueError, match="null value"):
        validate_focus_qc(df)


def test_non_numeric_metric_fails():
    df = _valid_df()
    df["interior_n_px"] = ["a", "b"]
    with pytest.raises(ValueError, match="must be numeric"):
        validate_focus_qc(df)


def test_diagnostic_only_focus_is_valid():
    df = _valid_df()
    df["focus_qc_applicability"] = "diagnostic_only"
    validate_focus_qc(df)
