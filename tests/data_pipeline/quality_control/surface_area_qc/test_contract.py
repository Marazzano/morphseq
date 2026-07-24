"""surface_area_qc contract tests — spine first, then the boolean sa_outlier_flag."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.surface_area_qc.contract import (
    SURFACE_AREA_QC_TABLE_COLUMNS,
    validate_surface_area_qc,
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


def _valid_df(n=2, flag=False):
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
                "sa_outlier_flag": flag,
                "surface_area_qc_applicability": "exclusion",
            }
        )
    df = pd.DataFrame(rows, columns=SURFACE_AREA_QC_TABLE_COLUMNS)
    df["sa_outlier_flag"] = df["sa_outlier_flag"].astype(bool)
    return df


def test_valid_df_passes():
    validate_surface_area_qc(_valid_df())


def test_missing_physical_embryo_id_fails():
    df = _valid_df().drop(columns=["physical_embryo_id"])
    with pytest.raises(ValueError, match="identity-spine"):
        validate_surface_area_qc(df)


def test_missing_flag_column_fails():
    df = _valid_df().drop(columns=["sa_outlier_flag"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_surface_area_qc(df)


def test_null_flag_fails():
    df = _valid_df()
    df["sa_outlier_flag"] = pd.Series([True, None], dtype="object")
    with pytest.raises(ValueError, match="null value"):
        validate_surface_area_qc(df)


def test_non_boolean_flag_fails():
    df = _valid_df()
    df["sa_outlier_flag"] = [1, 0]  # int, not bool
    with pytest.raises(ValueError, match="boolean dtype"):
        validate_surface_area_qc(df)


def test_check_sources_requires_registered_animal():
    df = _valid_df()
    validate_surface_area_qc(
        df, physical_embryo_registry_df=pd.DataFrame({"physical_embryo_id": [PHYS]}), check_sources=True
    )
    with pytest.raises(ValueError, match="not in the"):
        validate_surface_area_qc(
            df, physical_embryo_registry_df=pd.DataFrame({"physical_embryo_id": []}), check_sources=True
        )


def test_not_applicable_requires_false_flag():
    df = _valid_df()
    df["surface_area_qc_applicability"] = "not_applicable"
    validate_surface_area_qc(df)
    df.loc[0, "sa_outlier_flag"] = True
    with pytest.raises(ValueError, match="must carry sa_outlier_flag=False"):
        validate_surface_area_qc(df)
