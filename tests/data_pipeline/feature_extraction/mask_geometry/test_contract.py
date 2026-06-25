"""mask_geometry contract tests — spine first, then feature columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.feature_extraction.mask_geometry.contract import (
    MASK_GEOMETRY_TABLE_COLUMNS,
    validate_mask_geometry_features,
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
                "snip_id": snip_id,
                "embryo_id": embryo_id,
                "physical_embryo_id": PHYS,
                "experiment_id": EXP,
                "well_id": WELL,
                "image_id": image_id,
                "time_index": t,
                "channel_id": CHANNEL,
                "area_um2": 100.0,
                "perimeter_um": 40.0,
                "length_um": 12.0,
                "width_um": 8.0,
                "centroid_x_um": 5.0,
                "centroid_y_um": 5.0,
            }
        )
    return pd.DataFrame(rows, columns=MASK_GEOMETRY_TABLE_COLUMNS)


def test_valid_df_passes():
    validate_mask_geometry_features(_valid_df())


def test_missing_physical_embryo_id_fails():
    df = _valid_df().drop(columns=["physical_embryo_id"])
    with pytest.raises(ValueError, match="identity-spine"):
        validate_mask_geometry_features(df)


def test_missing_feature_column_fails():
    df = _valid_df().drop(columns=["area_um2"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_mask_geometry_features(df)


def test_null_feature_value_fails():
    df = _valid_df()
    df.loc[0, "perimeter_um"] = np.nan
    with pytest.raises(ValueError, match="null/non-numeric"):
        validate_mask_geometry_features(df)


def test_check_sources_requires_registered_animal():
    df = _valid_df()
    registry = pd.DataFrame({"physical_embryo_id": [PHYS]})
    validate_mask_geometry_features(
        df, physical_embryo_registry_df=registry, check_sources=True
    )

    empty_registry = pd.DataFrame({"physical_embryo_id": []})
    with pytest.raises(ValueError, match="not in the"):
        validate_mask_geometry_features(
            df, physical_embryo_registry_df=empty_registry, check_sources=True
        )
