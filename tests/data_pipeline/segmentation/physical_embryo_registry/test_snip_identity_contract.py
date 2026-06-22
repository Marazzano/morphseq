"""Tests for the identity-carrying spine validator ``validate_snip_grain_identity_columns``."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    validate_snip_grain_identity_columns,
)
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)


def _snip_row(experiment_id="20250912", well_index="B01", local_embryo_index=1, time_index=7):
    well_id = build_well_id(experiment_id, well_index)
    physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
    image_id = build_image_id(well_id, "BF", time_index)
    embryo_id = build_embryo_id(physical_embryo_id, image_id)
    snip_id = build_snip_id(embryo_id, image_id)
    return {
        "experiment_id": experiment_id,
        "well_id": well_id,
        "physical_embryo_id": physical_embryo_id,
        "embryo_id": embryo_id,
        "snip_id": snip_id,
        "image_id": image_id,
        "channel_id": "BF",
        "time_index": time_index,
    }


def _snip_df(n=2):
    return pd.DataFrame([_snip_row(local_embryo_index=1, time_index=t) for t in range(n)])


def test_valid_snip_grain_passes():
    validate_snip_grain_identity_columns(_snip_df(3), grain="snip")


def test_valid_embryo_grain_passes():
    df = pd.DataFrame([
        {
            "experiment_id": "20250912",
            "well_id": "20250912_B01",
            "physical_embryo_id": "20250912_B01_e01",
        },
        {
            "experiment_id": "20250912",
            "well_id": "20250912_B01",
            "physical_embryo_id": "20250912_B01_e02",
        },
    ])
    validate_snip_grain_identity_columns(df, grain="embryo")


def test_unknown_grain_raises():
    with pytest.raises(ValueError, match="unknown grain"):
        validate_snip_grain_identity_columns(_snip_df(1), grain="bogus")


def test_missing_spine_column_raises():
    df = _snip_df(2).drop(columns=["physical_embryo_id"])
    with pytest.raises(ValueError, match="missing required identity-spine column"):
        validate_snip_grain_identity_columns(df, grain="snip")


def test_null_spine_value_raises():
    df = _snip_df(2)
    df.loc[0, "physical_embryo_id"] = None
    with pytest.raises(ValueError, match="null"):
        validate_snip_grain_identity_columns(df, grain="snip")


def test_embryo_id_disagrees_with_physical_embryo_id_raises():
    # snip_id/embryo_id syntactically valid, but embryo_id points at a different animal.
    df = _snip_df(1)
    other = _snip_row(local_embryo_index=2)
    df.loc[0, "embryo_id"] = other["embryo_id"]
    with pytest.raises(ValueError, match="embryo_id .*encodes physical_embryo_id|not enough"):
        validate_snip_grain_identity_columns(df, grain="snip")


def test_snip_id_disagrees_with_embryo_id_raises():
    df = _snip_df(1)
    other = _snip_row(local_embryo_index=2)
    df.loc[0, "snip_id"] = other["snip_id"]
    with pytest.raises(ValueError, match="snip_id .*encodes embryo_id|not enough"):
        validate_snip_grain_identity_columns(df, grain="snip")


def test_channel_id_disagrees_with_image_id_raises():
    df = _snip_df(1)
    df.loc[0, "channel_id"] = "DAPI"
    with pytest.raises(ValueError, match="channel_id column .*disagrees"):
        validate_snip_grain_identity_columns(df, grain="snip")


def test_duplicate_snip_id_raises():
    df = pd.concat([_snip_df(1), _snip_df(1)], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        validate_snip_grain_identity_columns(df, grain="snip")


def test_check_sources_passes_when_registered():
    df = _snip_df(2)
    registry = pd.DataFrame({"physical_embryo_id": ["20250912_B01_e01"]})
    validate_snip_grain_identity_columns(
        df, grain="snip", physical_embryo_registry_df=registry, check_sources=True
    )


def test_check_sources_fails_when_unregistered():
    df = _snip_df(2)
    registry = pd.DataFrame({"physical_embryo_id": ["20250912_C04_e09"]})
    with pytest.raises(ValueError, match="not in the physical_embryo_registry"):
        validate_snip_grain_identity_columns(
            df, grain="snip", physical_embryo_registry_df=registry, check_sources=True
        )


def test_check_sources_requires_registry_df():
    with pytest.raises(ValueError, match="requires physical_embryo_registry_df"):
        validate_snip_grain_identity_columns(_snip_df(1), grain="snip", check_sources=True)
