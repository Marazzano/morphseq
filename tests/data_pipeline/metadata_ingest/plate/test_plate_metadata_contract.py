"""Tests for plate_metadata_contract (L2 schema validation)."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.metadata_ingest.plate.plate_metadata_contract import (
    REQUIRED_PLATE_METADATA_COLUMNS,
    REQUIRED_PLATE_METADATA_FIELDS,
    validate_plate_metadata,
)
from data_pipeline.shared.identifiers import build_well_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_df(n: int = 3) -> pd.DataFrame:
    experiment_id = "20250101"
    well_indices = [f"A{i:02d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "experiment_id": experiment_id,
        "well_index": well_indices,
        "well_id": [build_well_id(experiment_id, w) for w in well_indices],
        "genotype": ["wt"] * n,
        "start_age_hpf": [24.0] * n,
        "temperature": [28.5] * n,
        "medium": ["E3"] * n,
    })


# ---------------------------------------------------------------------------
# Shape of the constants
# ---------------------------------------------------------------------------

def test_required_columns_is_tuple():
    assert isinstance(REQUIRED_PLATE_METADATA_COLUMNS, tuple)


def test_required_fields_is_tuple():
    assert isinstance(REQUIRED_PLATE_METADATA_FIELDS, tuple)


def test_required_fields_are_subset_of_required_columns():
    for f in REQUIRED_PLATE_METADATA_FIELDS:
        assert f in REQUIRED_PLATE_METADATA_COLUMNS


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_df_passes():
    validate_plate_metadata(_make_valid_df())


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

def test_missing_genotype_raises_with_hint():
    df = _make_valid_df().drop(columns=["genotype"])
    with pytest.raises(ValueError, match="genotype"):
        validate_plate_metadata(df)


def test_missing_temperature_raises():
    """Guards against re-introducing the silent temperature=28.5 default."""
    df = _make_valid_df().drop(columns=["temperature"])
    with pytest.raises(ValueError, match="temperature"):
        validate_plate_metadata(df)


def test_missing_start_age_hpf_raises():
    df = _make_valid_df().drop(columns=["start_age_hpf"])
    with pytest.raises(ValueError, match="start_age_hpf"):
        validate_plate_metadata(df)


def test_missing_medium_raises():
    df = _make_valid_df().drop(columns=["medium"])
    with pytest.raises(ValueError, match="medium"):
        validate_plate_metadata(df)


def test_error_message_includes_fix_hint():
    df = _make_valid_df().drop(columns=["genotype"])
    with pytest.raises(ValueError, match="Add an 8×12"):
        validate_plate_metadata(df)


# ---------------------------------------------------------------------------
# Null biological values are ALLOWED at L2
# ---------------------------------------------------------------------------

def test_required_column_with_null_values_passes():
    df = _make_valid_df()
    df.loc[0, "temperature"] = None
    validate_plate_metadata(df)


def test_required_column_all_null_still_passes():
    df = _make_valid_df()
    df["genotype"] = None
    validate_plate_metadata(df)


# ---------------------------------------------------------------------------
# well_id consistency
# ---------------------------------------------------------------------------

def test_inconsistent_well_id_raises():
    df = _make_valid_df()
    df.loc[0, "well_id"] = "wrong_well_id"
    with pytest.raises(ValueError, match="well_id is inconsistent"):
        validate_plate_metadata(df)


# ---------------------------------------------------------------------------
# Duplicate key check
# ---------------------------------------------------------------------------

def test_duplicate_experiment_id_well_id_raises():
    df = _make_valid_df()
    dup_row = df.iloc[[0]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_plate_metadata(df)


# ---------------------------------------------------------------------------
# Legacy schema re-export
# ---------------------------------------------------------------------------

def test_legacy_schema_import_still_works():
    from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
    assert "genotype" in REQUIRED_COLUMNS_PLATE_METADATA
    assert "temperature" in REQUIRED_COLUMNS_PLATE_METADATA
