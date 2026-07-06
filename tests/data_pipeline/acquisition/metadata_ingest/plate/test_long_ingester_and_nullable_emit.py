"""Step 4 — long biology ingester (no-plate path) + canonical nullable emit."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_loader import ingest_long_well_table
from data_pipeline.acquisition.metadata_ingest.plate.plate_processing import _ensure_canonical_columns
from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract import (
    REQUIRED_PLATE_METADATA_COLUMNS,
    validate_plate_metadata,
)
from data_pipeline.shared.identifiers import build_well_id


# --- long ingester --------------------------------------------------------------------------

def test_long_ingests_and_normalizes_well_key():
    df = pd.DataFrame({"well": ["A1", "B12"], "genotype": ["wt", "cep290"]})
    out = ingest_long_well_table(df, page_name="biology")
    assert list(out["well_index"]) == ["A01", "B12"]
    assert list(out["genotype"]) == ["wt", "cep290"]


def test_long_accepts_well_name_alias():
    df = pd.DataFrame({"well_name": ["A01"], "medium": ["E3"]})
    out = ingest_long_well_table(df, page_name="biology")
    assert list(out["well_index"]) == ["A01"]
    assert "medium" in out.columns


def test_long_age_alias_normalizes_to_start_age_hpf():
    df = pd.DataFrame({"well_index": ["A01"], "age_hpf": [24.0]})
    out = ingest_long_well_table(df, page_name="biology")
    assert "start_age_hpf" in out.columns
    assert "age_hpf" not in out.columns
    assert out["start_age_hpf"].iloc[0] == 24.0


def test_long_conflicting_age_aliases_fail():
    df = pd.DataFrame({"well_index": ["A01"], "start_age_hpf": [24.0], "age_hpf": [30.0]})
    with pytest.raises(ValueError, match="disagreeing"):
        ingest_long_well_table(df, page_name="biology")


def test_long_agreeing_age_aliases_collapse_to_one():
    df = pd.DataFrame({"well_index": ["A01"], "start_age_hpf": [24.0], "age_hpf": [24.0]})
    out = ingest_long_well_table(df, page_name="biology")
    assert "start_age_hpf" in out.columns
    assert "age_hpf" not in out.columns


def test_long_supplied_well_id_is_dropped_not_trusted():
    df = pd.DataFrame({"well_index": ["A01"], "well_id": ["bogus_value"], "genotype": ["wt"]})
    out = ingest_long_well_table(df, page_name="biology")
    assert "well_id" not in out.columns  # identity minted downstream, never trusted as authored


def test_long_no_well_key_fails():
    df = pd.DataFrame({"genotype": ["wt"]})
    with pytest.raises(ValueError, match="no well-key column"):
        ingest_long_well_table(df, page_name="biology")


def test_long_no_value_columns_fails():
    df = pd.DataFrame({"well_index": ["A01", "B01"]})
    with pytest.raises(ValueError, match="no value columns"):
        ingest_long_well_table(df, page_name="biology")


def test_long_duplicate_well_index_fails():
    df = pd.DataFrame({"well_index": ["A01", "A01"], "genotype": ["wt", "homo"]})
    with pytest.raises(ValueError, match="duplicate well_index"):
        ingest_long_well_table(df, page_name="biology")


def test_long_out_of_range_well_fails():
    df = pd.DataFrame({"well_index": ["I01"], "genotype": ["wt"]})  # I is outside A-H
    with pytest.raises(ValueError):
        ingest_long_well_table(df, page_name="biology")


def test_long_from_csv_path(tmp_path):
    csv = tmp_path / "well_metadata.csv"
    pd.DataFrame({"well": ["A01"], "genotype": ["wt"]}).to_csv(csv, index=False)
    out = ingest_long_well_table(csv, page_name="biology")
    assert list(out["well_index"]) == ["A01"]


# --- canonical nullable emit ----------------------------------------------------------------

def test_ensure_canonical_columns_fills_absent_with_na():
    # A user supplied only genotype; medium/temperature/start_age_hpf must still appear (all-NA).
    df = pd.DataFrame({
        "experiment_id": ["20250912"],
        "well_index": ["A01"],
        "well_id": [build_well_id("20250912", "A01")],
        "genotype": ["wt"],
    })
    out = _ensure_canonical_columns(df)
    for col in REQUIRED_PLATE_METADATA_COLUMNS:
        assert col in out.columns
    assert out["medium"].isna().all()
    assert out["temperature"].isna().all()
    assert out["start_age_hpf"].isna().all()
    assert out["genotype"].iloc[0] == "wt"


def test_l2_passes_on_all_na_but_present_skeleton():
    # The whole point: a sparse external biology table still passes L2 (nulls allowed at plate level).
    df = pd.DataFrame({
        "experiment_id": ["20250912"],
        "well_index": ["A01"],
        "well_id": [build_well_id("20250912", "A01")],
        "genotype": ["wt"],
    })
    out = _ensure_canonical_columns(df)
    validate_plate_metadata(out)  # must not raise
