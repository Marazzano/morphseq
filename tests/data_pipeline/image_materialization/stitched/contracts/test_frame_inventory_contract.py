"""Tests for frame_inventory_contract.py.

Three coverage areas:
  1. Import + public surface — the module is importable and exposes every expected symbol.
  2. Required columns — REQUIRED_FRAME_INVENTORY_COLUMNS contains exactly the expected atoms.
  3. Derived-id recompute logic — assert_derived_ids_consistent passes on correct ids and raises
     (loud) on atom/id disagreements for both well_id and image_id.
"""

import pytest
import pandas as pd

from data_pipeline.image_materialization.stitched.contracts.frame_inventory_contract import (
    REQUIRED_FRAME_INVENTORY_COLUMNS,
    DERIVED_FRAME_INVENTORY_COLUMNS,
    FrameInventorySpec,
    StitchedHandoffSpec,
    WellHandoff,
    assert_derived_ids_consistent,
    derive_well_id,
    derive_image_id,
)


# ---------------------------------------------------------------------------
# 1. Import and public surface
# ---------------------------------------------------------------------------


def test_module_importable():
    """The module loads without error."""
    # Import already happened at the top of this file; reaching here means it worked.
    assert REQUIRED_FRAME_INVENTORY_COLUMNS is not None


def test_dataclasses_importable():
    assert FrameInventorySpec is not None
    assert StitchedHandoffSpec is not None
    assert WellHandoff is not None


# ---------------------------------------------------------------------------
# 2. Required columns
# ---------------------------------------------------------------------------

_FORBIDDEN_IN_REQUIRED = {"well_id", "image_id"}

_FOUR_FRAME_KEY_ATOMS = {"experiment_id", "well_index", "channel_id", "time_index"}


def test_required_columns_contain_atoms():
    assert _FOUR_FRAME_KEY_ATOMS.issubset(set(REQUIRED_FRAME_INVENTORY_COLUMNS))


def test_derived_ids_not_in_required():
    assert not _FORBIDDEN_IN_REQUIRED.intersection(set(REQUIRED_FRAME_INVENTORY_COLUMNS))


def test_derived_columns_tuple():
    assert set(DERIVED_FRAME_INVENTORY_COLUMNS) == {"well_id", "image_id"}


def test_required_columns_is_tuple():
    assert isinstance(REQUIRED_FRAME_INVENTORY_COLUMNS, tuple)


def test_derived_columns_is_tuple():
    assert isinstance(DERIVED_FRAME_INVENTORY_COLUMNS, tuple)


def test_frame_inventory_spec_defaults():
    spec = FrameInventorySpec()
    assert spec.required_columns == REQUIRED_FRAME_INVENTORY_COLUMNS
    assert spec.required_channel == "BF"


# ---------------------------------------------------------------------------
# 3. Derived-id recompute logic
# ---------------------------------------------------------------------------

def _make_df(**overrides) -> pd.DataFrame:
    base = {
        "experiment_id": "20250912",
        "well_index": "B01",
        "channel_id": "BF",
        "time_index": 0,
        "well_id": "20250912_B01",
        "image_id": "20250912_B01_BF_t0000",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_consistent_ids_pass():
    df = _make_df()
    assert_derived_ids_consistent(df)  # must not raise


def test_bad_well_id_raises():
    df = _make_df(well_id="WRONG_ID")
    with pytest.raises(ValueError, match="well_id inconsistent"):
        assert_derived_ids_consistent(df)


def test_bad_image_id_raises():
    df = _make_df(image_id="WRONG_IMAGE_ID")
    with pytest.raises(ValueError, match="image_id inconsistent"):
        assert_derived_ids_consistent(df)


def test_no_derived_columns_is_noop():
    """If the producer didn't supply well_id / image_id, assert_derived_ids_consistent is silent."""
    df = pd.DataFrame([{
        "experiment_id": "20250912",
        "well_index": "B01",
        "channel_id": "BF",
        "time_index": 0,
    }])
    assert_derived_ids_consistent(df)  # must not raise


def test_derive_well_id():
    assert derive_well_id("20250912", "B01") == "20250912_B01"


def test_derive_image_id():
    assert derive_image_id("20250912_B01", "BF", 0) == "20250912_B01_BF_t0000"
    assert derive_image_id("20250912_B01", "GFP", 12) == "20250912_B01_GFP_t0012"


def test_multiple_rows_all_consistent():
    rows = [
        {"experiment_id": "20250912", "well_index": "B01", "channel_id": "BF", "time_index": i,
         "well_id": "20250912_B01", "image_id": f"20250912_B01_BF_t{i:04d}"}
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    assert_derived_ids_consistent(df)


def test_one_bad_row_in_batch_raises():
    rows = [
        {"experiment_id": "20250912", "well_index": "B01", "channel_id": "BF", "time_index": i,
         "well_id": "20250912_B01", "image_id": f"20250912_B01_BF_t{i:04d}"}
        for i in range(4)
    ]
    # Corrupt the last row's image_id
    rows.append({"experiment_id": "20250912", "well_index": "B01", "channel_id": "BF", "time_index": 4,
                 "well_id": "20250912_B01", "image_id": "WRONG"})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="image_id inconsistent"):
        assert_derived_ids_consistent(df)
