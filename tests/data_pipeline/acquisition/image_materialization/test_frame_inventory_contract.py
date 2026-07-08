"""Tests for frame_inventory_contract.py.

Three coverage areas:
  1. Import + public surface — the module is importable and exposes every expected symbol.
  2. Required columns — REQUIRED_FRAME_INVENTORY_COLUMNS contains exactly the expected atoms.
  3. Derived-id recompute logic — assert_derived_ids_consistent passes on correct ids and raises
     (loud) on atom/id disagreements for both well_id and image_id.
"""

import pytest
import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    DOWNSTREAM_FRAME_IDENTITY_BLOCK,
    FRAME_IDENTITY_NULLABLE,
    REQUIRED_FRAME_INVENTORY_COLUMNS,
    DERIVED_FRAME_INVENTORY_COLUMNS,
    FrameInventorySpec,
    StitchedHandoffSpec,
    WellHandoff,
    assert_derived_ids_consistent,
    assert_product_columns_consistent,
    derive_well_id,
    derive_image_id,
    frame_inventory_image_ids,
    frame_inventory_product_aware_keys,
    frame_inventory_product_keys,
    validate_frame_identity_block,
    validate_frame_inventory_identity_contract,
)
from data_pipeline.acquisition.image_materialization.materialized_image_write_policy import (
    MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
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

_FRAME_KEY_ATOMS = {"experiment_id", "well_index", "channel_id", "time_index", "z_index"}


def test_required_columns_contain_atoms():
    assert _FRAME_KEY_ATOMS.issubset(set(REQUIRED_FRAME_INVENTORY_COLUMNS))


def test_required_columns_include_writer_policy_column_family():
    assert set(MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS).issubset(
        set(REQUIRED_FRAME_INVENTORY_COLUMNS)
    )


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
        "z_index": pd.NA,
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
    assert derive_image_id("20250912_B01", "BF", 0, z_index=3) == "20250912_B01_BF_z0003_t0000"


def test_multiple_rows_all_consistent():
    rows = [
        {"experiment_id": "20250912", "well_index": "B01", "channel_id": "BF", "time_index": i,
         "z_index": pd.NA, "well_id": "20250912_B01", "image_id": f"20250912_B01_BF_t{i:04d}"}
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    assert_derived_ids_consistent(df)


def test_one_bad_row_in_batch_raises():
    rows = [
        {"experiment_id": "20250912", "well_index": "B01", "channel_id": "BF", "time_index": i,
         "z_index": pd.NA, "well_id": "20250912_B01", "image_id": f"20250912_B01_BF_t{i:04d}"}
        for i in range(4)
    ]
    # Corrupt the last row's image_id
    rows.append({"experiment_id": "20250912", "well_index": "B01", "channel_id": "BF", "time_index": 4,
                 "z_index": pd.NA, "well_id": "20250912_B01", "image_id": "WRONG"})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="image_id inconsistent"):
        assert_derived_ids_consistent(df)


def test_z_stack_image_id_is_consistent_with_z_index_atom():
    df = _make_df(z_index=3, image_id="20250912_B01_BF_z0003_t0000")
    assert_derived_ids_consistent(df)
    assert frame_inventory_image_ids(df).tolist() == ["20250912_B01_BF_z0003_t0000"]


def test_bad_z_stack_image_id_fails_loud():
    df = _make_df(z_index=3, image_id="20250912_B01_BF_z0002_t0000")
    with pytest.raises(ValueError, match="image_id inconsistent"):
        assert_derived_ids_consistent(df)


# ---------------------------------------------------------------------------
# 4. Downstream frame-identity block + validator
# ---------------------------------------------------------------------------

_EXP = "20250912"
_WELL_ID = f"{_EXP}_B01"
_WIDTH, _HEIGHT = 1024, 768


def _ref_inventory(n_frames: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment_id": _EXP,
        "well_index": "B01",
        "channel_id": "BF",
        "time_index": t,
        "source_image_path": f"images/{_WELL_ID}_BF_t{t:04d}.png",
        "image_width_px": _WIDTH,
        "image_height_px": _HEIGHT,
    } for t in range(n_frames)])


def _identity_df(n_frames: int, **overrides) -> pd.DataFrame:
    rows = []
    for t in range(n_frames):
        row = {
            "experiment_id": _EXP,
            "well_id": _WELL_ID,
            "image_id": f"{_WELL_ID}_BF_t{t:04d}",
            "time_index": t,
            "z_index": pd.NA,
            "channel_id": "BF",
            "source_image_path": f"images/{_WELL_ID}_BF_t{t:04d}.png",
            "image_width_px": _WIDTH,
            "image_height_px": _HEIGHT,
        }
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def test_identity_block_includes_z_index_as_nullable():
    assert "z_index" in DOWNSTREAM_FRAME_IDENTITY_BLOCK
    assert "z_index" in FRAME_IDENTITY_NULLABLE


def test_identity_block_is_tuple_with_derived_ids():
    assert isinstance(DOWNSTREAM_FRAME_IDENTITY_BLOCK, tuple)
    # The downstream block (unlike the producer atom list) DOES carry the derived ids.
    assert {"well_id", "image_id"}.issubset(set(DOWNSTREAM_FRAME_IDENTITY_BLOCK))


def test_identity_validator_passes_with_na_z_index():
    df = _identity_df(2)
    assert df["z_index"].isna().all()
    validate_frame_identity_block(df, _ref_inventory(2))  # must not raise


def test_identity_validator_synthesizes_missing_z_index():
    df = _identity_df(2).drop(columns=["z_index"])
    validate_frame_identity_block(df, _ref_inventory(2))  # must not raise


def test_identity_validator_unknown_image_id_fails():
    df = _identity_df(3)
    with pytest.raises(ValueError, match="not present in reference_frame_inventory"):
        validate_frame_identity_block(df, _ref_inventory(1))


def test_identity_validator_carried_mismatch_fails():
    df = _identity_df(1, image_height_px=999)
    with pytest.raises(ValueError, match="disagreeing with reference_frame_inventory"):
        validate_frame_identity_block(df, _ref_inventory(1))


def test_identity_validator_multiple_wells_fails():
    df = _identity_df(2)
    df.loc[1, "well_id"] = "20250912_C02"
    with pytest.raises(ValueError, match="multiple well_id"):
        validate_frame_identity_block(df, _ref_inventory(2))


def test_identity_validator_uses_context_in_error():
    df = _identity_df(1, source_image_path="images/wrong.png")
    with pytest.raises(ValueError, match=r"\[my_product\]"):
        validate_frame_identity_block(df, _ref_inventory(1), context="my_product")


# ---------------------------------------------------------------------------
# 4. The identity-contract gate — product-aware frame identity, enforced once
# ---------------------------------------------------------------------------

def _product_row(*, time_index=0, z_index=pd.NA, image_product_type="projection",
                 projection_method="focus_stack", well_index="B01") -> dict:
    return {
        "experiment_id": "20250912",
        "well_index": well_index,
        "channel_id": "BF",
        "time_index": time_index,
        "z_index": z_index,
        "image_product_type": image_product_type,
        "projection_method": projection_method,
    }


def _z_row(**kw) -> dict:
    kw.setdefault("z_index", 0)
    kw.setdefault("image_product_type", "z_stack")
    kw.setdefault("projection_method", pd.NA)
    return _product_row(**kw)


def test_product_keys_compose_projection_and_z_stack():
    df = pd.DataFrame([_product_row(), _z_row()])
    keys = list(frame_inventory_product_keys(df))
    assert keys == ["BF__projection__focus_stack", "BF__z_stack"]


def test_two_projection_products_same_time_are_distinct_frames():
    # The whole point: focus_stack vs max_intensity at the same well/channel/time share an image_id
    # but are DIFFERENT frames — the product-aware key must NOT collide them.
    df = pd.DataFrame([
        _product_row(projection_method="focus_stack"),
        _product_row(projection_method="max_intensity"),
    ])
    keys = frame_inventory_product_aware_keys(df)
    assert keys.nunique() == 2
    validate_frame_inventory_identity_contract(df)  # must not raise


def test_identity_contract_rejects_true_duplicate_product_frame():
    df = pd.DataFrame([_product_row(), _product_row()])  # same product, same time
    with pytest.raises(ValueError, match="(?i)duplicate"):
        validate_frame_inventory_identity_contract(df)


def test_identity_contract_distinct_z_planes_pass():
    df = pd.DataFrame([_z_row(z_index=0), _z_row(z_index=1)])
    validate_frame_inventory_identity_contract(df)  # planes differ → not duplicate


def test_product_consistency_projection_requires_method():
    df = pd.DataFrame([_product_row(projection_method=pd.NA)])
    with pytest.raises(ValueError, match="projection"):
        assert_product_columns_consistent(df)


def test_product_consistency_z_stack_rejects_method():
    df = pd.DataFrame([_z_row(projection_method="focus_stack")])
    with pytest.raises(ValueError, match="z_stack"):
        assert_product_columns_consistent(df)


def test_product_consistency_projection_rejects_z_index():
    df = pd.DataFrame([_product_row(z_index=0)])  # projection must have null z_index
    with pytest.raises(ValueError, match="projection"):
        assert_product_columns_consistent(df)


def test_identity_contract_runs_product_consistency():
    # The gate composes the product-consistency check — a contradictory row fails through the gate.
    df = pd.DataFrame([_z_row(projection_method="focus_stack")])
    with pytest.raises(ValueError, match="z_stack"):
        validate_frame_inventory_identity_contract(df)
