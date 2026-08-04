"""Tests for the identity-carrying spine validator ``validate_snip_grain_identity_columns``."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    EMBRYO_ID_SPINE_COLUMNS,
    PHYSICAL_EMBRYO_ID_SPINE_COLUMNS,
    SNIP_FRAME_PROVENANCE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
    SNIP_INVENTORY_COLUMNS,
    validate_snip_grain_identity_columns,
    validate_snip_inventory_contract,
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


def test_spine_constants_are_additive():
    assert PHYSICAL_EMBRYO_ID_SPINE_COLUMNS == ("experiment_id", "well_id", "physical_embryo_id")
    assert EMBRYO_ID_SPINE_COLUMNS == PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + ("embryo_id",)
    assert SNIP_ID_SPINE_COLUMNS == EMBRYO_ID_SPINE_COLUMNS + ("snip_id",)


def test_valid_snip_grain_passes():
    validate_snip_grain_identity_columns(_snip_df(3), grain="snip_id")


def test_valid_embryo_id_grain_passes():
    df = pd.DataFrame([
        {
            "experiment_id": "20250912",
            "well_id": "20250912_B01",
            "physical_embryo_id": "20250912_B01_e01",
            "embryo_id": "20250912_B01_e01__BF",
        },
        {
            "experiment_id": "20250912",
            "well_id": "20250912_B01",
            "physical_embryo_id": "20250912_B01_e02",
            "embryo_id": "20250912_B01_e02__BF",
        },
    ])
    validate_snip_grain_identity_columns(df, grain="embryo_id")


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
    validate_snip_grain_identity_columns(df, grain="physical_embryo_id")


def test_unknown_grain_raises():
    with pytest.raises(ValueError, match="unknown grain"):
        validate_snip_grain_identity_columns(_snip_df(1), grain="bogus")


def test_missing_spine_column_raises():
    df = _snip_df(2).drop(columns=["physical_embryo_id"])
    with pytest.raises(ValueError, match="missing required identity-spine column"):
        validate_snip_grain_identity_columns(df, grain="snip_id")


def test_null_spine_value_raises():
    df = _snip_df(2)
    df.loc[0, "physical_embryo_id"] = None
    with pytest.raises(ValueError, match="null"):
        validate_snip_grain_identity_columns(df, grain="snip_id")


def test_embryo_id_disagrees_with_physical_embryo_id_raises():
    # snip_id/embryo_id syntactically valid, but embryo_id points at a different animal.
    df = _snip_df(1)
    other = _snip_row(local_embryo_index=2)
    df.loc[0, "embryo_id"] = other["embryo_id"]
    with pytest.raises(ValueError, match="embryo_id .*encodes physical_embryo_id|not enough"):
        validate_snip_grain_identity_columns(df, grain="snip_id")


def test_snip_id_disagrees_with_embryo_id_raises():
    df = _snip_df(1)
    other = _snip_row(local_embryo_index=2)
    df.loc[0, "snip_id"] = other["snip_id"]
    with pytest.raises(ValueError, match="snip_id .*encodes embryo_id|not enough"):
        validate_snip_grain_identity_columns(df, grain="snip_id")


def test_channel_id_disagrees_with_image_id_raises():
    df = _snip_df(1)
    df.loc[0, "channel_id"] = "DAPI"
    with pytest.raises(ValueError, match="channel_id column .*disagrees"):
        validate_snip_grain_identity_columns(df, grain="snip_id")


def test_duplicate_snip_id_raises():
    df = pd.concat([_snip_df(1), _snip_df(1)], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        validate_snip_grain_identity_columns(df, grain="snip_id")


def test_check_sources_passes_when_registered():
    df = _snip_df(2)
    registry = pd.DataFrame({"physical_embryo_id": ["20250912_B01_e01"]})
    validate_snip_grain_identity_columns(
        df, grain="snip_id", physical_embryo_registry_df=registry, check_sources=True
    )


def test_check_sources_fails_when_unregistered():
    df = _snip_df(2)
    registry = pd.DataFrame({"physical_embryo_id": ["20250912_C04_e09"]})
    with pytest.raises(ValueError, match="not in the physical_embryo_registry"):
        validate_snip_grain_identity_columns(
            df, grain="snip_id", physical_embryo_registry_df=registry, check_sources=True
        )


def test_check_sources_requires_registry_df():
    with pytest.raises(ValueError, match="requires physical_embryo_registry_df"):
        validate_snip_grain_identity_columns(_snip_df(1), grain="snip_id", check_sources=True)


# ── validate_snip_inventory_contract — the one public gate the task verb calls ─────────────────

def _snip_inventory_row(**kw):
    """A snip_row extended with the snip_inventory product columns."""
    row = _snip_row(**kw)
    row.update({
        "mask_id": f"{row['image_id']}_m0001",
        "track_id": f"{row['well_id']}_track0000",
        "image_path": "images/src.png",
        # The authoritative product-keyed location, plus the compatibility alias and the product
        # identity that together make a multi-product inventory readable.
        "processed_snip_path": (
            "snips/BF__projection__focus_stack__clahe_blend/20250912_B01_e01/out.png"
        ),
        "legacy_flat_snip_path": "snips/20250912_B01_e01/out.png",
        "snip_product_key": "BF__projection__focus_stack__clahe_blend",
        "embryo_mask": "snips/out_mask.png",
        "embryo_mask_snip_path": "snips/out_mask.png",
        "crop_x_min_px": 0,
        "crop_y_min_px": 0,
        "crop_x_max_px": 255,
        "crop_y_max_px": 575,
        "crop_width_px": 256,
        "crop_height_px": 576,
        "is_valid_snip": True,
        "error_message": "",
        # Construction provenance: HOW this snip was geometrically built. The full replayable
        # evidence lives in the per-well snip transform table, referenced by snip_transform_id.
        "crop_x_min_um": 0.0,
        "crop_y_min_um": 0.0,
        "crop_x_max_um": 1996.8,
        "crop_y_max_um": 4492.8,
        "orientation_policy": "pca_major_axis_yolk_down",
        "orientation_source": "embryo_mass_distribution",
        "no_yolk_policy": "fallback_mass_distribution",
        "rotation_angle_rad": 0.0,
        "flip_x": False,
        "crop_center_um_x": 998.4,
        "crop_center_um_y": 2246.4,
        "source_height_px": 2189,
        "source_width_px": 1152,
        "source_um_per_px": 3.230785,
        "target_um_per_px": 7.8,
        "output_height_px": 576,
        "output_width_px": 256,
        "border_mode": "constant",
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "realized_scale_y": 0.414344,
        "realized_scale_x": 0.414344,
        "centering": "legacy_latched",
        # Channel-independent FK: sibling products of this embryo-time share this id.
        "snip_transform_id": f"{row['physical_embryo_id']}_t{row['time_index']:04d}",
    })
    return row


def test_snip_inventory_contract_passes_on_complete_shard():
    df = pd.DataFrame([_snip_inventory_row(time_index=t) for t in range(3)])
    validate_snip_inventory_contract(df)  # must not raise


def test_empty_snip_inventory_round_trips_through_csv_with_canonical_schema(tmp_path):
    path = tmp_path / "empty_snip_inventory.csv"
    pd.DataFrame(columns=SNIP_INVENTORY_COLUMNS).to_csv(path, index=False)

    reloaded = pd.read_csv(path)
    assert reloaded.empty
    assert tuple(reloaded.columns) == SNIP_INVENTORY_COLUMNS
    validate_snip_inventory_contract(reloaded)


def test_snip_inventory_contract_rejects_missing_product_column():
    df = pd.DataFrame([_snip_inventory_row()]).drop(columns=["mask_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_snip_inventory_contract(df)


def test_snip_inventory_contract_rejects_missing_frame_derived_column():
    df = pd.DataFrame([_snip_inventory_row()]).drop(columns=["image_id"])
    with pytest.raises(ValueError, match="missing required columns|identity-spine"):
        validate_snip_inventory_contract(df)


def test_snip_inventory_contract_rejects_identity_disagreement():
    # A snip_id that doesn't agree with its embryo_id must fail through the composed gate.
    df = pd.DataFrame([_snip_inventory_row()])
    df.loc[0, "snip_id"] = "20250912_B01_e02_BF_t0007"  # embryo e02 ≠ row's e01
    with pytest.raises(ValueError):
        validate_snip_inventory_contract(df)


def test_snip_inventory_contract_rejects_duplicate_snip_id():
    df = pd.DataFrame([_snip_inventory_row(time_index=0), _snip_inventory_row(time_index=0)])
    with pytest.raises(ValueError, match="unique"):
        validate_snip_inventory_contract(df)


def test_one_snip_id_may_appear_once_per_product():
    """Snip-grain uniqueness is (snip_id, snip_product_key), not bare snip_id.

    One physical embryo-time renders into several products -- BF clahe_blend, RFP no_change, ... --
    so a bare snip_id rule would REJECT a legitimate multi-product table rather than catch a defect.
    snip_id stays the physical embryo-time identity; the product is a separate column and a separate
    path level, deliberately NOT encoded into snip_id (that would make every cross-channel join
    awkward). Mirrors (image_id, product_key) on frame_inventory.
    """
    bf = _snip_inventory_row()
    rfp = dict(bf)
    rfp["snip_product_key"] = "RFP__projection__max__no_change"
    rfp["processed_snip_path"] = "snips/RFP__projection__max__no_change/20250912_B01_e01/out.png"

    # Same snip_id, two products: valid.
    validate_snip_grain_identity_columns(
        pd.DataFrame([bf, rfp]), grain="snip_id", scope_label="two_products"
    )

    # Same snip_id AND same product: a genuine duplicate, still rejected.
    with pytest.raises(ValueError, match="must be unique"):
        validate_snip_grain_identity_columns(
            pd.DataFrame([bf, dict(bf)]), grain="snip_id", scope_label="true_duplicate"
        )
