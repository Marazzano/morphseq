"""Tests for snip_auxiliary_masks contract validator."""

import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    SNIP_AUXILIARY_MASKS_REQUIRED_COLUMNS,
    validate_snip_auxiliary_masks,
    validate_snip_auxiliary_masks_against_snip_inventory,
)


def _make_valid_df(n: int = 2) -> pd.DataFrame:
    mask_types = ["via", "yolk", "focus", "bubble"]
    rows = []
    for i in range(n):
        snip_id = f"20250912_B0{i+1}_e01_BF_t0007"
        physical_embryo_id = f"20250912_B0{i+1}_e01"
        for mt in mask_types:
            rows.append({
                "snip_id": snip_id,
                "physical_embryo_id": physical_embryo_id,
                "embryo_id": f"embryo_{i}",
                "experiment_id": "20250912",
                "well_id": f"20250912_B0{i+1}",
                "image_id": f"20250912_B0{i+1}_t0007",
                "time_index": 7,
                "channel_id": "BF",
                "auxiliary_mask_type": mt,
                "auxiliary_mask_path": f"/out/{snip_id}/{mt}.png",
                "auxiliary_mask_format": "png",
                "model_backend": "unet_snip",
                "model_id": "unet_v1",
                "checkpoint_path": f"weights/{mt}.pth",
                "snip_height_px": 64,
                "snip_width_px": 32,
                "mask_height_px": 64,
                "mask_width_px": 32,
                "is_valid_auxiliary_mask": True,
                "error_message": None,
            })
    df = pd.DataFrame(rows)
    df["is_valid_auxiliary_mask"] = df["is_valid_auxiliary_mask"].astype(bool)
    return df


def test_valid_df_passes():
    validate_snip_auxiliary_masks(_make_valid_df())


def test_missing_column_raises():
    df = _make_valid_df().drop(columns=["physical_embryo_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_snip_auxiliary_masks(df)


def test_invalid_mask_type_raises():
    df = _make_valid_df(1)
    df.loc[df["auxiliary_mask_type"] == "via", "auxiliary_mask_type"] = "unknown"
    with pytest.raises(ValueError, match="unknown auxiliary_mask_type"):
        validate_snip_auxiliary_masks(df)


def test_duplicate_snip_mask_type_raises():
    df = _make_valid_df(1)
    dup = df[df["auxiliary_mask_type"] == "yolk"].copy()
    df = pd.concat([df, dup], ignore_index=True)
    df["is_valid_auxiliary_mask"] = df["is_valid_auxiliary_mask"].astype(bool)
    with pytest.raises(ValueError, match="duplicate"):
        validate_snip_auxiliary_masks(df)


def test_mask_path_null_when_valid_raises():
    df = _make_valid_df(1)
    df.loc[df["auxiliary_mask_type"] == "yolk", "auxiliary_mask_path"] = None
    with pytest.raises(ValueError, match="null for valid rows"):
        validate_snip_auxiliary_masks(df)


def test_mask_path_non_null_when_invalid_raises():
    df = _make_valid_df(1)
    # Mark one row invalid but leave its path non-null
    mask = df["auxiliary_mask_type"] == "yolk"
    df.loc[mask, "is_valid_auxiliary_mask"] = False
    with pytest.raises(ValueError, match="non-null for invalid rows"):
        validate_snip_auxiliary_masks(df)


def test_validate_against_inventory_id_mismatch_raises():
    df = _make_valid_df(1)
    inv = pd.DataFrame([{
        "snip_id": df["snip_id"].iloc[0],
        "physical_embryo_id": "WRONG_ID",
        "embryo_id": df["embryo_id"].iloc[0],
        "image_id": df["image_id"].iloc[0],
        "time_index": df["time_index"].iloc[0],
        "channel_id": df["channel_id"].iloc[0],
    }])
    with pytest.raises(ValueError, match="physical_embryo_id.*disagrees"):
        validate_snip_auxiliary_masks_against_snip_inventory(df, inv)
