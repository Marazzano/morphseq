"""Contract for the snip_auxiliary_masks product.

One row per (snip_id, auxiliary_mask_type).
Primary key: (snip_id, auxiliary_mask_type).

Produced by: run_unet_for_snip_inventory in run_unet_snip.py
Consumed by: QC, feature extraction (join back to physical embryo via snip_inventory)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNIP_AUXILIARY_MASKS_REQUIRED_COLUMNS = [
    # Identity — inherited from snip_inventory
    "snip_id",
    "physical_embryo_id",
    "embryo_id",
    "experiment_id",
    "well_id",
    "image_id",
    "time_index",
    "channel_id",
    # Auxiliary mask identity
    "auxiliary_mask_type",
    # Output
    "auxiliary_mask_path",
    "auxiliary_mask_format",
    # Model provenance
    "model_backend",
    "model_id",
    "checkpoint_path",
    # Shape
    "snip_height_px",
    "snip_width_px",
    "mask_height_px",
    "mask_width_px",
    # Validity
    "is_valid_auxiliary_mask",
    "error_message",
]

# The UNet auxiliary-mask families. NO "foreground": the whole-embryo mask is created by
# snip_processing (the cropped frame_masks RLE, embryo_mask), not predicted here —
# fraction_alive reads foreground from snip_processing and only `via` from this product.
ALLOWED_AUXILIARY_MASK_TYPES = ("via", "yolk", "focus", "bubble")


def validate_snip_auxiliary_masks(df: pd.DataFrame) -> None:
    """Validate a snip_auxiliary_masks shard.

    Raises ValueError on the first violation found.
    """
    missing = [c for c in SNIP_AUXILIARY_MASKS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"snip_auxiliary_masks missing required columns: {missing}")

    for col in ("snip_id", "physical_embryo_id", "embryo_id"):
        if df[col].isnull().any():
            raise ValueError(f"snip_auxiliary_masks: column '{col}' has null values")

    bad_types = df.loc[~df["auxiliary_mask_type"].isin(ALLOWED_AUXILIARY_MASK_TYPES), "auxiliary_mask_type"].unique()
    if len(bad_types):
        raise ValueError(
            f"snip_auxiliary_masks: unknown auxiliary_mask_type values: {list(bad_types)}. "
            f"Allowed: {ALLOWED_AUXILIARY_MASK_TYPES}"
        )

    dup = df.duplicated(subset=["snip_id", "auxiliary_mask_type"])
    if dup.any():
        dupes = df.loc[dup, ["snip_id", "auxiliary_mask_type"]].head(5).to_dict("records")
        raise ValueError(f"snip_auxiliary_masks: duplicate (snip_id, auxiliary_mask_type) rows: {dupes}")

    if not pd.api.types.is_bool_dtype(df["is_valid_auxiliary_mask"]):
        raise ValueError("snip_auxiliary_masks: 'is_valid_auxiliary_mask' must be boolean dtype")

    valid_rows = df["is_valid_auxiliary_mask"]
    path_null_when_valid = valid_rows & df["auxiliary_mask_path"].isnull()
    if path_null_when_valid.any():
        raise ValueError(
            "snip_auxiliary_masks: 'auxiliary_mask_path' is null for valid rows"
        )

    path_non_null_when_invalid = ~valid_rows & df["auxiliary_mask_path"].notnull()
    if path_non_null_when_invalid.any():
        raise ValueError(
            "snip_auxiliary_masks: 'auxiliary_mask_path' is non-null for invalid rows"
        )

    dim_mismatch = valid_rows & (
        (df["mask_height_px"] != df["snip_height_px"])
        | (df["mask_width_px"] != df["snip_width_px"])
    )
    if dim_mismatch.any():
        raise ValueError(
            "snip_auxiliary_masks: mask dims do not match snip dims for valid rows"
        )


def validate_snip_auxiliary_masks_against_snip_inventory(
    auxiliary_masks: pd.DataFrame,
    snip_inventory: pd.DataFrame,
) -> None:
    """Cross-validate auxiliary_masks against the snip_inventory it was built from.

    Checks that every snip_id is known and that inherited identity columns agree.
    """
    known_snip_ids = set(snip_inventory["snip_id"])
    unknown = set(auxiliary_masks["snip_id"]) - known_snip_ids
    if unknown:
        raise ValueError(
            f"snip_auxiliary_masks: snip_id values not found in snip_inventory: {sorted(unknown)[:5]}"
        )

    check_cols = ["physical_embryo_id", "embryo_id", "image_id", "time_index", "channel_id"]
    inv_index = snip_inventory.set_index("snip_id")[check_cols]
    merged = auxiliary_masks[["snip_id"] + check_cols].merge(
        inv_index.rename(columns={c: f"_inv_{c}" for c in check_cols}),
        left_on="snip_id",
        right_index=True,
        how="left",
    )
    for col in check_cols:
        mismatch = merged[col] != merged[f"_inv_{col}"]
        if mismatch.any():
            raise ValueError(
                f"snip_auxiliary_masks: column '{col}' disagrees with snip_inventory "
                f"for {mismatch.sum()} rows"
            )


def load_snip_auxiliary_masks(csv_path: Path) -> pd.DataFrame:
    """Read a snip_auxiliary_masks CSV shard and return a validated DataFrame.

    Enforces bool dtype on is_valid_auxiliary_mask and runs the contract validator.
    Raises ValueError if the file fails validation.
    """
    df = pd.read_csv(csv_path)
    df["is_valid_auxiliary_mask"] = df["is_valid_auxiliary_mask"].astype(bool)
    validate_snip_auxiliary_masks(df)
    return df
