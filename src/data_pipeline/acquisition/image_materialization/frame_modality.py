"""Optional first-class acquisition-modality context for frame inventories.

The canonical ``image_product_type`` describes where a materialized image is routed
inside the pipeline (projection versus z-stack).  It does not always describe the
physical acquisition that produced the pixels.  SeaHub is the motivating example:
one arbitrary focal plane is intentionally routed through the projection product
slot so the existing segmentation path can consume it.

When any modality column is present, all modality columns are required and validated.
Legacy/native inventories that predate this block remain valid; consumers use
``frame_modality_for_image`` to obtain conservative legacy defaults.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    frame_inventory_image_ids,
    frame_inventory_product_keys,
)

FRAME_MODALITY_COLUMNS: tuple[str, ...] = (
    "source_scope",
    "image_kind",
    "z_position",
    "calibration_status",
)

IMAGE_KIND_PROJECTION = "projection"
IMAGE_KIND_SINGLE_Z = "single_z"
IMAGE_KIND_Z_STACK = "z_stack"
ALLOWED_IMAGE_KINDS: frozenset[str] = frozenset(
    {IMAGE_KIND_PROJECTION, IMAGE_KIND_SINGLE_Z, IMAGE_KIND_Z_STACK}
)

CALIBRATION_STATUS_CALIBRATED = "calibrated"
CALIBRATION_STATUS_PLACEHOLDER = "placeholder"
ALLOWED_CALIBRATION_STATUSES: frozenset[str] = frozenset(
    {CALIBRATION_STATUS_CALIBRATED, CALIBRATION_STATUS_PLACEHOLDER}
)


def validate_frame_modality_block(
    df: pd.DataFrame,
    *,
    scope_label: str = "frame_inventory",
) -> None:
    """Validate the optional modality block when a producer declares it."""
    present = [column for column in FRAME_MODALITY_COLUMNS if column in df.columns]
    if not present:
        return

    missing = [column for column in FRAME_MODALITY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"[{scope_label}] partial frame modality block: present={present}, missing={missing}. "
            "Declare all of source_scope, image_kind, z_position, and calibration_status, "
            "or omit the complete block for a legacy inventory."
        )

    for column in ("source_scope", "image_kind", "calibration_status"):
        nullish = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if nullish.any():
            raise ValueError(
                f"[{scope_label}] modality column {column!r} must be non-null and non-empty."
            )

    image_kinds = set(df["image_kind"].astype(str))
    unknown_kinds = sorted(image_kinds - ALLOWED_IMAGE_KINDS)
    if unknown_kinds:
        raise ValueError(
            f"[{scope_label}] unknown image_kind value(s) {unknown_kinds}; "
            f"allowed={sorted(ALLOWED_IMAGE_KINDS)}."
        )

    calibration_statuses = set(df["calibration_status"].astype(str))
    unknown_calibration = sorted(
        calibration_statuses - ALLOWED_CALIBRATION_STATUSES
    )
    if unknown_calibration:
        raise ValueError(
            f"[{scope_label}] unknown calibration_status value(s) {unknown_calibration}; "
            f"allowed={sorted(ALLOWED_CALIBRATION_STATUSES)}."
        )

    single_z = df["image_kind"].astype(str).eq(IMAGE_KIND_SINGLE_Z)
    if single_z.any():
        bad_product = single_z & ~df["image_product_type"].astype(str).eq("projection")
        if bad_product.any():
            raise ValueError(
                f"[{scope_label}] single_z rows must use the projection compatibility product "
                "slot (image_product_type='projection')."
            )
        if "z_index" in df.columns and df.loc[single_z, "z_index"].notna().any():
            raise ValueError(
                f"[{scope_label}] single_z compatibility rows must leave z_index null. "
                "The unknown physical z belongs in nullable z_position."
            )


def frame_modality_for_image(
    frame_inventory_df: pd.DataFrame,
    *,
    image_id: str,
    product_key: str | None = None,
) -> dict[str, object]:
    """Resolve one image's modality, with conservative defaults for legacy rows.

    ``image_id`` may be shared by multiple projection products.  ``product_key``
    narrows the match when supplied.  Z-stack plane rows use z-bearing image IDs
    and therefore do not collide with their projection source.
    """
    inventory = frame_inventory_df.copy()
    if not all(column in inventory.columns for column in FRAME_MODALITY_COLUMNS):
        # Legacy tables do not declare the physical modality. Keep their historical
        # behavior without demanding newer atom columns from old downstream fixtures.
        exact = (
            inventory["image_id"].astype(str).eq(str(image_id))
            if "image_id" in inventory.columns
            else pd.Series(False, index=inventory.index)
        )
        exact_types = set(
            inventory.loc[exact, "image_product_type"].dropna().astype(str)
        )
        return {
            "source_scope": "legacy_unspecified",
            "image_kind": (
                IMAGE_KIND_Z_STACK
                if exact_types == {"z_stack"}
                else IMAGE_KIND_PROJECTION
            ),
            "z_position": pd.NA,
            "calibration_status": CALIBRATION_STATUS_CALIBRATED,
        }

    inventory["_derived_image_id"] = (
        inventory["image_id"].astype(str)
        if "image_id" in inventory.columns
        else frame_inventory_image_ids(
            inventory, scope_label="frame_modality"
        ).astype(str)
    )
    matches = inventory[inventory["_derived_image_id"].eq(str(image_id))]
    if product_key is not None:
        inventory_keys = frame_inventory_product_keys(
            matches, scope_label="frame_modality"
        )
        matches = matches[inventory_keys.eq(str(product_key))]
    if len(matches) != 1:
        raise ValueError(
            "frame_modality: expected exactly one frame row for "
            f"image_id={image_id!r}, product_key={product_key!r}; found {len(matches)}."
        )

    row = matches.iloc[0]
    return {column: row[column] for column in FRAME_MODALITY_COLUMNS}


__all__ = [
    "ALLOWED_CALIBRATION_STATUSES",
    "ALLOWED_IMAGE_KINDS",
    "CALIBRATION_STATUS_CALIBRATED",
    "CALIBRATION_STATUS_PLACEHOLDER",
    "FRAME_MODALITY_COLUMNS",
    "IMAGE_KIND_PROJECTION",
    "IMAGE_KIND_SINGLE_Z",
    "IMAGE_KIND_Z_STACK",
    "frame_modality_for_image",
    "validate_frame_modality_block",
]
