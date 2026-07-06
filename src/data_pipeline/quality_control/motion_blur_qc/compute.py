"""motion_blur_qc compute — adjacent-z mask-pixel NCC, one row per snip.

Pixels are read through materialized-image readers (resolve via frame_inventory, never a
reconstructed path). Masks are decoded from canonical frame_masks RLE and aligned to the loaded
z-plane dimensions with the shared nearest-neighbor mask resize helper, which handles downsampled
materialized z stacks without changing label semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.acquisition.image_materialization.materialized_image_readers import (
    load_z_stack_images_from_image_id,
)
from data_pipeline.object_extraction.segmentation.masks.mask_resize import resize_binary_mask_to_shape
from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_FRAME_PROVENANCE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
)

from .config import MotionBlurQCConfig

_REQUIRED_FRAME_COLUMNS: tuple[str, ...] = ("mask_id",)


def compute_mask_pixel_motion_metrics(
    z_stack_zyx: np.ndarray,
    mask: np.ndarray,
    *,
    config: MotionBlurQCConfig,
) -> dict[str, float | int | bool]:
    """Return adjacent-z NCC summary metrics for one snip and one aligned mask."""
    stack = _coerce_z_stack(z_stack_zyx)
    aligned_mask = resize_binary_mask_to_shape(mask, stack.shape[1:3])
    n_mask_pixels = int(np.sum(aligned_mask))
    if n_mask_pixels == 0:
        raise ValueError("motion_blur_qc: aligned mask contains zero pixels.")

    n_z_planes = int(stack.shape[0])
    if n_z_planes < 2:
        raise ValueError(
            f"motion_blur_qc: z stack must contain at least two planes, got {n_z_planes}."
        )

    valid_ncc: list[float] = []
    pair_bad_status: list[bool | None] = []
    n_flat_z_pairs = 0
    flat_eps = float(config.flat_pair_variance_epsilon)

    for z in range(n_z_planes - 1):
        a = stack[z][aligned_mask].astype(np.float64)
        b = stack[z + 1][aligned_mask].astype(np.float64)
        var_a = float(np.var(a))
        var_b = float(np.var(b))
        if var_a <= flat_eps or var_b <= flat_eps:
            n_flat_z_pairs += 1
            pair_bad_status.append(None)
            continue
        ncc = _normalized_cross_correlation(a, b, var_a=var_a, var_b=var_b)
        valid_ncc.append(ncc)
        pair_bad_status.append(ncc < config.bad_z_pair_ncc_threshold)

    if not valid_ncc:
        raise ValueError(
            "motion_blur_qc: no valid adjacent z-plane pairs remain after excluding "
            "flat/constant in-mask pairs."
        )

    ncc_array = np.asarray(valid_ncc, dtype=np.float64)
    bad_array = ncc_array < config.bad_z_pair_ncc_threshold
    bad_pair_frac = float(np.mean(bad_array))

    return {
        "mask_pixel_ncc_mean": float(np.mean(ncc_array)),
        "mask_pixel_ncc_min": float(np.min(ncc_array)),
        "mask_pixel_ncc_p05": float(np.percentile(ncc_array, 5)),
        "mask_pixel_bad_pair_frac": bad_pair_frac,
        "mask_pixel_longest_bad_run": int(_longest_bad_run(pair_bad_status)),
        "n_z_planes": n_z_planes,
        "n_z_pairs": n_z_planes - 1,
        "n_valid_z_pairs": int(len(valid_ncc)),
        "n_flat_z_pairs": int(n_flat_z_pairs),
        "n_mask_pixels": n_mask_pixels,
        "motion_blur_flag": bool(bad_pair_frac > config.bad_pair_frac_threshold),
    }


def compute_motion_blur_qc(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    config: MotionBlurQCConfig,
    image_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return one motion_blur_qc row per snip."""
    required = (*SNIP_ID_SPINE_COLUMNS, *SNIP_FRAME_PROVENANCE_COLUMNS, *_REQUIRED_FRAME_COLUMNS)
    missing_cols = [c for c in required if c not in snip_inventory_df.columns]
    if missing_cols:
        raise ValueError(
            f"motion_blur_qc: snip_inventory missing required column(s): "
            f"{', '.join(missing_cols)}."
        )

    frame_masks_by_mask = frame_masks_df.set_index("mask_id")
    metrics_by_snip: dict[str, dict[str, float | int | bool]] = {}
    image_root_path = Path(image_root) if image_root is not None else None

    for _, snip in snip_inventory_df.iterrows():
        snip_id = str(snip["snip_id"])
        image_id = str(snip["image_id"])
        mask_id = str(snip["mask_id"])

        mask = _decode_snip_mask(frame_masks_by_mask, mask_id, image_id, snip_id, config)

        try:
            z_planes, _z_rows = load_z_stack_images_from_image_id(
                frame_inventory_df,
                image_id=image_id,
                product_key=config.z_stack_product_key,
                image_root=image_root_path,
            )
        except ValueError as exc:
            if config.missing_z_stack_policy == "fail":
                raise ValueError(
                    f"motion_blur_qc: could not load z stack for snip {snip_id!r} "
                    f"(image_id={image_id!r}, product_key={config.z_stack_product_key!r}): {exc}"
                ) from exc
            raise

        metrics_by_snip[snip_id] = compute_mask_pixel_motion_metrics(
            np.stack(z_planes, axis=0),
            mask,
            config=config,
        )

    out = snip_inventory_df[list(SNIP_ID_SPINE_COLUMNS)].copy()
    snip_ids = out["snip_id"].astype(str)
    for col in (
        "mask_pixel_ncc_mean",
        "mask_pixel_ncc_min",
        "mask_pixel_ncc_p05",
        "mask_pixel_bad_pair_frac",
        "mask_pixel_longest_bad_run",
        "n_z_planes",
        "n_z_pairs",
        "n_valid_z_pairs",
        "n_flat_z_pairs",
        "n_mask_pixels",
    ):
        out[col] = [metrics_by_snip[s][col] for s in snip_ids]
    out["motion_blur_flag"] = pd.array(
        [metrics_by_snip[s]["motion_blur_flag"] for s in snip_ids],
        dtype=bool,
    )
    return out


def _coerce_z_stack(z_stack_zyx: np.ndarray) -> np.ndarray:
    stack = np.asarray(z_stack_zyx)
    if stack.ndim == 4:
        stack = stack.mean(axis=3)
    if stack.ndim != 3:
        raise ValueError(
            f"motion_blur_qc: z_stack_zyx must be a 3-D (z, height, width) array "
            f"or 4-D with channels, got shape {stack.shape!r}."
        )
    return stack


def _normalized_cross_correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    var_a: float,
    var_b: float,
) -> float:
    centered_a = a - float(np.mean(a))
    centered_b = b - float(np.mean(b))
    return float(np.mean(centered_a * centered_b) / np.sqrt(var_a * var_b))


def _longest_bad_run(pair_bad_status: list[bool | None]) -> int:
    longest = 0
    current = 0
    for status in pair_bad_status:
        if status is True:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _decode_snip_mask(frame_masks_by_mask, mask_id, image_id, snip_id, config):
    """Decode one snip's canonical mask, failing loud on a missing mask."""
    if mask_id not in frame_masks_by_mask.index:
        raise ValueError(
            f"motion_blur_qc: mask_id {mask_id!r} (snip {snip_id!r}) not found in frame_masks. "
            "snip_inventory and frame_masks must agree on mask identity."
        )
    mask_row = frame_masks_by_mask.loc[mask_id]
    if str(mask_row["image_id"]) != image_id:
        raise ValueError(
            f"motion_blur_qc: mask_id {mask_id!r} maps to image_id {mask_row['image_id']!r} in "
            f"frame_masks but snip {snip_id!r} declares image_id {image_id!r}."
        )
    rle = mask_row["mask_rle"]
    if pd.isna(rle) if np.isscalar(rle) else (rle is None):
        if config.missing_mask_policy == "fail":
            raise ValueError(
                f"motion_blur_qc: snip {snip_id!r} (mask_id {mask_id!r}) has no mask_rle payload."
            )
        return None
    rle = json.loads(str(rle)) if isinstance(rle, str) else rle
    return decode_binary_mask_rle(rle)
