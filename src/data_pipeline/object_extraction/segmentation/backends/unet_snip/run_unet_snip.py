"""Per-snip auxiliary mask runner for the unet_snip backend.

Step 1: predictor interface only — no Torch, no FishModel, no checkpoint loading.
Step 2 will add FishModelSnipPredictor in model_loader.py.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
    validate_snip_auxiliary_masks,
)
from data_pipeline.object_extraction.snip_processing.snip_frame_masks import assert_on_snip_frame

# Predictor contract:
#   input  — H×W uint8 grayscale snip on artifact_shape (asserted by the runner before the
#            per-mask try/except, so an off-grid snip kills the shard rather than scattering)
#   output — H×W bool mask on artifact_shape (the promise, not the input's incidental shape)
AuxiliaryMaskPredictor = Callable[[np.ndarray], np.ndarray]

UNET_SNIP_BACKEND_LABEL = "unet_snip"
AUXILIARY_MASK_FORMAT = "png"


def _is_existing_path(value: object) -> bool:
    """Return whether ``value`` is a nonblank, path-like existing path."""
    if not isinstance(value, (str, os.PathLike)):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return Path(value).exists()


def run_auxiliary_mask_predictors_for_snip(
    predictors: Mapping[str, AuxiliaryMaskPredictor],
    snip_image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run snip-local auxiliary mask predictors on one processed snip image."""
    return {mask_type: predictor(snip_image) for mask_type, predictor in predictors.items()}


def write_auxiliary_mask_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(str(output_path), mask.astype(np.uint8) * 255, check_contrast=False)


def _mask_output_path(
    output_dir: Path,
    experiment_id: str,
    well_id: str,
    physical_embryo_id: str,
    snip_id: str,
    auxiliary_mask_type: str,
) -> Path:
    # well_id is the GLOBAL well_id (e.g. 20250912_B01), passed from the snip row — never
    # re-derived by splitting snip_id (that yielded the LOCAL slug B01 and broke the grain).
    return (
        output_dir
        / str(experiment_id)
        / "snip_auxiliary_masks"
        / "per_well"
        / str(well_id)
        / str(physical_embryo_id)
        / str(snip_id)
        / f"{auxiliary_mask_type}.{AUXILIARY_MASK_FORMAT}"
    )


def run_unet_for_snip_inventory(
    snip_inventory: pd.DataFrame,
    predictors: Mapping[str, AuxiliaryMaskPredictor],
    output_dir: Path,
    model_id: str,
    model_backend: str,
    checkpoint_paths: Mapping[str, str],
    artifact_shape: tuple[int, int],
) -> pd.DataFrame:
    """Run UNet auxiliary mask predictors over all valid snips.

    Only processes rows where is_valid_snip == True and processed_snip_path exists.
    Invalid snip → no row in output.
    Valid snip + predictor failure → row with is_valid_auxiliary_mask=False.

    ``artifact_shape`` (= snip_frame_shape) is the law: each snip image is asserted to be ON it
    BEFORE the per-mask try/except. An off-grid snip is a systemic upstream contract violation
    affecting every snip in the well identically, so it raises out of this runner and kills the
    shard — never laundered into scattered per-mask is_valid=False noise.
    """
    artifact_shape = (int(artifact_shape[0]), int(artifact_shape[1]))

    valid_snips = snip_inventory[
        snip_inventory["is_valid_snip"].astype(bool)
        & snip_inventory["processed_snip_path"].map(_is_existing_path)
    ]

    rows: list[dict] = []

    for _, snip_row in valid_snips.iterrows():
        snip_id = snip_row["snip_id"]
        physical_embryo_id = snip_row["physical_embryo_id"]
        experiment_id = snip_row["experiment_id"]
        snip_path = Path(snip_row["processed_snip_path"])

        snip_image = skio.imread(str(snip_path))
        if snip_image.ndim == 3:
            snip_image = snip_image[:, :, 0]
        # Guard the input grid BEFORE the per-mask try/except: a wrong-grid snip is a shard-wide
        # bug, so let it raise rather than be swallowed as a per-mask failure.
        assert_on_snip_frame(snip_image, artifact_shape, label=f"snip {snip_id}")
        snip_h, snip_w = snip_image.shape

        for mask_type in ALLOWED_AUXILIARY_MASK_TYPES:
            base_row = {
                "snip_id": snip_id,
                "physical_embryo_id": physical_embryo_id,
                "embryo_id": snip_row["embryo_id"],
                "experiment_id": experiment_id,
                "well_id": snip_row["well_id"],
                "image_id": snip_row["image_id"],
                "time_index": snip_row["time_index"],
                "channel_id": snip_row["channel_id"],
                "auxiliary_mask_type": mask_type,
                "auxiliary_mask_format": AUXILIARY_MASK_FORMAT,
                "model_backend": model_backend,
                "model_id": model_id,
                "checkpoint_path": checkpoint_paths.get(mask_type, ""),
                "snip_height_px": snip_h,
                "snip_width_px": snip_w,
            }

            if mask_type not in predictors:
                rows.append({
                    **base_row,
                    "auxiliary_mask_path": None,
                    "mask_height_px": snip_h,
                    "mask_width_px": snip_w,
                    "is_valid_auxiliary_mask": False,
                    "error_message": f"no predictor registered for mask_type '{mask_type}'",
                })
                continue

            try:
                mask = predictors[mask_type](snip_image)
                out_path = _mask_output_path(
                    output_dir,
                    experiment_id,
                    snip_row["well_id"],
                    physical_embryo_id,
                    snip_id,
                    mask_type,
                )
                write_auxiliary_mask_png(mask, out_path)
                rows.append({
                    **base_row,
                    "auxiliary_mask_path": str(out_path),
                    "mask_height_px": mask.shape[0],
                    "mask_width_px": mask.shape[1],
                    "is_valid_auxiliary_mask": True,
                    "error_message": None,
                })
            except Exception:
                rows.append({
                    **base_row,
                    "auxiliary_mask_path": None,
                    "mask_height_px": snip_h,
                    "mask_width_px": snip_w,
                    "is_valid_auxiliary_mask": False,
                    "error_message": traceback.format_exc(limit=3),
                })

    if not rows:
        df = pd.DataFrame(columns=list(snip_inventory.columns) + [
            "auxiliary_mask_type", "auxiliary_mask_path", "auxiliary_mask_format",
            "model_backend", "model_id", "checkpoint_path",
            "snip_height_px", "snip_width_px", "mask_height_px", "mask_width_px",
            "is_valid_auxiliary_mask", "error_message",
        ])
        df["is_valid_auxiliary_mask"] = df["is_valid_auxiliary_mask"].astype(bool)
        return df

    df = pd.DataFrame(rows)
    df["is_valid_auxiliary_mask"] = df["is_valid_auxiliary_mask"].astype(bool)
    validate_snip_auxiliary_masks(df)
    return df
