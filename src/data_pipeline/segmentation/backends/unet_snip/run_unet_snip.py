"""Per-snip auxiliary mask runner for the unet_snip backend.

Step 1: predictor interface only — no Torch, no FishModel, no checkpoint loading.
Step 2 will add FishModelSnipPredictor in model_loader.py.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
    validate_snip_auxiliary_masks,
)

# Predictor contract:
#   input  — H×W uint8 grayscale snip image (original snip dimensions)
#   output — H×W bool mask (same spatial dimensions as input)
AuxiliaryMaskPredictor = Callable[[np.ndarray], np.ndarray]

UNET_SNIP_BACKEND_LABEL = "unet_snip"
AUXILIARY_MASK_FORMAT = "png"


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
    physical_embryo_id: str,
    snip_id: str,
    auxiliary_mask_type: str,
) -> Path:
    return (
        output_dir
        / experiment_id
        / "snip_auxiliary_masks"
        / "per_well"
        / snip_id.split("_")[1]  # well_id = second token of snip_id (e.g. B01 from 20250912_B01_...)
        / physical_embryo_id
        / snip_id
        / f"{auxiliary_mask_type}.{AUXILIARY_MASK_FORMAT}"
    )


def run_unet_for_snip_inventory(
    snip_inventory: pd.DataFrame,
    predictors: Mapping[str, AuxiliaryMaskPredictor],
    output_dir: Path,
    model_id: str,
    model_backend: str,
    checkpoint_paths: Mapping[str, str],
) -> pd.DataFrame:
    """Run UNet auxiliary mask predictors over all valid snips.

    Only processes rows where is_valid_snip == True and processed_snip_path exists.
    Invalid snip → no row in output.
    Valid snip + predictor failure → row with is_valid_auxiliary_mask=False.
    """
    valid_snips = snip_inventory[
        snip_inventory["is_valid_snip"].astype(bool)
        & snip_inventory["processed_snip_path"].notnull()
        & snip_inventory["processed_snip_path"].apply(lambda p: Path(p).exists() if p else False)
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
                    output_dir, experiment_id, physical_embryo_id, snip_id, mask_type
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
