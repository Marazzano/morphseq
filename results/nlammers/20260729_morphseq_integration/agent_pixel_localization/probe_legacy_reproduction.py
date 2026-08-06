"""Probe whether the checked-in legacy algorithm reproduces one stored CLAHE-only snip."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.exposure
import skimage.io as skio
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.transform import rescale, resize

from src.core.functions.image_utils import (
    crop_embryo_image,
    get_embryo_angle,
    process_masks,
    rotate_image,
)


DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
EXPERIMENT = "20250612_30hpf_ctrl_atf6"
WELL = "A01"


def jpeg_roundtrip(array: np.ndarray, quality: int) -> np.ndarray:
    handle = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(
        handle, format="JPEG", quality=quality
    )
    handle.seek(0)
    return np.asarray(Image.open(handle).convert("L"))


def main() -> None:
    metadata_path = (
        DATA_ROOT
        / "metadata"
        / "embryo_metadata_files"
        / f"{EXPERIMENT}_embryo_metadata.csv"
    )
    metadata = pd.read_csv(metadata_path)
    matching = metadata.loc[metadata["well"].astype(str) == WELL]
    if len(matching):
        row = matching.iloc[0]
    else:
        # The per-experiment metadata currently contains only two retained rows, but the
        # acquisition geometry and single-embryo region label are plate-wide constants.
        row = pd.Series(
            {
                "Height (um)": float(metadata["Height (um)"].iloc[0]),
                "Height (px)": float(metadata["Height (px)"].iloc[0]),
                "region_label": 1,
            }
        )

    ff_path = (
        DATA_ROOT
        / "built_image_data"
        / "stitched_FF_images"
        / EXPERIMENT
        / f"{WELL}_t0000_stitch.jpg"
    )
    mask_path = (
        DATA_ROOT
        / "segmentation"
        / "mask_v0_0100_predictions"
        / EXPERIMENT
        / f"{WELL}_t0000_stitch.jpg.jpg"
    )
    yolk_path = (
        DATA_ROOT
        / "segmentation"
        / "yolk_v1_0050_predictions"
        / EXPERIMENT
        / f"{WELL}_t0000_stitch.jpg.jpg"
    )
    stored_path = (
        DATA_ROOT
        / "training_data"
        / "bf_embryo_snips_uncropped"
        / EXPERIMENT
        / f"{EXPERIMENT}_{WELL}_e00_t0000.jpg"
    )

    image = skio.imread(ff_path)
    embryo_mask, yolk_mask = process_masks(
        skio.imread(mask_path), skio.imread(yolk_path), row
    )
    pixel_size = float(row["Height (um)"]) / float(row["Height (px)"])
    scale = pixel_size / 6.5
    image_rescaled = rescale(
        image, (scale, scale), order=1, preserve_range=True
    )
    embryo_rescaled = resize(
        embryo_mask.astype(float), image_rescaled.shape, order=1
    )
    yolk_rescaled = resize(
        yolk_mask.astype(float), image_rescaled.shape, order=1
    )
    angle = get_embryo_angle(
        (embryo_rescaled > 0.5).astype(np.uint8),
        (yolk_rescaled > 0.5).astype(np.uint8),
    )
    image_rotated = rotate_image(image_rescaled, np.rad2deg(angle))
    embryo_rotated = rotate_image(embryo_rescaled, np.rad2deg(angle))
    yolk_rotated = rotate_image(yolk_rescaled, np.rad2deg(angle))
    raw_crop, mask_crop, _ = crop_embryo_image(
        image_rotated,
        embryo_rotated,
        yolk_rotated,
        outshape=(576, 256),
    )
    clahe = (
        skimage.exposure.equalize_adapthist(raw_crop) * 255
    ).astype(np.uint8)
    stored = skio.imread(stored_path)

    print(f"pixel_size={pixel_size:.12f} angle_deg={np.rad2deg(angle):.6f}")
    print(
        "mask_area generated / stored pixels >30:",
        int((mask_crop > 0.5).sum()),
        int((stored > 30).sum()),
    )
    for quality in (75, 90, 95):
        encoded = jpeg_roundtrip(clahe, quality)
        mae = float(np.mean(np.abs(encoded.astype(float) - stored.astype(float))))
        rmse = float(
            np.sqrt(np.mean((encoded.astype(float) - stored.astype(float)) ** 2))
        )
        corr = float(np.corrcoef(encoded.ravel(), stored.ravel())[0, 1])
        ssim = float(
            structural_similarity(encoded, stored, data_range=255)
        )
        print(
            f"quality={quality}: mae={mae:.4f} rmse={rmse:.4f} "
            f"corr={corr:.6f} ssim={ssim:.6f}"
        )
    print(
        "generated percentiles:",
        np.percentile(clahe[mask_crop > 0.5], [1, 5, 50, 95, 99, 99.9]),
    )
    print(
        "stored percentiles (generated mask):",
        np.percentile(stored[mask_crop > 0.5], [1, 5, 50, 95, 99, 99.9]),
    )
    print(
        "saturation generated/stored within generated mask:",
        float(np.mean(clahe[mask_crop > 0.5] >= 250)),
        float(np.mean(stored[mask_crop > 0.5] >= 250)),
    )


if __name__ == "__main__":
    main()
