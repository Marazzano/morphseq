"""Factorial probe of the legacy/current snip saturation difference.

The probe reconstructs the July-2025 legacy crop path and then swaps one input
family at a time:

* legacy source + legacy mask/rotation (reconstruction control)
* current source + legacy mask/rotation (source-image effect)
* legacy source + current mask/rotation (mask/crop effect)
* current source + current mask/rotation (full current path at 6.5 um/px)

All four variants then receive the exact same CLAHE call.  This localizes a
post-CLAHE difference without relying on full-frame histogram summaries.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage
import skimage.io as skio
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.transform import rescale, resize

from data_pipeline.object_extraction.segmentation.masks.mask_rle import (
    decode_binary_mask_rle,
)


EXPERIMENT_ID = "20250612_30hpf_ctrl_atf6"
DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
OUTPUT_ROOT = DATA_ROOT / "pipeline" / "output"
OBJECT_ROOT = OUTPUT_ROOT / "object_extraction" / EXPERIMENT_ID
LEGACY_IMAGE_ROOT = (
    DATA_ROOT / "built_image_data" / "stitched_FF_images" / EXPERIMENT_ID
)
LEGACY_MASK_ROOT = (
    DATA_ROOT / "segmentation" / "mask_v0_0100_predictions" / EXPERIMENT_ID
)
LEGACY_YOLK_ROOT = (
    DATA_ROOT / "segmentation" / "yolk_v1_0050_predictions" / EXPERIMENT_ID
)
LEGACY_SNIP_ROOT = (
    DATA_ROOT / "training_data" / "bf_embryo_snips_uncropped" / EXPERIMENT_ID
)
LEGACY_SNIP_MASK_ROOT = DATA_ROOT / "training_data" / "bf_embryo_masks"
LEGACY_METADATA = (
    DATA_ROOT
    / "metadata"
    / "embryo_metadata_files"
    / f"{EXPERIMENT_ID}_embryo_metadata.csv"
)
HERE = Path(__file__).resolve().parent
OUTSHAPE = (576, 256)
OUTSCALE = 6.5


def rotate_image(mat: np.ndarray, angle: float) -> np.ndarray:
    """Exact legacy OpenCV expanded-canvas rotation."""
    height, width = mat.shape[:2]
    center = (width / 2, height / 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rotation[0, 0])
    abs_sin = abs(rotation[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation[0, 2] += bound_w / 2 - center[0]
    rotation[1, 2] += bound_h / 2 - center[1]
    return cv2.warpAffine(mat, rotation, (bound_w, bound_h))


def legacy_process_masks(
    embryo: np.ndarray,
    yolk: np.ndarray,
    region_label: int,
    close_radius: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact process_masks implementation from commit a4e9401d (2025-07-03)."""
    embryo_u8 = np.round(embryo / 255 * 2 - 1).astype(np.uint8)
    embryo_labels = label(embryo_u8)

    yolk_u8 = np.round(yolk / 255 * 2 - 1).astype(np.uint8)
    if np.any(yolk_u8 == 1):
        yolk_u8 = remove_small_objects(
            yolk_u8.astype(bool), min_size=75
        ).astype(int)

    if int(region_label) == 0:
        raise ValueError("legacy region_label must be nonzero")
    embryo_binary = (embryo_labels == int(region_label)).astype(int)
    embryo_binary = binary_closing(
        embryo_binary, disk(close_radius)
    ).astype(int)

    intersection = yolk_u8 * embryo_binary
    if np.sum(intersection) < 10:
        yolk_binary = np.zeros(yolk_u8.shape, dtype=int)
    else:
        yolk_labels = label(yolk_u8)
        labels_under_intersection = np.unique(yolk_labels[np.where(intersection)])
        if len(labels_under_intersection) == 1:
            yolk_binary = (
                yolk_labels == labels_under_intersection[0]
            ).astype(int)
        else:
            intersection_labels = label(intersection)
            props = regionprops(intersection_labels)
            max_index = int(np.argmax([prop.area for prop in props]))
            keep = np.unique(
                yolk_labels[np.where(intersection_labels == max_index + 1)]
            )
            yolk_binary = (yolk_labels == keep[0]).astype(int)
    return embryo_binary, yolk_binary


def legacy_angle(embryo: np.ndarray, yolk: np.ndarray) -> float:
    props = regionprops(embryo)
    if not props:
        return 0.0
    angle = props[0].orientation
    embryo_rotated = rotate_image(embryo, np.rad2deg(-angle))
    embryo_center = scipy.ndimage.center_of_mass(embryo_rotated, labels=1)
    if np.any(yolk):
        yolk_rotated = rotate_image(yolk, np.rad2deg(-angle))
        yolk_center = scipy.ndimage.center_of_mass(yolk_rotated, labels=1)
        return -angle if embryo_center[0] - yolk_center[0] >= 0 else -angle + np.pi
    y_indices = np.where(np.max(embryo_rotated, axis=1))[0]
    vertical_ratio = np.sum(y_indices > embryo_center[0]) / len(y_indices)
    return -angle if vertical_ratio >= 0.5 else -angle + np.pi


def legacy_crop(
    image: np.ndarray,
    embryo: np.ndarray,
    yolk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_indices = np.where(np.max(embryo, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(embryo, axis=0) > 0.5)[0]
    y_mean, x_mean = int(np.mean(y_indices)), int(np.mean(x_indices))

    raw_y = [y_mean - OUTSHAPE[0] // 2, y_mean + OUTSHAPE[0] // 2]
    from_y = np.asarray([max(raw_y[0], 0), min(raw_y[1], embryo.shape[0])])
    to_y = [from_y[0] - raw_y[0], OUTSHAPE[0] + from_y[1] - raw_y[1]]

    raw_x = [x_mean - OUTSHAPE[1] // 2, x_mean + OUTSHAPE[1] // 2]
    from_x = np.asarray([max(raw_x[0], 0), min(raw_x[1], embryo.shape[1])])
    to_x = [from_x[0] - raw_x[0], OUTSHAPE[1] + from_x[1] - raw_x[1]]

    image_crop = np.zeros(OUTSHAPE, dtype=np.uint8)
    mask_crop = np.zeros(OUTSHAPE)
    yolk_crop = np.zeros(OUTSHAPE)
    source = np.s_[from_y[0] : from_y[1], from_x[0] : from_x[1]]
    target = np.s_[to_y[0] : to_y[1], to_x[0] : to_x[1]]
    image_crop[target] = image[source]
    mask_crop[target] = embryo[source]
    yolk_crop[target] = yolk[source]
    return image_crop, mask_crop, yolk_crop


def transform(
    image: np.ndarray,
    embryo: np.ndarray,
    yolk: np.ndarray,
    pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = pixel_size_um / OUTSCALE
    image_rescaled = rescale(
        image, (scale, scale), order=1, preserve_range=True
    )
    embryo_rescaled = resize(
        embryo.astype(float), image_rescaled.shape, order=1
    )
    yolk_rescaled = resize(
        yolk.astype(float), image_rescaled.shape, order=1
    )
    angle = legacy_angle(
        (embryo_rescaled > 0.5).astype(np.uint8),
        (yolk_rescaled > 0.5).astype(np.uint8),
    )
    rotated = rotate_image(image_rescaled, np.rad2deg(angle))
    embryo_rotated = rotate_image(embryo_rescaled, np.rad2deg(angle))
    yolk_rotated = rotate_image(yolk_rescaled, np.rad2deg(angle))
    crop, mask_crop, _ = legacy_crop(rotated, embryo_rotated, yolk_rotated)
    return crop, mask_crop > 0.5


def clahe(image: np.ndarray) -> np.ndarray:
    return (skimage.exposure.equalize_adapthist(image) * 255).astype(np.uint8)


def jpeg_roundtrip(image: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"))


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    return {
        "n_pixels": int(values.size),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "frac_ge_250": float(np.mean(values >= 250)),
        "frac_eq_255": float(np.mean(values == 255)),
    }


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    af, bf = a.astype(float).ravel(), b.astype(float).ravel()
    return {
        "mae": float(np.mean(np.abs(af - bf))),
        "rmse": float(np.sqrt(np.mean((af - bf) ** 2))),
        "correlation": float(np.corrcoef(af, bf)[0, 1]),
    }


def main() -> None:
    metadata = pd.read_csv(LEGACY_METADATA)
    metadata["well"] = metadata["well"].astype(str)
    inventory = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(
                (OBJECT_ROOT / "snips" / "per_well").glob(
                    "*/*_snip_inventory.csv"
                )
            )
        ],
        ignore_index=True,
    )
    inventory = inventory[inventory["is_valid_snip"].astype(bool)]

    values_by_variant: dict[str, list[np.ndarray]] = {
        "observed_legacy_uncropped": [],
        "reconstructed_legacy_memory": [],
        "reconstructed_legacy_jpeg": [],
        "current_source_legacy_mask": [],
        "legacy_source_current_mask": [],
        "current_source_current_mask": [],
    }
    reconstruction_pairs: list[dict[str, float]] = []
    source_raw_pairs: list[dict[str, float]] = []
    per_well: list[dict[str, object]] = []

    for well_id, rows in inventory.groupby("well_id", sort=True):
        if len(rows) != 1:
            continue
        well = str(well_id).rsplit("_", 1)[-1]
        legacy_uncropped_path = (
            LEGACY_SNIP_ROOT / f"{EXPERIMENT_ID}_{well}_e00_t0000.jpg"
        )
        legacy_snip_mask_path = (
            LEGACY_SNIP_MASK_ROOT
            / f"emb_{EXPERIMENT_ID}_{well}_e00_t0000.jpg"
        )
        legacy_image_path = LEGACY_IMAGE_ROOT / f"{well}_t0000_stitch.jpg"
        legacy_mask_path = LEGACY_MASK_ROOT / f"{well}_t0000_stitch.jpg.jpg"
        legacy_yolk_path = LEGACY_YOLK_ROOT / f"{well}_t0000_stitch.jpg.jpg"
        required = [
            legacy_uncropped_path,
            legacy_snip_mask_path,
            legacy_image_path,
            legacy_mask_path,
            legacy_yolk_path,
        ]
        if not all(path.is_file() for path in required):
            continue

        # The legacy metadata CSV was later incrementally overwritten and now
        # retains only two wells.  These single-embryo plates used region 1 and
        # one experiment-wide Keyence calibration; recover the calibration
        # from the retained rows and use it for every reconstructed well.
        meta_rows = metadata.loc[metadata["well"] == well]
        calibration_row = (
            meta_rows.iloc[0] if len(meta_rows) == 1 else metadata.iloc[0]
        )
        pixel_size = float(calibration_row["Height (um)"]) / float(
            calibration_row["Height (px)"]
        )
        region_label = (
            int(meta_rows.iloc[0]["region_label"]) if len(meta_rows) == 1 else 1
        )

        row = rows.iloc[0]
        current_image = skio.imread(str(row["image_path"]))
        if current_image.ndim == 3:
            current_image = current_image[:, :, 0]
        legacy_image = skio.imread(str(legacy_image_path))
        if legacy_image.ndim == 3:
            legacy_image = legacy_image[:, :, 0]
        if legacy_image.shape[0] < legacy_image.shape[1]:
            legacy_image = legacy_image.transpose(1, 0)
        if current_image.shape != legacy_image.shape:
            continue

        old_mask_raw = skio.imread(str(legacy_mask_path))
        old_yolk_raw = skio.imread(str(legacy_yolk_path))
        old_mask, old_yolk = legacy_process_masks(
            old_mask_raw, old_yolk_raw, region_label
        )

        frame_masks_path = (
            OBJECT_ROOT
            / "frame_masks"
            / "per_well"
            / str(well_id)
            / f"{well_id}_frame_masks.csv"
        )
        frame_masks = pd.read_csv(frame_masks_path)
        current_mask_row = frame_masks.loc[
            frame_masks["mask_id"].astype(str) == str(row["mask_id"])
        ].iloc[0]
        current_mask = decode_binary_mask_rle(
            json.loads(str(current_mask_row["mask_rle"]))
        ).astype(np.uint8)
        empty_yolk = np.zeros_like(current_mask)

        raw_ll, mask_ll = transform(
            legacy_image, old_mask, old_yolk, pixel_size
        )
        raw_cl, mask_cl = transform(
            current_image, old_mask, old_yolk, pixel_size
        )
        raw_lc, mask_lc = transform(
            legacy_image, current_mask, empty_yolk, pixel_size
        )
        raw_cc, mask_cc = transform(
            current_image, current_mask, empty_yolk, pixel_size
        )

        out_ll = clahe(raw_ll)
        out_cl = clahe(raw_cl)
        out_lc = clahe(raw_lc)
        out_cc = clahe(raw_cc)
        out_ll_jpeg = jpeg_roundtrip(out_ll)
        observed = skio.imread(str(legacy_uncropped_path))
        # Legacy masks were written as uint8 {0,1} JPEGs rather than {0,255}.
        # The decoded foreground is consequently 1-2, not >127.
        observed_mask = skio.imread(str(legacy_snip_mask_path)) > 0

        variants = {
            "observed_legacy_uncropped": (observed, observed_mask),
            "reconstructed_legacy_memory": (out_ll, mask_ll),
            "reconstructed_legacy_jpeg": (out_ll_jpeg, mask_ll),
            "current_source_legacy_mask": (out_cl, mask_cl),
            "legacy_source_current_mask": (out_lc, mask_lc),
            "current_source_current_mask": (out_cc, mask_cc),
        }
        row_out: dict[str, object] = {"well": well}
        for name, (image, mask) in variants.items():
            values_by_variant[name].append(image[mask])
            for metric, value in summary(image[mask]).items():
                row_out[f"{name}__{metric}"] = value

        reconstruction = pair_stats(observed, out_ll_jpeg)
        reconstruction_pairs.append(reconstruction)
        for metric, value in reconstruction.items():
            row_out[f"reconstruction__{metric}"] = value
        raw_source = pair_stats(raw_ll, raw_cl)
        source_raw_pairs.append(raw_source)
        for metric, value in raw_source.items():
            row_out[f"source_raw__{metric}"] = value
        per_well.append(row_out)

    if not per_well:
        raise RuntimeError("No wells completed the factorial probe")

    aggregate = {
        name: summary(np.concatenate(chunks))
        for name, chunks in values_by_variant.items()
    }
    aggregate["reconstruction_pair_mean"] = {
        key: float(np.mean([row[key] for row in reconstruction_pairs]))
        for key in reconstruction_pairs[0]
    }
    aggregate["source_raw_pair_mean"] = {
        key: float(np.mean([row[key] for row in source_raw_pairs]))
        for key in source_raw_pairs[0]
    }
    result = {
        "skimage_version": skimage.__version__,
        "n_wells": len(per_well),
        "aggregate": aggregate,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "factorial_stage_probe.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    pd.DataFrame(per_well).to_csv(
        HERE / "factorial_stage_probe_per_well.csv", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
