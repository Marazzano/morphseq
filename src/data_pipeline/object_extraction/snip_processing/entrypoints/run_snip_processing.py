"""Per-well snip processing entrypoint.

Reads a validated frame_masks shard, the matching frame_inventory shard, and the
per-well physical_embryo_registry shard; JOINS physical_embryo_id onto each valid
mask (it is NOT minted here anymore — the registry is the identity-origination
boundary), builds the crop-level snip identifiers, runs the
extraction/rotation/augmentation stack, and writes the per-well snip_inventory
CSV + pixel PNG files.

Masks are stored as RLE in frame_masks — decoded to numpy here before passing
to the core stack. Yolk masks are not yet wired; rotation falls back to the
mass-distribution heuristic and extraction uses a zero yolk mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.shared.identifiers.constructors import (
    build_embryo_id,
    build_snip_id,
)
from data_pipeline.shared.identifiers.parsers import parse_image_id
from data_pipeline.object_extraction.snip_processing.augmentation import augment_snip
from data_pipeline.object_extraction.snip_processing.extraction import crop_to_embryo_bounds, extract_embryo_crop
from data_pipeline.object_extraction.snip_processing.rotation import apply_rotation_to_snip


def _physical_embryo_id_by_track(
    physical_embryo_registry: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    """Build the ``(well_id, track_id) -> physical_embryo_id`` lookup from the registry.

    The registry is the single source of identity; snip_processing JOINS against it rather than
    re-minting. Returns an exact-match dict keyed by the join columns.
    """
    return {
        (str(row["well_id"]), str(row["track_id"])): str(row["physical_embryo_id"])
        for _, row in physical_embryo_registry.iterrows()
    }


def _estimate_background(
    valid_masks: pd.DataFrame,
    inventory_index: pd.DataFrame,
    n_samples: int = 50,
    seed: int = 309,
) -> tuple[float, float]:
    """Sample background pixels (outside embryo mask) to estimate mean/std.

    Matches the legacy build03A definition: pixels in the full-frame source
    image where the embryo mask == 0.
    """
    np.random.seed(seed)
    indices = valid_masks.index.tolist()
    sample_idx = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

    bkg_pixels: list[float] = []
    for i in sample_idx:
        row = valid_masks.loc[i]
        image_id = str(row["image_id"])
        if image_id not in inventory_index.index:
            continue
        try:
            src = Path(str(inventory_index.loc[image_id]["source_image_path"]))
            img = skio.imread(str(src))
            if img.ndim == 3:
                img = img[:, :, 0]
            rle = json.loads(str(row["mask_rle"]))
            mask = decode_binary_mask_rle(rle).astype(bool)
            bkg = img[~mask].astype(float)
            if bkg.size > 0:
                bkg_pixels.extend(bkg[:5000].tolist())
        except Exception:
            continue

    if not bkg_pixels:
        return 128.0, 30.0
    return float(np.mean(bkg_pixels)), float(np.std(bkg_pixels))


def run_snip_processing(
    *,
    frame_masks_csv: Path,
    frame_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    snips_dir: Path,
    output_root: Path,
    target_pixel_size_um: float = 7.8,
    output_height_px: int = 576,
    output_width_px: int = 256,
    background_noise_scale: float = 0.1,
) -> None:
    frame_masks = pd.read_csv(frame_masks_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    physical_embryo_registry = pd.read_csv(physical_embryo_registry_csv)

    valid_masks = frame_masks[frame_masks["is_valid_mask"].astype(bool)].copy()
    inventory_index = frame_inventory.set_index("image_id")

    # Identity is JOINED from the registry, never minted here. A valid mask whose track has no
    # registry row is a contract violation (the registry is built from frame_masks, so it must
    # cover every detected track) — fail loud rather than silently drop a real embryo.
    physical_embryo_id_by_track = _physical_embryo_id_by_track(physical_embryo_registry)

    output_shape = (output_height_px, output_width_px)
    snips_dir = Path(snips_dir)
    output_root = Path(output_root)

    _bg_mean, _bg_std = _estimate_background(valid_masks, inventory_index)
    background_mean = background_noise_scale * _bg_mean
    background_std = background_noise_scale * _bg_std

    rows: list[dict[str, Any]] = []

    for _, mask_row in valid_masks.iterrows():
        image_id = str(mask_row["image_id"])
        track_id = str(mask_row["track_id"])
        mask_id = str(mask_row["mask_id"])
        well_id = str(mask_row["well_id"])

        _, channel_id, time_index = parse_image_id(image_id)
        experiment_id = str(mask_row.get("experiment_id", ""))

        physical_embryo_id = physical_embryo_id_by_track.get((well_id, track_id))
        if physical_embryo_id is None:
            raise ValueError(
                f"No physical_embryo_registry entry for (well_id={well_id!r}, "
                f"track_id={track_id!r}) — every valid frame_masks track must be registered. "
                f"Rebuild the physical_embryo_registry shard for {well_id!r} from the SAME "
                f"frame_masks shard before running snip_processing."
            )
        # embryo_id / snip_id are crop-product naming and legitimately stay here.
        embryo_id = build_embryo_id(physical_embryo_id, image_id)
        snip_id = build_snip_id(embryo_id, image_id)

        out: dict[str, Any] = {
            "snip_id": snip_id,
            "embryo_id": embryo_id,
            "physical_embryo_id": physical_embryo_id,
            "experiment_id": experiment_id,
            "well_id": well_id,
            "image_id": image_id,
            "time_index": time_index,
            "channel_id": channel_id,
            "mask_id": mask_id,
            "track_id": track_id,
            "source_image_path": None,
            "processed_snip_path": None,
            "embryo_mask_snip_path": None,
            "crop_x_min_px": None,
            "crop_y_min_px": None,
            "crop_x_max_px": None,
            "crop_y_max_px": None,
            "crop_width_px": output_width_px,
            "crop_height_px": output_height_px,
            "is_valid_snip": False,
            "error_message": None,
        }

        try:
            if image_id not in inventory_index.index:
                raise KeyError(f"image_id {image_id!r} not found in frame_inventory")
            inv_row = inventory_index.loc[image_id]

            source_image_path = Path(str(inv_row["source_image_path"]))
            pixel_size_um = float(inv_row.get("micrometers_per_pixel", inv_row.get("source_micrometers_per_pixel", 2.17)))
            out["source_image_path"] = str(inv_row["source_image_path"])

            # Decode RLE mask from frame_masks row.
            rle = json.loads(str(mask_row["mask_rle"]))
            embryo_mask = decode_binary_mask_rle(rle).astype(np.uint8)

            image = skio.imread(str(source_image_path))
            if image.ndim == 3:
                image = image[:, :, 0]

            # No yolk mask yet — falls back gracefully in rotation + extraction.
            yolk_mask = np.zeros_like(embryo_mask)

            image_rescaled, mask_rescaled, yolk_rescaled = extract_embryo_crop(
                image, embryo_mask, yolk_mask, output_shape, pixel_size_um, target_pixel_size_um,
            )
            image_rotated, mask_rotated, yolk_rotated, _ = apply_rotation_to_snip(
                image_rescaled, mask_rescaled, yolk_rescaled,
            )
            image_cropped, mask_cropped, _ = crop_to_embryo_bounds(
                image_rotated, mask_rotated, yolk_rotated, output_shape,
            )

            augmented, _ = augment_snip(image_cropped, mask_cropped, background_mean, background_std)

            embryo_snips_dir = snips_dir / physical_embryo_id
            embryo_snips_dir.mkdir(parents=True, exist_ok=True)
            processed_path = embryo_snips_dir / f"{snip_id}.png"
            skio.imsave(str(processed_path), augmented, check_contrast=False)

            # Persist the cropped embryo mask in the SAME snip coordinate space as the snip image
            # (same crop transform, so they are pixel-aligned by construction). This is the embryo
            # mask fraction_alive ANDs against the per-snip via mask — no model, no re-prediction.
            embryo_mask_path = embryo_snips_dir / f"{snip_id}_embryo.png"
            skio.imsave(str(embryo_mask_path), (mask_cropped > 0).astype(np.uint8) * 255, check_contrast=False)

            try:
                out["processed_snip_path"] = processed_path.relative_to(output_root).as_posix()
            except ValueError:
                out["processed_snip_path"] = str(processed_path)
            try:
                out["embryo_mask_snip_path"] = embryo_mask_path.relative_to(output_root).as_posix()
            except ValueError:
                out["embryo_mask_snip_path"] = str(embryo_mask_path)

            out["is_valid_snip"] = True

        except Exception as exc:
            out["error_message"] = f"{type(exc).__name__}: {exc}"
            out["is_valid_snip"] = False

        rows.append(out)

    result_df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
