"""Extract matched current-pipeline raw crops for a cross-version CLAHE probe.

This is diagnostic-only.  It deliberately stops immediately before CLAHE so the
same exact uint8 arrays can be evaluated by each historical scikit-image
environment without confounding crop, resize, or rotation behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.object_extraction.segmentation.masks.mask_rle import (
    decode_binary_mask_rle,
)
from data_pipeline.object_extraction.snip_processing.extraction import (
    crop_to_embryo_bounds,
    extract_embryo_crop,
)
from data_pipeline.object_extraction.snip_processing.rotation import (
    apply_rotation_to_snip,
)


EXPERIMENT_ID = "20250612_30hpf_ctrl_atf6"
DATA_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
OUTPUT_ROOT = DATA_ROOT / "pipeline" / "output"
OBJECT_ROOT = OUTPUT_ROOT / "object_extraction" / EXPERIMENT_ID
LEGACY_ROOT = DATA_ROOT / "training_data" / "bf_embryo_snips" / EXPERIMENT_ID
HERE = Path(__file__).resolve().parent


def main() -> None:
    inventory_files = sorted(
        (OBJECT_ROOT / "snips" / "per_well").glob("*/*_snip_inventory.csv")
    )
    inventory = pd.concat(
        [pd.read_csv(path) for path in inventory_files], ignore_index=True
    )
    inventory = inventory[inventory["is_valid_snip"].astype(bool)].copy()

    raw_crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    wells: list[str] = []

    for well_id, well_rows in inventory.groupby("well_id", sort=True):
        well = str(well_id).rsplit("_", 1)[-1]
        legacy_path = LEGACY_ROOT / f"{EXPERIMENT_ID}_{well}_e00_t0000.jpg"
        if not legacy_path.is_file() or len(well_rows) != 1:
            continue

        row = well_rows.iloc[0]
        frame_masks_path = (
            OBJECT_ROOT
            / "frame_masks"
            / "per_well"
            / str(well_id)
            / f"{well_id}_frame_masks.csv"
        )
        frame_masks = pd.read_csv(frame_masks_path)
        mask_row = frame_masks.loc[
            frame_masks["mask_id"].astype(str) == str(row["mask_id"])
        ].iloc[0]
        embryo_mask = decode_binary_mask_rle(
            json.loads(str(mask_row["mask_rle"]))
        ).astype(np.uint8)

        image = skio.imread(str(row["image_path"]))
        if image.ndim == 3:
            image = image[:, :, 0]

        image_rescaled, mask_rescaled, yolk_rescaled = extract_embryo_crop(
            image=image,
            mask=embryo_mask,
            yolk_mask=np.zeros_like(embryo_mask),
            target_shape=(576, 256),
            pixel_size_um=1.8872090277777778,
            target_pixel_size_um=6.5,
        )
        image_rotated, mask_rotated, yolk_rotated, _ = apply_rotation_to_snip(
            image_rescaled, mask_rescaled, yolk_rescaled
        )
        image_cropped, mask_cropped, _ = crop_to_embryo_bounds(
            image_rotated, mask_rotated, yolk_rotated, (576, 256)
        )

        raw_crops.append(np.asarray(image_cropped, dtype=np.uint8))
        masks.append(np.asarray(mask_cropped > 0.5, dtype=bool))
        wells.append(well)

    if not raw_crops:
        raise RuntimeError("No matched raw crops were extracted")

    HERE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        HERE / "current_raw_crops_6p5.npz",
        raw=np.stack(raw_crops),
        mask=np.stack(masks),
        wells=np.asarray(wells),
    )
    print(f"saved {len(wells)} matched raw crops")


if __name__ == "__main__":
    main()
