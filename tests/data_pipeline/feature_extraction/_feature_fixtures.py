"""Shared synthetic inputs for feature-product tests (snip_inventory + frame_masks + frame_inventory)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.masks.mask_rle import encode_binary_mask_rle
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_mask_id,
    build_physical_embryo_id,
    build_snip_id,
    build_track_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"
PHYS = build_physical_embryo_id(WELL, 1)
PIXEL_SIZE_UM = 2.0


def make_inputs(*, time_indices=(0, 1, 2), mask_side=10):
    """Return (snip_inventory, frame_masks, frame_inventory, registry) for one well/track."""
    track_id = build_track_id(WELL, 0)
    snip_rows, mask_rows, inv_rows = [], [], []
    for t in time_indices:
        image_id = build_image_id(WELL, CHANNEL, t)
        embryo_id = build_embryo_id(PHYS, image_id)
        snip_id = build_snip_id(embryo_id, image_id)
        mask_id = build_mask_id(image_id, 0)

        mask = np.zeros((40, 40), dtype=bool)
        # Offset the square a little each frame so kinematics are non-zero.
        x0 = 5 + t
        mask[x0 : x0 + mask_side, x0 : x0 + mask_side] = True
        rle = encode_binary_mask_rle(mask)

        snip_rows.append({
            "snip_id": snip_id, "embryo_id": embryo_id, "physical_embryo_id": PHYS,
            "experiment_id": EXP, "well_id": WELL, "image_id": image_id,
            "time_index": t, "channel_id": CHANNEL, "mask_id": mask_id,
            "track_id": track_id, "is_valid_snip": True,
        })
        mask_rows.append({"mask_id": mask_id, "image_id": image_id, "mask_rle": json.dumps(rle)})
        inv_rows.append({
            "image_id": image_id, "source_micrometers_per_pixel": PIXEL_SIZE_UM,
            "elapsed_time_s": float(t * 100),
        })

    registry = pd.DataFrame({"physical_embryo_id": [PHYS]})
    return (
        pd.DataFrame(snip_rows),
        pd.DataFrame(mask_rows),
        pd.DataFrame(inv_rows),
        registry,
    )


def make_plate_metadata():
    return pd.DataFrame({"well_id": [WELL], "start_age_hpf": [11.0], "temperature": [30.0]})
