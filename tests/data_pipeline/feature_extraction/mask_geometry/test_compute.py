"""mask_geometry compute tests — synthetic RLE masks, deterministic geometry."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_pipeline.feature_extraction.mask_geometry.compute import (
    compute_mask_geometry_features,
)
from data_pipeline.feature_extraction.mask_geometry.contract import (
    validate_mask_geometry_features,
)
from data_pipeline.segmentation.masks.mask_rle import encode_binary_mask_rle
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_mask_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"
PHYS = build_physical_embryo_id(WELL, 1)
PIXEL_SIZE_UM = 2.0


def _one_well_inputs(*, time_indices=(0, 1, 2), mask_size=10):
    """Build matching snip_inventory / frame_masks / frame_inventory for one well."""
    snip_rows, mask_rows, inv_rows = [], [], []
    for t in time_indices:
        image_id = build_image_id(WELL, CHANNEL, t)
        embryo_id = build_embryo_id(PHYS, image_id)
        snip_id = build_snip_id(embryo_id, image_id)
        mask_id = build_mask_id(image_id, 0)

        # A solid square mask of known pixel count.
        mask = np.zeros((20, 20), dtype=bool)
        mask[5 : 5 + mask_size, 5 : 5 + mask_size] = True
        rle = encode_binary_mask_rle(mask)

        snip_rows.append(
            {
                "snip_id": snip_id,
                "embryo_id": embryo_id,
                "physical_embryo_id": PHYS,
                "experiment_id": EXP,
                "well_id": WELL,
                "image_id": image_id,
                "time_index": t,
                "channel_id": CHANNEL,
                "mask_id": mask_id,
                "track_id": f"{WELL}_track0000",
                "is_valid_snip": True,
            }
        )
        mask_rows.append(
            {"mask_id": mask_id, "image_id": image_id, "mask_rle": json.dumps(rle)}
        )
        inv_rows.append(
            {"image_id": image_id, "source_micrometers_per_pixel": PIXEL_SIZE_UM}
        )

    return (
        pd.DataFrame(snip_rows),
        pd.DataFrame(mask_rows),
        pd.DataFrame(inv_rows),
        mask_size,
    )


def test_compute_one_row_per_snip_with_known_area():
    snip, masks, inv, side = _one_well_inputs()
    df = compute_mask_geometry_features(snip, masks, inv)

    assert len(df) == len(snip)
    # area_um2 = pixel_count * pixel_size**2 = side*side * 2.0**2
    expected_area = side * side * (PIXEL_SIZE_UM**2)
    assert np.allclose(df["area_um2"], expected_area)
    validate_mask_geometry_features(df)


def test_compute_emits_row_for_invalid_snip_no_filtering():
    # An invalid snip with an empty mask still produces a row with 0.0 features (QC excludes, not us).
    snip, masks, inv, _ = _one_well_inputs(time_indices=(0,))
    snip.loc[0, "is_valid_snip"] = False
    empty = np.zeros((20, 20), dtype=bool)
    masks.loc[0, "mask_rle"] = json.dumps(encode_binary_mask_rle(empty))

    df = compute_mask_geometry_features(snip, masks, inv)
    assert len(df) == 1
    assert float(df.iloc[0]["area_um2"]) == 0.0
    validate_mask_geometry_features(df)


def test_compute_falls_back_to_legacy_pixel_size_column():
    snip, masks, inv, side = _one_well_inputs(time_indices=(0,))
    inv = inv.rename(columns={"source_micrometers_per_pixel": "micrometers_per_pixel"})
    df = compute_mask_geometry_features(snip, masks, inv)
    expected_area = side * side * (PIXEL_SIZE_UM**2)
    assert np.allclose(df["area_um2"], expected_area)


def test_compute_fails_loud_on_missing_pixel_size():
    snip, masks, inv, _ = _one_well_inputs(time_indices=(0,))
    inv = inv.drop(columns=["source_micrometers_per_pixel"])
    with pytest.raises(ValueError, match="micron calibration"):
        compute_mask_geometry_features(snip, masks, inv)


def test_compute_fails_loud_on_mask_image_mismatch():
    snip, masks, inv, _ = _one_well_inputs(time_indices=(0,))
    masks.loc[0, "image_id"] = build_image_id(WELL, CHANNEL, 99)
    with pytest.raises(ValueError, match="frame_masks but snip"):
        compute_mask_geometry_features(snip, masks, inv)
