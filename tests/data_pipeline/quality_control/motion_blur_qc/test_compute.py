"""motion_blur_qc compute tests.

Covers the pure adjacent-z NCC behavior and the batch path that reads z-stack planes from
frame_inventory.image_path while masks come from canonical frame_masks RLE.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from data_pipeline.quality_control.motion_blur_qc.compute import (
    compute_mask_pixel_motion_metrics,
    compute_motion_blur_qc,
)
from data_pipeline.quality_control.motion_blur_qc.config import resolve_config
from data_pipeline.quality_control.motion_blur_qc.contract import validate_motion_blur_qc
from data_pipeline.object_extraction.segmentation.masks.mask_rle import encode_binary_mask_rle
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
PRODUCT_KEY = "BF__z_stack"
H = W = 12


def _mask(shape=(H, W)):
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return m


def _gradient(shape=(H, W)):
    y, x = np.indices(shape)
    return (x * 7 + y * 11).astype("uint8")


def _inverse_gradient(shape=(H, W)):
    return (255 - _gradient(shape)).astype("uint8")


def _snip_rows(tmp_path, *, mask, planes):
    inv_rows, mask_rows, fi_rows = [], [], []
    phys = build_physical_embryo_id(WELL, 1)
    image_id = build_image_id(WELL, CHANNEL, 0)
    embryo_id = build_embryo_id(phys, image_id)
    snip_id = build_snip_id(embryo_id, image_id)
    mask_id = build_mask_id(image_id, 1)
    inv_rows.append(
        {
            "experiment_id": EXP,
            "well_id": WELL,
            "physical_embryo_id": phys,
            "embryo_id": embryo_id,
            "snip_id": snip_id,
            "image_id": image_id,
            "time_index": 0,
            "channel_id": CHANNEL,
            "mask_id": mask_id,
        }
    )
    mask_rows.append(
        {"mask_id": mask_id, "image_id": image_id, "mask_rle": encode_binary_mask_rle(mask)}
    )
    for z, plane in enumerate(planes):
        z_image_id = build_image_id(WELL, CHANNEL, 0, z_index=z)
        path = tmp_path / f"{z_image_id}.png"
        cv2.imwrite(str(path), plane)
        fi_rows.append(
            {
                "well_id": WELL,
                "channel_id": CHANNEL,
                "time_index": 0,
                "z_index": z,
                "image_id": z_image_id,
                "image_product_type": "z_stack",
                "projection_method": pd.NA,
                "image_path": str(path),
                # Required by _resample_to_qc_resolution once config.qc_micrometers_per_pixel is
                # set (it is, by default). Pinned EQUAL to that QC target so resampling is a
                # deliberate no-op and these metric assertions stay about motion, not rescaling.
                "image_micrometers_per_pixel": resolve_config().qc_micrometers_per_pixel,
            }
        )
    return pd.DataFrame(inv_rows), pd.DataFrame(mask_rows), pd.DataFrame(fi_rows)


def test_identical_nonflat_planes_pass():
    config = resolve_config()
    base = _gradient()
    out = compute_mask_pixel_motion_metrics(
        np.stack([base, base, base], axis=0),
        _mask(),
        config=config,
    )
    assert out["mask_pixel_ncc_min"] == pytest.approx(1.0)
    assert out["mask_pixel_bad_pair_frac"] == 0.0
    assert out["motion_blur_flag"] == False  # noqa: E712


def test_low_ncc_pair_flags_when_bad_pair_fraction_exceeds_threshold():
    config = resolve_config()
    out = compute_mask_pixel_motion_metrics(
        np.stack([_gradient(), _inverse_gradient()], axis=0),
        _mask(),
        config=config,
    )
    assert out["mask_pixel_ncc_min"] < config.bad_z_pair_ncc_threshold
    assert out["mask_pixel_bad_pair_frac"] == 1.0
    assert out["mask_pixel_longest_bad_run"] == 1
    assert out["motion_blur_flag"] == True  # noqa: E712


def test_flat_pairs_are_skipped_and_counted():
    config = resolve_config()
    flat = np.full((H, W), 7, dtype="uint8")
    base = _gradient()
    out = compute_mask_pixel_motion_metrics(
        np.stack([flat, base, base], axis=0),
        _mask(),
        config=config,
    )
    assert out["n_z_pairs"] == 2
    assert out["n_flat_z_pairs"] == 1
    assert out["n_valid_z_pairs"] == 1
    assert out["mask_pixel_ncc_min"] == pytest.approx(1.0)


def test_all_flat_pairs_fail_loud():
    config = resolve_config()
    flat = np.full((H, W), 7, dtype="uint8")
    with pytest.raises(ValueError, match="no valid adjacent"):
        compute_mask_pixel_motion_metrics(
            np.stack([flat, flat], axis=0),
            _mask(),
            config=config,
        )


def test_empty_aligned_mask_fails_loud():
    config = resolve_config()
    with pytest.raises(ValueError, match="zero pixels"):
        compute_mask_pixel_motion_metrics(
            np.stack([_gradient(), _gradient()], axis=0),
            np.zeros((H, W), dtype=bool),
            config=config,
        )


def test_batch_compute_loads_z_stack_rows_and_aligns_mask(tmp_path):
    small_planes = [_gradient((6, 6)), _gradient((6, 6)), _inverse_gradient((6, 6))]
    inv, masks, fi = _snip_rows(tmp_path, mask=_mask((H, W)), planes=small_planes)

    out = compute_motion_blur_qc(inv, masks, fi, config=resolve_config())

    row = out.iloc[0]
    assert row["n_z_planes"] == 3
    assert row["n_z_pairs"] == 2
    assert row["n_valid_z_pairs"] == 2
    assert row["n_mask_pixels"] > 0
    assert row["motion_blur_flag"] == True  # noqa: E712
    validate_motion_blur_qc(out)


def test_missing_z_stack_row_fails_loud(tmp_path):
    inv, masks, fi = _snip_rows(tmp_path, mask=_mask(), planes=[_gradient(), _gradient()])
    fi = fi.iloc[0:0]
    with pytest.raises(ValueError, match="could not load z stack"):
        compute_motion_blur_qc(inv, masks, fi, config=resolve_config())


def test_missing_mask_fails_loud(tmp_path):
    inv, masks, fi = _snip_rows(tmp_path, mask=_mask(), planes=[_gradient(), _gradient()])
    masks = masks.iloc[0:0]
    with pytest.raises(ValueError, match="not found in frame_masks"):
        compute_motion_blur_qc(inv, masks, fi, config=resolve_config())
