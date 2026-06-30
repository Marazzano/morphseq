"""focus_qc compute tests.

Covers: a textured interior passes (high interior_strong_edge_fraction), a flat/ghost interior
trips focus_flag, and fail-loud on a missing projection row / missing mask. Pixels are read
through materialized_image_readers (via frame_inventory), masks via the canonical frame_masks RLE.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from data_pipeline.quality_control.focus_qc.compute import (
    compute_focus_qc,
    compute_interior_strong_edge_fraction,
)
from data_pipeline.quality_control.focus_qc.config import resolve_config
from data_pipeline.quality_control.focus_qc.contract import validate_focus_qc
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
PRODUCT_KEY = "BF__projection__focus_stack"
H = W = 40


def _embryo_mask():
    m = np.zeros((H, W), dtype=bool)
    m[8:32, 8:32] = True
    return m


def _textured_image():
    """A checkerboard pattern inside the mask region — plenty of strong edges."""
    img = np.zeros((H, W), dtype="uint8")
    block = 4
    for i in range(H):
        for j in range(W):
            img[i, j] = 220 if ((i // block) + (j // block)) % 2 == 0 else 30
    return img


def _flat_image():
    """A genuinely constant image — no internal structure at all, a "ghost" embryo."""
    return np.full((H, W), 128, dtype="uint8")


def _snip_rows(tmp_path, specs):
    """specs: list of (phys_index, time_index, mask, image). Returns (snip_inventory_df,
    frame_masks_df, frame_inventory_df)."""
    inv_rows, mask_rows, fi_rows = [], [], []
    for phys_index, t, mask, image in specs:
        phys = build_physical_embryo_id(WELL, phys_index)
        image_id = build_image_id(WELL, CHANNEL, t)
        embryo_id = build_embryo_id(phys, image_id)
        snip_id = build_snip_id(embryo_id, image_id)
        mask_id = build_mask_id(image_id, phys_index)
        inv_rows.append(
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": phys,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "image_id": image_id,
                "time_index": t,
                "channel_id": CHANNEL,
                "mask_id": mask_id,
            }
        )
        mask_rows.append(
            {"mask_id": mask_id, "image_id": image_id, "mask_rle": encode_binary_mask_rle(mask)}
        )
        if image is not None:
            path = tmp_path / f"{image_id}.png"
            cv2.imwrite(str(path), image)
            fi_rows.append(
                {
                    "well_id": WELL,
                    "channel_id": CHANNEL,
                    "time_index": t,
                    "z_index": None,
                    "image_id": image_id,
                    "image_product_type": "projection",
                    "projection_method": "focus_stack",
                    "source_image_path": str(path),
                }
            )
    return pd.DataFrame(inv_rows), pd.DataFrame(mask_rows), pd.DataFrame(fi_rows)


def _run(tmp_path, specs):
    inv, masks, fi = _snip_rows(tmp_path, specs)
    return compute_focus_qc(inv, masks, fi, config=resolve_config())


# ── pure-function tests ───────────────────────────────────────────────────────────────────────


def test_textured_interior_high_fraction():
    config = resolve_config()
    fraction, n_px = compute_interior_strong_edge_fraction(
        _textured_image(), _embryo_mask(), config=config
    )
    assert n_px > 0
    assert fraction > config.interior_strong_edge_fraction_threshold


def test_flat_interior_low_fraction():
    config = resolve_config()
    fraction, n_px = compute_interior_strong_edge_fraction(
        _flat_image(), _embryo_mask(), config=config
    )
    assert n_px > 0
    assert fraction < config.interior_strong_edge_fraction_threshold


def test_shape_mismatch_fails_loud():
    config = resolve_config()
    small_image = np.zeros((10, 10), dtype="uint8")
    with pytest.raises(ValueError, match="does not match mask"):
        compute_interior_strong_edge_fraction(small_image, _embryo_mask(), config=config)


# ── per-snip batch tests ────────────────────────────────────────────────────────────────────────


def test_textured_snip_passes(tmp_path):
    out = _run(tmp_path, [(1, 0, _embryo_mask(), _textured_image())])
    row = out.iloc[0]
    assert row["focus_flag"] == False  # noqa: E712
    validate_focus_qc(out)


def test_flat_snip_trips_focus_flag(tmp_path):
    out = _run(tmp_path, [(1, 0, _embryo_mask(), _flat_image())])
    row = out.iloc[0]
    assert row["focus_flag"] == True  # noqa: E712
    validate_focus_qc(out)


def test_missing_projection_row_fails_loud(tmp_path):
    inv, masks, fi = _snip_rows(tmp_path, [(1, 0, _embryo_mask(), _textured_image())])
    fi = fi.iloc[0:0]  # drop all frame_inventory rows
    with pytest.raises(ValueError, match="could not load projection"):
        compute_focus_qc(inv, masks, fi, config=resolve_config())


def test_missing_mask_fails_loud(tmp_path):
    inv, masks, fi = _snip_rows(tmp_path, [(1, 0, _embryo_mask(), _textured_image())])
    masks = masks.iloc[0:0]  # drop all mask rows
    with pytest.raises(ValueError, match="not found in frame_masks"):
        compute_focus_qc(inv, masks, fi, config=resolve_config())


def test_output_validates_full_spine(tmp_path):
    out = _run(
        tmp_path,
        [
            (1, 0, _embryo_mask(), _textured_image()),
            (2, 1, _embryo_mask(), _flat_image()),
        ],
    )
    for col in ("experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id"):
        assert col in out.columns
    validate_focus_qc(out)
