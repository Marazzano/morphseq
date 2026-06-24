"""mask_quality_qc compute tests.

Covers the spec "Done When": an edge-touching mask, a two-component mask, an overlapping pair of
DISTINCT physical embryos in one image (both flagged), a same-physical-embryo overlap (NOT
flagged), and a clean mask that trips none. Input is canonical frame_masks RLE via the shared
decoder — no SAM2 mask_rle path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.quality_control.mask_quality_qc.compute import (
    compute_discontinuous_flag,
    compute_edge_flag,
    compute_mask_quality_qc_flags,
)
from data_pipeline.quality_control.mask_quality_qc.config import resolve_config
from data_pipeline.quality_control.mask_quality_qc.contract import validate_mask_quality_qc
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
H = W = 20


def _clean_mask():
    m = np.zeros((H, W), dtype=bool)
    m[8:12, 8:12] = True  # centered block, away from edges
    return m


def _edge_mask():
    m = np.zeros((H, W), dtype=bool)
    m[0:3, 8:12] = True  # touches top edge
    return m


def _two_component_mask():
    m = np.zeros((H, W), dtype=bool)
    m[8:12, 2:6] = True   # blob 1
    m[8:12, 14:18] = True  # blob 2 (equal size -> both significant)
    return m


def _overlap_mask_a():
    m = np.zeros((H, W), dtype=bool)
    m[6:14, 6:14] = True
    return m


def _overlap_mask_b():
    m = np.zeros((H, W), dtype=bool)
    m[8:16, 8:16] = True  # heavily overlaps A
    return m


# ── pure-function tests ───────────────────────────────────────────────────────────────────────

def test_edge_flag_pure():
    assert compute_edge_flag(_edge_mask(), margin_pixels=2) is True
    assert compute_edge_flag(_clean_mask(), margin_pixels=2) is False


def test_discontinuous_flag_pure():
    assert compute_discontinuous_flag(_two_component_mask(), min_component_fraction=0.05) is True
    assert compute_discontinuous_flag(_clean_mask(), min_component_fraction=0.05) is False


# ── per-snip batch tests ────────────────────────────────────────────────────────────────────────

def _snip_rows(specs):
    """specs: list of (phys_index, time_index, mask). Returns (snip_inventory_df, frame_masks_df)."""
    inv_rows, mask_rows = [], []
    for phys_index, t, mask in specs:
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
                "mask_id": mask_id,
            }
        )
        mask_rows.append({"mask_id": mask_id, "image_id": image_id, "mask_rle": encode_binary_mask_rle(mask)})
    return pd.DataFrame(inv_rows), pd.DataFrame(mask_rows)


def _run(specs):
    inv, masks = _snip_rows(specs)
    return compute_mask_quality_qc_flags(inv, masks, config=resolve_config())


def test_clean_mask_trips_nothing():
    out = _run([(1, 0, _clean_mask())])
    row = out.iloc[0]
    assert not row["edge_flag"] and not row["discontinuous_mask_flag"] and not row["overlapping_mask_flag"]
    validate_mask_quality_qc(out)


def test_edge_and_discontinuous_per_snip():
    out = _run([(1, 0, _edge_mask()), (2, 1, _two_component_mask())])
    by_phys = {r["physical_embryo_id"]: r for _, r in out.iterrows()}
    assert by_phys[build_physical_embryo_id(WELL, 1)]["edge_flag"]
    assert by_phys[build_physical_embryo_id(WELL, 2)]["discontinuous_mask_flag"]
    # neither trips overlap (different images)
    assert not out["overlapping_mask_flag"].any()


def test_distinct_embryo_overlap_flags_both():
    # Two distinct physical embryos, SAME image (same time_index) -> both flagged.
    out = _run([(1, 0, _overlap_mask_a()), (2, 0, _overlap_mask_b())])
    assert out["overlapping_mask_flag"].tolist() == [True, True]


def test_same_embryo_overlap_not_flagged():
    # Same physical embryo appearing twice in one image (degenerate) overlapping -> NOT ID confusion.
    phys = 1
    image_id = build_image_id(WELL, CHANNEL, 0)
    embryo_id = build_embryo_id(build_physical_embryo_id(WELL, phys), image_id)
    inv_rows, mask_rows = [], []
    for k, mask in enumerate((_overlap_mask_a(), _overlap_mask_b())):
        # two distinct snips/masks but SAME physical embryo
        snip_id = build_snip_id(embryo_id, image_id) + f"_dup{k}"
        mask_id = build_mask_id(image_id, phys) + f"_{k}"
        inv_rows.append(
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": build_physical_embryo_id(WELL, phys),
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "image_id": image_id,
                "mask_id": mask_id,
            }
        )
        mask_rows.append({"mask_id": mask_id, "image_id": image_id, "mask_rle": encode_binary_mask_rle(mask)})
    out = compute_mask_quality_qc_flags(pd.DataFrame(inv_rows), pd.DataFrame(mask_rows), config=resolve_config())
    assert out["overlapping_mask_flag"].tolist() == [False, False]


def test_missing_mask_fails_loud():
    inv, masks = _snip_rows([(1, 0, _clean_mask())])
    masks = masks.iloc[0:0]  # drop all mask rows
    with pytest.raises(ValueError, match="not found in frame_masks"):
        compute_mask_quality_qc_flags(inv, masks, config=resolve_config())


def test_output_validates_full_spine():
    out = _run([(1, 0, _clean_mask()), (2, 0, _edge_mask())])
    for col in ("experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id"):
        assert col in out.columns
    validate_mask_quality_qc(out)
