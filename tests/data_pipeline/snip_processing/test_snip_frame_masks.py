import re
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.snip_processing.snip_frame_masks import (
    align_pair_to_snip_frame,
    assert_on_snip_frame,
    assert_same_shape,
    back_to_snip_frame,
    from_snip_frame,
    snip_image_to_model_grid,
    to_snip_frame,
)

SNIP_FRAME = (576, 256)


def test_to_snip_frame_brings_any_resolution_onto_grid() -> None:
    big = np.zeros((2189, 2189), dtype=np.uint8)
    big[500:1500, 500:1500] = 1

    out = to_snip_frame(big, SNIP_FRAME)

    assert out.shape == SNIP_FRAME
    assert out.dtype == bool


def test_to_snip_frame_noop_when_already_on_grid() -> None:
    mask = np.zeros(SNIP_FRAME, dtype=np.uint8)
    mask[10:20, 10:20] = 1

    out = to_snip_frame(mask, SNIP_FRAME)

    assert out.shape == SNIP_FRAME
    assert np.array_equal(out, mask.astype(bool))


def test_from_and_back_round_trip_shapes() -> None:
    on_frame = np.zeros(SNIP_FRAME, dtype=np.uint8)
    on_frame[100:200, 50:150] = 1

    model_grid = from_snip_frame(on_frame, (288, 128))
    assert model_grid.shape == (288, 128)

    back = back_to_snip_frame(model_grid, SNIP_FRAME)
    assert back.shape == SNIP_FRAME
    assert back.dtype == bool


def test_snip_image_to_model_grid_is_continuous_not_binarised() -> None:
    # The IMAGE seam (continuous interp) must NOT binarise a grayscale gradient like a mask would.
    img = np.tile(np.linspace(0, 255, SNIP_FRAME[1], dtype=np.uint8), (SNIP_FRAME[0], 1))

    model_img = snip_image_to_model_grid(img, (288, 128))

    assert model_img.shape == (288, 128)
    assert model_img.dtype == np.uint8
    assert len(np.unique(model_img)) > 2  # genuinely continuous


def test_predictor_has_no_inline_st_resize() -> None:
    # Discipline guard: the UNet snip predictor must route resizing through the shared seam, never
    # call skimage/cv2 resize inline. Fails loud if someone re-introduces an ad hoc resize.
    src = Path(__file__).resolve().parents[3] / "src"
    predictor = (
        src / "data_pipeline" / "segmentation" / "backends" / "unet_snip" / "model_loader.py"
    ).read_text()
    assert "st.resize" not in predictor, "inline st.resize re-introduced in the UNet predictor"
    assert not re.search(r"\bcv2\.resize\b", predictor), "inline cv2.resize in the UNet predictor"


def test_assert_on_snip_frame_passes_when_on_grid() -> None:
    mask = np.zeros(SNIP_FRAME, dtype=bool)
    # Returns the mask unchanged (no exception).
    assert assert_on_snip_frame(mask, SNIP_FRAME) is mask


def test_assert_on_snip_frame_fails_loud_when_off_grid() -> None:
    off = np.zeros((100, 100), dtype=bool)
    with pytest.raises(ValueError, match="not on the snip frame"):
        assert_on_snip_frame(off, SNIP_FRAME, label="embryo_mask")


def test_assert_same_shape_passes_and_fails() -> None:
    assert_same_shape(np.zeros((5, 5)), np.ones((5, 5)))  # no raise
    with pytest.raises(ValueError, match="shape mismatch"):
        assert_same_shape(np.zeros((5, 5)), np.ones((6, 6)), a_label="embryo", b_label="via")


def test_align_pair_brings_both_onto_grid_and_matches() -> None:
    embryo = np.zeros((2189, 2189), dtype=np.uint8)
    embryo[800:1400, 800:1200] = 1
    via = np.zeros(SNIP_FRAME, dtype=np.uint8)
    via[100:200, 50:150] = 1

    a, b = align_pair_to_snip_frame(embryo, via, SNIP_FRAME, a_label="embryo", b_label="via")

    assert a.shape == b.shape == SNIP_FRAME
    # Safe to AND now — exercises the post-condition the helper guarantees.
    overlap = np.logical_and(a, b)
    assert overlap.shape == SNIP_FRAME
