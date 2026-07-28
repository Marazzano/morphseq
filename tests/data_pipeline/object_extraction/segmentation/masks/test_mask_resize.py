import numpy as np
import pytest

from data_pipeline.object_extraction.segmentation.masks.mask_resize import (
    align_binary_masks,
    resize_binary_mask_to_shape,
    resize_image_to_shape,
)


# ── binary mask resize ────────────────────────────────────────────────────────────────────────

def test_binary_resize_returns_exact_target_shape_and_bool() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1

    out = resize_binary_mask_to_shape(mask, (50, 50))

    assert out.shape == (50, 50)
    assert out.dtype == bool
    assert set(np.unique(out)) <= {False, True}


def test_binary_resize_stays_binary_not_blurred() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, 5] = 1

    out = resize_binary_mask_to_shape(mask, (20, 20))

    assert out.dtype == bool
    assert set(np.unique(out)) <= {False, True}


def test_binary_resize_noop_when_already_target_shape() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    out = resize_binary_mask_to_shape(mask, (8, 8))

    assert np.array_equal(out, mask.astype(bool))


def test_binary_resize_height_width_order_not_swapped() -> None:
    out = resize_binary_mask_to_shape(np.ones((4, 4), dtype=np.uint8), (10, 3))
    assert out.shape == (10, 3)


# ── continuous image resize ───────────────────────────────────────────────────────────────────

def test_image_resize_returns_exact_target_shape() -> None:
    img = (np.arange(100 * 80, dtype=np.uint8).reshape(100, 80)) % 255

    out = resize_image_to_shape(img, (50, 40))

    assert out.shape == (50, 40)


def test_image_resize_preserves_dtype() -> None:
    img = np.full((20, 20), 128, dtype=np.uint8)

    out = resize_image_to_shape(img, (10, 10))

    assert out.dtype == np.uint8


def test_image_resize_preserves_intensity_range_not_binarised() -> None:
    # A smooth gradient must keep a spread of intensities (NOT collapse to {0, 255} like a mask).
    img = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))

    out = resize_image_to_shape(img, (32, 32))

    assert out.min() < 64 and out.max() > 192
    assert len(np.unique(out)) > 2  # genuinely continuous, not binarised


def test_image_resize_noop_when_already_target_shape() -> None:
    img = np.full((12, 12), 7, dtype=np.uint8)
    out = resize_image_to_shape(img, (12, 12))
    assert out is img


def test_image_resize_height_width_order_not_swapped() -> None:
    out = resize_image_to_shape(np.zeros((4, 4), dtype=np.uint8), (9, 3))
    assert out.shape == (9, 3)


# ── shared validation: 2-D only, two-int shape, no silent channel squeeze ─────────────────────

@pytest.mark.parametrize("fn", [resize_binary_mask_to_shape, resize_image_to_shape])
def test_rejects_non_2d_input_never_squeezes_channels(fn) -> None:
    # Both helpers must reject a 3-D (channelled) array rather than silently dropping a channel.
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D|two-dimensional"):
        fn(rgb, (5, 5))


@pytest.mark.parametrize("fn", [resize_binary_mask_to_shape, resize_image_to_shape])
def test_rejects_nonpositive_target(fn) -> None:
    with pytest.raises(ValueError):
        fn(np.ones((4, 4), dtype=np.uint8), (0, 5))


@pytest.mark.parametrize("fn", [resize_binary_mask_to_shape, resize_image_to_shape])
def test_rejects_wrong_length_target(fn) -> None:
    with pytest.raises(ValueError, match="two-item"):
        fn(np.ones((4, 4), dtype=np.uint8), (5, 5, 5))


# ── align ─────────────────────────────────────────────────────────────────────────────────────

def test_align_uses_smallest_shape_by_default() -> None:
    a, b = align_binary_masks(np.ones((100, 80), np.uint8), np.ones((50, 40), np.uint8))
    assert a.shape == b.shape == (50, 40)


def test_align_to_explicit_target_shape() -> None:
    a, b = align_binary_masks(
        np.ones((100, 80), np.uint8), np.ones((50, 40), np.uint8), target_shape=(20, 20)
    )
    assert a.shape == b.shape == (20, 20)


def test_align_requires_at_least_one_mask() -> None:
    with pytest.raises(ValueError):
        align_binary_masks()
