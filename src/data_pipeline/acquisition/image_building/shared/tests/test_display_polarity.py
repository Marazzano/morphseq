"""Unit tests for the shared display-polarity module.

Run with:
    PYTHONPATH=src pytest \
      src/data_pipeline/acquisition/image_building/shared/tests/test_display_polarity.py
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.acquisition.image_building.shared.display_polarity import (
    INVERT_FOR_DISPLAY,
    apply_display_polarity,
)


def test_canonical_polarity_is_inverted():
    # Downstream snip_processing assumes a DARK background, so the canonical materialized polarity
    # inverts raw bright-field. Guard the constant so a silent flip is caught.
    assert INVERT_FOR_DISPLAY is True


def test_uint8_inversion_matches_255_minus():
    img = np.array([[0, 1, 128, 254, 255]], dtype=np.uint8)
    out = apply_display_polarity(img)
    np.testing.assert_array_equal(out, np.array([[255, 254, 127, 1, 0]], dtype=np.uint8))
    assert out.dtype == np.uint8


def test_uint16_inversion_matches_dtype_max_minus():
    # z-stack mosaics are uint16; inversion must use 65535, matching legacy iinfo(dtype).max - out.
    img = np.array([[0, 1000, 65535]], dtype=np.uint16)
    out = apply_display_polarity(img)
    np.testing.assert_array_equal(out, np.array([[65535, 64535, 0]], dtype=np.uint16))
    assert out.dtype == np.uint16


def test_invert_false_returns_unchanged():
    img = np.array([[10, 20, 30]], dtype=np.uint8)
    out = apply_display_polarity(img, invert=False)
    np.testing.assert_array_equal(out, img)


def test_double_inversion_is_identity():
    img = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)
    np.testing.assert_array_equal(apply_display_polarity(apply_display_polarity(img)), img)


def test_tile_then_stitch_equivalent_to_stitch_then_invert():
    # Polarity is a pure per-pixel op, so inverting each tile then concatenating equals
    # concatenating then inverting — the property that lets us own it in one shared place
    # regardless of where composition happens.
    a = np.random.randint(0, 256, size=(4, 4), dtype=np.uint8)
    b = np.random.randint(0, 256, size=(4, 4), dtype=np.uint8)
    per_tile = np.concatenate([apply_display_polarity(a), apply_display_polarity(b)], axis=1)
    post = apply_display_polarity(np.concatenate([a, b], axis=1))
    np.testing.assert_array_equal(per_tile, post)


def test_rejects_non_integer_dtype():
    with pytest.raises(TypeError):
        apply_display_polarity(np.zeros((2, 2), dtype=np.float32))
