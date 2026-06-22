"""Run-length encoding utilities for two-dimensional binary masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RLE_COUNTS_KEY = "counts"
RLE_SHAPE_KEY = "shape"
MASK_NDIM = 2


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` as bool after validating its shape, dtype, and values."""
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a numpy.ndarray.")
    if mask.ndim != MASK_NDIM:
        raise ValueError("mask must be a two-dimensional binary array.")
    if mask.dtype == np.bool_:
        return mask
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("mask dtype must be bool or an integer type containing only 0/1 values.")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask values must be binary: use only 0 and 1.")
    return mask.astype(bool, copy=False)


def encode_binary_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a two-dimensional binary mask as row-major run lengths."""
    binary_mask = validate_binary_mask(mask)
    flat = binary_mask.ravel(order="C")

    counts: list[int] = []
    current_value = False
    run_length = 0
    for value in flat:
        value_bool = bool(value)
        if value_bool == current_value:
            run_length += 1
            continue
        counts.append(run_length)
        current_value = value_bool
        run_length = 1
    counts.append(run_length)

    return {
        RLE_SHAPE_KEY: [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        RLE_COUNTS_KEY: counts,
    }


def decode_binary_mask_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode a row-major binary-mask RLE payload into a bool mask."""
    shape = _parse_rle_shape(rle)
    counts = _parse_rle_counts(rle)
    total_pixels = shape[0] * shape[1]

    flat = np.zeros(total_pixels, dtype=bool)
    cursor = 0
    value = False
    for run_length in counts:
        next_cursor = cursor + run_length
        if next_cursor > total_pixels:
            raise ValueError("RLE counts exceed the mask shape; fix counts or shape.")
        if value:
            flat[cursor:next_cursor] = True
        cursor = next_cursor
        value = not value

    if cursor != total_pixels:
        raise ValueError("RLE counts do not fill the mask shape; fix counts or shape.")

    return flat.reshape(shape, order="C")


def _parse_rle_shape(rle: Mapping[str, Any]) -> tuple[int, int]:
    if not isinstance(rle, Mapping):
        raise ValueError("rle must be a mapping with 'shape' and 'counts' keys.")
    shape = rle.get(RLE_SHAPE_KEY)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != MASK_NDIM:
        raise ValueError("RLE shape must be a two-item sequence: [height, width].")

    height, width = shape
    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError("RLE shape entries must be positive integers.")
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("RLE shape entries must be positive integers.")
    if height <= 0 or width <= 0:
        raise ValueError("RLE shape entries must be positive integers.")

    return height, width


def _parse_rle_counts(rle: Mapping[str, Any]) -> list[int]:
    counts = rle.get(RLE_COUNTS_KEY)
    if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)) or not counts:
        raise ValueError("RLE counts must be a non-empty sequence of non-negative integers.")

    parsed_counts: list[int] = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("RLE counts must be non-negative integers.")
        if count < 0:
            raise ValueError("RLE counts must be non-negative integers.")
        parsed_counts.append(count)

    return parsed_counts
