"""All snip-frame mask coordinate operations in ONE place.

Every mask that participates in snip-level work (the cropped embryo mask, the per-snip via/yolk/
focus/bubble masks) must live on the single grid defined by ``snip_frame_shape`` (the source of
truth in :mod:`snip_processing.snip_frame_shape`). This module owns the small set of operations
that move masks onto / off / back onto that grid, and the guards that verify they fit — so the
"is this mask on the right grid, and does it match that other mask" logic is not scattered across
fraction_alive, the UNet predictor, and snip_processing.

The actual pixel resampling is delegated to the shared resamplers in
:mod:`data_pipeline.object_extraction.segmentation.masks.mask_resize` (``resize_binary_mask_to_shape`` nearest-
neighbour for masks; ``resize_image_to_shape`` interpolated for continuous images). This module
adds the
*snip-frame-aware* layer on top — every call site reads as the narrative step it is, never as a
raw resize with interpolation flags:

    to_snip_frame(mask, snip_frame_shape)        — bring ANY mask onto the snip grid
    from_snip_frame(mask, target_shape)          — take a snip-grid mask to another grid (e.g. model)
    back_to_snip_frame(mask, snip_frame_shape)    — round-trip a model-grid mask back to the snip grid
    snip_image_to_model_grid(image, model_shape)  — put a snip IMAGE onto a model's input grid (continuous)
    assert_on_snip_frame(mask, snip_frame_shape)  — guard: mask is ALREADY on the snip grid (no resize)
    assert_same_shape(a, b, ...)                  — guard: two masks share a shape (safe to AND/OR)
    align_pair_to_snip_frame(a, b, snip_frame_shape) — bring two masks onto the snip grid + verify

The "model grid" is an internal, transient detail of one inference call — a model's required input
size. It is encapsulated here (``snip_image_to_model_grid`` going in, ``back_to_snip_frame`` coming
out) so the model's preferred dimension never leaks into any on-disk artifact: on disk, only the
snip frame exists.

DOCTRINE — the model may visit another grid; artifacts must come home; every trip changes shape
through a guarded gate. No resize inline, no silent dimension changes: binary masks resize
nearest-neighbour (never invent fractional pixels), continuous images resize interpolated (preserve
intensities), and both validate the target is a two-int ``(height, width)`` and that the output
shape exactly equals it.
"""

from __future__ import annotations

import numpy as np

from data_pipeline.object_extraction.segmentation.masks.mask_resize import (
    resize_binary_mask_to_shape,
    resize_image_to_shape,
)

__all__ = [
    "to_snip_frame",
    "from_snip_frame",
    "back_to_snip_frame",
    "snip_image_to_model_grid",
    "assert_on_snip_frame",
    "assert_same_shape",
    "align_pair_to_snip_frame",
]


def _as_hw(shape) -> tuple[int, int]:
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape must be a positive (height, width); got {shape!r}.")
    return h, w


def to_snip_frame(mask: np.ndarray, snip_frame_shape) -> np.ndarray:
    """Bring ``mask`` (at any resolution) ONTO the snip grid as a boolean array.

    A mask already on the snip grid is returned unchanged (as bool); otherwise it is
    nearest-neighbour resampled to ``snip_frame_shape``. Use this for the "to" direction — e.g.
    cropping a full-frame embryo mask down onto the snip grid.
    """
    return resize_binary_mask_to_shape(mask, _as_hw(snip_frame_shape))


def from_snip_frame(mask: np.ndarray, target_shape) -> np.ndarray:
    """Take a snip-grid ``mask`` to another grid ``target_shape`` (the "from" direction).

    Typical use: resize a snip-frame mask to a model's input dimensions before inference.
    """
    return resize_binary_mask_to_shape(mask, _as_hw(target_shape))


def snip_image_to_model_grid(image: np.ndarray, model_shape) -> np.ndarray:
    """Put a snip IMAGE (continuous intensities) onto a model's input grid for inference.

    The image counterpart of :func:`from_snip_frame`: same "leave the snip frame for the model"
    step, but for a grayscale intensity image, so it uses interpolated (not nearest) resampling via
    the shared ``resize_image``. The returned image's mask is then predicted and brought back with
    :func:`back_to_snip_frame`, so the model grid never escapes the predictor.
    """
    return resize_image_to_shape(image, _as_hw(model_shape))


def back_to_snip_frame(mask: np.ndarray, snip_frame_shape) -> np.ndarray:
    """Round-trip a mask that left the snip grid (e.g. a model output) BACK onto the snip grid.

    Semantically identical to :func:`to_snip_frame`; named separately so call sites read as the
    explicit "and now back to the snip frame" step of a to → infer → back round-trip.
    """
    return resize_binary_mask_to_shape(mask, _as_hw(snip_frame_shape))


def assert_on_snip_frame(mask: np.ndarray, snip_frame_shape, *, label: str = "mask") -> np.ndarray:
    """Guard: fail loud unless ``mask`` is ALREADY on the snip grid (no resampling performed).

    Use before an operation that *requires* a mask to be snip-native (so a silent resize never
    hides an upstream grid bug). Returns the mask unchanged on success.
    """
    h, w = _as_hw(snip_frame_shape)
    if tuple(mask.shape[:2]) != (h, w):
        raise ValueError(
            f"{label} is not on the snip frame: shape {tuple(mask.shape[:2])} != "
            f"snip_frame_shape {(h, w)}. Resize it with to_snip_frame() upstream, or fix the "
            "producer so it emits snip-frame masks."
        )
    return mask


def assert_same_shape(a: np.ndarray, b: np.ndarray, *, a_label: str = "a", b_label: str = "b") -> None:
    """Guard: fail loud unless ``a`` and ``b`` share a 2-D shape (precondition for AND/OR)."""
    if tuple(a.shape[:2]) != tuple(b.shape[:2]):
        raise ValueError(
            f"mask shape mismatch: {a_label} {tuple(a.shape[:2])} != {b_label} {tuple(b.shape[:2])}. "
            "Bring both onto the snip frame with align_pair_to_snip_frame() before combining them."
        )


def align_pair_to_snip_frame(
    a: np.ndarray,
    b: np.ndarray,
    snip_frame_shape,
    *,
    a_label: str = "a",
    b_label: str = "b",
) -> tuple[np.ndarray, np.ndarray]:
    """Bring two masks onto the snip grid and verify they match — the safe pre-AND step.

    Resizes each to ``snip_frame_shape`` (no-op when already there) and asserts the result shapes
    agree, so a caller can ``np.logical_and`` the returned pair without re-checking.
    """
    aa = to_snip_frame(a, snip_frame_shape)
    bb = to_snip_frame(b, snip_frame_shape)
    assert_same_shape(aa, bb, a_label=a_label, b_label=b_label)
    return aa, bb
