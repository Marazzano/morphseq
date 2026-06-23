"""The one controlled seam for pixel-grid resizing — masks AND images.

THE LAW: no resize inline, no silent dimension changes. Every operation that changes a 2-D pixel
grid's ``(height, width)`` goes through a named helper here, never a raw ``cv2.resize`` /
``skimage.resize`` sprinkled in a consumer. Centralising it keeps the shape validation in one
tested place and keeps the two interpolation regimes from ever being mixed up:

* :func:`resize_binary_mask_to_shape` — for BINARY masks. Nearest-neighbour, so a label never
  becomes a fractional 0.5; preserves bool / 0-1 semantics. Returns bool.
* :func:`resize_image_to_shape` — for CONTINUOUS images (e.g. a grayscale snip fed to a model).
  Interpolated (area when shrinking, linear when growing), which is correct for intensities and
  WRONG for labels; preserves the input dtype/range.

Every helper: validates the target is a two-int ``(height, width)`` tuple, validates the input is
2-D (it never silently squeezes/drops channels), and validates the OUTPUT shape exactly equals the
target before returning — so a wrong-order or partial resize fails loud rather than slipping
through. Names use ``shape`` (= ``(height, width)``), never ``dims``, because "which dimension order
are we even talking about" is the whole bug class this module exists to kill.

``align_binary_masks`` resizes a group of masks to one common shape so callers can
``np.logical_and`` / ``np.logical_or`` them safely.
"""

from __future__ import annotations

import cv2
import numpy as np

from .mask_rle import validate_binary_mask

__all__ = [
    "resize_binary_mask_to_shape",
    "resize_image_to_shape",
    "align_binary_masks",
]


def _validate_target_shape(target_shape: tuple[int, int]) -> tuple[int, int]:
    """Return ``target_shape`` as a validated two-int ``(height, width)`` tuple."""
    if len(target_shape) != 2:
        raise ValueError(f"target_shape must be a two-item (height, width); got {target_shape!r}.")
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_shape must be positive (height, width); got {target_shape!r}.")
    return target_h, target_w


def _require_2d(array: np.ndarray, *, what: str) -> None:
    if array.ndim != 2:
        raise ValueError(
            f"{what} must be a 2-D (height, width) array; got shape {array.shape!r}. This seam "
            "never silently squeezes or drops channels — pass a single 2-D plane."
        )


def _assert_output_shape(out: np.ndarray, target_hw: tuple[int, int], *, what: str) -> None:
    if tuple(out.shape) != tuple(target_hw):
        raise AssertionError(
            f"{what} resize post-condition failed: output shape {out.shape!r} != target "
            f"{tuple(target_hw)!r}. (Likely an axis-order bug between (H, W) and OpenCV's (W, H).)"
        )


def resize_binary_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Return ``mask`` resampled to ``target_shape`` (height, width) as a boolean array.

    Uses ``cv2.INTER_NEAREST`` so a binary label stays binary (never interpolated to 0.5). A mask
    already at ``target_shape`` is returned as bool without resampling. ``target_shape`` is (H, W)
    numpy order; OpenCV's ``dsize`` is (W, H), so the axes are swapped on the way in and the output
    shape is asserted before returning.
    """
    binary = validate_binary_mask(mask)
    _require_2d(binary, what="mask")
    target_h, target_w = _validate_target_shape(target_shape)
    if binary.shape == (target_h, target_w):
        return binary
    resized = cv2.resize(
        binary.astype(np.uint8),
        (target_w, target_h),  # cv2 dsize is (width, height)
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)
    _assert_output_shape(resized, (target_h, target_w), what="binary mask")
    return resized


def resize_image_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Return a CONTINUOUS image resampled to ``target_shape`` (height, width).

    For intensity images (NOT binary masks — use :func:`resize_binary_mask_to_shape` for those).
    Chooses ``cv2.INTER_AREA`` when shrinking (the correct anti-aliased downsampler) and
    ``INTER_LINEAR`` when growing. The dtype of ``image`` is preserved. An image already at
    ``target_shape`` is returned unchanged. ``target_shape`` is (H, W); OpenCV's ``dsize`` is
    (W, H), so the axes are swapped on the way in and the output shape is asserted before returning.
    """
    _require_2d(image, what="image")
    target_h, target_w = _validate_target_shape(target_shape)
    if image.shape == (target_h, target_w):
        return image
    shrinking = target_h * target_w < image.shape[0] * image.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interp).astype(
        image.dtype, copy=False
    )
    _assert_output_shape(resized, (target_h, target_w), what="image")
    return resized


def align_binary_masks(
    *masks: np.ndarray,
    target_shape: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Resize every mask to one common shape so they can be combined elementwise.

    With ``target_shape`` given, every mask is resized to it. Without it, the smallest mask's shape
    is used as the target (upsampling a coarse mask to a fine grid invents detail it never had;
    resizing the precise mask down to the coarse grid is the honest choice).
    """
    if not masks:
        raise ValueError("align_binary_masks requires at least one mask.")
    binaries = [validate_binary_mask(m) for m in masks]
    if target_shape is None:
        target_shape = min((b.shape for b in binaries), key=lambda s: s[0] * s[1])
    return [resize_binary_mask_to_shape(b, target_shape) for b in binaries]
