"""Bounds-expanding rotation geometry.

Pure arithmetic: given a canvas shape and an angle, what rotation matrix maps that canvas onto the
smallest axis-aligned canvas that still contains every source pixel? This is textbook geometry with
no knowledge of embryos, masks, or snips, which is why it lives here rather than in any one consumer.

WHAT THIS MODULE DOES *NOT* DO: it never touches pixels. It returns a matrix and a shape; the caller
chooses the resampling kernel. That separation is deliberate — the callers of this arithmetic
disagree about interpolation (one deliberately reproduces a legacy bilinear-on-binary-mask defect),
and folding a `warpAffine` in here would erase a distinction those callers depend on.

THE ``int()`` TRUNCATION IS PART OF THE CONTRACT. The bound expressions truncate toward zero rather
than rounding or taking a ceiling. That loses up to a pixel of canvas versus the exact bound, so a
rotated corner can fall marginally outside the output. Every historical copy of this block did the
same thing, and snip placement is pinned to the resulting dimensions by a byte-for-byte pixel-hash
test, so the truncation is reproduced exactly rather than corrected. Changing it to ``ceil`` would
be geometrically defensible and would move every snip.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["bounds_expanded_rotation_matrix", "expanded_rotation_bounds_wh"]


def expanded_rotation_bounds_wh(
    *, shape_hw: tuple[int, int], abs_cos: float, abs_sin: float
) -> tuple[int, int]:
    """Return the ``(width, height)`` canvas that contains ``shape_hw`` rotated by the given angle.

    ``abs_cos`` / ``abs_sin`` are the absolute values of the rotation matrix's top-row entries, i.e.
    ``|cos(angle)|`` and ``|sin(angle)|``. They are taken as arguments rather than recomputed from an
    angle so that a caller holding a matrix gets bounds consistent with THAT matrix, with no risk of
    a degrees/radians or sign mismatch reintroducing a half-pixel disagreement.

    Truncating via ``int()`` is intentional — see the module docstring.
    """
    height, width = shape_hw
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    return bound_w, bound_h


def bounds_expanded_rotation_matrix(
    *, shape_hw: tuple[int, int], angle_deg: float
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return ``(rotation_2x3, (bound_w, bound_h))`` rotating ``shape_hw`` without cropping.

    The matrix rotates about the source canvas center and is then translated so the rotated content
    is centered on the expanded output canvas. Apply it with
    ``cv2.warpAffine(img, mat, (bound_w, bound_h))`` — the returned bounds are already in OpenCV's
    ``dsize`` (width, height) order.

    The caller picks the interpolation flag. This function is coordinate truth only; it deliberately
    does not resample.
    """
    height, width = shape_hw
    # getRotationMatrix2D takes (x, y) = (width, height) order, the reverse of numpy's shape.
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w, bound_h = expanded_rotation_bounds_wh(
        shape_hw=(height, width), abs_cos=abs_cos, abs_sin=abs_sin
    )

    # Re-center: undo the source center, then add the expanded canvas center.
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return rotation_mat, (bound_w, bound_h)
