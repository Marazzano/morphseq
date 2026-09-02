"""Geometry helpers for two-dimensional binary masks."""

from __future__ import annotations

import numpy as np

from image_geometry import BoxYX

from .mask_rle import validate_binary_mask

EMPTY_BOUNDING_BOX_XYXY_PX = (0, 0, 0, 0)
EMPTY_CENTROID_XY_PX = (0.0, 0.0)
PIXEL_CENTER_OFFSET = 0.5


def mask_area_px(mask: np.ndarray) -> int:
    """Return the number of foreground pixels in a binary mask."""
    binary_mask = validate_binary_mask(mask)
    return int(np.count_nonzero(binary_mask))


def mask_bounding_box_xyxy_px(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the half-open ``(x_min, y_min, x_max, y_max)`` foreground box in pixels.

    The box arithmetic is :meth:`image_geometry.BoxYX.from_mask` — identical half-open convention,
    identical ``+1`` on the maxima. This function remains the domain-facing seam because it owns two
    things the generic primitive must not: ``validate_binary_mask`` (masks entering the metrics path
    are crisp by contract, so ``> 0`` is the correct threshold here), and the
    ``EMPTY_BOUNDING_BOX_XYXY_PX`` sentinel that metrics rows depend on in place of ``None``.
    """
    binary_mask = validate_binary_mask(mask)
    box = BoxYX.from_mask(binary_mask)
    if box is None:
        return EMPTY_BOUNDING_BOX_XYXY_PX
    return (box.x0, box.y0, box.x1, box.y1)


def mask_centroid_xy_px(mask: np.ndarray) -> tuple[float, float]:
    """Return the foreground centroid in pixel-center coordinates as ``(x, y)``."""
    binary_mask = validate_binary_mask(mask)
    y_coords, x_coords = np.where(binary_mask)
    if len(x_coords) == 0:
        return EMPTY_CENTROID_XY_PX

    return (
        float(np.mean(x_coords + PIXEL_CENTER_OFFSET)),
        float(np.mean(y_coords + PIXEL_CENTER_OFFSET)),
    )


def mask_geometry(mask: np.ndarray) -> dict[str, int | float]:
    """Return frame-mask geometry fields for a binary mask."""
    area_px = mask_area_px(mask)
    bbox_x_min_px, bbox_y_min_px, bbox_x_max_px, bbox_y_max_px = mask_bounding_box_xyxy_px(mask)
    centroid_x_px, centroid_y_px = mask_centroid_xy_px(mask)
    return {
        "area_px": area_px,
        "bbox_x_min_px": bbox_x_min_px,
        "bbox_y_min_px": bbox_y_min_px,
        "bbox_x_max_px": bbox_x_max_px,
        "bbox_y_max_px": bbox_y_max_px,
        "centroid_x_px": centroid_x_px,
        "centroid_y_px": centroid_y_px,
    }
