"""Binary mask utilities for segmentation contracts."""

from .mask_geometry import (
    mask_area_px,
    mask_bounding_box_xyxy_px,
    mask_centroid_xy_px,
    mask_geometry,
)
from .mask_rle import decode_binary_mask_rle, encode_binary_mask_rle, validate_binary_mask

__all__ = [
    "decode_binary_mask_rle",
    "encode_binary_mask_rle",
    "mask_area_px",
    "mask_bounding_box_xyxy_px",
    "mask_centroid_xy_px",
    "mask_geometry",
    "validate_binary_mask",
]
