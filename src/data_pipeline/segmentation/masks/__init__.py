"""Binary mask utilities for segmentation contracts."""

from .mask_geometry import (
    mask_area_px,
    mask_bounding_box_xyxy_px,
    mask_centroid_xy_px,
    mask_geometry,
)
from .mask_resize import (
    align_binary_masks,
    resize_binary_mask_to_shape,
    resize_image_to_shape,
)
from .mask_rle import decode_binary_mask_rle, encode_binary_mask_rle, validate_binary_mask

__all__ = [
    "align_binary_masks",
    "decode_binary_mask_rle",
    "encode_binary_mask_rle",
    "mask_area_px",
    "mask_bounding_box_xyxy_px",
    "mask_centroid_xy_px",
    "mask_geometry",
    "resize_binary_mask_to_shape",
    "resize_image_to_shape",
    "validate_binary_mask",
]
