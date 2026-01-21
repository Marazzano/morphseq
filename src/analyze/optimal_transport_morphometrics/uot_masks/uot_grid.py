"""
Canonical grid standardization for UOT mask transport.

Provides transformation to a fixed reference grid (576×256 @ 7.8 μm/px) matching
the snip export pipeline (src/build/build03A_process_images.py).

This enables:
- Resolution-independent comparisons across embryos
- Interpretable physical units (micrometers)
- Consistent orientation using yolk-based alignment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from scipy import ndimage
from skimage.measure import regionprops

from src.data_pipeline.snip_processing.rotation import (
    get_embryo_rotation_angle,
    rotate_image,
)


@dataclass
class CanonicalGridConfig:
    """
    Configuration for canonical grid standardization.

    IMPORTANT: These defaults match the snip export pipeline in
    src/build/build03A_process_images.py (lines 1436-1467).
    Any changes here should be coordinated with snip export parameters.
    """
    reference_um_per_pixel: float = 7.8       # Match snip export default
    grid_shape_hw: tuple[int, int] = (256, 576)  # Match snip output shape (H, W)
    padding_um: float = 50.0                  # Padding in micrometers
    align_mode: str = "yolk"                  # "yolk" | "centroid" | "none"
    downsample_factor: int = 1                # 1 = no downsampling (recommended for 576×256)


@dataclass
class GridTransform:
    """Records transformation from source to canonical grid."""
    source_um_per_pixel: float      # From CSV: Height(um) / Height(px)
    target_um_per_pixel: float      # = reference_um_per_pixel (7.8)
    scale_factor: float             # = source / target
    rotation_rad: float             # Yolk-based rotation angle
    offset_yx_px: tuple[int, int]   # Translation offset in target pixels
    grid_shape_hw: tuple[int, int]  # (256, 576)
    downsample_factor: int          # Additional downsampling after grid (1 = none)
    effective_um_per_pixel: float   # = target_um_per_pixel * downsample_factor


def compute_grid_transform(
    mask: np.ndarray,
    source_um_per_pixel: float,
    yolk_mask: Optional[np.ndarray],
    config: CanonicalGridConfig,
) -> GridTransform:
    """
    Compute transform to place mask on canonical grid.

    Steps:
    1. Compute rotation angle (yolk-based or centroid-based)
    2. Compute scale factor to reach target resolution
    3. Compute crop offsets to center on mask centroid

    Args:
        mask: Binary mask in source resolution
        source_um_per_pixel: Physical resolution of source mask
        yolk_mask: Binary yolk mask (None if unavailable)
        config: Canonical grid configuration

    Returns:
        GridTransform capturing the full transformation
    """
    # 1. Compute rotation angle
    rotation_rad = 0.0
    if config.align_mode == "yolk":
        yolk_for_rotation = yolk_mask if yolk_mask is not None else np.zeros_like(mask)
        rotation_rad = get_embryo_rotation_angle(
            mask.astype(np.uint8),
            yolk_for_rotation.astype(np.uint8)
        )
    elif config.align_mode == "centroid":
        # Use PCA orientation without yolk guidance
        rp = regionprops(mask.astype(int))
        if rp:
            rotation_rad = -rp[0].orientation
    # else: align_mode == "none", rotation_rad = 0.0

    # 2. Compute scale factor
    scale_factor = source_um_per_pixel / config.reference_um_per_pixel

    # 3. Apply rotation and scaling to compute centroid in target space
    # (We'll compute offset when actually applying the transform)
    # For now, record the parameters
    effective_um_per_pixel = config.reference_um_per_pixel * config.downsample_factor

    return GridTransform(
        source_um_per_pixel=source_um_per_pixel,
        target_um_per_pixel=config.reference_um_per_pixel,
        scale_factor=scale_factor,
        rotation_rad=rotation_rad,
        offset_yx_px=(0, 0),  # Will be computed during apply
        grid_shape_hw=config.grid_shape_hw,
        downsample_factor=config.downsample_factor,
        effective_um_per_pixel=effective_um_per_pixel,
    )


def apply_grid_transform(
    mask: np.ndarray,
    transform: GridTransform,
) -> np.ndarray:
    """
    Apply transform to resample mask onto canonical grid.

    Steps:
    1. Rotate by transform.rotation_rad
    2. Rescale by transform.scale_factor
    3. Crop/pad to transform.grid_shape_hw centered on centroid

    Args:
        mask: Binary mask in source resolution
        transform: Grid transform computed by compute_grid_transform

    Returns:
        Resampled mask on canonical grid (shape = transform.grid_shape_hw)
    """
    # 1. Rotate
    if abs(transform.rotation_rad) > 1e-6:
        angle_deg = np.rad2deg(transform.rotation_rad)
        mask_rotated = rotate_image(mask.astype(float), angle_deg)
        mask_rotated = (mask_rotated > 0.5).astype(np.uint8)
    else:
        mask_rotated = mask.astype(np.uint8)

    # 2. Rescale
    if abs(transform.scale_factor - 1.0) > 1e-6:
        new_shape = (
            int(mask_rotated.shape[0] / transform.scale_factor),
            int(mask_rotated.shape[1] / transform.scale_factor),
        )
        mask_scaled = cv2.resize(
            mask_rotated.astype(float),
            (new_shape[1], new_shape[0]),  # cv2 uses (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        mask_scaled = (mask_scaled > 0.5).astype(np.uint8)
    else:
        mask_scaled = mask_rotated

    # 3. Crop/pad to canonical grid centered on centroid
    target_h, target_w = transform.grid_shape_hw

    # Compute centroid in scaled space
    if mask_scaled.sum() > 0:
        cy, cx = ndimage.center_of_mass(mask_scaled)
        cy, cx = int(cy), int(cx)
    else:
        # Empty mask - use center
        cy, cx = mask_scaled.shape[0] // 2, mask_scaled.shape[1] // 2

    # Compute crop window centered on centroid
    y0 = cy - target_h // 2
    x0 = cx - target_w // 2
    y1 = y0 + target_h
    x1 = x0 + target_w

    # Create output canvas
    canonical = np.zeros((target_h, target_w), dtype=np.uint8)

    # Compute overlap region
    src_y0 = max(0, y0)
    src_y1 = min(mask_scaled.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(mask_scaled.shape[1], x1)

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    # Copy data
    if src_y1 > src_y0 and src_x1 > src_x0:
        canonical[dst_y0:dst_y1, dst_x0:dst_x1] = mask_scaled[src_y0:src_y1, src_x0:src_x1]

    # Apply final downsampling if needed
    if transform.downsample_factor > 1:
        d = transform.downsample_factor
        downsampled_h = target_h // d
        downsampled_w = target_w // d
        canonical = cv2.resize(
            canonical.astype(float),
            (downsampled_w, downsampled_h),
            interpolation=cv2.INTER_LINEAR
        )
        canonical = (canonical > 0.5).astype(np.uint8)

    return canonical


def transform_coords_to_um(
    coords_yx: np.ndarray,
    transform: GridTransform,
    origin: str = "grid_center",
) -> np.ndarray:
    """
    Convert pixel coordinates to micrometers in canonical space.

    Args:
        coords_yx: (N, 2) array of (y, x) coordinates in canonical grid pixels
        transform: Grid transform
        origin: "grid_center" or "top_left"

    Returns:
        (N, 2) array of (y, x) coordinates in micrometers
    """
    coords_um = coords_yx * transform.effective_um_per_pixel

    if origin == "grid_center":
        # Shift origin to center of grid
        h, w = transform.grid_shape_hw
        if transform.downsample_factor > 1:
            h = h // transform.downsample_factor
            w = w // transform.downsample_factor
        center_um = np.array([h / 2, w / 2]) * transform.effective_um_per_pixel
        coords_um -= center_um

    return coords_um


def rescale_velocity_to_um(
    velocity_yx_hw2: np.ndarray,
    transform: GridTransform,
) -> np.ndarray:
    """
    Rescale velocity field from pixels to micrometers.

    Args:
        velocity_yx_hw2: (H, W, 2) velocity field in pixels
        transform: Grid transform

    Returns:
        (H, W, 2) velocity field in micrometers
    """
    return velocity_yx_hw2 * transform.effective_um_per_pixel


def rescale_distance_to_um(
    distance: float,
    transform: GridTransform,
) -> float:
    """
    Rescale a distance from pixels to micrometers.

    Args:
        distance: Distance in canonical grid pixels
        transform: Grid transform

    Returns:
        Distance in micrometers
    """
    return distance * transform.effective_um_per_pixel
