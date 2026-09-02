"""
PCA-based embryo rotation alignment.

Rotates embryos to standard orientation using principal component analysis.
Extracted from build03A_process_images.py and image_utils.py.
"""

import numpy as np
import cv2
import scipy.ndimage
from skimage.measure import regionprops
from typing import Tuple

from image_geometry import bounds_expanded_rotation_matrix


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate image by angle in degrees, expanding bounds to avoid cropping.

    Args:
        image: Input image array
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)

    Returns:
        Rotated image with expanded bounds
    """
    height, width = image.shape[:2]
    rotation_mat, bounds_wh = bounds_expanded_rotation_matrix(
        shape_hw=(height, width), angle_deg=angle_degrees
    )
    return cv2.warpAffine(image, rotation_mat, bounds_wh)


def get_embryo_rotation_angle(
    embryo_mask: np.ndarray,
    yolk_mask: np.ndarray,
) -> float:
    """
    Compute PCA-based rotation angle to align embryo with yolk at top.

    Based on build03A get_embryo_angle function.

    Args:
        embryo_mask: Binary embryo mask
        yolk_mask: Binary yolk mask (can be empty)

    Returns:
        Rotation angle in radians
    """
    # Get orientation from PCA
    rp = regionprops(embryo_mask.astype(int))
    if not rp:
        return 0.0  # No regions found

    angle = rp[0].orientation

    # Determine correct orientation to place yolk at top
    rotated_embryo = rotate_image(embryo_mask, np.rad2deg(-angle))
    embryo_com = scipy.ndimage.center_of_mass(rotated_embryo)

    if np.any(yolk_mask):
        # Use yolk position to orient
        rotated_yolk = rotate_image(yolk_mask, np.rad2deg(-angle))
        yolk_com = scipy.ndimage.center_of_mass(rotated_yolk)

        if (embryo_com[0] - yolk_com[0]) >= 0:
            angle_to_use = -angle
        else:
            angle_to_use = -angle + np.pi
    else:
        # Use embryo mass distribution when yolk unavailable
        y_indices = np.where(np.max(rotated_embryo, axis=1))[0]
        if len(y_indices) > 0:
            vert_ratio = np.sum(y_indices > embryo_com[0]) / len(y_indices)
            if vert_ratio >= 0.5:
                angle_to_use = -angle
            else:
                angle_to_use = -angle + np.pi
        else:
            angle_to_use = -angle

    return angle_to_use


def apply_rotation_to_snip(
    image: np.ndarray,
    mask: np.ndarray,
    yolk_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply PCA-based rotation to embryo snip.

    Args:
        image: Embryo image
        mask: Embryo mask
        yolk_mask: Yolk mask

    Returns:
        Tuple of (rotated_image, rotated_mask, rotated_yolk, angle_radians)
    """
    # Compute rotation angle
    angle_rad = get_embryo_rotation_angle(
        (mask > 0.5).astype(np.uint8),
        (yolk_mask > 0.5).astype(np.uint8)
    )

    # Apply rotation
    angle_deg = np.rad2deg(angle_rad)
    image_rotated = rotate_image(image, angle_deg)
    mask_rotated = rotate_image(mask, angle_deg)
    yolk_rotated = rotate_image(yolk_mask, angle_deg)

    return image_rotated, mask_rotated, yolk_rotated, angle_rad
