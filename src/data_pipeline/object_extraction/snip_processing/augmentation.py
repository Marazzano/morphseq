"""
Embryo snip augmentation for model-ready processing.

Applies CLAHE equalization, background noise injection, and edge blending.
Extracted from build03A_process_images.py (lines 401-407).
"""

import numpy as np
import skimage.exposure
import skimage.filters
import scipy.ndimage
from scipy.stats import truncnorm
from typing import Tuple

from data_pipeline.object_extraction.snip_processing.defaults import (
    DEFAULT_BLEND_RADIUS_UM,
)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Args:
        image: Input image (uint8)

    Returns:
        CLAHE-equalized image (uint8)
    """
    # Apply CLAHE and scale to uint8
    equalized = skimage.exposure.equalize_adapthist(image) * 255
    return equalized.astype(np.uint8)


def generate_background_noise(
    shape: Tuple[int, int],
    background_mean: float,
    background_std: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate truncated normal noise matching background statistics.

    Args:
        shape: Output shape (height, width)
        background_mean: Mean background intensity
        background_std: Background standard deviation
        seed: Random seed for reproducibility

    Returns:
        Noise array with background statistics
    """
    np.random.seed(seed)

    # A constant background is a valid input (notably for center-padded SeaHub frames).
    # scipy's standardized truncation bounds divide by the scale, so handle this
    # degenerate distribution explicitly.
    if background_std == 0:
        return np.full(shape, max(float(background_mean), 0.0), dtype=float)

    # Generate truncated normal distribution (no negative values)
    a = -background_mean / background_std  # Lower bound in standard deviations
    b = 4  # Upper bound (4 std above mean)

    noise_normalized = truncnorm.rvs(
        a, b,
        size=shape[0] * shape[1]
    )

    # Rescale to background statistics
    noise = np.reshape(noise_normalized, shape) * background_std + background_mean
    noise[noise < 0] = 0

    return noise


def blend_with_background_noise(
    image: np.ndarray,
    mask: np.ndarray,
    background_mean: float,
    background_std: float,
    blend_radius_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """
    Blend embryo image with background noise using Gaussian edge weighting.

    Args:
        image: CLAHE-equalized embryo image
        mask: Binary embryo mask
        background_mean: Background pixel mean
        background_std: Background pixel std
        blend_radius_um: Gaussian blur radius in micrometers
        pixel_size_um: Pixel size in micrometers

    Returns:
        Blended image with soft edges
    """
    # Fill holes in mask for smooth blending
    mask_filled = scipy.ndimage.binary_fill_holes(mask > 0.5).astype(np.uint8)

    # Generate background noise
    noise = generate_background_noise(
        image.shape,
        background_mean,
        background_std
    )

    # Create Gaussian blending weights
    blur_sigma = blend_radius_um / pixel_size_um
    mask_blurred = skimage.filters.gaussian(
        mask_filled.astype(float),
        sigma=blur_sigma
    )

    # Blend: weighted combination of image and noise
    blended = (
        image.astype(float) * mask_blurred +
        noise * (1 - mask_blurred)
    )

    return blended.astype(np.uint8)


def augment_snip(
    image: np.ndarray,
    mask: np.ndarray,
    background_mean: float,
    background_std: float,
    blend_radius_um: float = DEFAULT_BLEND_RADIUS_UM,
    pixel_size_um: float = 2.17,
    use_clahe: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete augmentation pipeline: optional CLAHE + noise blending.

    Args:
        image: Cropped embryo image (uint8)
        mask: Cropped embryo mask
        background_mean: Background intensity mean
        background_std: Background intensity std
        blend_radius_um: Edge blending radius in micrometers
        pixel_size_um: Pixel size for blur conversion
        use_clahe: Apply the legacy CLAHE normalization before blending. Defaults
            to True to preserve the Keyence/YX1 and checkpoint-training contract.
            SeaHub disables it because its already-8-bit inverted source images
            otherwise acquire a strongly amplified near-white tail.

    Returns:
        Tuple of (augmented_image, contrast_normalized_image). The second value
        is the CLAHE image when ``use_clahe`` is True and the unchanged uint8
        input image otherwise.
    """
    # CLAHE remains the default because the established microscopy pipelines and
    # morphology checkpoint were built around it. SeaHub can explicitly bypass
    # this step: its materializer has already converted and inverted an 8-bit
    # display image, and another full-range local equalization blows out dark
    # source anatomy after inversion.
    contrast_image = (
        apply_clahe(image)
        if use_clahe
        else np.asarray(image, dtype=np.uint8)
    )

    # Blend with background noise
    augmented = blend_with_background_noise(
        contrast_image,
        mask,
        background_mean,
        background_std,
        blend_radius_um,
        pixel_size_um
    )

    return augmented, contrast_image
