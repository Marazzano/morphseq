"""Boundary smoothing applied to an embryo mask before centerline extraction.

Ported verbatim (behavior-preserving) from the validated legacy body-axis pipeline
(``segmentation_sandbox/scripts/body_axis_analysis/mask_preprocessing.py``). A Gaussian blur
followed by a high re-threshold shaves fine boundary protrusions (notably fins) that would
otherwise pull the skeleton off the body axis. ``sigma=15`` / ``threshold=0.7`` were tuned
empirically on multiple embryos to remove fins without eroding body structure — do not change them
without re-validating against the legacy benchmark set.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SMOOTHING_SIGMA_PX: float = 15.0
SMOOTHING_THRESHOLD: float = 0.7


def smooth_mask_boundary(
    mask: np.ndarray,
    *,
    sigma_px: float = SMOOTHING_SIGMA_PX,
    threshold: float = SMOOTHING_THRESHOLD,
) -> np.ndarray:
    """Return ``mask`` with its boundary smoothed by blur-then-re-threshold.

    Blurring spreads the mask; re-thresholding at a high value keeps only the dense core, so thin
    protrusions (fins) fall below threshold and disappear while the body is preserved.
    """
    blurred = gaussian_filter(mask.astype(float), sigma=sigma_px)
    return (blurred > threshold).astype(np.uint8)
