"""Shared materialized-image display polarity — owned in ONE place for all microscopes.

Canonical polarity for every materialized brightfield frame is **embryo bright on a dark
background** (i.e. the display is inverted relative to raw bright-field). Downstream stages —
snip_processing background estimation, frame masks, and the VAE embeddings — assume a DARK
background, so all microscopes must agree on this polarity.

This is deliberately its OWN module, not a side effect of focus-stacking or stitching. The
regression this prevents: the inversion used to be a hidden ``max - out`` inside the Keyence
stitcher (``frame_tiler``), which YX1's single-tile identity path never went through — so YX1
silently drifted to the opposite polarity. Polarity is a distinct display decision; it lives
here and is applied ONCE by each materializer at its final composition boundary (post-stitch for
Keyence, post-projection for YX1).
"""

from __future__ import annotations

import numpy as np

# Canonical materialized-image polarity for ALL microscopes (bright embryo / dark background).
INVERT_FOR_DISPLAY = True


def apply_display_polarity(image: np.ndarray, *, invert: bool = INVERT_FOR_DISPLAY) -> np.ndarray:
    """Return the canonical materialized-image polarity for an integer frame.

    ``invert=True`` maps ``v -> dtype_max - v`` (bright embryo on dark background). It is
    dtype-aware — ``dtype_max`` is the max of the array's integer dtype (255 for uint8, 65535
    for uint16) — so it matches the legacy ``np.iinfo(dtype).max - out`` inversion for both the
    uint8 projection mosaics and the uint16 z-stack mosaics. Pure per-pixel op, so it is
    equivalent whether applied to a single tile or a fully stitched mosaic; callers apply it
    ONCE at the final composition boundary.

    Args:
        image: an integer image (single frame or stitched mosaic), uint8 or uint16.
        invert: whether to invert; defaults to the canonical :data:`INVERT_FOR_DISPLAY`.

    Returns:
        An array (same dtype) with the requested polarity. When ``invert`` is False the input is
        returned unchanged.
    """
    arr = np.asarray(image)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"apply_display_polarity expects an integer image; got {arr.dtype}.")
    if not invert:
        return arr
    return (np.iinfo(arr.dtype).max - arr).astype(arr.dtype)
