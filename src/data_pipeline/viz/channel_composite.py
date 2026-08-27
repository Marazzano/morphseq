"""Multi-channel composite views of materialized frames — DISPLAY ONLY.

Renders two materialized single-channel frames into one RGB image: a grayscale base channel with a
color-tinted overlay channel on top. The tint comes from ``CHANNEL_ID_COLORS`` (the canonical channel
vocabulary), so a channel looks the same everywhere it is drawn.

**Nothing here writes a pipeline product.** Materialized image products are single-channel INTENSITY
data — a fluorescence product like ``RFP__projection__max`` is a quantitative measurement stored at
native resolution in uint16, deliberately un-inverted. Color and contrast stretching are applied for
HUMAN VIEWING at the moment of rendering; the stored frames are never modified. Anything produced
here is a derived view, safe to delete and regenerate.

Why the two channels are stretched INDEPENDENTLY: brightfield and fluorescence occupy wildly
different intensity ranges (measured on a real pbx well: BF mean 18193 vs RFP mean 815, ~22x). Under
one shared stretch the brightfield swamps the fluorescence completely and the overlay shows nothing.
Each channel is therefore normalized against its own percentiles before compositing.

This module is generic over ``(base_channel_id, overlay_channel_id)`` — it is not BF/RFP specific,
because every fluorescence channel wants exactly this view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from data_pipeline.shared.channel_vocabulary import color_for_channel_id

# Default display stretch. The high end is 99.5 rather than 100 so a handful of hot pixels cannot
# compress the entire visible range; the low end trims sensor floor without clipping real signal.
DEFAULT_STRETCH_PERCENTILES: tuple[float, float] = (1.0, 99.5)


def hex_to_rgb_fractions(hex_color: str) -> tuple[float, float, float]:
    """Convert ``#RRGGBB`` to per-channel fractions in ``[0, 1]``."""
    text = hex_color.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected a #RRGGBB hex color, got {hex_color!r}.")
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def stretch_to_unit(
    image: np.ndarray, percentiles: tuple[float, float] = DEFAULT_STRETCH_PERCENTILES
) -> np.ndarray:
    """Percentile-stretch one single-channel image to ``[0, 1]`` floats.

    DISPLAY transform only — never write the result back as a product. A flat image (no spread
    between the percentiles) returns all zeros rather than dividing by zero.
    """
    if image.ndim != 2:
        raise ValueError(f"stretch_to_unit expects a 2D single-channel image; got {image.shape}.")
    lo_pct, hi_pct = percentiles
    data = image.astype(np.float64)
    lo, hi = np.percentile(data, lo_pct), np.percentile(data, hi_pct)
    if hi <= lo:
        return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def composite_two_channels(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    *,
    base_channel_id: str,
    overlay_channel_id: str,
    percentiles: tuple[float, float] = DEFAULT_STRETCH_PERCENTILES,
    base_gain: float = 1.0,
) -> np.ndarray:
    """Composite a grayscale base channel with a color-tinted overlay channel → RGB uint8.

    Args:
        base_image: 2D single-channel frame drawn as the grayscale base (typically brightfield).
        overlay_image: 2D single-channel frame drawn tinted on top (typically fluorescence).
        base_channel_id: canonical channel_id of ``base_image`` (validated; currently informational).
        overlay_channel_id: canonical channel_id of ``overlay_image`` — supplies the tint via
            ``CHANNEL_ID_COLORS``.
        percentiles: display stretch applied to EACH channel independently.
        base_gain: scales the base after stretching. Lower it (~0.4) to make the overlay dominate
            when the question is "where is the signal", rather than "what does the animal look like".

    Returns:
        ``(H, W, 3)`` uint8 RGB. Additive composite, clipped at white where both are bright.
    """
    if base_image.shape != overlay_image.shape:
        raise ValueError(
            f"Channel frames must share a shape to composite; got base {base_image.shape} "
            f"and overlay {overlay_image.shape}. They must come from the same well/time and the "
            "same write policy (a downsampled product cannot be composited with a native one)."
        )
    # Validate both ids even though only the overlay is tinted: a typo'd base channel should fail
    # here, not silently label the output.
    color_for_channel_id(base_channel_id)
    tint = hex_to_rgb_fractions(color_for_channel_id(overlay_channel_id))

    base = stretch_to_unit(base_image, percentiles) * float(base_gain)
    overlay = stretch_to_unit(overlay_image, percentiles)

    rgb = np.repeat(base[:, :, np.newaxis], 3, axis=2)
    rgb = rgb + overlay[:, :, np.newaxis] * np.asarray(tint)[np.newaxis, np.newaxis, :]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def tint_single_channel(
    image: np.ndarray,
    *,
    channel_id: str,
    percentiles: tuple[float, float] = DEFAULT_STRETCH_PERCENTILES,
) -> np.ndarray:
    """Render ONE channel in its vocabulary color → RGB uint8 (display only)."""
    tint = hex_to_rgb_fractions(color_for_channel_id(channel_id))
    stretched = stretch_to_unit(image, percentiles)
    rgb = stretched[:, :, np.newaxis] * np.asarray(tint)[np.newaxis, np.newaxis, :]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def read_materialized_frame(path: Path | str) -> np.ndarray:
    """Read one materialized single-channel frame as a 2D array, dtype preserved."""
    with Image.open(path) as im:
        array = np.asarray(im)
    if array.ndim != 2:
        raise ValueError(
            f"{path} is not a single-channel frame (shape {array.shape}). Materialized image "
            "products are single-channel intensity data; a 3-channel file is already a rendered view."
        )
    return array
