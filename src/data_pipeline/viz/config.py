"""Rendering configuration and color constants for data_pipeline.viz.

Color values are BGR tuples (OpenCV convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2


# Colorblind-safe palette — same values as segmentation.video_generation.video_config.
COLORBLIND_PALETTE: dict[str, tuple[int, int, int]] = {
    "light_blue": (173, 216, 230),
    "light_green": (144, 238, 144),
    "light_coral": (240, 128, 128),
    "light_yellow": (255, 215, 0),
    "light_purple": (221, 160, 221),
    "light_orange": (255, 218, 185),
    "light_cyan": (0, 255, 255),
    "light_rose": (255, 182, 193),
    "light_mint": (152, 255, 152),
    "light_lavender": (186, 135, 186),
}

OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "detection": COLORBLIND_PALETTE["light_blue"],
    "mask": COLORBLIND_PALETTE["light_green"],
    "qc_good": COLORBLIND_PALETTE["light_green"],
    "qc_warning": COLORBLIND_PALETTE["light_yellow"],
    "qc_error": COLORBLIND_PALETTE["light_coral"],
}


@dataclass
class RenderConfig:
    fps: int = 5
    codec: str = "mp4v"
    font: int = field(default_factory=lambda: cv2.FONT_HERSHEY_SIMPLEX)
    font_scale: float = 0.8
    font_thickness: int = 2
    bbox_thickness: int = 2
    mask_alpha: float = 0.4
    # Contour thickness for mask outlines, in NATIVE frame pixels. The outline is what makes
    # OVERLAPPING masks readable -- two alpha-blended fills stack into an ambiguous colour, but
    # each mask's own-colour border stays traceable. Raise this when the frame will be scaled down
    # for a contact sheet, or a 1px border disappears in the resize.
    mask_outline_thickness: int = 1
    banner_height_px: int = 40
    text_color: tuple[int, int, int] = (255, 255, 255)
    banner_color: tuple[int, int, int] = (0, 0, 0)
    # Label toggles — all default on
    show_well_id: bool = True
    show_embryo_labels: bool = True
