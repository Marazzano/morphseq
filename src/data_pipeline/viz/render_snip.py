"""Per-snip auxiliary mask contact sheet rendering.

render_snip_auxiliary_masks  — for one snip, renders the original crop alongside
                                each of its auxiliary mask overlays as a single PNG
                                row: [snip | foreground | via | yolk | focus | bubble]

render_snip_auxiliary_masks_contact_sheet — renders all snips for a well as a grid,
                                            one snip per row, writing a single PNG.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_pipeline.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
)
from data_pipeline.viz.config import COLORBLIND_PALETTE, RenderConfig
from data_pipeline.viz.overlay import draw_banner


def _load_gray_as_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _load_mask_png(path: Path) -> np.ndarray | None:
    """Load an auxiliary mask PNG (uint8 0/255) and return a bool H×W array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def _overlay_mask_on_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    out = bgr.copy()
    colored = np.zeros_like(out)
    colored[mask] = color
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    blended = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)
    return np.where(mask_3ch, blended, out).astype(np.uint8)


def _mask_type_color(mask_type: str) -> tuple[int, int, int]:
    palette = list(COLORBLIND_PALETTE.values())
    idx = list(ALLOWED_AUXILIARY_MASK_TYPES).index(mask_type) if mask_type in ALLOWED_AUXILIARY_MASK_TYPES else 0
    return palette[idx % len(palette)]


def render_snip_auxiliary_masks(
    snip_id: str,
    snip_image_path: Path,
    auxiliary_masks_rows: pd.DataFrame,
    output_path: Path,
    *,
    config: RenderConfig | None = None,
) -> Path:
    """Render one snip + its auxiliary mask overlays as a horizontal strip PNG.

    Layout: [original snip | foreground overlay | via overlay | yolk overlay | focus overlay | bubble overlay]
    Missing or invalid masks render as the plain snip (no overlay).
    """
    cfg = config or RenderConfig()
    base = _load_gray_as_bgr(snip_image_path)
    if base is None:
        raise FileNotFoundError(f"Could not load snip image: {snip_image_path}")

    panels = [base.copy()]

    for mask_type in ALLOWED_AUXILIARY_MASK_TYPES:
        row = auxiliary_masks_rows[
            (auxiliary_masks_rows["auxiliary_mask_type"] == mask_type)
            & auxiliary_masks_rows["is_valid_auxiliary_mask"].astype(bool)
        ]
        panel = base.copy()
        if len(row) == 1:
            mask_path = row.iloc[0].get("auxiliary_mask_path")
            if pd.notna(mask_path):
                mask = _load_mask_png(Path(str(mask_path)))
                if mask is not None:
                    color = _mask_type_color(mask_type)
                    panel = _overlay_mask_on_bgr(panel, mask, color, cfg.mask_alpha)
        # Label the panel with mask_type
        cv2.putText(
            panel,
            mask_type,
            (4, panel.shape[0] - 6),
            cfg.font,
            cfg.font_scale * 0.6,
            cfg.text_color,
            cfg.font_thickness,
            cv2.LINE_AA,
        )
        panels.append(panel)

    strip = np.hstack(panels)
    strip = draw_banner(strip, snip_id, cfg=cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), strip)
    return output_path


def render_snip_auxiliary_masks_contact_sheet(
    snip_inventory: pd.DataFrame,
    auxiliary_masks: pd.DataFrame,
    output_path: Path,
    *,
    well_id: str | None = None,
    config: RenderConfig | None = None,
    max_snips: int | None = None,
) -> Path:
    """Render a contact sheet for all valid snips in a well.

    One row per snip: [original | foreground | via | yolk | focus | bubble].
    Writes a single PNG at output_path.

    snip_inventory must have columns: snip_id, processed_snip_path, is_valid_snip.
    auxiliary_masks is the snip_auxiliary_masks DataFrame for the same well.
    """
    cfg = config or RenderConfig()

    valid = snip_inventory[snip_inventory["is_valid_snip"].astype(bool)].copy()
    if max_snips is not None:
        valid = valid.head(max_snips)

    rows_rendered: list[np.ndarray] = []

    for _, snip_row in valid.iterrows():
        snip_id = snip_row["snip_id"]
        snip_path = Path(str(snip_row["processed_snip_path"]))
        if not snip_path.exists():
            continue

        base = _load_gray_as_bgr(snip_path)
        if base is None:
            continue

        panels = [base.copy()]
        snip_masks = auxiliary_masks[auxiliary_masks["snip_id"] == snip_id]

        for mask_type in ALLOWED_AUXILIARY_MASK_TYPES:
            row = snip_masks[
                (snip_masks["auxiliary_mask_type"] == mask_type)
                & snip_masks["is_valid_auxiliary_mask"].astype(bool)
            ]
            panel = base.copy()
            if len(row) == 1:
                mask_path = row.iloc[0].get("auxiliary_mask_path")
                if pd.notna(mask_path):
                    mask = _load_mask_png(Path(str(mask_path)))
                    if mask is not None:
                        color = _mask_type_color(mask_type)
                        panel = _overlay_mask_on_bgr(panel, mask, color, cfg.mask_alpha)
            cv2.putText(
                panel,
                mask_type,
                (4, panel.shape[0] - 6),
                cfg.font,
                cfg.font_scale * 0.6,
                cfg.text_color,
                cfg.font_thickness,
                cv2.LINE_AA,
            )
            panels.append(panel)

        strip = np.hstack(panels)
        label = f"{snip_id}" + (f"  [{well_id}]" if well_id else "")
        strip = draw_banner(strip, label, cfg=cfg)
        rows_rendered.append(strip)

    if not rows_rendered:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return output_path

    sheet = np.vstack(rows_rendered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path
