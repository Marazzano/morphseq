"""Core overlay primitives for contract-native frame rendering.

All functions operate on a single BGR frame (H×W×3 uint8 numpy array) and a pre-filtered
DataFrame slice for that frame.  Callers are responsible for filtering to one frame before
calling; no joins happen here.
"""

from __future__ import annotations

import ast
from typing import Any

import cv2
import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.viz.config import COLORBLIND_PALETTE, OVERLAY_COLORS, RenderConfig


def color_for_key(key: Any) -> tuple[int, int, int]:
    """Deterministic, palette-stable color for an arbitrary key (track_id, embryo_id, etc.)."""
    palette = list(COLORBLIND_PALETTE.values())
    return palette[abs(hash(str(key))) % len(palette)]


def draw_banner(
    frame: np.ndarray,
    image_id: str,
    *,
    well_id: str | None = None,
    cfg: RenderConfig | None = None,
) -> np.ndarray:
    """Draw a solid top-banner and return the vertically expanded frame.

    When ``well_id`` is supplied and ``cfg.show_well_id`` is True, the banner
    shows ``well_id  |  image_id``; otherwise just ``image_id``.
    """
    cfg = cfg or RenderConfig()
    h, w = frame.shape[:2]
    banner = np.full((cfg.banner_height_px, w, 3), cfg.banner_color, dtype=np.uint8)

    if well_id is not None and cfg.show_well_id:
        text = f"{well_id}  |  {image_id}"
    else:
        text = image_id

    cv2.putText(
        banner,
        text,
        (8, cfg.banner_height_px - 10),
        cfg.font,
        cfg.font_scale,
        cfg.text_color,
        cfg.font_thickness,
        cv2.LINE_AA,
    )
    return np.vstack([banner, frame])


def draw_boxes(
    frame: np.ndarray,
    detections: pd.DataFrame,
    *,
    cfg: RenderConfig | None = None,
) -> np.ndarray:
    """Draw bounding boxes and confidence scores from a per-frame ``frame_detections`` slice.

    ``detections`` must already be filtered to the rows for this frame and to ``is_kept == True``.
    """
    cfg = cfg or RenderConfig()
    out = frame.copy()
    for _, row in detections.iterrows():
        x1 = int(row["bbox_x_min_px"])
        y1 = int(row["bbox_y_min_px"])
        x2 = int(row["bbox_x_max_px"])
        y2 = int(row["bbox_y_max_px"])
        color = OVERLAY_COLORS["detection"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, cfg.bbox_thickness)
        conf = row.get("confidence", float("nan"))
        label = row.get("class_label", "")
        if pd.notna(conf):
            text = f"{label} {float(conf):.2f}" if label else f"{float(conf):.2f}"
            cv2.putText(
                out,
                text,
                (x1, max(y1 - 4, 0)),
                cfg.font,
                cfg.font_scale * 0.7,
                color,
                cfg.font_thickness,
                cv2.LINE_AA,
            )
    return out


def draw_masks(
    frame: np.ndarray,
    masks: pd.DataFrame,
    *,
    cfg: RenderConfig | None = None,
) -> np.ndarray:
    """Alpha-blend segmentation masks from a per-frame ``frame_masks`` slice.

    ``masks`` must already be filtered to the rows for this frame; rows with ``is_valid_mask``
    false (no-mask placeholders) are skipped automatically.

    When ``cfg.show_embryo_labels`` is True, the ``track_id`` (or ``physical_embryo_id`` if
    present) is drawn at the mask centroid.
    """
    cfg = cfg or RenderConfig()
    out = frame.copy()
    valid = masks[masks["is_valid_mask"].astype(bool)]
    for _, row in valid.iterrows():
        rle_raw = row.get("mask_rle")
        if pd.isna(rle_raw):
            continue
        rle = ast.literal_eval(rle_raw) if isinstance(rle_raw, str) else rle_raw
        try:
            binary = decode_binary_mask_rle(rle)
        except Exception:
            continue

        # Use physical_embryo_id if available, fall back to track_id, then mask_id.
        embryo_key = row.get("physical_embryo_id") or row.get("track_id") or row.get("mask_id", "")
        color = color_for_key(embryo_key)

        colored = np.zeros_like(out)
        colored[binary] = color
        mask_3ch = np.stack([binary, binary, binary], axis=-1)
        out = np.where(mask_3ch, cv2.addWeighted(out, 1 - cfg.mask_alpha, colored, cfg.mask_alpha, 0), out)

        # Contour outline
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)

        # Embryo label at centroid
        if cfg.show_embryo_labels and pd.notna(embryo_key) and str(embryo_key):
            cx = row.get("centroid_x_px")
            cy = row.get("centroid_y_px")
            if pd.notna(cx) and pd.notna(cy):
                # Use the short suffix (e.g. "track0000") to keep the label compact.
                label = str(embryo_key).split("_")[-1] if "_" in str(embryo_key) else str(embryo_key)
                cv2.putText(
                    out,
                    label,
                    (int(cx), int(cy)),
                    cfg.font,
                    cfg.font_scale * 0.7,
                    cfg.text_color,
                    cfg.font_thickness,
                    cv2.LINE_AA,
                )

    return out.astype(np.uint8)
