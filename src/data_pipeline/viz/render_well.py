"""Contract-native per-well overlay video rendering.

Entry points:
  render_detection_video    — bounding-box overlay from frame_detections contract
  render_segmentation_video — mask overlay from frame_masks contract
  render_combined_video     — boxes + masks on the same frames

All data arguments (frame_inventory, frame_detections, frame_masks) are optional (``None``
skips that overlay layer).  At least one must be supplied so the frame dimensions are known.

Label toggles (well_id banner, per-embryo track labels) are controlled via ``RenderConfig``:
  cfg.show_well_id       — include well_id in the top banner (default True)
  cfg.show_embryo_labels — draw track_id / physical_embryo_id at mask centroids (default True)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_pipeline.viz.config import RenderConfig
from data_pipeline.viz.overlay import draw_banner, draw_boxes, draw_masks


def _load_frame(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _open_writer(output_path: Path, width: int, height: int, cfg: RenderConfig) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(output_path), fourcc, cfg.fps, (width, height + cfg.banner_height_px))


def _frame_dims(inv_row: pd.Series) -> tuple[int, int]:
    return int(inv_row.get("image_width_px", 512)), int(inv_row.get("image_height_px", 512))


def render_detection_video(
    well_id: str,
    frame_inventory: pd.DataFrame | None,
    frame_detections: pd.DataFrame | None,
    images_root: Path | None,
    output_path: Path,
    *,
    config: RenderConfig | None = None,
) -> Path:
    """Render an MP4 with bounding-box overlays from ``frame_detections``."""
    cfg = config or RenderConfig()
    kept = frame_detections[frame_detections["is_kept"].astype(bool)] if frame_detections is not None else None
    return _render(
        well_id=well_id,
        frame_inventory=frame_inventory,
        images_root=images_root,
        output_path=output_path,
        cfg=cfg,
        detections=kept,
        masks=None,
    )


def render_segmentation_video(
    well_id: str,
    frame_inventory: pd.DataFrame | None,
    frame_masks: pd.DataFrame | None,
    images_root: Path | None,
    output_path: Path,
    *,
    config: RenderConfig | None = None,
) -> Path:
    """Render an MP4 with mask overlays from ``frame_masks``."""
    cfg = config or RenderConfig()
    return _render(
        well_id=well_id,
        frame_inventory=frame_inventory,
        images_root=images_root,
        output_path=output_path,
        cfg=cfg,
        detections=None,
        masks=frame_masks,
    )


def render_combined_video(
    well_id: str,
    frame_inventory: pd.DataFrame | None,
    frame_detections: pd.DataFrame | None,
    frame_masks: pd.DataFrame | None,
    images_root: Path | None,
    output_path: Path,
    *,
    config: RenderConfig | None = None,
) -> Path:
    """Render an MP4 with bounding boxes and/or mask overlays.

    Any of ``frame_inventory``, ``frame_detections``, ``frame_masks`` may be ``None`` to skip
    that layer.  The frame timeline is driven by ``frame_inventory`` when supplied; otherwise
    it falls back to the union of image_ids found in the non-None data tables.
    """
    cfg = config or RenderConfig()
    kept = frame_detections[frame_detections["is_kept"].astype(bool)] if frame_detections is not None else None
    return _render(
        well_id=well_id,
        frame_inventory=frame_inventory,
        images_root=images_root,
        output_path=output_path,
        cfg=cfg,
        detections=kept,
        masks=frame_masks,
    )


def _build_frame_order(
    frame_inventory: pd.DataFrame | None,
    detections: pd.DataFrame | None,
    masks: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return a DataFrame with columns [image_id, source_image_path, image_width_px, image_height_px]
    in frame order, derived from whichever inputs are available."""
    if frame_inventory is not None:
        cols = {c: frame_inventory[c] for c in ["image_id", "source_image_path", "image_width_px", "image_height_px"] if c in frame_inventory.columns}
        return frame_inventory[list(cols)].drop_duplicates("image_id")

    # Fall back: collect image_id ordering from data tables
    sources: list[pd.DataFrame] = []
    for df in (detections, masks):
        if df is not None and "image_id" in df.columns:
            sub_cols = [c for c in ["image_id", "source_image_path", "image_width_px", "image_height_px", "time_index"] if c in df.columns]
            sources.append(df[sub_cols].drop_duplicates("image_id"))
    if not sources:
        raise ValueError("At least one of frame_inventory, frame_detections, or frame_masks must be provided.")
    combined = pd.concat(sources).drop_duplicates("image_id")
    if "time_index" in combined.columns:
        combined = combined.sort_values("time_index")
    return combined.reset_index(drop=True)


def _render(
    *,
    well_id: str,
    frame_inventory: pd.DataFrame | None,
    images_root: Path | None,
    output_path: Path,
    cfg: RenderConfig,
    detections: pd.DataFrame | None,
    masks: pd.DataFrame | None,
) -> Path:
    frame_order = _build_frame_order(frame_inventory, detections, masks)
    writer: cv2.VideoWriter | None = None

    for _, inv_row in frame_order.iterrows():
        image_id = str(inv_row["image_id"])

        src_raw = inv_row.get("source_image_path")
        frame: np.ndarray | None = None
        if pd.notna(src_raw):
            src = Path(str(src_raw))
            if not src.is_absolute() and images_root is not None:
                src = images_root / src
            frame = _load_frame(src)

        if frame is None:
            w, h = _frame_dims(inv_row)
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        if detections is not None:
            frame = draw_boxes(frame, detections[detections["image_id"] == image_id], cfg=cfg)

        if masks is not None:
            frame = draw_masks(frame, masks[masks["image_id"] == image_id], cfg=cfg)

        frame = draw_banner(frame, image_id, well_id=well_id, cfg=cfg)

        if writer is None:
            h_frame, w_frame = frame.shape[:2]
            writer = _open_writer(output_path, w_frame, h_frame - cfg.banner_height_px, cfg)

        writer.write(frame)

    if writer is not None:
        writer.release()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

    return output_path
