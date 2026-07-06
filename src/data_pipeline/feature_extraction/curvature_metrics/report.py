"""curvature_metrics report — feature histogram grid (D) + centerline-overlay gallery (E+overlay).
TERMINAL leaf (report_world.md).

Consumes curvature_metrics's own merged output (the numbers) + snip_inventory (for the gallery's
snip image path AND its per-snip embryo mask, ``embryo_mask``; legacy alias ``embryo_mask_snip_path``) + frame_inventory (for the
micron calibration the spine's arc length needs).

The overlay redraws the geodesic spine PURELY FOR DISPLAY. It recomputes the centerline from
``embryo_mask`` — the cropped-and-rotated embryo mask snip_processing persists in the SAME
pixel frame as the snip image (see run_snip_processing.py: image and mask go through one
``extract_embryo_crop -> rotate -> crop`` together). Computing the spine on that snip-space mask
lands its coordinates directly in snip-pixel space, pixel-aligned with the snip by construction —
no crop/rotate/resize math here. This is a redraw for the eye, not the literal pixels the merged
numbers came from (those are measured on the full-frame RLE mask); the curvature shape is
rotation/scale-covariant, so the drawn spine tracks the same animal the numbers describe.

Consumed by nothing; imported by nothing but its own tasks.py subcommand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_and_mask_paths
from data_pipeline.feature_extraction.curvature_metrics.geodesic_centerline import (
    extract_geodesic_centerline,
)
from data_pipeline.feature_extraction.curvature_metrics.head_tail_orientation import (
    orient_centerline_head_to_tail,
)
from data_pipeline.feature_extraction.curvature_metrics.mask_preprocessing import smooth_mask_boundary
from data_pipeline.feature_extraction.shared.feature_table_utils import pixel_size_for_image
from data_pipeline.viz.reporting import (
    draw_centerline_on_snip,
    plot_histogram_grid,
    render_value_quartile_gallery_with_overlay,
)

# The most interpretable payload columns to eyeball; baseline_deviation_normalized leads (headline).
CURVATURE_FEATURE_COLUMNS = [
    "baseline_deviation_normalized",
    "mean_curvature_per_um",
    "total_length_um",
    "baseline_deviation_um",
    "arc_length_ratio",
    "centerline_point_count",
]
RANK_COLUMN = "baseline_deviation_normalized"


def _geodesic_spine_layers(mask: np.ndarray, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoothed head-oriented B-spline, raw skeleton centerline) for the overlay.
    Either array is empty when the skeleton was too short to fit a spline, or when it was too
    degenerate to skeletonize at all (mirrors compute.py's per-mask null-metrics handling)."""
    smoothed = smooth_mask_boundary(mask)
    try:
        centerline = extract_geodesic_centerline(smoothed, pixel_size_um=pixel_size_um)
    except (ValueError, RuntimeError):
        return np.empty((0, 2)), np.empty((0, 2))
    if len(centerline.smoothed_xy) == 0:
        return np.empty((0, 2)), centerline.raw_xy
    spline_xy = orient_centerline_head_to_tail(centerline.smoothed_xy, smoothed)
    return spline_xy, centerline.raw_xy


def _load_snip_mask(mask_path: str) -> np.ndarray:
    """Load the per-snip embryo mask PNG as a boolean array (snip pixel space, alpha 0/255)."""
    return np.asarray(Image.open(mask_path)) > 0


def _overlay_fn(paths_by_snip: dict[str, tuple[str, str, float]]):
    """Return image_fn(row) that draws the B-spline (bold) over the raw skeleton (faint) + mask
    outline on the snip. ``paths_by_snip``: snip_id -> (snip_image_path, mask_path, pixel_size_um).

    ``mask_path`` is ``embryo_mask`` — the embryo mask already carried through snip_processing's
    exact crop+rotate into the SAME pixel frame as the snip image. The spine is computed on that
    snip-space mask, so its coordinates land directly on the snip: no offset, rotation, or resize
    (which is why the earlier full-frame-RLE-then-crop-resize path drew nothing — it ignored the
    rotation and re-skeletonized a different, edge-clipped mask)."""

    def image_fn(row: pd.Series) -> Image.Image | None:
        entry = paths_by_snip.get(row["snip_id"])
        if entry is None:
            return None
        snip_path, mask_path, pixel_size_um = entry
        if not Path(snip_path).exists() or not Path(mask_path).exists():
            return None
        mask = _load_snip_mask(mask_path)
        spline_xy, raw_xy = _geodesic_spine_layers(mask, pixel_size_um)
        return draw_centerline_on_snip(
            Path(snip_path), spline_xy, raw_centerline_xy_px=raw_xy, contour_mask=mask
        )

    return image_fn


def build_curvature_metrics_report(
    *,
    curvature_metrics_csv: Path,
    snip_inventory_csv: Path,
    frame_inventory_csv: Path,
    output_root: Path,
    output_feature_grid_png: Path,
    output_gallery_png: Path,
) -> list[Path]:
    curv = pd.read_csv(curvature_metrics_csv)
    if curv.empty:
        return []

    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)

    for output_png in (output_feature_grid_png, output_gallery_png):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    grid = plot_histogram_grid(
        curv,
        CURVATURE_FEATURE_COLUMNS,
        title="curvature_metrics — feature distributions",
        output_path=output_feature_grid_png,
    )

    resolved_paths = resolve_snip_inventory_image_and_mask_paths(snip_inventory, output_root=Path(output_root))
    frame_inventory_by_image = frame_inventory.set_index("image_id")
    resolved_by_snip = resolved_paths.set_index("snip_id")

    paths_by_snip: dict[str, tuple[str, str, float]] = {}
    for snip_id in curv["snip_id"]:
        if snip_id not in resolved_by_snip.index:
            continue
        snip_row = resolved_by_snip.loc[snip_id]
        image_id = snip_row["image_id"]
        snip_path = snip_row["resolved_image_path"]
        mask_path = snip_row["resolved_mask_path"]
        if pd.isna(snip_path) or pd.isna(mask_path) or image_id not in frame_inventory_by_image.index:
            continue
        pixel_size_um = pixel_size_for_image(frame_inventory_by_image, image_id, snip_id)
        paths_by_snip[snip_id] = (str(snip_path), str(mask_path), pixel_size_um)

    gallery = render_value_quartile_gallery_with_overlay(
        curv.merge(
            resolved_paths[["snip_id", "resolved_image_path"]], on="snip_id", how="left"
        ).dropna(subset=[RANK_COLUMN]),
        RANK_COLUMN,
        image_fn=_overlay_fn(paths_by_snip),
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="curvature_metrics — baseline_deviation_normalized quartiles (geodesic spine overlaid)",
        output_path=output_gallery_png,
    )
    return [grid, gallery]
