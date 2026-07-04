"""curvature_metrics report — feature histogram grid (D) + centerline-overlay gallery (E+overlay).

curvature_metrics is not yet materialized in the 20250912 tree, so this report RECOMPUTES the
per-snip geodesic centerline + curvature at render time from each snip's embryo mask — exactly the
pipeline's own compute path (smooth -> geodesic centerline -> orient -> summary), so what we plot is
what the step would persist. A legitimate report-only derivation: recomputable from the step's mask
input, shown only in the report, consumed by nothing.

Two artifacts:
- histogram grid over the curvature payload columns (length, mean curvature, baseline-deviation
  family, keypoints);
- a value-quartile gallery ranked by baseline_deviation_normalized (the headline curvature feature),
  with the SMOOTHED GEODESIC SPINE + mask outline drawn on each snip — the named ask. The overlay is
  the whole point: a curvature summary hides whether the spine it measured actually tracks the
  animal head-to-tail, so we let the eye check it.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.feature_extraction.curvature_metrics.compute import compute_curvature_for_mask
from data_pipeline.feature_extraction.curvature_metrics.geodesic_centerline import (
    extract_geodesic_centerline,
)
from data_pipeline.feature_extraction.curvature_metrics.head_tail_orientation import (
    orient_centerline_head_to_tail,
)
from data_pipeline.feature_extraction.curvature_metrics.mask_preprocessing import smooth_mask_boundary
from data_pipeline.viz.reporting import (
    draw_centerline_on_snip,
    plot_histogram_grid,
    render_value_quartile_gallery_with_overlay,
)

from .._loaders import (
    pixel_size_by_image_id,
    snip_image_paths,
    snip_mask_paths,
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


def _load_mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def _curvature_for_one(task: tuple[str, str, float]) -> dict | None:
    """Worker: (snip_id, mask_path, pixel_size_um) -> curvature payload dict (or None if unreadable).
    Module-level so it is picklable for ProcessPoolExecutor."""
    snip_id, mask_path, pixel_size_um = task
    if not Path(mask_path).exists():
        return None
    metrics = compute_curvature_for_mask(_load_mask(mask_path), pixel_size_um)
    metrics["snip_id"] = snip_id
    return metrics


def _recompute_curvature(masks: pd.DataFrame, pixel_size: pd.DataFrame) -> pd.DataFrame:
    """One row per snip: the curvature payload, recomputed from the snip's embryo mask.

    The geodesic centerline (Dijkstra + B-spline per mask) is CPU-bound, so this fans the per-mask
    work out across a process pool. Worker count follows ``CURVATURE_REPORT_WORKERS`` (set by the
    SGE submit script to the number of granted slots), defaulting to the local CPU count.
    """
    px = pixel_size.set_index("image_id")["pixel_size_um"]
    tasks = [
        (m.snip_id, m.resolved_mask_path, float(px.loc[m.image_id]))
        for m in masks.itertuples(index=False)
        if m.image_id in px.index
    ]

    n_workers = int(os.environ.get("CURVATURE_REPORT_WORKERS", os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        rows = [r for r in pool.map(_curvature_for_one, tasks, chunksize=8) if r is not None]
    return pd.DataFrame(rows)


def _geodesic_spine_layers(mask: np.ndarray, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoothed head-oriented B-spline, raw skeleton centerline) for the overlay.
    Either array is empty when the skeleton was too short to fit a spline."""
    smoothed = smooth_mask_boundary(mask)
    centerline = extract_geodesic_centerline(smoothed, pixel_size_um=pixel_size_um)
    if len(centerline.smoothed_xy) == 0:
        return np.empty((0, 2)), centerline.raw_xy
    spline_xy = orient_centerline_head_to_tail(centerline.smoothed_xy, smoothed)
    return spline_xy, centerline.raw_xy


def _overlay_fn(paths_by_snip: dict[str, tuple[str, str, float]]):
    """Return image_fn(row) that draws the B-spline (bold) over the raw skeleton (faint) + mask
    outline on the snip. ``paths_by_snip``: snip_id -> (snip_image_path, mask_path, pixel_size_um)."""

    def image_fn(row: pd.Series) -> Image.Image | None:
        entry = paths_by_snip.get(row["snip_id"])
        if entry is None:
            return None
        snip_path, mask_path, pixel_size_um = entry
        if not Path(snip_path).exists() or not Path(mask_path).exists():
            return None
        mask = _load_mask(mask_path)
        spline_xy, raw_xy = _geodesic_spine_layers(mask, pixel_size_um)
        return draw_centerline_on_snip(
            Path(snip_path), spline_xy, raw_centerline_xy_px=raw_xy, contour_mask=mask
        )

    return image_fn


# Artifacts this report emits (used by both the fresh build and the placeholder-reuse path).
_ARTIFACT_NAMES = ("curvature_feature_grid.png", "curvature_centerline_gallery.png")


def build(output_dir: Path, *, reuse_existing: bool = True) -> list[Path]:
    # PLACEHOLDER SEAM: curvature_metrics is not materialized in the 20250912 tree yet, so a full
    # build recomputes the geodesic centerline for every snip (~15k Dijkstra+B-spline fits — the
    # expensive step). To keep a whole-index rebuild cheap, reuse already-rendered PNGs when they
    # exist. This is a *placeholder*: once the curvature_metrics step actually writes its merged
    # output, this report should read that real output (like every other report reads its step's
    # merged artifact) instead of recomputing or reusing stale PNGs. Pass reuse_existing=False to
    # force a fresh recompute.
    existing = [output_dir / name for name in _ARTIFACT_NAMES]
    if reuse_existing and all(p.exists() for p in existing):
        print(f"[curvature_metrics] reusing {len(existing)} existing PNG(s) "
              f"(placeholder — real report should read materialized curvature_metrics output)")
        return existing

    masks = snip_mask_paths()
    snips = snip_image_paths()
    px = pixel_size_by_image_id()

    curv = _recompute_curvature(masks, px)
    if curv.empty:
        return []

    grid = plot_histogram_grid(
        curv,
        CURVATURE_FEATURE_COLUMNS,
        title="curvature_metrics — feature distributions (geodesic, recomputed from snip masks)",
        output_path=output_dir / "curvature_feature_grid.png",
    )

    px_by_image = px.set_index("image_id")["pixel_size_um"]
    snip_path_by_id = dict(zip(snips["snip_id"], snips["resolved_image_path"]))
    paths_by_snip = {
        m.snip_id: (snip_path_by_id[m.snip_id], m.resolved_mask_path, float(px_by_image[m.image_id]))
        for m in masks.itertuples(index=False)
        if m.snip_id in snip_path_by_id and m.image_id in px_by_image.index
    }

    gallery = render_value_quartile_gallery_with_overlay(
        curv.merge(snips, on="snip_id", how="left").dropna(subset=[RANK_COLUMN]),
        RANK_COLUMN,
        image_fn=_overlay_fn(paths_by_snip),
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="curvature_metrics — baseline_deviation_normalized quartiles (geodesic spine overlaid)",
        output_path=output_dir / "curvature_centerline_gallery.png",
    )
    return [grid, gallery]
