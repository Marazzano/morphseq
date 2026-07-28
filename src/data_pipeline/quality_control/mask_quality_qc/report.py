"""mask_quality_qc report — histograms + cutoff-relative galleries with snip overlays. TERMINAL leaf.

The report is intentionally report-only: it derives three continuous severity views from the
binary QC table and renders both a raw histogram and a cutoff-relative gallery for each flag.
Every gallery cell overlays the snip-space mask on top of the snip image so reviewers can inspect
the spatial failure mode directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import label

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_and_mask_paths
from data_pipeline.viz.reporting import (
    draw_mask_on_snip,
    plot_metric_histogram,
    render_quartile_gallery_with_overlay,
)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], *, label_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label_name}: missing required column(s): {', '.join(missing)}")


def _decode_mask_payload(mask_rle: object) -> np.ndarray:
    if isinstance(mask_rle, str):
        mask_rle = json.loads(str(mask_rle))
    return decode_binary_mask_rle(mask_rle)


def _edge_margin_px(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    bottom = int(mask.shape[0] - 1 - ys.max())
    left = int(xs.min())
    right = int(mask.shape[1] - 1 - xs.max())
    return float(min(top, bottom, left, right))


def _second_largest_component_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    labeled = label(mask)
    num_components = int(labeled.max())
    if num_components <= 1:
        return 0.0
    areas = sorted((int(np.sum(labeled == idx)) for idx in range(1, num_components + 1)), reverse=True)
    largest = areas[0]
    second_largest = areas[1] if len(areas) > 1 else 0
    return float(second_largest / largest) if largest > 0 else 0.0


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union else 0.0


def _build_report_frame(
    mask_quality_qc: pd.DataFrame,
    snip_inventory: pd.DataFrame,
    frame_masks: pd.DataFrame,
    *,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    _require_columns(mask_quality_qc, ("snip_id",), label_name="mask_quality_qc")
    _require_columns(
        snip_inventory,
        ("snip_id", "mask_id", "processed_snip_path", "image_id"),
        label_name="snip_inventory",
    )
    _require_columns(frame_masks, ("mask_id", "mask_rle"), label_name="frame_masks")

    mask_quality_qc = mask_quality_qc.copy()
    snip_inventory = snip_inventory.copy()
    frame_masks = frame_masks.copy()
    mask_quality_qc["snip_id"] = mask_quality_qc["snip_id"].astype("string")
    snip_inventory["snip_id"] = snip_inventory["snip_id"].astype("string")
    snip_inventory["mask_id"] = snip_inventory["mask_id"].astype("string")
    snip_inventory["image_id"] = snip_inventory["image_id"].astype("string")
    snip_inventory["physical_embryo_id"] = snip_inventory["physical_embryo_id"].astype("string")
    frame_masks["mask_id"] = frame_masks["mask_id"].astype("string")

    resolved_paths = resolve_snip_inventory_image_and_mask_paths(snip_inventory, output_root=output_root)
    df = mask_quality_qc.merge(
        snip_inventory[["snip_id", "mask_id"]],
        on="snip_id",
        how="left",
    ).merge(
        resolved_paths,
        on="snip_id",
        how="left",
    ).merge(
        frame_masks[["mask_id", "mask_rle"]],
        on="mask_id",
        how="left",
    )

    missing = df.loc[
        df["snip_id"].isna()
        | df["image_id"].isna()
        | df["physical_embryo_id"].isna()
        | df["mask_id"].isna()
        | df["resolved_image_path"].isna()
        | df["resolved_mask_path"].isna()
        | df["mask_rle"].isna(),
        "snip_id",
    ].head(10).tolist()
    if missing:
        raise ValueError(
            "mask_quality_qc_report: unable to resolve mask/image inputs for snip_id(s) "
            f"{missing}. The report requires merged snip_inventory, frame_masks, and readable "
            "processed snip images."
        )

    mask_cache: dict[str, np.ndarray] = {}
    overlay_cache: dict[str, np.ndarray] = {}
    edge_margin_px: dict[str, float] = {}
    second_component_fraction: dict[str, float] = {}
    max_overlap_iou: dict[str, float] = {}

    for image_id, group in df.groupby("image_id", sort=False):
        rows: list[tuple[str, str, np.ndarray]] = []
        for row in group.itertuples(index=False):
            snip_id = str(row.snip_id)
            if snip_id not in mask_cache:
                mask_cache[snip_id] = _decode_mask_payload(row.mask_rle)
            mask = mask_cache[snip_id]
            edge_margin_px[snip_id] = _edge_margin_px(mask)
            second_component_fraction[snip_id] = _second_largest_component_fraction(mask)
            rows.append((snip_id, str(row.physical_embryo_id), mask))

        for snip_id, physical_embryo_id, mask in rows:
            best = 0.0
            for other_snip_id, other_physical_embryo_id, other_mask in rows:
                if other_snip_id == snip_id or other_physical_embryo_id == physical_embryo_id:
                    continue
                best = max(best, _iou(mask, other_mask))
            max_overlap_iou[snip_id] = best

    df["edge_margin_px"] = df["snip_id"].map(edge_margin_px)
    df["second_largest_component_fraction"] = df["snip_id"].map(second_component_fraction)
    df["max_overlap_iou"] = df["snip_id"].map(max_overlap_iou).fillna(0.0)
    df["resolved_image_path"] = df["resolved_image_path"].map(lambda value: str(value))
    df["resolved_mask_path"] = df["resolved_mask_path"].map(lambda value: str(value))
    return df, {"canonical": mask_cache, "overlay": overlay_cache}


def _load_overlay_mask(mask_path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(mask_path).convert("L")) > 0


def _overlay_image_fn(overlay_cache: dict[str, np.ndarray]):
    def _draw(row):
        snip_id = str(row.snip_id)
        if snip_id not in overlay_cache:
            overlay_cache[snip_id] = _load_overlay_mask(Path(str(row.resolved_mask_path)))
        return draw_mask_on_snip(Path(str(row.resolved_image_path)), overlay_cache[snip_id])

    return _draw


def build_mask_quality_qc_report(
    *,
    mask_quality_qc_csv: Path,
    snip_inventory_csv: Path,
    frame_masks_csv: Path,
    output_root: Path,
    output_edge_flag_histogram_png: Path,
    output_edge_flag_gallery_png: Path,
    output_discontinuous_mask_flag_histogram_png: Path,
    output_discontinuous_mask_flag_gallery_png: Path,
    output_overlapping_mask_flag_histogram_png: Path,
    output_overlapping_mask_flag_gallery_png: Path,
) -> list[Path]:
    mask_quality_qc = pd.read_csv(mask_quality_qc_csv)
    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_masks = pd.read_csv(frame_masks_csv)
    report_df, mask_caches = _build_report_frame(
        mask_quality_qc,
        snip_inventory,
        frame_masks,
        output_root=Path(output_root),
    )

    for output_png in (
        output_edge_flag_histogram_png,
        output_edge_flag_gallery_png,
        output_discontinuous_mask_flag_histogram_png,
        output_discontinuous_mask_flag_gallery_png,
        output_overlapping_mask_flag_histogram_png,
        output_overlapping_mask_flag_gallery_png,
    ):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    overlay = _overlay_image_fn(mask_caches["overlay"])

    edge_histogram = plot_metric_histogram(
        report_df["edge_margin_px"],
        2.0,
        fail_direction="below",
        title="mask_quality_qc — edge_margin_px distribution",
        output_path=output_edge_flag_histogram_png,
        xlabel="edge_margin_px (px)",
    )
    edge_gallery = render_quartile_gallery_with_overlay(
        report_df,
        "edge_margin_px",
        2.0,
        fail_direction="below",
        image_fn=overlay,
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="mask_quality_qc — edge_flag gallery (fail if edge_margin_px < 2 px)",
        output_path=output_edge_flag_gallery_png,
    )
    discontinuous_histogram = plot_metric_histogram(
        report_df["second_largest_component_fraction"],
        0.05,
        fail_direction="above",
        title="mask_quality_qc — second_largest_component_fraction distribution",
        output_path=output_discontinuous_mask_flag_histogram_png,
        xlabel="second_largest_component_fraction",
    )
    discontinuous_gallery = render_quartile_gallery_with_overlay(
        report_df,
        "second_largest_component_fraction",
        0.05,
        fail_direction="above",
        image_fn=overlay,
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="mask_quality_qc — discontinuous_mask_flag gallery (fail if second_largest_component_fraction > 0.05)",
        output_path=output_discontinuous_mask_flag_gallery_png,
    )
    overlapping_histogram = plot_metric_histogram(
        report_df["max_overlap_iou"],
        0.10,
        fail_direction="above",
        title="mask_quality_qc — max_overlap_iou distribution",
        output_path=output_overlapping_mask_flag_histogram_png,
        xlabel="max_overlap_iou",
    )
    overlapping_gallery = render_quartile_gallery_with_overlay(
        report_df,
        "max_overlap_iou",
        0.10,
        fail_direction="above",
        image_fn=overlay,
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="mask_quality_qc — overlapping_mask_flag gallery (fail if max_overlap_iou > 0.10)",
        output_path=output_overlapping_mask_flag_gallery_png,
    )
    return [
        edge_histogram,
        edge_gallery,
        discontinuous_histogram,
        discontinuous_gallery,
        overlapping_histogram,
        overlapping_gallery,
    ]
