"""Shared shape for scalar-cutoff QC products (focus_qc, motion_blur_qc): a metric histogram (C)
plus a cutoff-relative image gallery (E), both driven by the product's QC_REPORT_SPECS entry.

This is the common body those per-product report.py files delegate to — the product script just
names its spec and its stage/product/suffix. In the wired form the loader seam becomes artifact_path.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.quality_control.reporting import QC_REPORT_SPECS
from data_pipeline.viz.reporting import plot_metric_histogram, render_quartile_gallery

from ._loaders import merged, snip_image_paths

# QC product -> on-disk per-well filename suffix where it diverges from the product folder name.
_SUFFIX = {"death_detection": "death_detection_qc"}


def build_scalar_qc(qc_product: str, output_dir: Path, *, metrics_df=None) -> list[Path]:
    spec = QC_REPORT_SPECS[qc_product]
    if metrics_df is None:
        metrics_df = merged("quality_control", qc_product, _SUFFIX.get(qc_product, qc_product))
    merged_df = metrics_df.merge(snip_image_paths(), on="snip_id", how="left")

    hist = plot_metric_histogram(
        merged_df[spec.metric_col],
        spec.cutoff,
        fail_direction=spec.fail_direction,
        title=f"{qc_product} ({spec.metric_label})",
        output_path=output_dir / f"{qc_product}_histogram.png",
        xlabel=spec.metric_label,
    )
    gallery = render_quartile_gallery(
        merged_df,
        spec.metric_col,
        spec.cutoff,
        fail_direction=spec.fail_direction,
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title=f"{qc_product} ({spec.metric_label}, cutoff={spec.cutoff:g})",
        output_path=output_dir / f"{qc_product}_gallery.png",
    )
    return [hist, gallery]
