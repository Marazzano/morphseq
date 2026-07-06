"""motion_blur_qc report — metric histogram + cutoff-relative gallery. TERMINAL leaf.

The report follows the same scalar-QC pattern as ``focus_qc``: a metric histogram plus a
quartile gallery around the actual decision threshold. The motion-blur metric is
``mask_pixel_bad_pair_frac`` and the configured fail direction is ``above`` with cutoff ``0.0``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_paths
from data_pipeline.quality_control.reporting import QC_REPORT_SPECS
from data_pipeline.viz.reporting import plot_metric_histogram, render_quartile_gallery


def build_motion_blur_qc_report(
    *,
    motion_blur_qc_csv: Path,
    snip_inventory_csv: Path,
    output_root: Path,
    output_histogram_png: Path,
    output_gallery_png: Path,
) -> list[Path]:
    spec = QC_REPORT_SPECS["motion_blur_qc"]
    df = pd.read_csv(motion_blur_qc_csv).merge(
        resolve_snip_inventory_image_paths(pd.read_csv(snip_inventory_csv), output_root=Path(output_root)),
        on="snip_id",
        how="left",
    )

    for output_png in (output_histogram_png, output_gallery_png):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    hist = plot_metric_histogram(
        df[spec.metric_col],
        spec.cutoff,
        fail_direction=spec.fail_direction,
        title=f"{spec.qc_product} ({spec.metric_label})",
        output_path=output_histogram_png,
        xlabel=spec.metric_label,
    )
    gallery = render_quartile_gallery(
        df,
        spec.metric_col,
        spec.cutoff,
        fail_direction=spec.fail_direction,
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title=f"{spec.qc_product} ({spec.metric_label}, cutoff={spec.cutoff:g})",
        output_path=output_gallery_png,
    )
    return [hist, gallery]
