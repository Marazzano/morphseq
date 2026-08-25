"""Per-QC-product histogram + quartile-gallery report specs.

Each ``QC_REPORT_SPECS`` entry names the RAW metric column and the actual scalar cutoff that
judges it — the histogram plots the raw metric with the true cutoff value drawn on it, so axis
values mean what they say. Products without a persisted continuous metric (``mask_quality_qc``,
``snip_qc``) are deliberately omitted — see docs/data_pipeline/specs/target/specs/viz/
for why.

``surface_area_qc`` does NOT fit the single-scalar-cutoff shape: its real threshold is a
stage-interpolated reference band (``[k_lower*p5(stage), k_upper*p95(stage)]``), so a plain
histogram against one fixed cutoff is misleading. It gets its own spec
(``SURFACE_AREA_QC_REPORT_SPEC``) describing the covariate (stage), the per-row band columns, and
the reference curve, consumed by ``viz.reporting.plot_metric_vs_reference`` /
``render_quartile_gallery_vs_band`` instead of the scalar-cutoff functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QCReportSpec:
    qc_product: str
    flag_col: str
    metric_col: str
    cutoff: float
    fail_direction: Literal["below", "above"]
    metric_label: str


@dataclass(frozen=True)
class SurfaceAreaQCReportSpec:
    qc_product: str
    flag_col: str
    metric_col: str
    x_col: str
    lower_col: str
    upper_col: str
    reference_x_col: str
    reference_lower_col: str
    reference_upper_col: str


QC_REPORT_SPECS: dict[str, QCReportSpec] = {
    "death_detection": QCReportSpec(
        qc_product="death_detection",
        flag_col="viability_dead_flag",
        metric_col="fraction_alive",
        cutoff=0.90,
        fail_direction="below",
        metric_label="fraction_alive",
    ),
    "focus_qc": QCReportSpec(
        qc_product="focus_qc",
        flag_col="focus_flag",
        metric_col="interior_strong_edge_fraction",
        cutoff=0.50,
        fail_direction="below",
        metric_label="interior_strong_edge_fraction",
    ),
    "motion_blur_qc": QCReportSpec(
        qc_product="motion_blur_qc",
        flag_col="motion_blur_flag",
        metric_col="mask_pixel_bad_pair_frac",
        # Matches the QC gate: any bad z-pair at all is a fail.
        cutoff=0.0,
        fail_direction="above",
        metric_label="mask_pixel_bad_pair_frac",
    ),
}

SURFACE_AREA_QC_REPORT_SPEC = SurfaceAreaQCReportSpec(
    qc_product="surface_area_qc",
    flag_col="sa_outlier_flag",
    metric_col="area_um2",
    x_col="predicted_stage_hpf",
    lower_col="sa_lower_threshold",
    upper_col="sa_upper_threshold",
    reference_x_col="stage_hpf",
    reference_lower_col="p5_scaled",
    reference_upper_col="p95_scaled",
)
