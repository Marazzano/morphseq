"""surface_area_qc report — area-vs-stage scatter + stage-banded gallery (F+E). TIER-1.

Terminal report. surface_area_qc's threshold is a stage-interpolated reference band, not a scalar,
so a plain histogram is misleading — hence renderer F (metric-vs-reference) instead of C. Consumes
mask_geometry (area_um2), stage_predictions (predicted_stage_hpf), the SA-QC flag table, and the
packaged reference — all already surface_area_qc inputs. Migrated verbatim from the v1 driver.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.quality_control.reporting import SURFACE_AREA_QC_REPORT_SPEC
from data_pipeline.quality_control.surface_area_qc.config import resolve_config
from data_pipeline.quality_control.surface_area_qc.reference import (
    interpolate_reference_band,
    load_packaged_surface_area_reference,
)
from data_pipeline.viz.reporting import plot_metric_vs_reference, render_quartile_gallery_vs_band

from .._loaders import merged, snip_image_paths

_SA_CONFIG = resolve_config()
SA_K_LOWER, SA_K_UPPER = _SA_CONFIG.k_lower, _SA_CONFIG.k_upper


def _metrics() -> pd.DataFrame:
    mask_geometry = merged("feature_extraction", "mask_geometry", "mask_geometry")
    stage_predictions = merged("feature_extraction", "stage_predictions", "stage_predictions")
    sa_qc = merged("quality_control", "surface_area_qc", "surface_area_qc")
    reference = load_packaged_surface_area_reference(version="v1")

    df = sa_qc.merge(mask_geometry[["snip_id", "area_um2"]], on="snip_id", how="left").merge(
        stage_predictions[["snip_id", "predicted_stage_hpf"]], on="snip_id", how="left"
    )
    band = [interpolate_reference_band(float(s), reference) for s in df["predicted_stage_hpf"]]
    df["sa_reference_p5"] = [p5 for p5, _ in band]
    df["sa_reference_p95"] = [p95 for _, p95 in band]
    df["sa_lower_threshold"] = SA_K_LOWER * df["sa_reference_p5"]
    df["sa_upper_threshold"] = SA_K_UPPER * df["sa_reference_p95"]
    return df


def _reference_curve() -> pd.DataFrame:
    ref = load_packaged_surface_area_reference(version="v1").copy()
    ref["p5_scaled"] = SA_K_LOWER * ref["p5"]
    ref["p95_scaled"] = SA_K_UPPER * ref["p95"]
    return ref


def build(output_dir: Path) -> list[Path]:
    spec = SURFACE_AREA_QC_REPORT_SPEC
    df = _metrics().merge(snip_image_paths(), on="snip_id", how="left")

    scatter = plot_metric_vs_reference(
        df, spec.x_col, spec.metric_col,
        fail_col=spec.flag_col,
        reference_df=_reference_curve(),
        reference_x_col=spec.reference_x_col,
        reference_lower_col=spec.reference_lower_col,
        reference_upper_col=spec.reference_upper_col,
        title=f"surface_area_qc (area_um2 vs stage, band = [{SA_K_LOWER:g}*p5, {SA_K_UPPER:g}*p95])",
        output_path=output_dir / "surface_area_qc_vs_stage.png",
        xlabel="predicted_stage_hpf", ylabel="area_um2",
    )
    gallery = render_quartile_gallery_vs_band(
        df, spec.metric_col, spec.lower_col, spec.upper_col,
        image_path_col="resolved_image_path", label_col="snip_id",
        title="surface_area_qc (area_um2 vs. per-row stage-banded threshold)",
        output_path=output_dir / "surface_area_qc_gallery.png",
    )
    return [scatter, gallery]
