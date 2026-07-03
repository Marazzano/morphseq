"""surface_area_qc report — area-vs-stage scatter + stage-banded gallery (F+E). TERMINAL leaf.

surface_area_qc's threshold is a stage-interpolated reference band, not a scalar, so a plain
histogram is misleading — hence renderer F (metric-vs-reference) instead of C. Consumes
mask_geometry (area_um2), stage_predictions (predicted_stage_hpf), the SA-QC flag table, and the
packaged reference — all already surface_area_qc inputs. Consumed by nothing; imported by nothing
but its own tasks.py subcommand.
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

_SA_CONFIG = resolve_config()
SA_K_LOWER, SA_K_UPPER = _SA_CONFIG.k_lower, _SA_CONFIG.k_upper


def _resolve_snip_image_paths(snip_inventory: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """snip_id -> absolute processed-snip image path, resolved against output_root."""
    def _abs(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        p = Path(str(value))
        return str(p if p.is_absolute() else (output_root / p))

    resolved = snip_inventory[["snip_id", "processed_snip_path"]].copy()
    resolved["resolved_image_path"] = resolved["processed_snip_path"].map(_abs)
    return resolved[["snip_id", "resolved_image_path"]]


def _metrics(
    mask_geometry: pd.DataFrame, stage_predictions: pd.DataFrame, sa_qc: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
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


def build_surface_area_qc_report(
    *,
    surface_area_qc_csv: Path,
    mask_geometry_csv: Path,
    stage_predictions_csv: Path,
    snip_inventory_csv: Path,
    output_root: Path,
    output_vs_stage_png: Path,
    output_gallery_png: Path,
) -> list[Path]:
    spec = SURFACE_AREA_QC_REPORT_SPEC
    reference = load_packaged_surface_area_reference(version="v1")
    df = _metrics(
        pd.read_csv(mask_geometry_csv), pd.read_csv(stage_predictions_csv),
        pd.read_csv(surface_area_qc_csv), reference,
    ).merge(
        _resolve_snip_image_paths(pd.read_csv(snip_inventory_csv), Path(output_root)),
        on="snip_id", how="left",
    )

    for output_png in (output_vs_stage_png, output_gallery_png):
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    scatter = plot_metric_vs_reference(
        df, spec.x_col, spec.metric_col,
        fail_col=spec.flag_col,
        reference_df=_reference_curve(),
        reference_x_col=spec.reference_x_col,
        reference_lower_col=spec.reference_lower_col,
        reference_upper_col=spec.reference_upper_col,
        title=f"surface_area_qc (area_um2 vs stage, band = [{SA_K_LOWER:g}*p5, {SA_K_UPPER:g}*p95])",
        output_path=output_vs_stage_png,
        xlabel="predicted_stage_hpf", ylabel="area_um2",
    )
    gallery = render_quartile_gallery_vs_band(
        df, spec.metric_col, spec.lower_col, spec.upper_col,
        image_path_col="resolved_image_path", label_col="snip_id",
        title="surface_area_qc (area_um2 vs. per-row stage-banded threshold)",
        output_path=output_gallery_png,
    )
    return [scatter, gallery]
