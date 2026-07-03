"""Ad-hoc QC reporting v1: histogram + cutoff-relative quartile gallery per QC product.

Reads real per-well output for experiment 20250912 (no Snakemake wiring yet), merges each QC
product's metrics onto snip image paths, and writes PNGs under ./output/<qc_product>/.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260702_qc_reporting_v1/generate_qc_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data_pipeline.quality_control.reporting import (  # noqa: E402
    QC_REPORT_SPECS,
    SURFACE_AREA_QC_REPORT_SPEC,
)
from data_pipeline.quality_control.surface_area_qc.reference import (  # noqa: E402
    interpolate_reference_band,
    load_packaged_surface_area_reference,
)
from data_pipeline.viz.reporting import (  # noqa: E402
    plot_metric_histogram,
    plot_metric_vs_reference,
    render_quartile_gallery,
    render_quartile_gallery_vs_band,
)

EXPERIMENT_ID = "20250912"
DATA_ROOT = _REPO_ROOT / "data_pipeline_output"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# QC product -> on-disk per-well filename suffix, where it diverges from the product name.
QC_FILE_SUFFIX = {"death_detection": "death_detection_qc"}

# surface_area_qc canonical thresholds (must match src/data_pipeline/quality_control/
# surface_area_qc/config.py SURFACE_AREA_QC_DEFAULTS).
SA_K_LOWER, SA_K_UPPER = 0.7, 1.4


def _concat_per_well(stage: str, product: str, suffix: str) -> pd.DataFrame:
    root = DATA_ROOT / stage / EXPERIMENT_ID / product / "per_well"
    paths = sorted(root.glob(f"*/*_{suffix}.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-well {suffix} files found under {root}")
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)


def _load_snip_inventory() -> pd.DataFrame:
    root = DATA_ROOT / "object_extraction" / EXPERIMENT_ID / "snips" / "per_well"
    paths = sorted(root.glob("*/*_snip_inventory.csv"))
    inventory = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    inventory["resolved_image_path"] = inventory["processed_snip_path"].apply(
        lambda rel: str(DATA_ROOT / rel)
    )
    return inventory[["snip_id", "resolved_image_path"]]


def _build_surface_area_metrics() -> pd.DataFrame:
    mask_geometry = _concat_per_well("feature_extraction", "mask_geometry", "mask_geometry")
    stage_predictions = _concat_per_well("feature_extraction", "stage_predictions", "stage_predictions")
    sa_qc = _concat_per_well("quality_control", "surface_area_qc", "surface_area_qc")

    reference = load_packaged_surface_area_reference(version="v1")

    merged = sa_qc.merge(
        mask_geometry[["snip_id", "area_um2"]], on="snip_id", how="left"
    ).merge(
        stage_predictions[["snip_id", "predicted_stage_hpf"]], on="snip_id", how="left"
    )

    p5_list, p95_list = [], []
    for stage in merged["predicted_stage_hpf"]:
        p5, p95 = interpolate_reference_band(float(stage), reference)
        p5_list.append(p5)
        p95_list.append(p95)
    merged["sa_reference_p5"] = p5_list
    merged["sa_reference_p95"] = p95_list
    merged["sa_lower_threshold"] = SA_K_LOWER * merged["sa_reference_p5"]
    merged["sa_upper_threshold"] = SA_K_UPPER * merged["sa_reference_p95"]
    return merged


def _build_surface_area_reference_curve() -> pd.DataFrame:
    reference = load_packaged_surface_area_reference(version="v1").copy()
    reference["p5_scaled"] = SA_K_LOWER * reference["p5"]
    reference["p95_scaled"] = SA_K_UPPER * reference["p95"]
    return reference


def _build_death_detection_metrics() -> pd.DataFrame:
    qc = _concat_per_well("quality_control", "death_detection", "death_detection_qc")
    fraction_alive = _concat_per_well("feature_extraction", "fraction_alive", "fraction_alive")
    return qc.merge(fraction_alive[["snip_id", "fraction_alive"]], on="snip_id", how="left")


def _load_qc_metrics(qc_product: str) -> pd.DataFrame:
    if qc_product == "death_detection":
        return _build_death_detection_metrics()
    suffix = QC_FILE_SUFFIX.get(qc_product, qc_product)
    return _concat_per_well("quality_control", qc_product, suffix)


def _run_scalar_cutoff_products(snip_paths: pd.DataFrame) -> None:
    for qc_product, spec in QC_REPORT_SPECS.items():
        metrics = _load_qc_metrics(qc_product)
        merged = metrics.merge(snip_paths, on="snip_id", how="left")

        out_dir = OUTPUT_ROOT / qc_product
        plot_metric_histogram(
            merged[spec.metric_col],
            spec.cutoff,
            fail_direction=spec.fail_direction,
            title=f"{qc_product} ({spec.metric_label})",
            output_path=out_dir / f"{qc_product}_histogram.png",
            xlabel=spec.metric_label,
        )
        render_quartile_gallery(
            merged,
            spec.metric_col,
            spec.cutoff,
            fail_direction=spec.fail_direction,
            image_path_col="resolved_image_path",
            label_col="snip_id",
            title=f"{qc_product} ({spec.metric_label}, cutoff={spec.cutoff:g})",
            output_path=out_dir / f"{qc_product}_gallery.png",
        )
        print(f"[{qc_product}] wrote histogram + gallery to {out_dir}")


def _run_surface_area_qc(snip_paths: pd.DataFrame) -> None:
    spec = SURFACE_AREA_QC_REPORT_SPEC
    merged = _build_surface_area_metrics().merge(snip_paths, on="snip_id", how="left")
    reference_curve = _build_surface_area_reference_curve()

    out_dir = OUTPUT_ROOT / spec.qc_product
    plot_metric_vs_reference(
        merged,
        spec.x_col,
        spec.metric_col,
        fail_col=spec.flag_col,
        reference_df=reference_curve,
        reference_x_col=spec.reference_x_col,
        reference_lower_col=spec.reference_lower_col,
        reference_upper_col=spec.reference_upper_col,
        title=f"{spec.qc_product} ({spec.metric_col} vs {spec.x_col}, "
              f"band = [{SA_K_LOWER:g}*p5, {SA_K_UPPER:g}*p95])",
        output_path=out_dir / f"{spec.qc_product}_vs_stage.png",
        xlabel="predicted_stage_hpf",
        ylabel="area_um2",
    )
    render_quartile_gallery_vs_band(
        merged,
        spec.metric_col,
        spec.lower_col,
        spec.upper_col,
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title=f"{spec.qc_product} ({spec.metric_col} vs. per-row stage-banded threshold)",
        output_path=out_dir / f"{spec.qc_product}_gallery.png",
    )
    print(f"[{spec.qc_product}] wrote area-vs-stage plot + gallery to {out_dir}")


def main() -> None:
    snip_paths = _load_snip_inventory()
    _run_scalar_cutoff_products(snip_paths)
    _run_surface_area_qc(snip_paths)


if __name__ == "__main__":
    main()
