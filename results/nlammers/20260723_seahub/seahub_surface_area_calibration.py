#!/usr/bin/env python
"""Estimate SeaHub effective pixel size from stage-matched wild-type surface area.

This is an analysis-only utility.  It reads the July 30 SeaHub reset quarantine,
recovers mask area in pixels from the known 7.8 um/px placeholder, and asks what
pixel size would align each reference-like SeaHub control with (a) the operational
packaged surface-area curve and (b) a stricter explicit-WT audit cohort.

The central identity is:

    area_um2 = area_px * pixel_size_um_per_px**2

so the per-embryo effective scale estimate is:

    sqrt(reference_p50_um2(stage) / observed_area_px)

The packaged curve's archived builder cohort is not clean WT because a universal
``control_flag`` admitted most source rows; the script therefore keeps the
operational and strict-WT estimates separate.  Neither is physical metrology.
No active pipeline products or pipeline code are modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

DEFAULT_PIPELINE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/"
    "stale_output/20260730_SeaHub_full_reset/active_pipeline"
)
DEFAULT_BUNDLE_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/"
    "stale/20260730_SeaHub_full_reset/bundle"
)
DEFAULT_REFERENCE_CSV = (
    REPO_ROOT
    / "src/data_pipeline/quality_control/surface_area_qc/references/"
    "surface_area_reference_v1.csv"
)
DEFAULT_OUTPUT_DIR = HERE / "outputs/surface_area_calibration"

REFERENCE_BUILD_SCRIPT = (
    REPO_ROOT
    / "src/data_pipeline/quality_control/generate_references/build_sa_reference.py"
)
REFERENCE_SOURCE_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/"
    "morphseq_playground/metadata/build04_output"
)
REFERENCE_ORIGINAL_CSV = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/metadata/sa_reference_curves.csv"
)

SA_K_LOWER = 0.9
SA_K_UPPER = 1.4
RNG_SEED = 20260730

# Independent audit of the 87 archived build04 inputs, using:
#   (genotype in {wik, ab, wik-ab, wik/ab, AB, WIK} OR phenotype == "wt")
#   AND no chemical perturbation AND use_embryo_flag
# then taking the raw median area_um2 within +/-0.25 hpf of each SeaHub stage.
# These are intentionally a compact, reviewable snapshot rather than a 1.9 GB
# re-scan every time the notebook is executed.
STRICT_WT_REFERENCE_STAGE_ROWS: tuple[tuple[float, int, float], ...] = (
    (12.0, 181, 468_565.6392),
    (14.0, 203, 476_034.2569),
    (15.0, 273, 500_940.1977),
    (18.0, 289, 608_000.6188),
    (24.0, 301, 785_964.5283),
    (36.0, 285, 935_560.6316),
    (48.0, 118, 1_090_671.7169),
    (72.0, 21, 1_033_776.9768),
    (96.0, 5, 1_068_921.6363),
)

STRICT_WT_REFERENCE_IMAGE_ROWS: tuple[tuple[float, str], ...] = (
    (
        12.0,
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/"
        "sam2_pipeline_files/raw_data_organized/20230615/images/20230615_A04/"
        "20230615_A04_ch00_t0005.jpg",
    ),
    (
        18.0,
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/"
        "sam2_pipeline_files/raw_data_organized/20230525/images/20230525_F01/"
        "20230525_F01_ch00_t0007.jpg",
    ),
    (
        24.0,
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/"
        "sam2_pipeline_files/raw_data_organized/20230615/images/20230615_F06/"
        "20230615_F06_ch00_t0031.jpg",
    ),
    (
        48.0,
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/"
        "sam2_pipeline_files/raw_data_organized/20260124/images/20260124_G09/"
        "20260124_G09_ch00_t0058.jpg",
    ),
    (
        72.0,
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/"
        "morphseq_playground/sam2_pipeline_files/raw_data_organized/20260213/"
        "images/20260213_A01/20260213_A01_ch00_t0136.jpg",
    ),
    (
        96.0,
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/"
        "morphseq_playground/sam2_pipeline_files/raw_data_organized/20260306/"
        "images/20260306_E12/20260306_E12_ch00_t0077.jpg",
    ),
)

REFERENCE_COHORT_AUDIT = {
    "source_rows": 724_816,
    "control_flag_true_rows": 724_816,
    "legacy_builder_mask_before_required_value_filter": 351_818,
    "strict_wt_mask_before_required_value_filter": 31_144,
    "legacy_builder_rows_after_required_value_filter": 285_844,
    "strict_wt_rows_after_required_value_filter": 22_958,
    "legacy_rows_admitted_only_by_control_flag": 262_886,
    "legacy_unique_embryos": 6_369,
}


@dataclass(frozen=True)
class AnalysisPaths:
    pipeline_root: Path = DEFAULT_PIPELINE_ROOT
    bundle_root: Path = DEFAULT_BUNDLE_ROOT
    reference_csv: Path = DEFAULT_REFERENCE_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _norm_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _interpolate_reference(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    stage = rows["predicted_stage_hpf"].to_numpy(dtype=float)
    ref_stage = reference["stage_hpf"].to_numpy(dtype=float)
    for percentile in ("p5", "p50", "p95"):
        rows[f"reference_{percentile}_um2"] = np.interp(
            stage, ref_stage, reference[percentile].to_numpy(dtype=float)
        )
    return rows


def _classify_controls(rows: pd.DataFrame) -> pd.DataFrame:
    genotype = _norm_text(rows["genotype"])
    chemical = _norm_text(rows["chem_perturbation"])
    domain = _norm_text(rows["perturbation_domain"])

    genetic_control = domain.eq("genetic") & genotype.eq("ctrl-inj")
    chemical_vehicle = (
        domain.eq("chemical")
        & genotype.eq("ctrl")
        & chemical.isin({"ctrl", "dmso", "dmso_24hpf"})
    )
    nominal_temperature_control = (
        domain.eq("chemical")
        & genotype.eq("ctrl")
        & chemical.isin({"control_28c", "control_at_24hpf_28c"})
    )
    nonstandard_temperature_control = (
        domain.eq("chemical")
        & genotype.eq("ctrl")
        & chemical.isin({"control_24c", "control_34c"})
    )

    rows["control_class"] = "not_reference_control"
    rows.loc[genetic_control, "control_class"] = "genetic ctrl-inj"
    rows.loc[chemical_vehicle, "control_class"] = "chemical vehicle/untreated"
    rows.loc[nominal_temperature_control, "control_class"] = "nominal 28C control"
    rows.loc[
        nonstandard_temperature_control, "control_class"
    ] = "nonstandard-temperature control"

    rows["is_genetic_control"] = genetic_control
    rows["is_chemical_vehicle_control"] = chemical_vehicle
    rows["is_nominal_temperature_control"] = nominal_temperature_control
    rows["is_nonstandard_temperature_control"] = nonstandard_temperature_control
    rows["is_wt_control_strict"] = (
        genetic_control | chemical_vehicle | nominal_temperature_control
    )
    rows["is_wt_control_broad"] = (
        rows["is_wt_control_strict"] | nonstandard_temperature_control
    )
    return rows


def _resolve_snip_path(value: object, pipeline_root: Path) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = pipeline_root / path
    return str(path)


def load_seahub_comparison_rows(paths: AnalysisPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all SeaHub rows that reached merged mask geometry and stage prediction."""
    _require_dir(paths.pipeline_root, "quarantined pipeline root")
    _require_dir(paths.bundle_root, "quarantined materialization bundle")
    _require_file(paths.reference_csv, "packaged surface-area reference")

    reference = pd.read_csv(paths.reference_csv)
    mask_files = sorted(
        paths.pipeline_root.glob(
            "feature_extraction/*/mask_geometry/*_mask_geometry.csv"
        )
    )
    if not mask_files:
        raise RuntimeError(
            f"No merged mask_geometry files found under {paths.pipeline_root}"
        )

    pieces: list[pd.DataFrame] = []
    input_rows: list[dict[str, object]] = []
    experiments_root = paths.bundle_root / "experiments"

    for mask_path in mask_files:
        experiment_id = mask_path.parents[1].name
        stage_path = (
            paths.pipeline_root
            / "feature_extraction"
            / experiment_id
            / "stage_predictions"
            / f"{experiment_id}_stage_predictions.csv"
        )
        metadata_path = experiments_root / experiment_id / "plate_metadata.csv"
        snip_path = (
            paths.pipeline_root
            / "object_extraction"
            / experiment_id
            / "snips"
            / f"{experiment_id}_snip_inventory.csv"
        )
        _require_file(stage_path, f"{experiment_id} stage predictions")
        _require_file(metadata_path, f"{experiment_id} plate metadata")

        geometry = pd.read_csv(mask_path)
        stages = pd.read_csv(stage_path)[["snip_id", "predicted_stage_hpf"]]
        metadata_columns = [
            "well_id",
            "genotype",
            "chem_perturbation",
            "perturbation",
            "perturbation_key",
            "perturbation_domain",
            "source_scope",
            "source_experiment_id",
            "source_fov_id",
            "source_embryo_id",
            "fov_label",
            "fov_position",
            "source_filename",
            "source_relative_path",
            "stage_hpf",
            "stage_source_label",
            "micrometers_per_pixel",
            "calibration_status",
            "calibration_issue",
        ]
        metadata = pd.read_csv(metadata_path)[metadata_columns]

        merged = geometry.merge(
            stages, on="snip_id", how="inner", validate="one_to_one"
        ).merge(metadata, on="well_id", how="inner", validate="many_to_one")

        if snip_path.is_file():
            snips = pd.read_csv(snip_path)[
                ["snip_id", "processed_snip_path", "is_valid_snip"]
            ]
            merged = merged.merge(
                snips, on="snip_id", how="left", validate="one_to_one"
            )
            merged["resolved_processed_snip_path"] = merged[
                "processed_snip_path"
            ].map(lambda value: _resolve_snip_path(value, paths.pipeline_root))
        else:
            merged["processed_snip_path"] = ""
            merged["is_valid_snip"] = pd.NA
            merged["resolved_processed_snip_path"] = ""

        pieces.append(merged)
        input_rows.append(
            {
                "experiment_id": experiment_id,
                "mask_geometry_csv": str(mask_path),
                "stage_predictions_csv": str(stage_path),
                "plate_metadata_csv": str(metadata_path),
                "snip_inventory_csv": str(snip_path) if snip_path.is_file() else "",
                "n_geometry_rows": len(geometry),
                "n_joined_rows": len(merged),
                "n_wells": merged["well_id"].nunique(),
            }
        )

    rows = pd.concat(pieces, ignore_index=True)
    rows = _interpolate_reference(rows, reference)
    rows = _classify_controls(rows)

    calibration = pd.to_numeric(rows["micrometers_per_pixel"], errors="coerce")
    if calibration.isna().any() or (calibration <= 0).any():
        raise ValueError("Cannot recover area_px: missing/non-positive placeholder calibration.")
    rows["area_px"] = rows["area_um2"] / calibration.pow(2)
    rows["effective_um_per_px_to_reference_p50"] = np.sqrt(
        rows["reference_p50_um2"] / rows["area_px"]
    )
    rows["effective_um_per_px_to_reference_p5"] = np.sqrt(
        rows["reference_p5_um2"] / rows["area_px"]
    )
    rows["effective_um_per_px_to_reference_p95"] = np.sqrt(
        rows["reference_p95_um2"] / rows["area_px"]
    )
    strict_lookup = strict_wt_reference_summary().set_index("stage_hpf")[
        "strict_wt_p50_um2"
    ]
    rows["strict_wt_reference_p50_um2"] = rows["predicted_stage_hpf"].map(
        strict_lookup
    )
    if rows["strict_wt_reference_p50_um2"].isna().any():
        missing_stages = sorted(
            rows.loc[
                rows["strict_wt_reference_p50_um2"].isna(),
                "predicted_stage_hpf",
            ].unique()
        )
        raise ValueError(
            f"Strict-WT reference snapshot lacks SeaHub stage(s): {missing_stages}"
        )
    rows["effective_um_per_px_to_strict_wt_p50"] = np.sqrt(
        rows["strict_wt_reference_p50_um2"] / rows["area_px"]
    )

    rows["is_primary_mask"] = rows["physical_embryo_id"].astype(str).str.endswith(
        "_e01"
    )
    rows["is_largest_mask_for_well"] = False
    largest_indices = rows.groupby("well_id", sort=False)["area_px"].idxmax()
    rows.loc[largest_indices, "is_largest_mask_for_well"] = True

    source_fov = rows["source_fov_id"].fillna("").astype(str)
    source_filename = rows["source_filename"].fillna("").astype(str)
    rows["calibration_fov_id"] = source_fov.where(
        source_fov.str.len() > 0, source_filename
    )
    missing_fov = rows["calibration_fov_id"].str.len().eq(0)
    rows.loc[missing_fov, "calibration_fov_id"] = (
        rows.loc[missing_fov, "source_experiment_id"].astype(str)
        + "::"
        + rows.loc[missing_fov, "well_id"].astype(str)
    )

    rows["placeholder_too_small"] = (
        rows["area_um2"] < SA_K_LOWER * rows["reference_p5_um2"]
    )
    rows["placeholder_too_large"] = (
        rows["area_um2"] > SA_K_UPPER * rows["reference_p95_um2"]
    )
    rows["placeholder_sa_outlier"] = (
        rows["placeholder_too_small"] | rows["placeholder_too_large"]
    )

    input_manifest = pd.DataFrame(input_rows).sort_values("experiment_id")
    return rows, input_manifest


def _bootstrap_median(
    values: np.ndarray, *, repeats: int = 5000, seed: int = RNG_SEED
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]).astype(float))


def strict_wt_reference_summary() -> pd.DataFrame:
    """Return the compact strict-WT stage-window audit snapshot."""
    return pd.DataFrame(
        STRICT_WT_REFERENCE_STAGE_ROWS,
        columns=["stage_hpf", "strict_wt_reference_n", "strict_wt_p50_um2"],
    )


def reference_cohort_audit() -> pd.DataFrame:
    """Return a one-row audit of the legacy reference builder cohort."""
    audit = dict(REFERENCE_COHORT_AUDIT)
    legacy_final = audit["legacy_builder_rows_after_required_value_filter"]
    strict_final = audit["strict_wt_rows_after_required_value_filter"]
    admitted_only = audit["legacy_rows_admitted_only_by_control_flag"]
    audit["strict_wt_fraction_of_legacy_final"] = strict_final / legacy_final
    audit["control_flag_only_fraction_of_legacy_final"] = admitted_only / legacy_final
    audit["mean_frames_per_legacy_embryo"] = (
        legacy_final / audit["legacy_unique_embryos"]
    )
    return pd.DataFrame([audit])


def strict_wt_reference_image_manifest() -> pd.DataFrame:
    """Return existence-checked strict-WT raw-image examples for the montage."""
    rows: list[dict[str, object]] = []
    for stage_hpf, path_text in STRICT_WT_REFERENCE_IMAGE_ROWS:
        path = Path(path_text)
        width = np.nan
        height = np.nan
        mode = ""
        if path.is_file():
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        rows.append(
            {
                "stage_hpf": stage_hpf,
                "reference_image_path": str(path),
                "exists": path.is_file(),
                "width_px": width,
                "height_px": height,
                "mode": mode,
                "selection_filter": (
                    "explicit WT genotype or phenotype=wt; no chemical "
                    "perturbation; use_embryo_flag=True; within +/-0.25 hpf"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_tables(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build FOV-, stage-, regime-, and sensitivity-level calibration summaries."""
    selected = rows[
        rows["is_wt_control_strict"] & rows["is_largest_mask_for_well"]
    ].copy()
    if selected.empty:
        raise RuntimeError("No strict reference-like controls reached mask geometry.")

    fov_summary = (
        selected.groupby(
            [
                "source_experiment_id",
                "predicted_stage_hpf",
                "calibration_fov_id",
                "source_filename",
                "control_class",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_embryos=("well_id", "nunique"),
            median_area_px=("area_px", "median"),
            median_effective_um_per_px=(
                "effective_um_per_px_to_reference_p50",
                "median",
            ),
            median_strict_wt_effective_um_per_px=(
                "effective_um_per_px_to_strict_wt_p50",
                "median",
            ),
            q25_effective_um_per_px=(
                "effective_um_per_px_to_reference_p50",
                lambda values: values.quantile(0.25),
            ),
            q75_effective_um_per_px=(
                "effective_um_per_px_to_reference_p50",
                lambda values: values.quantile(0.75),
            ),
        )
        .sort_values(["predicted_stage_hpf", "source_experiment_id", "source_filename"])
    )

    stage_records: list[dict[str, object]] = []
    for stage_hpf, stage_fovs in fov_summary.groupby(
        "predicted_stage_hpf", sort=True
    ):
        fov_values = stage_fovs["median_effective_um_per_px"].to_numpy(dtype=float)
        stage_rows = selected[selected["predicted_stage_hpf"].eq(stage_hpf)]
        if len(fov_values) >= 2:
            ci_low, ci_high = _bootstrap_median(
                fov_values, seed=RNG_SEED + int(round(stage_hpf * 10))
            )
            ci_basis = "FOV-cluster bootstrap"
        else:
            ci_low, ci_high = _bootstrap_median(
                stage_rows["effective_um_per_px_to_reference_p50"].to_numpy(
                    dtype=float
                ),
                seed=RNG_SEED + int(round(stage_hpf * 10)),
            )
            ci_basis = "within-FOV embryo bootstrap (only one FOV)"
        stage_records.append(
            {
                "stage_hpf": float(stage_hpf),
                "n_embryos": int(stage_rows["well_id"].nunique()),
                "n_fovs": int(stage_fovs["calibration_fov_id"].nunique()),
                "recommended_um_per_px": float(np.median(fov_values)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "fov_q25": float(np.quantile(fov_values, 0.25)),
                "fov_q75": float(np.quantile(fov_values, 0.75)),
                "ci_basis": ci_basis,
                "reference_p50_um2": float(stage_rows["reference_p50_um2"].median()),
                "median_control_area_px": float(stage_rows["area_px"].median()),
            }
        )
    stage_summary = pd.DataFrame(stage_records).merge(
        strict_wt_reference_summary(), on="stage_hpf", how="left", validate="one_to_one"
    )
    # The packaged curve is the operational comparator.  The strict-WT stage
    # snapshot is preferable for biological interpretation.  Both share the
    # same observed SeaHub pixel areas, so their effective scale differs by the
    # square root of the reference-area ratio.
    stage_summary = stage_summary.rename(
        columns={
            "recommended_um_per_px": "legacy_curve_effective_um_per_px",
            "bootstrap_ci95_low": "legacy_curve_bootstrap_ci95_low",
            "bootstrap_ci95_high": "legacy_curve_bootstrap_ci95_high",
        }
    )
    strict_ratio = np.sqrt(
        stage_summary["strict_wt_p50_um2"]
        / stage_summary["reference_p50_um2"]
    )
    stage_summary["strict_wt_effective_um_per_px"] = (
        stage_summary["legacy_curve_effective_um_per_px"] * strict_ratio
    )
    stage_summary["strict_wt_bootstrap_ci95_low"] = (
        stage_summary["legacy_curve_bootstrap_ci95_low"] * strict_ratio
    )
    stage_summary["strict_wt_bootstrap_ci95_high"] = (
        stage_summary["legacy_curve_bootstrap_ci95_high"] * strict_ratio
    )
    stage_summary["strict_vs_legacy_scale_ratio"] = strict_ratio
    # Preferred for a provisional biological calibration; this remains an
    # effective scale estimate, not physical metrology.
    stage_summary["recommended_um_per_px"] = stage_summary[
        "strict_wt_effective_um_per_px"
    ]

    def regime(stage: float) -> str:
        if stage <= 24:
            return "early (12-24 hpf)"
        if stage < 48:
            return "transition (36 hpf)"
        return "late (48-96 hpf)"

    stage_summary["capture_regime"] = stage_summary["stage_hpf"].map(regime)
    regime_summary = (
        stage_summary.groupby("capture_regime", sort=False, as_index=False)
        .agg(
            stage_range_hpf=(
                "stage_hpf",
                lambda values: f"{values.min():g}-{values.max():g}",
            ),
            n_stages=("stage_hpf", "nunique"),
            n_fovs=("n_fovs", "sum"),
            n_embryos=("n_embryos", "sum"),
            strict_wt_stage_balanced_um_per_px=(
                "strict_wt_effective_um_per_px",
                "median",
            ),
            strict_wt_stage_min_um_per_px=(
                "strict_wt_effective_um_per_px",
                "min",
            ),
            strict_wt_stage_max_um_per_px=(
                "strict_wt_effective_um_per_px",
                "max",
            ),
            legacy_curve_stage_balanced_um_per_px=(
                "legacy_curve_effective_um_per_px",
                "median",
            ),
        )
    )

    cohort_masks = {
        "chemical vehicle/untreated only": rows["is_chemical_vehicle_control"],
        "genetic ctrl-inj only": rows["is_genetic_control"],
        "nominal 28C control only": rows["is_nominal_temperature_control"],
        "strict reference-like controls": rows["is_wt_control_strict"],
        "broad controls (+24C/+34C)": rows["is_wt_control_broad"],
    }
    mask_strategies = {
        "largest mask per intended well": rows["is_largest_mask_for_well"],
        "e01 mask per intended well": rows["is_primary_mask"],
        "all masks": pd.Series(True, index=rows.index),
    }
    sensitivity_records: list[dict[str, object]] = []
    for cohort_name, cohort_mask in cohort_masks.items():
        for strategy_name, strategy_mask in mask_strategies.items():
            subset = rows[cohort_mask & strategy_mask].copy()
            if subset.empty:
                continue
            per_fov = subset.groupby(
                ["predicted_stage_hpf", "calibration_fov_id"], dropna=False
            )["effective_um_per_px_to_reference_p50"].median()
            per_stage = per_fov.groupby(level=0).median()
            sensitivity_records.append(
                {
                    "control_cohort": cohort_name,
                    "mask_selection": strategy_name,
                    "n_rows": len(subset),
                    "n_wells": subset["well_id"].nunique(),
                    "n_fovs": subset["calibration_fov_id"].nunique(),
                    "n_stages": subset["predicted_stage_hpf"].nunique(),
                    "row_weighted_median_um_per_px": subset[
                        "effective_um_per_px_to_reference_p50"
                    ].median(),
                    "fov_weighted_median_um_per_px": per_fov.median(),
                    "stage_balanced_median_um_per_px": per_stage.median(),
                    "row_q25_um_per_px": subset[
                        "effective_um_per_px_to_reference_p50"
                    ].quantile(0.25),
                    "row_q75_um_per_px": subset[
                        "effective_um_per_px_to_reference_p50"
                    ].quantile(0.75),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_records)
    return fov_summary, stage_summary, regime_summary, sensitivity


def project_sa_flags(
    rows: pd.DataFrame, stage_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project current-QC flags using calibration to its packaged legacy curve."""
    intended = rows[rows["is_largest_mask_for_well"]].copy()
    scale_lookup = stage_summary.set_index("stage_hpf")[
        "legacy_curve_effective_um_per_px"
    ]
    intended["stage_calibrated_um_per_px"] = intended[
        "predicted_stage_hpf"
    ].map(scale_lookup)
    if intended["stage_calibrated_um_per_px"].isna().any():
        missing = sorted(
            intended.loc[
                intended["stage_calibrated_um_per_px"].isna(),
                "predicted_stage_hpf",
            ].unique()
        )
        raise ValueError(f"No control-derived scale for stage(s): {missing}")

    intended["stage_calibrated_area_um2"] = intended["area_px"] * intended[
        "stage_calibrated_um_per_px"
    ].pow(2)
    intended["stage_calibrated_too_small"] = (
        intended["stage_calibrated_area_um2"]
        < SA_K_LOWER * intended["reference_p5_um2"]
    )
    intended["stage_calibrated_too_large"] = (
        intended["stage_calibrated_area_um2"]
        > SA_K_UPPER * intended["reference_p95_um2"]
    )
    intended["stage_calibrated_sa_outlier"] = (
        intended["stage_calibrated_too_small"]
        | intended["stage_calibrated_too_large"]
    )

    cohort_masks = {
        "all intended embryos": pd.Series(True, index=intended.index),
        "strict WT controls": intended["is_wt_control_strict"],
        "non-control embryos": ~intended["is_wt_control_strict"],
    }
    records: list[dict[str, object]] = []
    for cohort_name, cohort_mask in cohort_masks.items():
        subset = intended[cohort_mask]
        for calibration_name, prefix in (
            ("7.8 placeholder", "placeholder"),
            ("stage-matched to legacy curve", "stage_calibrated"),
        ):
            records.append(
                {
                    "cohort": cohort_name,
                    "calibration": calibration_name,
                    "n": len(subset),
                    "n_outlier": int(subset[f"{prefix}_sa_outlier"].sum()),
                    "outlier_fraction": float(
                        subset[f"{prefix}_sa_outlier"].mean()
                    ),
                    "too_small_fraction": float(
                        subset[f"{prefix}_too_small"].mean()
                    ),
                    "too_large_fraction": float(
                        subset[f"{prefix}_too_large"].mean()
                    ),
                }
            )
    return intended, pd.DataFrame(records)


def _configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_effective_scale(
    fov_summary: pd.DataFrame, stage_summary: pd.DataFrame, output_path: Path
) -> None:
    palette = {
        "genetic ctrl-inj": "#4472A8",
        "chemical vehicle/untreated": "#D9822B",
        "nominal 28C control": "#4C956C",
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.stripplot(
        data=fov_summary,
        x="predicted_stage_hpf",
        y="median_strict_wt_effective_um_per_px",
        hue="control_class",
        palette=palette,
        dodge=False,
        jitter=0.13,
        alpha=0.8,
        size=8,
        ax=ax,
    )
    x_positions = {
        stage: position
        for position, stage in enumerate(sorted(stage_summary["stage_hpf"]))
    }
    x = np.array([x_positions[value] for value in stage_summary["stage_hpf"]])
    y = stage_summary["strict_wt_effective_um_per_px"].to_numpy()
    yerr = np.vstack(
        [
            y - stage_summary["strict_wt_bootstrap_ci95_low"].to_numpy(),
            stage_summary["strict_wt_bootstrap_ci95_high"].to_numpy() - y,
        ]
    )
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="D",
        color="black",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=8,
        linewidth=2,
        capsize=4,
        label="strict-WT biological estimate",
        zorder=10,
    )
    ax.scatter(
        x,
        stage_summary["legacy_curve_effective_um_per_px"],
        marker="o",
        s=75,
        facecolor="white",
        edgecolor="#666666",
        linewidth=1.8,
        label="operational estimate vs legacy curve",
        zorder=9,
    )
    placeholder = 7.8
    ax.axhline(
        placeholder,
        color="#9E3D3F",
        linestyle="--",
        linewidth=2,
        label="current placeholder (7.8)",
    )
    ax.set(
        xlabel="Nominal developmental stage (hpf)",
        ylabel="Effective pixel size (µm/px)",
        title="SeaHub effective scale depends strongly on capture stage",
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        title="Control evidence",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_area_comparison(
    intended: pd.DataFrame, reference: pd.DataFrame, output_path: Path
) -> None:
    controls = intended[intended["is_wt_control_strict"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    stage_min = controls["predicted_stage_hpf"].min() - 2
    stage_max = controls["predicted_stage_hpf"].max() + 3
    ref = reference[
        reference["stage_hpf"].between(stage_min, stage_max)
    ].copy()

    panels = [
        ("area_um2", "Current 7.8 µm/px placeholder"),
        ("stage_calibrated_area_um2", "Stage-matched effective calibration"),
    ]
    for ax, (area_column, title) in zip(axes, panels):
        ax.fill_between(
            ref["stage_hpf"],
            SA_K_LOWER * ref["p5"],
            SA_K_UPPER * ref["p95"],
            color="#B7D7C2",
            alpha=0.45,
            label="QC inclusion band",
        )
        ax.plot(
            ref["stage_hpf"],
            ref["p50"],
            color="#295F4E",
            linewidth=2.5,
            label="legacy reference median",
        )
        sns.stripplot(
            data=controls,
            x="predicted_stage_hpf",
            y=area_column,
            hue="control_class",
            dodge=False,
            jitter=0.22,
            alpha=0.4,
            size=4,
            ax=ax,
            legend=False,
            native_scale=True,
        )
        ax.set_yscale("log")
        ax.set(
            xlabel="Nominal developmental stage (hpf)",
            ylabel="Mask area (µm²)",
            title=title,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc="upper left")
    fig.suptitle(
        "SeaHub controls versus the operational legacy SA curve (not a clean WT cohort)",
        y=1.03,
        fontsize=19,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_qc_projection(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.copy()
    plot_data["outlier_percent"] = 100 * plot_data["outlier_fraction"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_data,
        x="cohort",
        y="outlier_percent",
        hue="calibration",
        palette=["#A85B5B", "#4C956C"],
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=11)
    ax.set(
        xlabel="",
        ylabel="Surface-area outliers (%)",
        title="Illustrative SA-QC impact of stage-matched calibration",
    )
    ax.legend(title="", frameon=False)
    ax.set_ylim(0, max(5, plot_data["outlier_percent"].max() * 1.18))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _representative_rows(
    selected: pd.DataFrame, stage_summary: pd.DataFrame
) -> pd.DataFrame:
    targets = stage_summary.set_index("stage_hpf")["recommended_um_per_px"]
    selected = selected.copy()
    selected["target_scale"] = selected["predicted_stage_hpf"].map(targets)
    selected["distance_to_stage_estimate"] = (
        selected["effective_um_per_px_to_strict_wt_p50"]
        - selected["target_scale"]
    ).abs()
    valid_path = selected["resolved_processed_snip_path"].map(
        lambda value: bool(value) and Path(value).is_file()
    )
    selected = selected[valid_path]
    if selected.empty:
        return selected
    indices = selected.groupby("predicted_stage_hpf")[
        "distance_to_stage_estimate"
    ].idxmin()
    return selected.loc[indices].sort_values("predicted_stage_hpf")


def plot_representative_snips(
    selected: pd.DataFrame, stage_summary: pd.DataFrame, output_path: Path
) -> None:
    representatives = _representative_rows(selected, stage_summary)
    if representatives.empty:
        return
    n = len(representatives)
    columns = 3
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, representatives.iterrows()):
        image = Image.open(row["resolved_processed_snip_path"])
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"{row['predicted_stage_hpf']:g} hpf | "
            f"{row['effective_um_per_px_to_strict_wt_p50']:.2f} µm/px\n"
            f"{row['control_class']}",
            fontsize=11,
        )
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "Representative WT-control snips nearest each stage estimate",
        fontsize=18,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _show_grayscale_with_robust_contrast(ax: plt.Axes, image: Image.Image) -> None:
    array = np.asarray(image.convert("L"))
    finite = array[np.isfinite(array)]
    if finite.size:
        vmin, vmax = np.quantile(finite, [0.01, 0.995])
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = None, None
    ax.imshow(array, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")


def plot_seahub_vs_reference_images(
    selected: pd.DataFrame,
    stage_summary: pd.DataFrame,
    reference_images: pd.DataFrame,
    output_path: Path,
) -> None:
    """Show actual stage-matched SeaHub and strict-WT reference frames."""
    stages = reference_images.loc[reference_images["exists"], "stage_hpf"].tolist()
    representatives = _representative_rows(selected, stage_summary)
    representatives = representatives[
        representatives["predicted_stage_hpf"].isin(stages)
    ].set_index("predicted_stage_hpf")
    reference_images = reference_images[
        reference_images["stage_hpf"].isin(representatives.index)
    ].set_index("stage_hpf")
    if representatives.empty:
        return

    n = len(representatives)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.5 * n))
    if n == 1:
        axes = np.asarray([axes])
    for row_index, stage_hpf in enumerate(sorted(representatives.index)):
        seahub_row = representatives.loc[stage_hpf]
        reference_row = reference_images.loc[stage_hpf]
        seahub_path = Path(seahub_row["resolved_processed_snip_path"])
        reference_path = Path(reference_row["reference_image_path"])

        with Image.open(seahub_path) as image:
            _show_grayscale_with_robust_contrast(axes[row_index, 0], image)
            sea_width, sea_height = image.size
        with Image.open(reference_path) as image:
            _show_grayscale_with_robust_contrast(axes[row_index, 1], image)
            ref_width, ref_height = image.size

        axes[row_index, 0].set_title(
            f"SeaHub control, {stage_hpf:g} hpf\n"
            f"pipeline snip {sea_width}×{sea_height} px; "
            f"effective {seahub_row['effective_um_per_px_to_strict_wt_p50']:.2f} µm/px",
            fontsize=11,
        )
        axes[row_index, 1].set_title(
            f"Strict-WT reference example, {stage_hpf:g} hpf\n"
            f"raw frame {ref_width}×{ref_height} px",
            fontsize=11,
        )
    fig.suptitle(
        "Stage-matched image evidence (independent display scaling; no common scale bar)",
        fontsize=17,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_provenance(paths: AnalysisPaths, input_manifest: pd.DataFrame) -> pd.DataFrame:
    source_files = (
        sorted(REFERENCE_SOURCE_DIR.glob("qc_staged_*.csv"))
        if REFERENCE_SOURCE_DIR.is_dir()
        else []
    )
    records = [
        {
            "asset": "packaged_reference_used_by_QC",
            "path": str(paths.reference_csv),
            "exists": paths.reference_csv.is_file(),
            "detail": "Authoritative v1 p5/p50/p95 stage curve used in this analysis.",
        },
        {
            "asset": "original_reference_curve",
            "path": str(REFERENCE_ORIGINAL_CSV),
            "exists": REFERENCE_ORIGINAL_CSV.is_file(),
            "detail": "README states v1 was copied verbatim from this file.",
        },
        {
            "asset": "reference_build_script",
            "path": str(REFERENCE_BUILD_SCRIPT),
            "exists": REFERENCE_BUILD_SCRIPT.is_file(),
            "detail": (
                "Intended to filter WT/control rows, but control_flag is True "
                "for every archived source row; see reference_cohort_audit.csv."
            ),
        },
        {
            "asset": "reference_source_tables",
            "path": str(REFERENCE_SOURCE_DIR),
            "exists": REFERENCE_SOURCE_DIR.is_dir(),
            "detail": (
                f"{len(source_files)} qc_staged CSVs; "
                f"{sum(path.stat().st_size for path in source_files) / 1e9:.2f} GB."
                if source_files
                else "Source directory not accessible."
            ),
        },
        {
            "asset": "strict_WT_stage_snapshot",
            "path": str(paths.output_dir / "strict_wt_reference_stage_summary.csv"),
            "exists": True,
            "detail": (
                "Explicit WT genotype and/or phenotype=wt, no chemical "
                "perturbation, use_embryo=True, +/-0.25 hpf windows."
            ),
        },
        {
            "asset": "quarantined_SeaHub_pipeline_products",
            "path": str(paths.pipeline_root),
            "exists": paths.pipeline_root.is_dir(),
            "detail": (
                f"{len(input_manifest)} experiments with merged mask geometry "
                "and stage prediction."
            ),
        },
        {
            "asset": "quarantined_SeaHub_materialization_bundle",
            "path": str(paths.bundle_root),
            "exists": paths.bundle_root.is_dir(),
            "detail": "Plate metadata used to identify controls and source FOVs.",
        },
    ]
    return pd.DataFrame(records)


def run_analysis(paths: AnalysisPaths | None = None) -> dict[str, object]:
    paths = paths or AnalysisPaths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = paths.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _configure_plot_style()
    reference = pd.read_csv(paths.reference_csv)
    rows, input_manifest = load_seahub_comparison_rows(paths)
    fov_summary, stage_summary, regime_summary, sensitivity = (
        build_calibration_tables(rows)
    )
    experiment_stage_summary = (
        fov_summary.groupby(
            ["source_experiment_id", "predicted_stage_hpf"], as_index=False
        )
        .agg(
            n_fovs=("calibration_fov_id", "nunique"),
            n_control_embryos=("n_embryos", "sum"),
            strict_wt_effective_um_per_px=(
                "median_strict_wt_effective_um_per_px",
                "median",
            ),
            legacy_curve_effective_um_per_px=(
                "median_effective_um_per_px",
                "median",
            ),
            fov_min_strict_wt_um_per_px=(
                "median_strict_wt_effective_um_per_px",
                "min",
            ),
            fov_max_strict_wt_um_per_px=(
                "median_strict_wt_effective_um_per_px",
                "max",
            ),
        )
        .sort_values(["source_experiment_id", "predicted_stage_hpf"])
    )
    intended, qc_projection = project_sa_flags(rows, stage_summary)
    strict_reference = strict_wt_reference_summary()
    cohort_audit = reference_cohort_audit()
    reference_images = strict_wt_reference_image_manifest()
    provenance = build_provenance(paths, input_manifest)

    selected = rows[
        rows["is_wt_control_strict"] & rows["is_largest_mask_for_well"]
    ].copy()

    table_paths = {
        "all_comparison_rows": paths.output_dir / "seahub_sa_comparison_rows.csv",
        "wt_control_rows": paths.output_dir / "seahub_wt_control_rows.csv",
        "fov_summary": paths.output_dir / "fov_calibration_summary.csv",
        "stage_summary": paths.output_dir / "stage_calibration_summary.csv",
        "regime_summary": paths.output_dir / "capture_regime_summary.csv",
        "sensitivity": paths.output_dir / "calibration_sensitivity.csv",
        "experiment_stage_summary": paths.output_dir
        / "experiment_stage_calibration_summary.csv",
        "qc_projection": paths.output_dir / "sa_qc_projection_summary.csv",
        "input_manifest": paths.output_dir / "input_manifest.csv",
        "provenance": paths.output_dir / "reference_provenance.csv",
        "strict_reference": paths.output_dir
        / "strict_wt_reference_stage_summary.csv",
        "cohort_audit": paths.output_dir / "reference_cohort_audit.csv",
        "reference_images": paths.output_dir / "reference_image_manifest.csv",
    }
    rows.to_csv(table_paths["all_comparison_rows"], index=False)
    selected.to_csv(table_paths["wt_control_rows"], index=False)
    fov_summary.to_csv(table_paths["fov_summary"], index=False)
    stage_summary.to_csv(table_paths["stage_summary"], index=False)
    regime_summary.to_csv(table_paths["regime_summary"], index=False)
    sensitivity.to_csv(table_paths["sensitivity"], index=False)
    experiment_stage_summary.to_csv(
        table_paths["experiment_stage_summary"], index=False
    )
    qc_projection.to_csv(table_paths["qc_projection"], index=False)
    input_manifest.to_csv(table_paths["input_manifest"], index=False)
    provenance.to_csv(table_paths["provenance"], index=False)
    strict_reference.to_csv(table_paths["strict_reference"], index=False)
    cohort_audit.to_csv(table_paths["cohort_audit"], index=False)
    reference_images.to_csv(table_paths["reference_images"], index=False)

    figure_paths = {
        "effective_scale": figures_dir / "01_effective_um_per_px_by_stage.png",
        "area_comparison": figures_dir / "02_area_reference_comparison.png",
        "qc_projection": figures_dir / "03_sa_qc_projection.png",
        "representative_snips": figures_dir / "04_representative_wt_snips.png",
        "image_comparison": figures_dir
        / "05_seahub_vs_strict_wt_reference_images.png",
    }
    plot_effective_scale(fov_summary, stage_summary, figure_paths["effective_scale"])
    plot_area_comparison(intended, reference, figure_paths["area_comparison"])
    plot_qc_projection(qc_projection, figure_paths["qc_projection"])
    plot_representative_snips(
        selected, stage_summary, figure_paths["representative_snips"]
    )
    plot_seahub_vs_reference_images(
        selected,
        stage_summary,
        reference_images,
        figure_paths["image_comparison"],
    )

    return {
        "paths": paths,
        "rows": rows,
        "selected_controls": selected,
        "intended_rows": intended,
        "input_manifest": input_manifest,
        "fov_summary": fov_summary,
        "stage_summary": stage_summary,
        "regime_summary": regime_summary,
        "sensitivity": sensitivity,
        "experiment_stage_summary": experiment_stage_summary,
        "qc_projection": qc_projection,
        "strict_reference": strict_reference,
        "cohort_audit": cohort_audit,
        "reference_images": reference_images,
        "provenance": provenance,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
    }


def _print_summary(results: dict[str, object]) -> None:
    rows = results["rows"]
    selected = results["selected_controls"]
    stage_summary = results["stage_summary"]
    regime_summary = results["regime_summary"]
    qc_projection = results["qc_projection"]
    assert isinstance(rows, pd.DataFrame)
    assert isinstance(selected, pd.DataFrame)
    assert isinstance(stage_summary, pd.DataFrame)
    assert isinstance(regime_summary, pd.DataFrame)
    assert isinstance(qc_projection, pd.DataFrame)

    print(
        f"Loaded {len(rows):,} comparison rows from "
        f"{rows['experiment_id'].nunique()} completed/partial SeaHub shards."
    )
    print(
        f"Calibration cohort: {selected['well_id'].nunique():,} intended WT-control "
        f"embryos across {selected['calibration_fov_id'].nunique()} FOVs and "
        f"{selected['predicted_stage_hpf'].nunique()} stages."
    )
    print("\nStage-specific estimates (um/px):")
    print(
        stage_summary[
            [
                "stage_hpf",
                "n_embryos",
                "n_fovs",
                "strict_wt_effective_um_per_px",
                "strict_wt_bootstrap_ci95_low",
                "strict_wt_bootstrap_ci95_high",
                "legacy_curve_effective_um_per_px",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nCoarse capture regimes:")
    print(regime_summary.to_string(index=False))
    print("\nIllustrative SA-QC projection:")
    print(qc_projection.to_string(index=False))
    print(f"\nOutputs: {results['paths'].output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT
    )
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results = run_analysis(
        AnalysisPaths(
            pipeline_root=args.pipeline_root,
            bundle_root=args.bundle_root,
            reference_csv=args.reference_csv,
            output_dir=args.output_dir,
        )
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
