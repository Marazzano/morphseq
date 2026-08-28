#!/usr/bin/env python3
"""Read-only audit of current and legacy developmental-stage artifacts.

Experiments come only from the caller-supplied ordered list. Pipeline artifact
paths are constructed from those explicit IDs and the tracked path contract; the
pipeline output tree is never searched or globbed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


STAGE_COLUMNS = {
    "snip_id",
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "image_id",
    "time_index",
    "predicted_stage_hpf",
    "model_version",
}
STATUS_COLUMN = "stage_prediction_status"
CURRENT_MODEL_VERSION = "kimmel1995_temp_rate_v1"
STAGE_TERMS = (
    "predicted_stage_hpf",
    "inferred_stage_hpf_reg",
    "inferred_stage_hpf",
    "calc_stage_hpf",
    "stage_prediction_status",
    "manual_stage_hpf",
    "time_window",
    "age_key.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-list", type=Path, required=True)
    parser.add_argument("--experiment-limit", type=int, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--pipeline-output-root", type=Path, required=True)
    parser.add_argument("--legacy-age-key", type=Path, required=True)
    parser.add_argument("--surface-area-seam", type=Path, required=True)
    parser.add_argument("--surface-area-source-inventory", type=Path, required=True)
    parser.add_argument("--surface-area-reference", type=Path, required=True)
    parser.add_argument("--manual-curation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage-offsets-hpf",
        type=float,
        nargs="+",
        default=[-3.0, -1.5, 0.0, 1.5, 3.0],
    )
    parser.add_argument(
        "--metric-windows-hpf",
        type=float,
        nargs="+",
        default=[0.0, 0.75, 1.5, 2.25, 3.0, 4.0, 6.0, 12.0],
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def explicit_experiments(path: Path, limit: int) -> list[str]:
    if limit <= 0:
        raise ValueError("--experiment-limit must be positive")
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        raise ValueError(f"experiment authority has duplicate IDs: {duplicates[:5]}")
    if len(values) < limit:
        raise ValueError(
            f"experiment authority has {len(values)} entries, fewer than requested limit {limit}"
        )
    return values[:limit]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def inventory_row(
    path: Path,
    *,
    source: str,
    authority: str,
    experiment_id: str = "",
    row_count: int | None = None,
) -> dict:
    exists = path.is_file()
    stat = path.stat() if exists else None
    return {
        "experiment_id": experiment_id,
        "source": source,
        "authority": authority,
        "path": str(path.resolve() if exists else path.absolute()),
        "exists": exists,
        "row_count": row_count,
        "size_bytes": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "sha256": sha256_file(path) if exists else None,
    }


def stage_path(root: Path, experiment_id: str) -> Path:
    return (
        root
        / "feature_extraction"
        / experiment_id
        / "stage_predictions"
        / f"{experiment_id}_stage_predictions.csv"
    )


def collection_provenance_path(root: Path, experiment_id: str) -> Path:
    return (
        root
        / "acquisition"
        / experiment_id
        / "ingest_metadata"
        / "collection_provenance.json"
    )


def resolve_frame_timing(frame: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    require_columns(frame, {"image_id"}, f"{experiment_id} declared frame inventory")
    chosen: list[dict] = []
    for image_id, group in frame.groupby(frame["image_id"].astype(str), sort=False):
        values: list[tuple[str, float]] = []
        for column in ("elapsed_time_s", "experiment_time_s", "time_s"):
            if column not in group.columns:
                continue
            numeric = pd.to_numeric(group[column], errors="coerce").dropna().unique()
            if len(numeric) > 1:
                raise ValueError(
                    f"{experiment_id}: image_id {image_id!r} has conflicting {column}: "
                    f"{numeric[:5].tolist()}"
                )
            if len(numeric) == 1:
                values.append((column, float(numeric[0])))
        if not values:
            chosen.append(
                {"image_id": image_id, "elapsed_time_s_resolved": np.nan, "elapsed_time_source": "missing"}
            )
        else:
            column, value = values[0]
            chosen.append(
                {"image_id": image_id, "elapsed_time_s_resolved": value, "elapsed_time_source": column}
            )
    return pd.DataFrame(chosen)


def schema_variant(columns: Sequence[str]) -> str:
    present = set(columns)
    if STAGE_COLUMNS <= present and STATUS_COLUMN in present:
        return "S01_status_explicit"
    if STAGE_COLUMNS <= present and STATUS_COLUMN not in present:
        return "S02_status_absent"
    return "unrecognized"


def inspect_current_stage(
    *,
    experiments: Sequence[str],
    manifest: pd.DataFrame,
    pipeline_root: Path,
    source_inventory: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    manifest_by_id = manifest.set_index("experiment_id", drop=False)

    for experiment_id in experiments:
        if experiment_id not in manifest_by_id.index:
            raise ValueError(f"manifest has no row for explicit experiment {experiment_id!r}")
        manifest_row = manifest_by_id.loc[experiment_id]
        if isinstance(manifest_row, pd.DataFrame):
            raise ValueError(f"manifest has multiple rows for experiment {experiment_id!r}")

        current_stage_path = stage_path(pipeline_root, experiment_id)
        frame_path = Path(str(manifest_row["frame_inventory_csv"]))
        plate_path = Path(str(manifest_row["plate_metadata_csv"]))
        provenance_path = collection_provenance_path(pipeline_root, experiment_id)

        for path, source, authority in (
            (current_stage_path, "stage_predictions", "tracked_paths.py merged template"),
            (frame_path, "declared_frame_inventory", "experiment_manifest.csv"),
            (plate_path, "declared_plate_metadata", "experiment_manifest.csv"),
            (provenance_path, "collection_provenance", "tracked_paths.py experiment template"),
        ):
            source_inventory.append(
                inventory_row(
                    path,
                    source=source,
                    authority=authority,
                    experiment_id=experiment_id,
                )
            )

        missing = [str(path) for path in (current_stage_path, frame_path, plate_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{experiment_id}: required declared audit source(s) missing: {missing}"
            )

        stage = pd.read_csv(current_stage_path, low_memory=False)
        frame = pd.read_csv(frame_path, low_memory=False)
        plate = pd.read_csv(plate_path, low_memory=False)
        require_columns(stage, STAGE_COLUMNS, f"{experiment_id} stage_predictions")
        require_columns(plate, {"well_id", "start_age_hpf", "temperature"}, f"{experiment_id} plate")
        if plate["well_id"].astype(str).duplicated().any():
            raise ValueError(f"{experiment_id}: declared plate metadata has duplicate well_id")

        timing = resolve_frame_timing(frame, experiment_id)
        prepared_plate = plate[["well_id", "start_age_hpf", "temperature"]].copy()
        prepared_plate["well_id"] = prepared_plate["well_id"].astype(str)
        checked = stage.copy()
        checked["snip_id"] = checked["snip_id"].astype(str)
        checked["well_id"] = checked["well_id"].astype(str)
        checked["image_id"] = checked["image_id"].astype(str)
        checked = checked.merge(prepared_plate, on="well_id", how="left", validate="many_to_one")
        checked = checked.merge(timing, on="image_id", how="left", validate="many_to_one")
        checked["predicted_stage_hpf"] = pd.to_numeric(
            checked["predicted_stage_hpf"], errors="coerce"
        )
        checked["start_age_hpf"] = pd.to_numeric(checked["start_age_hpf"], errors="coerce")
        checked["incubation_temperature_c"] = pd.to_numeric(
            checked.pop("temperature"), errors="coerce"
        )
        checked["formula_expected_stage_hpf"] = checked["start_age_hpf"] + (
            checked["elapsed_time_s_resolved"] / 3600.0
        ) * (0.055 * checked["incubation_temperature_c"] - 0.57)
        checked["formula_residual_hpf"] = (
            checked["predicted_stage_hpf"] - checked["formula_expected_stage_hpf"]
        )
        checked["schema_variant"] = schema_variant(list(stage.columns))
        checked["collection_provenance_present"] = provenance_path.is_file()
        if STATUS_COLUMN not in checked.columns:
            checked[STATUS_COLUMN] = "unavailable_status_column_absent"

        track_sizes = checked.groupby("physical_embryo_id", dropna=False).size()
        monotonic_violations = 0
        comparable_transitions = 0
        for _, track in checked.groupby("physical_embryo_id", sort=False, dropna=False):
            finite = track.dropna(subset=["elapsed_time_s_resolved", "predicted_stage_hpf"]).sort_values(
                ["elapsed_time_s_resolved", "time_index"], kind="mergesort"
            )
            if len(finite) < 2:
                continue
            elapsed_delta = np.diff(finite["elapsed_time_s_resolved"].to_numpy(dtype=float))
            stage_delta = np.diff(finite["predicted_stage_hpf"].to_numpy(dtype=float))
            comparable_transitions += int((elapsed_delta >= 0).sum())
            monotonic_violations += int(((elapsed_delta >= 0) & (stage_delta < -1e-9)).sum())

        finite_stage = checked["predicted_stage_hpf"].dropna()
        finite_residual = checked["formula_residual_hpf"].dropna().abs()
        statuses = checked[STATUS_COLUMN].astype(str)
        summaries.append(
            {
                "experiment_id": experiment_id,
                "rows": len(checked),
                "unique_snip_ids": checked["snip_id"].nunique(),
                "duplicate_snip_rows": int(checked["snip_id"].duplicated(keep=False).sum()),
                "schema_variant": checked["schema_variant"].iloc[0],
                "status_values": "|".join(sorted(statuses.unique())),
                "predicted_status_rows": int(statuses.eq("predicted").sum()),
                "finite_stage_rows": int(finite_stage.size),
                "stage_min_hpf": float(finite_stage.min()) if len(finite_stage) else None,
                "stage_max_hpf": float(finite_stage.max()) if len(finite_stage) else None,
                "model_versions": "|".join(sorted(checked["model_version"].astype(str).unique())),
                "finite_formula_input_rows": int(
                    checked[
                        ["start_age_hpf", "elapsed_time_s_resolved", "incubation_temperature_c"]
                    ].notna().all(axis=1).sum()
                ),
                "max_abs_formula_residual_hpf": (
                    float(finite_residual.max()) if len(finite_residual) else None
                ),
                "physical_embryos": int(track_sizes.size),
                "multi_observation_physical_embryos": int((track_sizes > 1).sum()),
                "comparable_monotonic_transitions": comparable_transitions,
                "monotonicity_violations": monotonic_violations,
                "collection_provenance_present": provenance_path.is_file(),
            }
        )
        rows.append(checked)

    current = pd.concat(rows, ignore_index=True)
    duplicate_global = current["snip_id"].duplicated(keep=False)
    if duplicate_global.any():
        examples = current.loc[duplicate_global, ["experiment_id", "snip_id"]].head(5).to_dict("records")
        raise ValueError(f"explicit current sources have cross-experiment duplicate snip_id rows: {examples}")
    return current, pd.DataFrame(summaries)


def status_distribution(current: pd.DataFrame) -> pd.DataFrame:
    return (
        current.groupby(
            ["experiment_id", STATUS_COLUMN, "model_version"], dropna=False, sort=False
        )
        .agg(
            rows=("snip_id", "size"),
            finite_stage_rows=("predicted_stage_hpf", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .reset_index()
    )


def crosswalk_summary(current: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        legacy,
        {"snip_id", "embryo_id", "calc_stage_hpf", "inferred_stage_hpf_reg"},
        "legacy age key",
    )
    current_snips = set(current["snip_id"].astype(str))
    legacy_snips = set(legacy["snip_id"].astype(str))
    current_embryos = set(current["embryo_id"].astype(str))
    legacy_embryos = set(legacy["embryo_id"].astype(str))
    current_experiments = set(current["experiment_id"].astype(str))
    legacy_experiments = set(legacy["experiment_date"].astype(str))
    paired = current.merge(
        legacy[["snip_id", "inferred_stage_hpf_reg"]],
        on="snip_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) >= 2:
        differences = paired["predicted_stage_hpf"] - paired["inferred_stage_hpf_reg"]
        correlation = float(paired["predicted_stage_hpf"].corr(paired["inferred_stage_hpf_reg"]))
        bias = float(differences.mean())
        sd = float(differences.std(ddof=1))
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd
    else:
        correlation = bias = sd = loa_low = loa_high = None
    return pd.DataFrame(
        [
            {
                "current_rows": len(current),
                "current_unique_snip_ids": len(current_snips),
                "legacy_rows": len(legacy),
                "legacy_unique_snip_ids": len(legacy_snips),
                "exact_snip_id_crosswalk_rows": len(current_snips & legacy_snips),
                "exact_embryo_id_crosswalk_count": len(current_embryos & legacy_embryos),
                "exact_experiment_label_overlap_count": len(
                    current_experiments & legacy_experiments
                ),
                "paired_pearson": correlation,
                "paired_mean_bias_new_minus_legacy_hpf": bias,
                "paired_difference_sd_hpf": sd,
                "bland_altman_lower_hpf": loa_low,
                "bland_altman_upper_hpf": loa_high,
                "agreement_identifiability": "estimable" if len(paired) >= 2 else "not_identifiable_n_lt_2",
            }
        ]
    )


def legacy_provenance_summary(legacy: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        legacy,
        {
            "snip_id",
            "experiment_date",
            "embryo_id",
            "temperature",
            "calc_stage_hpf",
            "inferred_stage_hpf_reg",
            "train_dir",
            "model_name",
            "architecture_name",
        },
        "legacy age key",
    )
    return pd.DataFrame(
        [
            {
                "rows": len(legacy),
                "unique_snip_ids": legacy["snip_id"].astype(str).nunique(),
                "unique_embryo_ids": legacy["embryo_id"].astype(str).nunique(),
                "unique_experiment_dates": legacy["experiment_date"].astype(str).nunique(),
                "finite_calc_stage_rows": int(
                    pd.to_numeric(legacy["calc_stage_hpf"], errors="coerce").notna().sum()
                ),
                "finite_inferred_stage_rows": int(
                    pd.to_numeric(legacy["inferred_stage_hpf_reg"], errors="coerce").notna().sum()
                ),
                "temperature_values": "|".join(
                    sorted(legacy["temperature"].dropna().astype(str).unique())
                ),
                "train_dir_values": "|".join(
                    sorted(legacy["train_dir"].dropna().astype(str).unique())
                ),
                "model_name_values": "|".join(
                    sorted(legacy["model_name"].dropna().astype(str).unique())
                ),
                "architecture_name_values": "|".join(
                    sorted(legacy["architecture_name"].dropna().astype(str).unique())
                ),
                "persisted_reference_dataset_list": False,
                "persisted_model_weight_checksum": False,
                "persisted_estimator_hyperparameters": False,
            }
        ]
    )


def load_reference(path: Path) -> pd.DataFrame:
    reference = pd.read_csv(path)
    require_columns(reference, {"stage_hpf", "p5", "p95"}, "surface-area reference")
    reference = reference.sort_values("stage_hpf", kind="mergesort")
    return reference


def surface_sensitivity(
    seam: pd.DataFrame, reference: pd.DataFrame, offsets: Sequence[float]
) -> pd.DataFrame:
    require_columns(
        seam,
        {
            "snip_id",
            "experiment_id",
            "area_um2",
            "predicted_stage_hpf",
            "stage_hpf",
            "recomputed_sa_outlier_flag",
        },
        "surface-area stage seam",
    )
    stages_ref = reference["stage_hpf"].to_numpy(dtype=float)
    p5_ref = reference["p5"].to_numpy(dtype=float)
    p95_ref = reference["p95"].to_numpy(dtype=float)
    areas = pd.to_numeric(seam["area_um2"], errors="coerce").to_numpy(dtype=float)
    baseline = seam["recomputed_sa_outlier_flag"].astype(bool).to_numpy()
    axes: list[tuple[str, np.ndarray, str]] = [
        (
            "plate_metadata.stage_hpf",
            pd.to_numeric(seam["stage_hpf"], errors="coerce").to_numpy(dtype=float),
            "available_alternate_but_not_independent_anchor",
        )
    ]
    nominal = pd.to_numeric(seam["predicted_stage_hpf"], errors="coerce").to_numpy(dtype=float)
    for offset in offsets:
        axes.append(
            (
                f"predicted_stage_hpf_offset_{offset:+g}",
                nominal + float(offset),
                "diagnostic_uniform_offset_not_estimator",
            )
        )
    outputs: list[dict] = []
    for name, values, interpretation in axes:
        finite = np.isfinite(values) & np.isfinite(areas)
        lower = np.full(len(seam), np.nan)
        upper = np.full(len(seam), np.nan)
        lower[finite] = 0.9 * np.interp(values[finite], stages_ref, p5_ref)
        upper[finite] = 1.4 * np.interp(values[finite], stages_ref, p95_ref)
        flag = np.zeros(len(seam), dtype=bool)
        flag[finite] = (areas[finite] < lower[finite]) | (areas[finite] > upper[finite])
        outputs.append(
            {
                "axis": name,
                "interpretation": interpretation,
                "rows": len(seam),
                "finite_axis_rows": int(finite.sum()),
                "flagged_rows": int(flag.sum()),
                "changed_classification_rows_vs_nominal": int((flag != baseline).sum()),
                "not_applicable_rows_if_used": int((~finite).sum()),
            }
        )
    return pd.DataFrame(outputs)


def pair_counts(values: np.ndarray, embryo_ids: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    delta = np.abs(values[:, None] - values[None, :])
    age_match = (delta <= window) & finite[:, None] & finite[None, :]
    same_embryo = embryo_ids[:, None] == embryo_ids[None, :]
    self_options = age_match & same_embryo
    other_age_candidates = age_match & ~same_embryo
    return self_options, other_age_candidates


def metric_window_sensitivity(seam: pd.DataFrame, windows: Sequence[float]) -> pd.DataFrame:
    values = pd.to_numeric(seam["predicted_stage_hpf"], errors="coerce").to_numpy(dtype=float)
    embryo_ids = seam["physical_embryo_id"].astype(str).to_numpy()
    _, baseline = pair_counts(values, embryo_ids, 1.5)
    outputs: list[dict] = []
    for window in windows:
        self_options, other_candidates = pair_counts(values, embryo_ids, float(window))
        per_anchor = other_candidates.sum(axis=1)
        self_per_anchor = self_options.sum(axis=1)
        outputs.append(
            {
                "window_hpf": float(window),
                "role_at_current_defaults": (
                    "sampler_default" if float(window) == 1.5 else "loss_default" if float(window) == 3.0 else "sensitivity"
                ),
                "anchors": len(seam),
                "anchors_with_self_option": int((self_per_anchor > 0).sum()),
                "min_self_options_per_anchor": int(self_per_anchor.min()),
                "age_eligible_ordered_other_pairs": int(other_candidates.sum()),
                "anchors_with_any_other_age_candidate": int((per_anchor > 0).sum()),
                "min_other_age_candidates": int(per_anchor.min()),
                "median_other_age_candidates": float(np.median(per_anchor)),
                "max_other_age_candidates": int(per_anchor.max()),
                "ordered_pair_membership_changes_vs_sampler_1_5": int(np.logical_xor(other_candidates, baseline).sum()),
                "scientific_legality_status": "not_identifiable_without_relation_mapping_and_split",
            }
        )
    return pd.DataFrame(outputs)


def metric_axis_sensitivity(seam: pd.DataFrame, offsets: Sequence[float]) -> pd.DataFrame:
    nominal = pd.to_numeric(seam["predicted_stage_hpf"], errors="coerce").to_numpy(dtype=float)
    plate = pd.to_numeric(seam["stage_hpf"], errors="coerce").to_numpy(dtype=float)
    embryo_ids = seam["physical_embryo_id"].astype(str).to_numpy()
    axes = [("plate_metadata.stage_hpf", plate)] + [
        (f"predicted_stage_hpf_offset_{offset:+g}", nominal + float(offset)) for offset in offsets
    ]
    baseline_masks = {
        window: pair_counts(nominal, embryo_ids, window)[1] for window in (1.5, 3.0)
    }
    outputs: list[dict] = []
    for name, values in axes:
        for window in (1.5, 3.0):
            _, candidates = pair_counts(values, embryo_ids, window)
            outputs.append(
                {
                    "axis": name,
                    "window_hpf": window,
                    "age_eligible_ordered_other_pairs": int(candidates.sum()),
                    "ordered_pair_membership_changes_vs_nominal": int(
                        np.logical_xor(candidates, baseline_masks[window]).sum()
                    ),
                    "scientific_legality_status": "not_identifiable_without_relation_mapping_and_split",
                }
            )
    return pd.DataFrame(outputs)


def absence_sensitivity(seam: pd.DataFrame) -> pd.DataFrame:
    rows = len(seam)
    return pd.DataFrame(
        [
            {
                "consumer_or_policy": "planned Track A basic cohort",
                "stage_absent_result": "rows retained; stage_status=unavailable",
                "rows_retained_or_applicable": rows,
                "evidence_kind": "binding manifest contract",
            },
            {
                "consumer_or_policy": "planned Track A require-stage cohort",
                "stage_absent_result": "rows rejected by explicit policy",
                "rows_retained_or_applicable": 0,
                "evidence_kind": "binding manifest contract",
            },
            {
                "consumer_or_policy": "current surface_area_qc",
                "stage_absent_result": "flag false; applicability not_applicable",
                "rows_retained_or_applicable": rows,
                "evidence_kind": "production code",
            },
            {
                "consumer_or_policy": "current metric dataset age gate",
                "stage_absent_result": "no self/other options; random choice fails",
                "rows_retained_or_applicable": 0,
                "evidence_kind": "current legacy core code",
            },
            {
                "consumer_or_policy": "analysis/report stage axes",
                "stage_absent_result": "zero finite rows on stage plot/bin axis",
                "rows_retained_or_applicable": 0,
                "evidence_kind": "report code",
            },
        ]
    )


def fallback_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("live stage elapsed alias", "elapsed_time_s null/absent", "experiment_time_s then time_s", "silent selection; hard error if none", "yes", "src/data_pipeline/feature_extraction/stage_predictions/compute.py:34-45"),
            ("live collection age-map alias", "start_age_by_source_ordinal absent", "start_age_by_time_index", "silent compatibility fallback; hard error if key absent", "yes", "src/data_pipeline/feature_extraction/stage_predictions/compute.py:107-128"),
            ("live single missing start age", "plate start_age_hpf absent/null", "null stage + missing_start_age_hpf", "explicit status", "yes", "src/data_pipeline/feature_extraction/stage_predictions/compute.py:199-213"),
            ("live missing temperature", "plate temperature absent/null", "null stage + missing_temperature", "explicit status", "yes", "src/data_pipeline/feature_extraction/stage_predictions/compute.py:205-213"),
            ("batch helper time alias", "configured experiment_time_s absent/null", "time_s", "silent selection; hard error if none", "no repository caller found", "src/data_pipeline/feature_extraction/stage_inference.py:35-64"),
            ("legacy core metadata stage", "inferred_stage_hpf column absent", "predicted_stage_hpf renamed stage_hpf", "stdout warning", "current legacy Hydra path; no in planned Track A", "src/core/data/dataset_utils.py:26-38"),
            ("legacy core age key", "metadata/age_key.csv absent", "none", "hard error with misleading message", "current legacy Hydra path; no in planned Track A", "src/core/data/dataset_configs.py:38-66"),
            ("surface-area missing stage", "stage row exists but value null", "false flag + not_applicable", "explicit policy", "yes", "src/data_pipeline/quality_control/surface_area_qc/config.py:20-31; src/data_pipeline/quality_control/surface_area_qc/compute.py:87-103"),
            ("surface-area out-of-range stage", "stage outside reference range", "nearest reference endpoint", "silent numpy.interp clamp", "yes", "src/data_pipeline/quality_control/surface_area_qc/reference.py:40-51"),
            ("legacy morphology reference rows", "reference_datasets is None", "four hard-coded date/control selections", "silent default", "manual src/build launch only", "src/build/infer_developmental_age.py:46-70"),
            ("legacy morphology calibration", "fewer than n_ref same-experiment comparisons", "same-temperature nearest references", "prints ratio, no structured status", "manual src/build launch only", "src/build/infer_developmental_age.py:107-153"),
            ("legacy morphology snapshot", "max observations per embryo equals one", "copy nominal predicted_stage_hpf", "silent", "manual src/build launch only", "src/build/infer_developmental_age.py:123-156"),
            ("legacy combined QC stage", "predicted_stage_hpf column absent", "literal 0.0", "silent placeholder", "manual legacy main/results launch; no in planned Track A", "src/build/build04_perform_embryo_qc.py:1275-1300,1660-1678"),
            ("legacy Build03 formula", "formula input conversion raises", "leave column unchanged", "silent unless verbose", "manual legacy script; no in planned Track A", "src/build/build03A_process_images.py:1627-1674"),
        ],
        columns=[
            "path_or_behavior",
            "trigger",
            "selected_column_or_value",
            "signal",
            "reachability",
            "source_citation",
        ],
    )


def consumer_search_inventory(repo_root: Path) -> pd.DataFrame:
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "scripts", "results"], text=True
    ).splitlines()
    allowed = {".py", ".smk", ".yaml", ".yml", ".sh"}
    rows: list[dict] = []
    for relative in tracked:
        path = repo_root / relative
        if path.suffix not in allowed or not path.is_file() or "/_Archive/" in f"/{relative}/":
            continue
        text = path.read_text(errors="replace")
        matched = [term for term in STAGE_TERMS if term in text]
        if not matched:
            continue
        line_numbers = sorted(
            {
                number
                for number, line in enumerate(text.splitlines(), start=1)
                if any(term in line for term in matched)
            }
        )
        if relative.startswith("src/data_pipeline/feature_extraction/stage_prediction"):
            role = "live_producer_or_contract"
        elif "surface_area_qc" in relative or "surface_area_outlier" in relative:
            role = "qc_conditioning_or_qc_report"
        elif relative.startswith("src/core/data"):
            role = "core_loader_cohort_or_pair_sampler"
        elif relative.startswith("src/core/loss"):
            role = "core_metric_loss"
        elif relative.startswith("src/core"):
            role = "core_analysis_config_or_reporting"
        elif relative.startswith("src/data_pipeline/analysis_ready"):
            role = "analysis_ready_metadata_or_report_axis"
        elif relative.startswith("src/data_pipeline"):
            role = "pipeline_consumer_or_parallel_stage_formula"
        elif relative.startswith("src/build"):
            role = "legacy_build_producer_or_consumer"
        elif relative.startswith("src/analyze") or relative.startswith("src/app"):
            role = "analysis_or_visualization_axis"
        elif relative.startswith("src/vae"):
            role = "legacy_vae_consumer"
        elif relative.startswith("results"):
            role = "launch_or_analysis_script"
        else:
            role = "study_or_test"
        rows.append(
            {
                "path": relative,
                "matched_terms": "|".join(matched),
                "line_numbers": "|".join(map(str, line_numbers)),
                "role_class": role,
            }
        )
    return pd.DataFrame(rows).sort_values("path", kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    repo_root = Path(git_output("rev-parse", "--show-toplevel"))
    experiments = explicit_experiments(args.experiment_list, args.experiment_limit)
    manifest = pd.read_csv(args.experiment_manifest, low_memory=False)
    require_columns(
        manifest,
        {"experiment_id", "frame_inventory_csv", "plate_metadata_csv"},
        "experiment manifest",
    )
    manifest_ids = manifest["experiment_id"].astype(str).tolist()
    positions = [manifest_ids.index(experiment_id) for experiment_id in experiments]
    if positions != sorted(positions):
        raise ValueError("selected experiment order disagrees with experiment manifest order")

    source_inventory = [
        inventory_row(args.experiment_list, source="experiment_list", authority="caller argument"),
        inventory_row(args.experiment_manifest, source="experiment_manifest", authority="caller argument"),
        inventory_row(args.legacy_age_key, source="legacy_age_key", authority="caller argument"),
        inventory_row(args.surface_area_seam, source="surface_area_stage_seam", authority="Track D handoff"),
        inventory_row(
            args.surface_area_source_inventory,
            source="surface_area_source_inventory",
            authority="Track D handoff",
        ),
        inventory_row(
            args.surface_area_reference,
            source="surface_area_reference",
            authority="caller argument / packaged v1",
        ),
        inventory_row(
            args.manual_curation_csv,
            source="legacy_manual_curation_candidate",
            authority="src/build/build05_make_training_snips.py:21-48",
        ),
    ]

    current, integrity = inspect_current_stage(
        experiments=experiments,
        manifest=manifest,
        pipeline_root=args.pipeline_output_root,
        source_inventory=source_inventory,
    )
    legacy = pd.read_csv(args.legacy_age_key, low_memory=False)
    seam = pd.read_csv(args.surface_area_seam, low_memory=False)
    selected_set = set(experiments)
    unexpected = sorted(set(seam["experiment_id"].astype(str)) - selected_set)
    if unexpected:
        raise ValueError(f"Track D seam contains experiments outside selected authority: {unexpected}")
    reference = load_reference(args.surface_area_reference)

    outputs = {
        "selected_experiments.txt": "\n".join(experiments) + "\n",
        "source_inventory.csv": pd.DataFrame(source_inventory),
        "current_stage_formula_check.csv": current[
            [
                "experiment_id",
                "well_id",
                "physical_embryo_id",
                "embryo_id",
                "snip_id",
                "image_id",
                "time_index",
                "start_age_hpf",
                "elapsed_time_s_resolved",
                "elapsed_time_source",
                "incubation_temperature_c",
                "predicted_stage_hpf",
                "formula_expected_stage_hpf",
                "formula_residual_hpf",
                STATUS_COLUMN,
                "model_version",
                "schema_variant",
                "collection_provenance_present",
            ]
        ],
        "stage_integrity_by_experiment.csv": integrity,
        "stage_status_distribution.csv": status_distribution(current),
        "legacy_current_crosswalk.csv": crosswalk_summary(current, legacy),
        "legacy_provenance_summary.csv": legacy_provenance_summary(legacy),
        "surface_area_stage_sensitivity.csv": surface_sensitivity(
            seam, reference, args.stage_offsets_hpf
        ),
        "metric_window_sensitivity.csv": metric_window_sensitivity(
            seam, args.metric_windows_hpf
        ),
        "metric_axis_sensitivity.csv": metric_axis_sensitivity(
            seam, args.stage_offsets_hpf
        ),
        "stage_absence_sensitivity.csv": absence_sensitivity(seam),
        "fallback_inventory.csv": fallback_inventory(),
        "consumer_search_inventory.csv": consumer_search_inventory(repo_root),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        path = output_dir / name
        if isinstance(value, str):
            path.write_text(value)
        else:
            value.to_csv(path, index=False)

    crosswalk = outputs["legacy_current_crosswalk.csv"].iloc[0]
    summary = {
        "measurement_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "git_revision": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "experiment_list": str(args.experiment_list.absolute()),
        "experiment_list_sha256": sha256_file(args.experiment_list),
        "experiment_limit": args.experiment_limit,
        "selected_experiments": experiments,
        "pipeline_output_root": str(args.pipeline_output_root.absolute()),
        "current_stage_rows": len(current),
        "current_unique_snip_ids": int(current["snip_id"].nunique()),
        "current_physical_embryos": int(current["physical_embryo_id"].nunique()),
        "current_multi_observation_physical_embryos": int(
            (current.groupby("physical_embryo_id").size() > 1).sum()
        ),
        "current_finite_stage_rows": int(current["predicted_stage_hpf"].notna().sum()),
        "current_formula_complete_rows": int(current["formula_expected_stage_hpf"].notna().sum()),
        "current_max_abs_formula_residual_hpf": float(current["formula_residual_hpf"].abs().max()),
        "collection_provenance_present_experiments": int(
            integrity["collection_provenance_present"].sum()
        ),
        "legacy_rows": len(legacy),
        "legacy_unique_snip_ids": int(legacy["snip_id"].astype(str).nunique()),
        "exact_snip_id_crosswalk_rows": int(crosswalk["exact_snip_id_crosswalk_rows"]),
        "exact_embryo_id_crosswalk_count": int(crosswalk["exact_embryo_id_crosswalk_count"]),
        "surface_area_seam_rows": len(seam),
        "surface_area_seam_complete_asset_keys": int(
            seam["asset_key_status"].astype(str).eq("complete").sum()
            if "asset_key_status" in seam.columns
            else 0
        ),
        "manual_curation_candidate_present": args.manual_curation_csv.is_file(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
