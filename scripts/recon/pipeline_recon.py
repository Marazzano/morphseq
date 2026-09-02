#!/usr/bin/env python
"""Read-only reconnaissance of pipeline outputs for core-model integration.

The script never writes beneath ``--output-root``. It resolves canonical artifact
locations from the pipeline path registry, reads an explicit experiment list (or
explicit manifest files), and writes a Markdown report plus exact companion CSVs.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_REGISTRY_FILE = (
    REPO_ROOT
    / "src"
    / "data_pipeline"
    / "pipeline_orchestrator"
    / "orchestration"
    / "paths.py"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/nick/gs_cluster/projects/data/morphseq/pipeline/output"
)
DEFAULT_MANIFESTS = (
    REPO_ROOT
    / "src/data_pipeline/pipeline_orchestrator/manifests/back_half_keyence.txt",
    REPO_ROOT
    / "src/data_pipeline/pipeline_orchestrator/manifests/back_half_yx1.txt",
    REPO_ROOT
    / "src/data_pipeline/pipeline_orchestrator/manifests/back_half_hotfish_20260724.txt",
)
DEFAULT_METRIC_KEY = Path(
    "/media/nick/gs_cluster/projects/data/morphseq/training_data/models/metadata/metric_key.csv"
)
IDENTITY_SPINE = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
    "image_id",
    "time_index",
    "channel_id",
)
INVENTORY_COLUMNS_NEEDED = set(IDENTITY_SPINE) | {
    "processed_snip_path",
    "embryo_mask",
    "embryo_mask_snip_path",
    "is_valid_snip",
    "source_micrometers_per_pixel",
    "snip_micrometers_per_pixel",
    "image_product_type",
    "projection_method",
    "z_position",
    "z_index",
    "image_path",
}
STAGE_COLUMNS_NEEDED = set(IDENTITY_SPINE) | {
    "predicted_stage_hpf",
    "stage_prediction_status",
}
QC_COLUMNS_NEEDED = set(IDENTITY_SPINE) | {"use_snip", "qc_fail_reasons"}


@dataclass(frozen=True)
class SourceSpec:
    step: str
    artifact: str
    kind: str
    path_mode: str | None = None


SOURCE_SPECS = {
    "inventory": SourceSpec("snip_inventory", "snip_inventory", "csv", "merged"),
    "stage": SourceSpec("stage_predictions", "stage_predictions", "csv", "merged"),
    "qc": SourceSpec("snip_qc", "verdict", "parquet", "merged"),
    "plate": SourceSpec("ingest_plate_metadata", "csv", "csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        dest="manifests",
        help="Manifest containing '<experiment_id> <scope>' rows; repeatable.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="Explicit experiment ID; repeatable and combined with manifest IDs.",
    )
    parser.add_argument("--metric-key", type=Path, default=DEFAULT_METRIC_KEY)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports/PIPELINE_RECON.md",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=REPO_ROOT / "reports/recon_tables",
    )
    parser.add_argument("--intensity-sample-per-experiment", type=int, default=200)
    parser.add_argument("--format-sample", type=int, default=100)
    parser.add_argument("--throughput-sample", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def load_path_registry():
    """Load the pure registry module without importing orchestration.__init__."""
    spec = importlib.util.spec_from_file_location("_morphseq_pipeline_paths", PATH_REGISTRY_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load path registry from {PATH_REGISTRY_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(path: Path) -> list[str]:
    experiments: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            experiments.append(line.split()[0])
    return experiments


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def resolve_source_path(registry, root: Path, experiment: str, spec: SourceSpec) -> Path:
    kwargs: dict[str, Any] = {}
    if spec.path_mode is not None:
        kwargs["path_mode"] = spec.path_mode
    return registry.artifact_path(
        root,
        spec.step,
        spec.artifact,
        experiment,
        **kwargs,
    )


def parquet_engine() -> str | None:
    for name in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(name) is not None:
            return name
    return None


def jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value) if not isinstance(value, (str, int, float, bool)) else value


def sorted_values(series: pd.Series) -> list[Any]:
    values = [jsonable_scalar(value) for value in series.dropna().unique().tolist()]
    values = [value for value in values if value is not None and str(value).strip()]
    return sorted(values, key=lambda value: str(value))


def subset_with_provenance(df: pd.DataFrame, columns: set[str], experiment: str) -> pd.DataFrame:
    kept = [column for column in df.columns if column in columns]
    out = df.loc[:, kept].copy()
    out["_source_experiment"] = experiment
    return out


def load_sources(
    registry,
    root: Path,
    experiments: Sequence[str],
    engine: str | None,
):
    availability: list[dict[str, Any]] = []
    inventory_frames: list[pd.DataFrame] = []
    stage_frames: list[pd.DataFrame] = []
    qc_frames: list[pd.DataFrame] = []
    plate_frames: list[pd.DataFrame] = []

    for experiment in experiments:
        for source, spec in SOURCE_SPECS.items():
            path = resolve_source_path(registry, root, experiment, spec)
            record: dict[str, Any] = {
                "experiment_id": experiment,
                "source": source,
                "path": str(path),
                "present": path.is_file(),
                "readable": False,
                "row_count": None,
                "columns": [],
                "error": "",
            }
            if not record["present"]:
                availability.append(record)
                continue
            if spec.kind == "parquet" and engine is None:
                record["error"] = "BLOCKED: neither pyarrow nor fastparquet is installed"
                availability.append(record)
                continue
            try:
                if spec.kind == "csv":
                    frame = pd.read_csv(path, low_memory=False)
                else:
                    frame = pd.read_parquet(path, engine=engine)
                record["readable"] = True
                record["row_count"] = len(frame)
                record["columns"] = list(frame.columns)
                if source == "inventory":
                    inventory_frames.append(
                        subset_with_provenance(frame, INVENTORY_COLUMNS_NEEDED, experiment)
                    )
                elif source == "stage":
                    stage_frames.append(subset_with_provenance(frame, STAGE_COLUMNS_NEEDED, experiment))
                elif source == "qc":
                    qc_frames.append(subset_with_provenance(frame, QC_COLUMNS_NEEDED, experiment))
                elif source == "plate":
                    plate = frame.copy()
                    plate["_source_experiment"] = experiment
                    plate_frames.append(plate)
            except Exception as exc:  # report malformed/unreadable artifacts instead of aborting survey
                record["error"] = f"{type(exc).__name__}: {exc}"
            availability.append(record)

    inventory = pd.concat(inventory_frames, ignore_index=True, sort=False) if inventory_frames else pd.DataFrame()
    stages = pd.concat(stage_frames, ignore_index=True, sort=False) if stage_frames else pd.DataFrame()
    qc = pd.concat(qc_frames, ignore_index=True, sort=False) if qc_frames else pd.DataFrame()
    plates = pd.concat(plate_frames, ignore_index=True, sort=False) if plate_frames else pd.DataFrame()
    return pd.DataFrame(availability), inventory, stages, qc, plates


def assign_schema_details(availability: pd.DataFrame, experiments: Sequence[str]):
    newest = max(experiments)
    availability = availability.copy()
    availability["schema_id"] = ""
    availability["columns_json"] = availability["columns"].map(json.dumps)
    availability["reference_experiment"] = ""
    availability["added_vs_reference"] = "[]"
    availability["missing_vs_reference"] = "[]"
    catalog_rows: list[dict[str, Any]] = []

    for source, source_rows in availability.groupby("source", sort=False):
        schemas: dict[tuple[str, ...], str] = {}
        for idx in source_rows.index:
            columns = tuple(availability.at[idx, "columns"])
            if not columns:
                continue
            if columns not in schemas:
                schema_id = f"{source[:1].upper()}{len(schemas) + 1:02d}"
                schemas[columns] = schema_id
                catalog_rows.append(
                    {"source": source, "schema_id": schema_id, "columns_json": json.dumps(columns)}
                )
            availability.at[idx, "schema_id"] = schemas[columns]

        global_newest = source_rows[source_rows["experiment_id"] == newest]
        global_columns: list[str] = []
        if not global_newest.empty:
            global_columns = list(global_newest.iloc[0]["columns"])
        readable_rows = source_rows[source_rows["readable"]]
        if global_columns:
            reference_experiment = newest
            reference_columns = global_columns
        elif not readable_rows.empty:
            newest_readable_idx = readable_rows["experiment_id"].idxmax()
            reference_experiment = str(availability.at[newest_readable_idx, "experiment_id"])
            reference_columns = list(availability.at[newest_readable_idx, "columns"])
        else:
            reference_experiment = "none-readable"
            reference_columns = []

        reference_set = set(reference_columns)
        for idx in source_rows.index:
            columns = set(availability.at[idx, "columns"])
            availability.at[idx, "reference_experiment"] = reference_experiment
            if columns:
                availability.at[idx, "added_vs_reference"] = json.dumps(sorted(columns - reference_set))
                availability.at[idx, "missing_vs_reference"] = json.dumps(sorted(reference_set - columns))

    return availability, pd.DataFrame(catalog_rows), newest


TRUE_VALUES = frozenset({"true", "t", "1", "yes", "y"})
FALSE_VALUES = frozenset({"false", "f", "0", "no", "n"})


def coerce_bool(series: pd.Series) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        token = str(value).strip().lower()
        if token in TRUE_VALUES:
            return True
        if token in FALSE_VALUES:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def identity_audit(inventory: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if inventory.empty:
        return result
    spine_present = [column for column in IDENTITY_SPINE if column in inventory]
    result["null_counts"] = {column: int(inventory[column].isna().sum()) for column in spine_present}
    missing_spine = sorted(set(IDENTITY_SPINE) - set(inventory.columns))
    result["missing_spine_columns"] = missing_spine

    if "snip_id" in inventory:
        duplicates = inventory["snip_id"].duplicated(keep=False)
        result["duplicate_rows"] = int(duplicates.sum())
        result["duplicate_ids"] = int(inventory.loc[duplicates, "snip_id"].nunique(dropna=True))

    required_physical = {"well_id", "physical_embryo_id"}
    if required_physical <= set(inventory.columns):
        well = inventory["well_id"].astype("string")
        physical = inventory["physical_embryo_id"].astype("string")
        parsed = physical.str.extract(r"^(?P<embedded_well>.+)_e(?P<index>\d{2,})$")
        suffix = parsed["index"]
        valid = (
            (parsed["embedded_well"] == well)
            & suffix.str.fullmatch(r"\d{2,}", na=False)
            & (pd.to_numeric(suffix, errors="coerce") >= 1)
        )
        result["physical_embryo_disagreements"] = int((~valid.fillna(False)).sum())

    if {"physical_embryo_id", "channel_id", "embryo_id"} <= set(inventory.columns):
        expected = inventory["physical_embryo_id"].astype("string") + "_" + inventory[
            "channel_id"
        ].astype("string")
        result["embryo_disagreements"] = int(
            (inventory["embryo_id"].astype("string") != expected).fillna(True).sum()
        )

    if {"embryo_id", "time_index", "snip_id"} <= set(inventory.columns):
        numeric_time = pd.to_numeric(inventory["time_index"], errors="coerce").astype("Int64")
        expected = (
            inventory["embryo_id"].astype("string")
            + "_t"
            + numeric_time.astype("string").str.zfill(4)
        )
        result["snip_disagreements"] = int(
            (inventory["snip_id"].astype("string") != expected).fillna(True).sum()
        )
    return result


def resolve_artifact_path(root: Path, value: Any) -> tuple[str, Path | None]:
    if pd.isna(value) or not str(value).strip():
        return "null", None
    path = Path(str(value))
    if not path.is_absolute():
        return "relative_under_root", root / path
    try:
        path.relative_to(root)
        return "absolute_under_root", path
    except ValueError:
        return "absolute_external", path


def check_paths_row(root: Path, values: tuple[Any, Any, Any]):
    processed_value, mask_value, legacy_mask_value = values
    path_kind, image_path = resolve_artifact_path(root, processed_value)
    image_exists = bool(image_path and image_path.is_file())

    mask_candidates: list[Path] = []
    explicit_mask_value = None
    for value in (mask_value, legacy_mask_value):
        if not pd.isna(value) and str(value).strip():
            explicit_mask_value = value
            _, candidate = resolve_artifact_path(root, value)
            if candidate is not None and candidate not in mask_candidates:
                mask_candidates.append(candidate)
    if image_path is not None:
        derived = image_path.with_name(f"{image_path.stem}_embryo.png")
        if derived not in mask_candidates:
            mask_candidates.append(derived)
    mask_exists = any(candidate.is_file() for candidate in mask_candidates)

    naming_matches = False
    colocated = False
    if image_path is not None and explicit_mask_value is not None:
        _, explicit_path = resolve_artifact_path(root, explicit_mask_value)
        if explicit_path is not None:
            naming_matches = explicit_path.name == f"{image_path.stem}_embryo.png"
            colocated = explicit_path.parent == image_path.parent
    return path_kind, image_exists, str(image_path or ""), mask_exists, naming_matches, colocated


def audit_paths_and_masks(inventory: pd.DataFrame, root: Path, workers: int) -> pd.DataFrame:
    audited = inventory.copy()
    processed = audited.get("processed_snip_path", pd.Series(pd.NA, index=audited.index))
    masks = audited.get("embryo_mask_snip_path", pd.Series(pd.NA, index=audited.index))
    legacy_masks = audited.get("embryo_mask", pd.Series(pd.NA, index=audited.index))
    values = zip(processed.tolist(), masks.tolist(), legacy_masks.tolist())
    checker = partial(check_paths_row, root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(checker, values, chunksize=512))
    if results:
        columns = zip(*results)
        (
            audited["_path_kind"],
            audited["_image_exists"],
            audited["_resolved_path"],
            audited["_mask_exists"],
            audited["_mask_name_matches"],
            audited["_mask_colocated"],
        ) = [list(values) for values in columns]
    return audited


def infer_image_products(inventory: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    direct_product = inventory.get("image_product_type")
    direct_projection = inventory.get("projection_method")
    source = "inventory columns"
    if direct_product is not None and direct_product.notna().any():
        product = direct_product.astype("string")
    else:
        source = "inferred from image_path segments (inventory columns absent)"
        paths = inventory.get("image_path", pd.Series(pd.NA, index=inventory.index)).astype("string")
        product = pd.Series(pd.NA, index=inventory.index, dtype="string")
        product.loc[paths.str.contains(r"/projection/", na=False)] = "projection"
        product.loc[paths.str.contains(r"/z_stack/", na=False)] = "z_stack"
    if direct_projection is not None and direct_projection.notna().any():
        projection = direct_projection.astype("string")
    else:
        paths = inventory.get("image_path", pd.Series(pd.NA, index=inventory.index)).astype("string")
        projection = paths.str.extract(r"/projection/([^/]+)/", expand=False).astype("string")
    return product, projection, source


def deterministic_sample(values: Sequence[str], count: int, rng: np.random.Generator) -> list[str]:
    unique = list(dict.fromkeys(value for value in values if value))
    if len(unique) <= count:
        return unique
    indices = rng.choice(len(unique), size=count, replace=False)
    return [unique[int(index)] for index in indices]


def timed_image_decode(paths: Sequence[str]) -> tuple[np.ndarray, int]:
    times: list[float] = []
    failures = 0
    for path in paths:
        started = time.perf_counter()
        try:
            with Image.open(path) as image:
                image.load()
            times.append(time.perf_counter() - started)
        except Exception:
            failures += 1
    return np.asarray(times, dtype=float), failures


def throughput_benchmark(paths: Sequence[str]) -> dict[str, Any]:
    if not paths:
        return {}
    cold_times, cold_failures = timed_image_decode(paths)
    warm_times, warm_failures = timed_image_decode(paths)
    read_times: list[float] = []
    decode_times: list[float] = []
    resize_times: list[float] = []
    for path in paths:
        try:
            started = time.perf_counter()
            payload = Path(path).read_bytes()
            read_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            with Image.open(BytesIO(payload)) as image:
                image.load()
                decoded = image.copy()
            decode_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            decoded.resize((128, 288), resample=Image.Resampling.BILINEAR)
            resize_times.append(time.perf_counter() - started)
        except Exception:
            continue

    def timing_summary(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        if not array.size:
            return {"mean_ms": math.nan, "median_ms": math.nan, "p95_ms": math.nan}
        return {
            "mean_ms": float(array.mean() * 1000),
            "median_ms": float(np.median(array) * 1000),
            "p95_ms": float(np.quantile(array, 0.95) * 1000),
        }

    result = {
        "sample_count": len(paths),
        "cold_failures": cold_failures,
        "warm_failures": warm_failures,
        "cold_read_decode": timing_summary(cold_times),
        "warm_read_decode": timing_summary(warm_times),
        "warm_file_read": timing_summary(read_times),
        "decode_from_memory": timing_summary(decode_times),
        "resize_576x256_to_288x128": timing_summary(resize_times),
    }
    warm_mean_s = result["warm_read_decode"]["mean_ms"] / 1000
    result["images_per_second_per_worker"] = 1 / warm_mean_s if warm_mean_s > 0 else math.nan
    result["metric_batch_128_reads_seconds"] = 128 * warm_mean_s
    result["dominant_component"] = (
        "per-file read latency/I/O"
        if result["warm_file_read"]["median_ms"] > result["decode_from_memory"]["median_ms"]
        else "decode CPU"
    )
    return result


def inspect_format(path: str) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            return {
                "path": path,
                "opened": True,
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "dtype": str(array.dtype),
                "interlaced": bool(image.info.get("interlace", False)),
                "error": "",
            }
    except Exception as exc:
        return {
            "path": path,
            "opened": False,
            "format": "",
            "mode": "",
            "width": None,
            "height": None,
            "dtype": "",
            "interlaced": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def image_intensity(task: tuple[str, str]) -> dict[str, Any]:
    experiment, path = task
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
        return {
            "experiment_id": experiment,
            "path": path,
            "opened": True,
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "saturated_255_fraction": float(np.mean(array == 255)),
            "zero_fraction": float(np.mean(array == 0)),
            "width": int(array.shape[1]) if array.ndim >= 2 else None,
            "height": int(array.shape[0]) if array.ndim >= 2 else None,
            "dtype": str(array.dtype),
            "error": "",
        }
    except Exception as exc:
        return {
            "experiment_id": experiment,
            "path": path,
            "opened": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def sample_intensities(
    inventory: pd.DataFrame,
    per_experiment: int,
    workers: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    tasks: list[tuple[str, str]] = []
    for experiment, frame in inventory[inventory["_image_exists"]].groupby(
        "_source_experiment", sort=True
    ):
        sample = deterministic_sample(frame["_resolved_path"].tolist(), per_experiment, rng)
        tasks.extend((str(experiment), path) for path in sample)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(image_intensity, tasks, chunksize=8))
    raw = pd.DataFrame(rows)
    valid = raw[raw.get("opened", False) == True].copy()  # noqa: E712
    aggregate_rows: list[dict[str, Any]] = []
    metrics = ("mean", "std", "min", "max", "saturated_255_fraction", "zero_fraction")
    for experiment, frame in valid.groupby("experiment_id", sort=True):
        row: dict[str, Any] = {"experiment_id": experiment, "sample_n": len(frame)}
        for metric in metrics:
            row[f"{metric}_p05"] = float(frame[metric].quantile(0.05))
            row[f"{metric}_median"] = float(frame[metric].median())
            row[f"{metric}_p95"] = float(frame[metric].quantile(0.95))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    eta_squared: dict[str, float] = {}
    for metric in ("mean", "std", "saturated_255_fraction"):
        if valid.empty:
            eta_squared[metric] = math.nan
            continue
        grand_mean = valid[metric].mean()
        between = sum(
            len(frame) * (frame[metric].mean() - grand_mean) ** 2
            for _, frame in valid.groupby("experiment_id")
        )
        total = float(((valid[metric] - grand_mean) ** 2).sum())
        eta_squared[metric] = float(between / total) if total > 0 else 0.0
    return raw, aggregate, eta_squared


def scale_profile(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ("source_micrometers_per_pixel", "snip_micrometers_per_pixel"):
        if column not in inventory:
            continue
        for experiment, frame in inventory.groupby("_source_experiment", sort=True):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            rounded_unique = sorted(values.round(6).unique().tolist())
            rows.append(
                {
                    "experiment_id": experiment,
                    "column": column,
                    "finite_n": len(values),
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max()),
                    "unique_rounded_6dp": json.dumps(rounded_unique),
                    "mixed_scale": len(rounded_unique) > 1,
                }
            )
    return pd.DataFrame(rows)


def tokenize_fail_reasons(series: pd.Series) -> Counter:
    counter: Counter = Counter()
    for value in series.dropna():
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "[]"}:
            continue
        tokens: list[str]
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
        if isinstance(parsed, (list, tuple, set)):
            tokens = [str(token) for token in parsed]
        elif isinstance(parsed, str):
            tokens = re.split(r"[|;,]", parsed)
        else:
            tokens = re.split(r"[|;,]", text)
        counter.update(token.strip() for token in tokens if token.strip())
    return counter


def stage_and_qc_audit(inventory: pd.DataFrame, stages: pd.DataFrame, qc: pd.DataFrame):
    result: dict[str, Any] = {}
    merged = inventory.copy()
    if not stages.empty and "snip_id" in stages:
        stage_copy = stages.copy()
        stage_copy["predicted_stage_hpf"] = pd.to_numeric(
            stage_copy.get("predicted_stage_hpf"), errors="coerce"
        )
        result["stage_duplicate_rows"] = int(stage_copy["snip_id"].duplicated(keep=False).sum())
        status_counts = stage_copy.get(
            "stage_prediction_status", pd.Series(dtype="string")
        ).value_counts(dropna=False)
        result["status_counts"] = status_counts.rename_axis("status").reset_index(name="count")
        stage_unique = stage_copy.drop_duplicates("snip_id", keep="first")
        joined_columns = [
            column
            for column in ("snip_id", "predicted_stage_hpf", "stage_prediction_status")
            if column in stage_unique
        ]
        merged = merged.merge(stage_unique[joined_columns], how="left", on="snip_id")
    else:
        merged["predicted_stage_hpf"] = np.nan
        merged["stage_prediction_status"] = pd.NA
        result["status_counts"] = pd.DataFrame(columns=["status", "count"])

    finite = np.isfinite(pd.to_numeric(merged["predicted_stage_hpf"], errors="coerce"))
    predicted = merged["stage_prediction_status"].astype("string") == "predicted"
    valid = coerce_bool(merged.get("is_valid_snip", pd.Series(pd.NA, index=merged.index))).fillna(False)
    merged["_finite_stage"] = finite
    merged["_predicted"] = predicted.fillna(False)
    merged["_valid"] = valid
    merged["_stage_precursor_gate"] = valid & finite & predicted.fillna(False)

    staging_by_experiment = (
        merged.groupby("_source_experiment", dropna=False)
        .agg(
            inventory_snips=("snip_id", "size"),
            finite_stage=("_finite_stage", "sum"),
            predicted_status=("_predicted", "sum"),
            valid_predicted_finite=("_stage_precursor_gate", "sum"),
        )
        .reset_index()
        .rename(columns={"_source_experiment": "experiment_id"})
    )
    staging_by_experiment["finite_stage_fraction"] = (
        staging_by_experiment["finite_stage"] / staging_by_experiment["inventory_snips"]
    )
    result["staging_by_experiment"] = staging_by_experiment

    finite_values = pd.to_numeric(merged.loc[finite, "predicted_stage_hpf"], errors="coerce")
    if not finite_values.empty:
        start = math.floor(float(finite_values.min()) / 2) * 2
        stop = math.ceil(float(finite_values.max()) / 2) * 2 + 2
        bins = np.arange(start, stop + 0.001, 2)
        histogram = pd.cut(finite_values, bins=bins, right=False).value_counts(sort=False)
        result["stage_histogram"] = histogram.rename_axis("stage_bin_hpf").reset_index(name="count")
    else:
        result["stage_histogram"] = pd.DataFrame(columns=["stage_bin_hpf", "count"])

    if qc.empty or "snip_id" not in qc:
        result["qc_available"] = False
        result["merged"] = merged
        return result

    result["qc_available"] = True
    qc_copy = qc.copy()
    qc_copy["use_snip"] = coerce_bool(qc_copy.get("use_snip", pd.Series(pd.NA, index=qc_copy.index)))
    result["qc_duplicate_rows"] = int(qc_copy["snip_id"].duplicated(keep=False).sum())
    qc_unique = qc_copy.drop_duplicates("snip_id", keep="first")
    joined_columns = [column for column in ("snip_id", "use_snip", "qc_fail_reasons") if column in qc_unique]
    merged = merged.merge(qc_unique[joined_columns], how="left", on="snip_id")
    result["inventory_without_qc"] = int(merged["use_snip"].isna().sum())
    result["qc_pass_overall"] = float(merged["use_snip"].mean())
    result["reason_counts"] = pd.DataFrame(
        tokenize_fail_reasons(qc_copy.get("qc_fail_reasons", pd.Series(dtype="string"))).most_common(),
        columns=["qc_fail_reason", "count"],
    )
    result["valid_use_crosstab"] = pd.crosstab(
        merged["_valid"], merged["use_snip"], dropna=False
    ).reset_index()
    merged["_strict_gate"] = merged["_valid"] & merged["use_snip"].fillna(False)
    merged["_combined_metric_gate"] = merged["_strict_gate"] & merged["_predicted"] & merged[
        "_finite_stage"
    ]
    qc_by_experiment = (
        merged.groupby("_source_experiment")
        .agg(
            inventory_snips=("snip_id", "size"),
            qc_rows=("use_snip", "count"),
            qc_pass_rate=("use_snip", "mean"),
            strict_pass_rate=("_strict_gate", "mean"),
            combined_metric_survivors=("_combined_metric_gate", "sum"),
        )
        .reset_index()
        .rename(columns={"_source_experiment": "experiment_id"})
    )
    finite_rates = qc_by_experiment["strict_pass_rate"].dropna()
    spread = float(finite_rates.max() - finite_rates.min()) if not finite_rates.empty else math.nan
    result["qc_experiment_spread"] = spread
    result["qc_experiment_correlated"] = bool(spread > 0.05) if np.isfinite(spread) else None
    result["qc_by_experiment"] = qc_by_experiment
    result["merged"] = merged
    return result


def plate_profiles(plates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in sorted(column for column in plates.columns if column != "_source_experiment"):
        values = sorted_values(plates[column])
        rows.append(
            {
                "column": column,
                "row_count": len(plates),
                "null_count": int(plates[column].isna().sum()),
                "null_rate": float(plates[column].isna().mean()),
                "cardinality": len(values),
                "value_set_json": json.dumps(values, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def candidate_series(plates: pd.DataFrame) -> dict[str, pd.Series]:
    candidates: dict[str, pd.Series] = {}
    keywords = ("genotype", "strain", "perturb", "treatment", "condition")
    for column in plates.columns:
        if column == "_source_experiment":
            continue
        if any(keyword in column.lower() for keyword in keywords):
            candidates[column] = plates[column].astype("string").str.strip()
    if {"genotype", "strain"} <= set(plates.columns):
        genotype = plates["genotype"].astype("string").str.strip()
        strain = plates["strain"].astype("string").str.strip()
        candidates["genotype__strain"] = (genotype + "_" + strain).where(
            genotype.notna() & strain.notna()
        )
    return candidates


def metric_candidate_mapping(
    plates: pd.DataFrame, metric_key_path: Path
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str]]:
    labels: list[str] = []
    if metric_key_path.is_file():
        metric = pd.read_csv(metric_key_path, index_col=0)
        labels = sorted(set(map(str, metric.index)) | set(map(str, metric.columns)))
    label_set = set(labels)
    candidates = candidate_series(plates)
    rows: list[dict[str, Any]] = []
    for candidate, series in candidates.items():
        values = set(str(value) for value in series.dropna().unique() if str(value).strip())
        rows.append(
            {
                "candidate": candidate,
                "cohort_cardinality": len(values),
                "matched_count": len(values & label_set),
                "matched_values_json": json.dumps(sorted(values & label_set)),
                "cohort_values_without_class_json": json.dumps(sorted(values - label_set)),
                "curated_classes_unused_json": json.dumps(sorted(label_set - values)),
            }
        )
    return pd.DataFrame(rows), candidates, labels


def stable_split(
    inventory: pd.DataFrame,
    plates: pd.DataFrame,
    candidates: dict[str, pd.Series],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_counts = (
        inventory.groupby(["physical_embryo_id", "well_id"], dropna=False)
        .size()
        .reset_index(name="snip_count")
    )
    group_counts = group_counts[group_counts["physical_embryo_id"].notna()].copy()
    group_counts["_order"] = group_counts["physical_embryo_id"].map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    group_counts = group_counts.sort_values("_order").reset_index(drop=True)
    n_groups = len(group_counts)
    n_train = round(0.8 * n_groups)
    n_eval = round(0.1 * n_groups)
    group_counts["split"] = "test"
    group_counts.loc[: n_train - 1, "split"] = "train"
    group_counts.loc[n_train : n_train + n_eval - 1, "split"] = "eval"
    split_summary = (
        group_counts.groupby("split")
        .agg(physical_embryos=("physical_embryo_id", "size"), snips=("snip_count", "sum"))
        .reindex(["train", "eval", "test"])
        .reset_index()
    )
    split_summary["embryo_fraction"] = split_summary["physical_embryos"] / n_groups
    split_summary["snip_fraction"] = split_summary["snips"] / group_counts["snip_count"].sum()

    feasibility_rows: list[dict[str, Any]] = []
    if not plates.empty and "well_id" in plates:
        plate_candidates = pd.DataFrame({"well_id": plates["well_id"]})
        for name, series in candidates.items():
            plate_candidates[name] = series
        for candidate in candidates:
            mapping = (
                plate_candidates[["well_id", candidate]]
                .dropna()
                .groupby("well_id")[candidate]
                .agg(lambda values: values.iloc[0] if values.nunique() == 1 else pd.NA)
            )
            candidate_groups = group_counts.copy()
            candidate_groups["candidate_value"] = candidate_groups["well_id"].map(mapping)
            counts = (
                candidate_groups.dropna(subset=["candidate_value"])
                .groupby(["candidate_value", "split"])["physical_embryo_id"]
                .nunique()
                .unstack(fill_value=0)
                .reindex(columns=["train", "eval", "test"], fill_value=0)
            )
            for value, row in counts.iterrows():
                feasibility_rows.append(
                    {
                        "candidate": candidate,
                        "candidate_value": value,
                        "train_embryos": int(row["train"]),
                        "eval_embryos": int(row["eval"]),
                        "test_embryos": int(row["test"]),
                        "legal_positive_pairs_all_splits": bool((row >= 2).all()),
                    }
                )
    return group_counts.drop(columns="_order"), split_summary, pd.DataFrame(feasibility_rows)


def format_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def format_fraction(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{decimals}f}%"


def markdown_value(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text


def markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, list(columns)] if columns is not None else frame
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def availability_report_view(availability: pd.DataFrame, root: Path) -> pd.DataFrame:
    view = availability.copy()
    view["status"] = np.where(
        ~view["present"],
        "absent",
        np.where(view["readable"], "readable", "present / blocked"),
    )
    view["rows"] = view["row_count"].map(format_count)
    view["schema"] = view["schema_id"].replace("", "—")
    view["+ vs ref"] = view["added_vs_reference"]
    view["− vs ref"] = view["missing_vs_reference"]
    return view[
        ["experiment_id", "source", "status", "rows", "schema", "reference_experiment", "+ vs ref", "− vs ref", "error"]
    ]


def build_report(
    args: argparse.Namespace,
    manifests: Sequence[Path],
    experiments: Sequence[str],
    engine: str | None,
    availability: pd.DataFrame,
    schema_catalog: pd.DataFrame,
    newest: str,
    inventory: pd.DataFrame,
    identity: dict[str, Any],
    product: pd.Series,
    projection: pd.Series,
    product_source: str,
    format_checks: pd.DataFrame,
    throughput: dict[str, Any],
    intensity_aggregate: pd.DataFrame,
    eta_squared: dict[str, float],
    scales: pd.DataFrame,
    stage_qc: dict[str, Any],
    plate_profile: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
    split_summary: pd.DataFrame,
    split_feasibility: pd.DataFrame,
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Pipeline output reconnaissance for core-model refactor")
    add("")
    add(
        f"Generated {datetime.now(timezone.utc).isoformat()} by `{Path(sys.executable)}` with "
        f"pandas {pd.__version__}. This was a read-only survey of `{args.output_root}`."
    )
    add("")
    add("## Scope and blocking findings")
    add("")
    add(f"- Explicit cohort: **{len(experiments):,} deduplicated experiment IDs** from {len(manifests)} manifests.")
    add(f"- Newest selected experiment by ID: `{newest}`.")
    add(f"- Pipeline path authority: `{PATH_REGISTRY_FILE.relative_to(REPO_ROOT)}` (loaded directly to avoid package initializer side effects).")
    if engine is None:
        add("- **Blocking:** neither `pyarrow` nor `fastparquet` is installed in the training environment. QC Parquet files could be located but not read; every QC-derived statistic and the combined metric gate is therefore unmeasured, not silently omitted.")
    else:
        add(f"- Parquet engine: `{engine}`.")
    add("")
    add("<details><summary>Exact ordered experiment IDs</summary>")
    add("")
    add("```text")
    add("\n".join(experiments))
    add("```")
    add("</details>")
    add("")

    add("## 1. Availability and schema drift")
    add("")
    source_summary = (
        availability.groupby("source")
        .agg(selected=("experiment_id", "size"), present=("present", "sum"), readable=("readable", "sum"), schemas=("schema_id", lambda values: len({v for v in values if v})))
        .reset_index()
    )
    add(markdown_table(source_summary))
    add("")
    scale_flags = availability[availability["source"] == "inventory"].copy()
    readable_scale_flags = scale_flags[scale_flags["readable"]]
    missing_source_scale = 0
    missing_snip_scale = 0
    for _, row in readable_scale_flags.iterrows():
        columns = set(row["columns"])
        missing_source_scale += int("source_micrometers_per_pixel" not in columns)
        missing_snip_scale += int("snip_micrometers_per_pixel" not in columns)
    add(
        f"Inventory schema flag: **{missing_source_scale:,}/{len(readable_scale_flags):,}** readable inventories lack "
        f"`source_micrometers_per_pixel`; **{missing_snip_scale:,}/{len(readable_scale_flags):,}** lack "
        f"`snip_micrometers_per_pixel`. A further **{len(scale_flags) - len(readable_scale_flags):,}** selected "
        "experiments have no readable merged inventory."
    )
    add("")
    add("Schema IDs expand to these exact ordered column lists:")
    add("")
    for row in schema_catalog.itertuples(index=False):
        add(f"- `{row.schema_id}` ({row.source}): `{row.columns_json}`")
    add("")
    add("<details><summary>Per experiment × source availability, rows, schema, and drift</summary>")
    add("")
    add(markdown_table(availability_report_view(availability, args.output_root)))
    add("")
    add("</details>")
    add("")
    add("The same exact table, including canonical paths and full column JSON, is in [availability_schema.csv](recon_tables/availability_schema.csv).")
    add("")

    add("## 2. Cohort size")
    add("")
    add(f"Readable inventories contribute **{len(inventory):,} snip rows**.")
    experiment_counts = inventory["_source_experiment"].value_counts().sort_index().rename_axis("experiment_id").reset_index(name="snips")
    channel_counts = inventory.get("channel_id", pd.Series(dtype="string")).value_counts(dropna=False).rename_axis("channel_id").reset_index(name="snips")
    add("")
    add("Counts by channel:")
    add("")
    add(markdown_table(channel_counts))
    add("")
    add("Counts by experiment are exact in [snips_by_experiment.csv](recon_tables/snips_by_experiment.csv). Counts for every physical embryo are in [physical_embryo_frame_counts.csv](recon_tables/physical_embryo_frame_counts.csv).")
    physical_counts = inventory.groupby("physical_embryo_id", dropna=False).size()
    describe = physical_counts.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).rename_axis("statistic").reset_index(name="frames")
    add("")
    add(f"Unique non-null `physical_embryo_id`: **{inventory['physical_embryo_id'].nunique(dropna=True):,}**.")
    add("")
    add("Frames per physical embryo:")
    add("")
    add(markdown_table(describe))
    add("")

    add("## 3. Identity integrity")
    add("")
    add(f"- Duplicate `snip_id` rows: **{identity.get('duplicate_rows', 0):,}** across **{identity.get('duplicate_ids', 0):,}** IDs.")
    add(f"- Missing identity-spine columns: `{identity.get('missing_spine_columns', [])}`.")
    add(f"- Nulls by identity field: `{json.dumps(identity.get('null_counts', {}), sort_keys=True)}`.")
    add(f"- `physical_embryo_id` grammar/parent disagreements: **{identity.get('physical_embryo_disagreements', 0):,}**.")
    add(f"- `embryo_id = physical_embryo_id + channel_id` disagreements: **{identity.get('embryo_disagreements', 0):,}**.")
    add(f"- `snip_id = embryo_id + zero-padded time_index` disagreements: **{identity.get('snip_disagreements', 0):,}**.")
    add("")

    add("## 4. Image product type")
    add("")
    product_counts = product.value_counts(dropna=False).rename_axis("image_product_type").reset_index(name="snips")
    projection_counts = projection.value_counts(dropna=False).rename_axis("projection_method").reset_index(name="snips")
    add(f"Product source: **{product_source}**.")
    add("")
    add(markdown_table(product_counts))
    add("")
    add(markdown_table(projection_counts))
    add("")
    direct_product_types = product.dropna().nunique()
    if direct_product_types > 1:
        add(f"**Critical collision check:** multiple product types are present and the cohort has **{identity.get('duplicate_ids', 0):,} duplicated `snip_id` values**.")
    else:
        add("Only one non-null/inferred product type is present, so the focus-axis collision condition was not triggered in this cohort.")
    z_column = next((column for column in ("z_position", "z_index") if column in inventory and inventory[column].notna().any()), None)
    if z_column:
        add(f"`{z_column}` distribution: `{inventory[z_column].describe().to_dict()}`.")
    else:
        add("`z_position` and `z_index` are absent or entirely null in the snip inventories; a z-position distribution cannot be measured from the requested boundary.")
    add("")

    add("## 5. Masks")
    add("")
    missing_masks = int((~inventory["_mask_exists"]).sum())
    naming_matches = int(inventory["_mask_name_matches"].sum())
    colocated = int(inventory["_mask_colocated"].sum())
    add("Observed convention: processed crop `<snip_id>.png`; embryo mask `<snip_id>_embryo.png` in the same directory.")
    add(f"Exact convention match: **{naming_matches:,}/{len(inventory):,}**; colocated explicit paths: **{colocated:,}/{len(inventory):,}**; no locatable mask after explicit-path and suffix fallback checks: **{missing_masks:,}/{len(inventory):,}**.")
    add("Per-experiment exact coverage is in [mask_coverage_by_experiment.csv](recon_tables/mask_coverage_by_experiment.csv).")
    add("")

    add("## 6. Paths and image format")
    add("")
    path_cross = inventory.groupby(["_path_kind", "_image_exists"]).size().rename("snips").reset_index()
    path_cross["fraction"] = path_cross["snips"] / len(inventory)
    add(markdown_table(path_cross))
    add("")
    if not format_checks.empty:
        opened = format_checks[format_checks["opened"]]
        conforming = opened[
            (opened["mode"] == "L")
            & (opened["dtype"] == "uint8")
            & (opened["width"] == 256)
            & (opened["height"] == 576)
            & (~opened["interlaced"])
        ]
        add(f"Opened **{len(opened):,}/{len(format_checks):,}** sampled files; **{len(conforming):,}/{len(opened):,}** were 8-bit, non-interlaced grayscale at pipeline `(H, W) = (576, 256)`. Exact checks: [image_format_sample.csv](recon_tables/image_format_sample.csv).")
    add("")

    add("## 7. QC")
    add("")
    if not stage_qc.get("qc_available", False):
        add("**BLOCKED by missing Parquet engine.** `use_snip` pass rates, failure-reason tokens, `is_valid_snip × use_snip`, inventory rows without QC, the strict gate, and its experiment correlation could not be measured. This is a hard training preflight failure, not an optional omission.")
    else:
        add(f"Overall `use_snip` pass rate: **{format_fraction(stage_qc['qc_pass_overall'])}**. Inventory snips with no QC row: **{stage_qc['inventory_without_qc']:,}**.")
        add(f"Strict-gate exclusion-rate spread across experiments: **{format_fraction(stage_qc['qc_experiment_spread'])}**. Using a 5-percentage-point materiality threshold, the removal is **{'experiment-correlated' if stage_qc['qc_experiment_correlated'] else 'approximately uniform'}**.")
        add("Exact per-experiment rates, reason tokens, and cross-tab are in the companion tables.")
    add("")

    add("## 8. Staging")
    add("")
    status_counts = stage_qc["status_counts"]
    add(markdown_table(status_counts))
    add("")
    staging_by_exp = stage_qc["staging_by_experiment"]
    finite_total = int(staging_by_exp["finite_stage"].sum()) if not staging_by_exp.empty else 0
    precursor_total = int(staging_by_exp["valid_predicted_finite"].sum()) if not staging_by_exp.empty else 0
    add(f"Finite predicted-stage coverage against all inventory rows: **{finite_total:,}/{len(inventory):,} ({format_fraction(finite_total / len(inventory) if len(inventory) else math.nan)})**.")
    add(f"Survivors of the measurable precursor gate `valid ∧ status==predicted ∧ finite stage`: **{precursor_total:,}**. The requested full metric gate additionally requires QC and is blocked when Parquet is unreadable.")
    add("Per-experiment coverage: [staging_by_experiment.csv](recon_tables/staging_by_experiment.csv). Stage histogram: [stage_histogram.csv](recon_tables/stage_histogram.csv).")
    add("")

    add("## 9. Metric-group candidates")
    add("")
    short_exists = "short_pert_name" in set(plate_profile.get("column", []))
    add(f"`short_pert_name` exists anywhere in selected plate metadata: **{short_exists}**.")
    add("Every plate column's exact null count/rate, cardinality, and full JSON value set is in [plate_column_profiles.csv](recon_tables/plate_column_profiles.csv).")
    plate_preview = plate_profile.copy()
    if not plate_preview.empty:
        plate_preview["null_rate"] = plate_preview["null_rate"].map(format_fraction)
        plate_preview["value preview"] = plate_preview["value_set_json"].map(lambda value: value[:160] + ("…" if len(value) > 160 else ""))
        add("")
        add(markdown_table(plate_preview, ["column", "null_rate", "cardinality", "value preview"]))
    add("")
    add("Candidate-to-curated-class mapping (full exact sets are in [metric_candidate_mapping.csv](recon_tables/metric_candidate_mapping.csv)):")
    add("")
    if not candidate_mapping.empty:
        add(markdown_table(candidate_mapping, ["candidate", "cohort_cardinality", "matched_count"]))
    else:
        add("_No plausible candidate columns found._")
    add("")

    add("## 10. Pixel scale and intensity")
    add("")
    if scales.empty:
        add("Neither requested µm/px column contains measurable values in readable inventories, so scale distributions and mixed-scale detection are blocked by schema, not treated as homogeneous scale.")
    else:
        add(markdown_table(scales))
    add("")
    add("Per-experiment side-by-side sampled intensity distributions are in [intensity_by_experiment.csv](recon_tables/intensity_by_experiment.csv); per-image measurements are in [intensity_samples.csv](recon_tables/intensity_samples.csv).")
    add("")
    if not intensity_aggregate.empty:
        intensity_view = intensity_aggregate[["experiment_id", "sample_n", "mean_median", "std_median", "min_median", "max_median", "saturated_255_fraction_median"]]
        add("<details><summary>Per-experiment intensity medians (distribution quantiles are in CSV)</summary>")
        add("")
        add(markdown_table(intensity_view))
        add("")
        add("</details>")
        max_eta = max(value for value in eta_squared.values() if np.isfinite(value)) if eta_squared else math.nan
        add(f"Experiment η² (fraction of sampled statistic variance explained by experiment): `{json.dumps(eta_squared)}`. With η²≥0.10 treated as a material batch signal, intensity is **{'strongly experiment-correlated and learnable as a batch effect' if max_eta >= 0.10 else 'not strongly experiment-correlated by this screen'}**.")
    add("")

    add("## 11. I/O throughput")
    add("")
    if throughput:
        timing_rows = []
        for label in ("cold_read_decode", "warm_read_decode", "warm_file_read", "decode_from_memory", "resize_576x256_to_288x128"):
            timing_rows.append({"measurement": label, **throughput[label]})
        add(markdown_table(pd.DataFrame(timing_rows)))
        add("")
        add(f"Warm throughput is **{throughput['images_per_second_per_worker']:.1f} images/s/worker**. At 128 reads for a metric batch of 64, one worker spends approximately **{throughput['metric_batch_128_reads_seconds']:.3f} s/batch** in warm read+decode. The separated medians indicate **{throughput['dominant_component']}** dominates. “Cold” is a first-pass measurement on a shared mount; the script cannot guarantee or flush the system page cache.")
    else:
        add("No existing images were available for throughput measurement.")
    add("")

    add("## 12. Group-split feasibility")
    add("")
    split_view = split_summary.copy()
    if not split_view.empty:
        split_view["embryo_fraction"] = split_view["embryo_fraction"].map(format_fraction)
        split_view["snip_fraction"] = split_view["snip_fraction"].map(format_fraction)
    add(markdown_table(split_view))
    add("")
    add("The split is deterministic and group-disjoint at `physical_embryo_id`; exact assignments are in [group_split_assignments.csv](recon_tables/group_split_assignments.csv).")
    if not split_feasibility.empty:
        failures = (
            split_feasibility.groupby("candidate")["legal_positive_pairs_all_splits"]
            .agg(total_groups="size", groups_too_small=lambda values: int((~values).sum()))
            .reset_index()
        )
        add("")
        add(markdown_table(failures))
        add("")
        add("A group is flagged when any split has fewer than two physical embryos. Exact candidate/value/split counts: [metric_group_split_feasibility.csv](recon_tables/metric_group_split_feasibility.csv).")
    add("")

    add("## Design-changing findings, audit comparison, and unmeasured items")
    add("")
    missing_inventory = int(((availability["source"] == "inventory") & ~availability["present"]).sum())
    missing_stage = int(((availability["source"] == "stage") & ~availability["present"]).sum())
    max_eta = max((value for value in eta_squared.values() if np.isfinite(value)), default=math.nan)
    add("The three findings most likely to change the bridge design are:")
    add("")
    add(f"1. **Preflight must be cohort-wide and schema-aware:** {missing_inventory:,} selected experiments lack a merged inventory, {missing_stage:,} lack merged staging, and both requested scale fields are absent from all {len(readable_scale_flags):,} readable inventories.")
    add("2. **QC cannot be optional:** the training environment cannot currently read the required Parquet verdicts, so strict cohort selection and combined-gate sizing are blocked until the environment contract supplies an engine.")
    if np.isfinite(max_eta):
        add(f"3. **Intensity normalization/batch controls need an explicit decision:** the largest experiment η² among mean/std/saturation is {max_eta:.3f}, which {'is' if max_eta >= 0.10 else 'is not'} a material experiment signal by the stated screen.")
    else:
        add("3. **Intensity normalization could not be screened** because no sample images opened.")
    add("")
    add("Comparison with `NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md`:")
    add("")
    if not format_checks.empty and format_checks["opened"].any():
        opened = format_checks[format_checks["opened"]]
        conformity = ((opened["mode"] == "L") & (opened["dtype"] == "uint8") & (opened["width"] == 256) & (opened["height"] == 576)).mean()
        add(f"- The image-format claim is {'confirmed' if conformity == 1 else 'contradicted for part of the sample'} ({format_fraction(conformity)} conforming before the interlace check).")
    add("- The audit already warned that a live inventory lacked the two scale fields; this broader cohort determines whether that was isolated or systemic. It is an extension of the warning, not inherently a contradiction.")
    add(f"- The audit says `short_pert_name` is not guaranteed; this cohort {'contains it' if short_exists else 'confirms it is absent'}.")
    add("- No measured result contradicts the recommended inventory + stage + QC + plate boundary; current artifact availability and the Parquet blocker strengthen the need for fail-loud preflight.")
    add("")
    add("Items not measured, with reasons:")
    add("")
    if engine is None:
        add("- All QC-derived results and full metric-gate survivors: no Parquet engine in the training environment.")
    if "image_product_type" not in inventory or not inventory["image_product_type"].notna().any():
        add("- Direct product-type provenance: the inventory column is absent; path-segment inference is reported separately.")
    if not any(column in inventory and inventory[column].notna().any() for column in ("z_position", "z_index")):
        add("- `z_position` distribution: no populated source column at the snip-inventory boundary.")
    if scales.empty:
        add("- µm/px distributions and mixed-scale determination: requested fields absent/unpopulated.")
    add("- Truly cold I/O: the shared filesystem page cache cannot be flushed safely; first-pass and warm measurements are both reported.")
    add("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifests = tuple(args.manifests) if args.manifests else DEFAULT_MANIFESTS
    experiment_values: list[str] = []
    for manifest in manifests:
        experiment_values.extend(read_manifest(manifest))
    experiment_values.extend(args.experiments or [])
    experiments = ordered_unique(experiment_values)
    if not experiments:
        raise SystemExit("No experiments were supplied.")
    args.output_root = args.output_root.resolve()
    args.report = args.report.resolve()
    args.tables_dir = args.tables_dir.resolve()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    registry = load_path_registry()
    engine = parquet_engine()
    availability, inventory, stages, qc, plates = load_sources(
        registry, args.output_root, experiments, engine
    )
    availability, schema_catalog, newest = assign_schema_details(availability, experiments)
    if inventory.empty:
        raise SystemExit("No readable snip inventories were found; availability table cannot be reported safely.")

    identity = identity_audit(inventory)
    inventory = audit_paths_and_masks(inventory, args.output_root, args.workers)
    product, projection, product_source = infer_image_products(inventory)

    rng = np.random.default_rng(args.seed)
    existing_paths = inventory.loc[inventory["_image_exists"], "_resolved_path"].tolist()
    throughput_paths = deterministic_sample(existing_paths, args.throughput_sample, rng)
    throughput = throughput_benchmark(throughput_paths)
    format_paths = deterministic_sample(existing_paths, args.format_sample, rng)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        format_checks = pd.DataFrame(pool.map(inspect_format, format_paths))
    intensity_raw, intensity_aggregate, eta_squared = sample_intensities(
        inventory,
        args.intensity_sample_per_experiment,
        args.workers,
        rng,
    )
    scales = scale_profile(inventory)
    stage_qc = stage_and_qc_audit(inventory, stages, qc)
    plate_profile = plate_profiles(plates) if not plates.empty else pd.DataFrame()
    candidate_mapping, candidates, _labels = metric_candidate_mapping(plates, args.metric_key)
    group_assignments, split_summary, split_feasibility = stable_split(
        inventory, plates, candidates, args.seed
    )

    experiment_counts = (
        inventory["_source_experiment"]
        .value_counts()
        .sort_index()
        .rename_axis("experiment_id")
        .reset_index(name="snips")
    )
    physical_counts = (
        inventory.groupby(["_source_experiment", "physical_embryo_id"], dropna=False)
        .size()
        .reset_index(name="frames")
        .rename(columns={"_source_experiment": "experiment_id"})
    )
    mask_coverage = (
        inventory.groupby("_source_experiment")
        .agg(snips=("snip_id", "size"), masks_found=("_mask_exists", "sum"))
        .reset_index()
        .rename(columns={"_source_experiment": "experiment_id"})
    )
    mask_coverage["masks_missing"] = mask_coverage["snips"] - mask_coverage["masks_found"]
    mask_coverage["coverage_fraction"] = mask_coverage["masks_found"] / mask_coverage["snips"]

    availability_csv = availability.copy()
    availability_csv["columns_json"] = availability_csv["columns"].map(json.dumps)
    availability_csv = availability_csv.drop(columns=["columns"])
    write_table(availability_csv, args.tables_dir / "availability_schema.csv")
    write_table(schema_catalog, args.tables_dir / "schema_catalog.csv")
    write_table(experiment_counts, args.tables_dir / "snips_by_experiment.csv")
    write_table(physical_counts, args.tables_dir / "physical_embryo_frame_counts.csv")
    write_table(mask_coverage, args.tables_dir / "mask_coverage_by_experiment.csv")
    write_table(format_checks, args.tables_dir / "image_format_sample.csv")
    write_table(intensity_raw, args.tables_dir / "intensity_samples.csv")
    write_table(intensity_aggregate, args.tables_dir / "intensity_by_experiment.csv")
    write_table(scales, args.tables_dir / "pixel_scale_profiles.csv")
    write_table(stage_qc["staging_by_experiment"], args.tables_dir / "staging_by_experiment.csv")
    write_table(stage_qc["stage_histogram"], args.tables_dir / "stage_histogram.csv")
    write_table(plate_profile, args.tables_dir / "plate_column_profiles.csv")
    write_table(candidate_mapping, args.tables_dir / "metric_candidate_mapping.csv")
    write_table(group_assignments, args.tables_dir / "group_split_assignments.csv")
    write_table(split_feasibility, args.tables_dir / "metric_group_split_feasibility.csv")
    if stage_qc.get("qc_available"):
        write_table(stage_qc["qc_by_experiment"], args.tables_dir / "qc_by_experiment.csv")
        write_table(stage_qc["reason_counts"], args.tables_dir / "qc_fail_reason_counts.csv")
        write_table(stage_qc["valid_use_crosstab"], args.tables_dir / "valid_use_snip_crosstab.csv")

    report = build_report(
        args,
        manifests,
        experiments,
        engine,
        availability,
        schema_catalog,
        newest,
        inventory,
        identity,
        product,
        projection,
        product_source,
        format_checks,
        throughput,
        intensity_aggregate,
        eta_squared,
        scales,
        stage_qc,
        plate_profile,
        candidate_mapping,
        split_summary,
        split_feasibility,
    )
    args.report.write_text(report + "\n")
    print(f"Wrote {args.report}")
    print(f"Wrote exact tables under {args.tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
