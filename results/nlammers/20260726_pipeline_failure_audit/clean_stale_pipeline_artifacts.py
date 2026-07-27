#!/usr/bin/env python
"""Find and optionally remove stale Keyence/YX1 QC, stage, and Snakemake state.

The default mode is a dry run.  ``--apply`` removes only artifacts that fail a
current product contract, their validation sentinels, stale snip-QC resolver
plans, orphan validation sentinels, stale Snakemake locks, and incomplete-state
records (plus the exact output named by such a record).

SeaHub is intentionally out of scope: the experiment universe comes from
``front_half_archive.txt`` and is restricted to Keyence/YX1 rows.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from data_pipeline.feature_extraction.stage_predictions.contract import (
    STAGE_PREDICTION_TABLE_COLUMNS,
    validate_stage_prediction_features,
)
from data_pipeline.quality_control.death_detection.contract import (
    DEATH_DETECTION_QC_TABLE_COLUMNS,
    DEATH_EVENT_TABLE_COLUMNS,
    validate_death_detection_qc,
    validate_death_event,
)
from data_pipeline.quality_control.focus_qc.contract import (
    FOCUS_QC_TABLE_COLUMNS,
    validate_focus_qc,
)
from data_pipeline.quality_control.mask_quality_qc.contract import (
    MASK_QUALITY_QC_TABLE_COLUMNS,
    validate_mask_quality_qc,
)
from data_pipeline.quality_control.motion_blur_qc.contract import (
    MOTION_BLUR_QC_TABLE_COLUMNS,
    validate_motion_blur_qc,
)
from data_pipeline.quality_control.snip_qc.contract import (
    SNIP_QC_TABLE_COLUMNS,
    validate_snip_qc,
)
from data_pipeline.quality_control.surface_area_qc.contract import (
    SURFACE_AREA_QC_TABLE_COLUMNS,
    validate_surface_area_qc,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
)
MANIFEST = (
    REPO_ROOT
    / "src/data_pipeline/pipeline_orchestrator/manifests/front_half_archive.txt"
)
DEFAULT_REPORT = Path(__file__).with_name("stale_artifact_cleanup_report.csv")


@dataclass(frozen=True)
class ProductSpec:
    name: str
    stage_dir: str
    product_dir: str
    glob: str
    required_columns: tuple[str, ...]
    validator: Callable[..., None]


PRODUCTS = (
    ProductSpec(
        "focus_qc",
        "quality_control",
        "focus_qc",
        "*_focus_qc.csv",
        tuple(FOCUS_QC_TABLE_COLUMNS),
        validate_focus_qc,
    ),
    ProductSpec(
        "motion_blur_qc",
        "quality_control",
        "motion_blur_qc",
        "*_motion_blur_qc.csv",
        tuple(MOTION_BLUR_QC_TABLE_COLUMNS),
        validate_motion_blur_qc,
    ),
    ProductSpec(
        "surface_area_qc",
        "quality_control",
        "surface_area_qc",
        "*_surface_area_qc.csv",
        tuple(SURFACE_AREA_QC_TABLE_COLUMNS),
        validate_surface_area_qc,
    ),
    ProductSpec(
        "mask_quality_qc",
        "quality_control",
        "mask_quality_qc",
        "*_mask_quality_qc.csv",
        tuple(MASK_QUALITY_QC_TABLE_COLUMNS),
        validate_mask_quality_qc,
    ),
    ProductSpec(
        "death_detection_qc",
        "quality_control",
        "death_detection",
        "*_death_detection_qc.csv",
        tuple(DEATH_DETECTION_QC_TABLE_COLUMNS),
        validate_death_detection_qc,
    ),
    ProductSpec(
        "death_event",
        "quality_control",
        "death_detection",
        "*_death_event.csv",
        tuple(DEATH_EVENT_TABLE_COLUMNS),
        validate_death_event,
    ),
    ProductSpec(
        "snip_qc",
        "quality_control",
        "snip_qc",
        "*_snip_qc.parquet",
        tuple(SNIP_QC_TABLE_COLUMNS),
        validate_snip_qc,
    ),
    ProductSpec(
        "stage_predictions",
        "feature_extraction",
        "stage_predictions",
        "*_stage_predictions.csv",
        tuple(STAGE_PREDICTION_TABLE_COLUMNS),
        validate_stage_prediction_features,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the stale artifacts. Without this flag, only report them.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="CSV audit trail written in both dry-run and apply modes.",
    )
    parser.add_argument(
        "--active-experiment",
        action="append",
        default=[],
        help="Experiment to exclude from mutation because it is currently running.",
    )
    parser.add_argument(
        "--apply-existing-report",
        type=Path,
        help=(
            "Skip rescanning and apply the planned_action=delete rows from an "
            "existing cleanup report. Permission-blocked canonical product "
            "directories are quarantined outside the pipeline stage trees."
        ),
    )
    return parser.parse_args()


def experiment_universe() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        experiment, scope, *_ = line.split()
        if scope in {"Keyence", "YX1"}:
            result[experiment] = scope
    return result


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def file_columns(path: Path) -> tuple[str, ...]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return tuple(pq.read_schema(path).names)
    with path.open(newline="") as handle:
        return tuple(next(csv.reader(handle)))


def add_record(
    records: list[dict[str, str]],
    *,
    experiment: str,
    scope: str,
    product: str,
    kind: str,
    path: Path,
    reason: str,
    missing_columns: tuple[str, ...] = (),
) -> None:
    records.append(
        {
            "experiment_id": experiment,
            "scope": scope,
            "product": product,
            "kind": kind,
            "path": str(path),
            "reason": reason.replace("\n", " ")[:2000],
            "missing_columns": "|".join(missing_columns),
        }
    )


def paired_sentinel(path: Path) -> Path:
    return Path(f"{path}.validated")


def merged_artifacts(product_root: Path, glob: str) -> list[Path]:
    return [path for path in product_root.glob(glob) if path.is_file()]


def check_resolver_json(path: Path) -> str | None:
    expected = {
        "focus_qc": {"focus_qc_applicability"},
        "motion_blur_qc": {"motion_blur_qc_applicability"},
        "surface_area_qc": {"surface_area_qc_applicability"},
    }
    try:
        payload = json.loads(path.read_text())
        sources = {
            item["step"]: set(item.get("applicability_columns", ()))
            for item in payload["resolved_sources"]
        }
    except Exception as exc:
        return f"unreadable resolver plan: {type(exc).__name__}: {exc}"
    errors = []
    for step, columns in expected.items():
        if step in sources and not columns.issubset(sources[step]):
            errors.append(f"{step} missing {sorted(columns - sources[step])}")
    return "; ".join(errors) or None


def decode_incomplete_output(record: Path, incomplete_root: Path) -> Path | None:
    relative = record.relative_to(incomplete_root)
    token = "".join(relative.parts)
    try:
        token += "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(token).decode()
    except Exception:
        return None
    output = Path(decoded)
    return output if output.is_absolute() else None


def assert_safe(path: Path, experiments: dict[str, str]) -> None:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(OUTPUT_ROOT):
        raise RuntimeError(f"refusing path outside output root: {path}")
    if not any(experiment in resolved.parts for experiment in experiments):
        raise RuntimeError(f"refusing path outside Keyence/YX1 universe: {path}")


def canonical_product_root(path: Path, experiments: dict[str, str]) -> Path | None:
    """Return the stage/experiment/product root for a generated artifact."""
    relative = path.resolve(strict=False).relative_to(OUTPUT_ROOT)
    parts = relative.parts
    if len(parts) < 3:
        return None
    if parts[0] in {"quality_control", "feature_extraction"}:
        if parts[1] not in experiments:
            return None
        return OUTPUT_ROOT.joinpath(*parts[:3])
    if parts[0] == "work_directories" and parts[1] in experiments:
        return OUTPUT_ROOT / parts[0] / parts[1] / ".snakemake"
    return None


def apply_existing_report(report: Path, experiments: dict[str, str]) -> None:
    with report.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = sorted(
        {
            Path(row["path"])
            for row in rows
            if row.get("planned_action") == "delete"
        },
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in selected:
        assert_safe(path, experiments)

    deleted = 0
    blocked: list[tuple[Path, str]] = []
    for path in selected:
        if not path.exists():
            continue
        try:
            path.unlink()
            deleted += 1
        except PermissionError as exc:
            blocked.append((path, str(exc)))

    quarantine = OUTPUT_ROOT / ".stale_contract_quarantine_20260726"
    roots = sorted(
        {
            root
            for path, _error in blocked
            if (root := canonical_product_root(path, experiments)) is not None
            and root.exists()
        }
    )
    quarantined: dict[Path, Path] = {}
    still_blocked_roots: list[tuple[Path, str]] = []
    for root in roots:
        relative = root.relative_to(OUTPUT_ROOT)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(
                f"{destination.name}_{datetime.now().strftime('%H%M%S%f')}"
            )
        try:
            root.rename(destination)
            quarantined[root] = destination
        except PermissionError as exc:
            still_blocked_roots.append((root, str(exc)))

    blocked_report = report.with_name(
        f"{report.stem}_permission_blocked.csv"
    )
    with blocked_report.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "error", "canonical_root", "quarantine_destination"]
        )
        for path, error in blocked:
            root = canonical_product_root(path, experiments)
            writer.writerow(
                [
                    path,
                    error,
                    root or "",
                    quarantined.get(root, "") if root else "",
                ]
            )
        for root, error in still_blocked_roots:
            writer.writerow([root, error, root, ""])

    unresolved = [
        path
        for path, _error in blocked
        if (root := canonical_product_root(path, experiments)) is None
        or root not in quarantined
    ]
    print(f"report paths selected: {len(selected)}")
    print(f"files deleted now: {deleted}")
    print(f"canonical product directories quarantined: {len(quarantined)}")
    print(f"permission-blocked paths left canonical: {len(unresolved)}")
    print(f"permission report: {blocked_report}")
    if unresolved:
        raise SystemExit(5)


def main() -> None:
    args = parse_args()
    experiments = experiment_universe()
    if args.apply_existing_report is not None:
        apply_existing_report(args.apply_existing_report, experiments)
        return
    active = set(args.active_experiment)
    unknown_active = active - experiments.keys()
    if unknown_active:
        raise SystemExit(f"unknown active experiment(s): {sorted(unknown_active)}")

    records: list[dict[str, str]] = []
    delete_paths: set[Path] = set()

    for experiment, scope in sorted(experiments.items()):
        if experiment in active:
            continue

        for spec in PRODUCTS:
            product_root = (
                OUTPUT_ROOT / spec.stage_dir / experiment / spec.product_dir
            )
            if not product_root.is_dir():
                continue
            invalid_in_product = False
            for path in product_root.rglob(spec.glob):
                if not path.is_file():
                    continue
                columns: tuple[str, ...] = ()
                try:
                    columns = file_columns(path)
                    missing = tuple(
                        column
                        for column in spec.required_columns
                        if column not in columns
                    )
                    if missing:
                        invalid_in_product = True
                        add_record(
                            records,
                            experiment=experiment,
                            scope=scope,
                            product=spec.name,
                            kind="stale_schema",
                            path=path,
                            reason=(
                                "missing current contract columns: "
                                f"{list(missing)}"
                            ),
                            missing_columns=missing,
                        )
                        delete_paths.add(path)
                        sentinel = paired_sentinel(path)
                        if sentinel.exists():
                            delete_paths.add(sentinel)
                        continue
                    frame = read_table(path)
                    spec.validator(
                        frame,
                        check_sources=False,
                        scope_label=str(path),
                    )
                except Exception as exc:
                    missing = tuple(
                        column
                        for column in spec.required_columns
                        if column not in columns
                    )
                    # An unreadable/headerless artifact is safe to invalidate.  A
                    # current-schema semantic failure may reproduce identically,
                    # so report it but do not claim deletion is a repair.
                    should_delete = not columns or bool(missing)
                    invalid_in_product = invalid_in_product or should_delete
                    add_record(
                        records,
                        experiment=experiment,
                        scope=scope,
                        product=spec.name,
                        kind=(
                            "unreadable_or_headerless"
                            if should_delete
                            else "semantic_contract_anomaly"
                        ),
                        path=path,
                        reason=f"{type(exc).__name__}: {exc}",
                        missing_columns=missing,
                    )
                    if should_delete:
                        delete_paths.add(path)
                        sentinel = paired_sentinel(path)
                        if sentinel.exists():
                            delete_paths.add(sentinel)

            if invalid_in_product:
                for merged in merged_artifacts(product_root, spec.glob):
                    delete_paths.add(merged)
                    sentinel = paired_sentinel(merged)
                    if sentinel.exists():
                        delete_paths.add(sentinel)

            for sentinel in product_root.rglob("*.validated"):
                artifact = Path(str(sentinel)[: -len(".validated")])
                if not artifact.exists():
                    add_record(
                        records,
                        experiment=experiment,
                        scope=scope,
                        product=spec.name,
                        kind="orphan_validation_sentinel",
                        path=sentinel,
                        reason=f"validated sentinel has no artifact: {artifact}",
                    )
                    delete_paths.add(sentinel)

        snip_root = (
            OUTPUT_ROOT / "quality_control" / experiment / "snip_qc"
        )
        if snip_root.is_dir():
            for resolver in snip_root.rglob("*_snip_qc_resolved_sources.json"):
                error = check_resolver_json(resolver)
                if not error:
                    continue
                add_record(
                    records,
                    experiment=experiment,
                    scope=scope,
                    product="snip_qc",
                    kind="stale_resolver_plan",
                    path=resolver,
                    reason=error,
                )
                delete_paths.add(resolver)
                stem = resolver.name.replace(
                    "_snip_qc_resolved_sources.json", "_snip_qc.parquet"
                )
                verdict = resolver.with_name(stem)
                if verdict.exists():
                    delete_paths.add(verdict)
                sentinel = paired_sentinel(verdict)
                if sentinel.exists():
                    delete_paths.add(sentinel)
                for merged in snip_root.glob("*_snip_qc.parquet"):
                    delete_paths.add(merged)
                    merged_sentinel = paired_sentinel(merged)
                    if merged_sentinel.exists():
                        delete_paths.add(merged_sentinel)

        workdir = OUTPUT_ROOT / "work_directories" / experiment
        lock_root = workdir / ".snakemake" / "locks"
        if lock_root.is_dir():
            for lock in lock_root.rglob("*"):
                if lock.is_file():
                    add_record(
                        records,
                        experiment=experiment,
                        scope=scope,
                        product="snakemake",
                        kind="stale_lock",
                        path=lock,
                        reason="experiment is not active; lock is stale",
                    )
                    delete_paths.add(lock)

        incomplete_root = workdir / ".snakemake" / "incomplete"
        if incomplete_root.is_dir():
            for record in incomplete_root.rglob("*"):
                if not record.is_file():
                    continue
                output = decode_incomplete_output(record, incomplete_root)
                reason = "stale Snakemake incomplete-state record"
                if output is not None:
                    reason += f"; decoded output={output}"
                    if output.exists() and output.is_relative_to(OUTPUT_ROOT):
                        delete_paths.add(output)
                        sentinel = paired_sentinel(output)
                        if sentinel.exists():
                            delete_paths.add(sentinel)
                add_record(
                    records,
                    experiment=experiment,
                    scope=scope,
                    product="snakemake",
                    kind="stale_incomplete_record",
                    path=record,
                    reason=reason,
                )
                delete_paths.add(record)

    recorded_paths = {record["path"] for record in records}
    for path in sorted(delete_paths):
        if str(path) in recorded_paths:
            continue
        experiment = next(
            item for item in experiments if item in path.resolve(strict=False).parts
        )
        add_record(
            records,
            experiment=experiment,
            scope=experiments[experiment],
            product="dependent_or_validation_state",
            kind="dependent_invalidation",
            path=path,
            reason=(
                "removed with a stale upstream artifact so Snakemake cannot "
                "reuse a dependent or validation state generated from it"
            ),
        )

    for path in delete_paths:
        assert_safe(path, experiments)

    action_by_path = {str(path): "delete" for path in delete_paths}
    for record in records:
        record["planned_action"] = action_by_path.get(record["path"], "context")
        record["applied"] = str(bool(args.apply)).lower()
        record["audit_timestamp"] = datetime.now().astimezone().isoformat()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_id",
        "scope",
        "product",
        "kind",
        "path",
        "reason",
        "missing_columns",
        "planned_action",
        "applied",
        "audit_timestamp",
    ]
    with args.report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    print(f"experiments scanned: {len(experiments) - len(active)}")
    print(f"issues recorded: {len(records)}")
    print(f"paths selected for deletion: {len(delete_paths)}")
    print(f"report: {args.report}")
    if not args.apply:
        print("dry run only; pass --apply to delete selected paths")
        return

    deleted = 0
    for path in sorted(delete_paths, key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted += 1
    print(f"paths deleted: {deleted}")


if __name__ == "__main__":
    main()
