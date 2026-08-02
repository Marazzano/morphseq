"""Fail-fast production checks for materialized SeaHub shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_pipeline.acquisition.metadata_ingest.frame_inventory.frame_inventory_validation import (
    validate_frame_inventory,
)


def _resolved_manifest_path(value: Any, *, field: str) -> Path:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"SeaHub experiment manifest has no {field!r} value.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"SeaHub experiment manifest {field!r} must be absolute; got {path}."
        )
    return path.resolve()


def _require_exact_path(*, actual: Path, expected: Path, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"SeaHub manifest {field!r} points outside the selected fresh bundle: "
            f"{actual}; expected {expected}."
        )


def preflight_materialized_experiment(
    *, bundle_root: str | Path, experiment_id: str
) -> Path:
    """Validate one shard immediately before standard-pipeline execution.

    The check intentionally reopens every materialized image through the shared
    frame-inventory validator.  It also strengthens the generic drop-in contract:
    SeaHub paths must be absolute and contained by this bundle's experiment root.
    This prevents a new run from silently following a stale manifest into an older
    acquisition tree.
    """

    bundle_root = Path(bundle_root).expanduser().resolve()
    manifest_path = bundle_root / "integration" / "experiment_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SeaHub experiment manifest does not exist: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)
    if "experiment_id" not in manifest.columns:
        raise ValueError(
            f"SeaHub experiment manifest has no experiment_id column: {manifest_path}"
        )
    selected = manifest[
        manifest["experiment_id"].astype(str).eq(str(experiment_id))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for {experiment_id!r}; found "
            f"{len(selected)}."
        )

    row = selected.iloc[0]
    experiment_root = (bundle_root / "experiments" / str(experiment_id)).resolve()
    expected = {
        "image_root": experiment_root,
        "frame_inventory_csv": experiment_root / "dropin_frame_inventory.csv",
        "plate_metadata_csv": experiment_root / "plate_metadata.csv",
        "runtime_config_yaml": experiment_root / "runtime_config.yaml",
    }
    actual: dict[str, Path] = {}
    for field, expected_path in expected.items():
        actual[field] = _resolved_manifest_path(row.get(field), field=field)
        _require_exact_path(
            actual=actual[field], expected=expected_path, field=field
        )

    frame_inventory_csv = actual["frame_inventory_csv"]
    required_files = [
        frame_inventory_csv,
        actual["plate_metadata_csv"],
        actual["runtime_config_yaml"],
        frame_inventory_csv.with_suffix(".csv.validated"),
        actual["plate_metadata_csv"].with_suffix(".csv.validated"),
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"SeaHub shard {experiment_id!r} is incomplete; missing: {missing}"
        )

    frame_inventory = pd.read_csv(frame_inventory_csv)
    if "image_path" not in frame_inventory.columns:
        raise ValueError(
            f"SeaHub frame inventory has no image_path column: {frame_inventory_csv}"
        )
    for index, raw_path in frame_inventory["image_path"].items():
        image_path = Path(str(raw_path)).expanduser()
        if not image_path.is_absolute():
            raise ValueError(
                "SeaHub frame inventories require absolute image_path values; "
                f"row {index} contains {raw_path!r}."
            )
        resolved = image_path.resolve()
        try:
            resolved.relative_to(experiment_root)
        except ValueError as exc:
            raise ValueError(
                "SeaHub frame inventory points outside its fresh experiment root; "
                f"row {index} resolves to {resolved}, root={experiment_root}."
            ) from exc

    runtime = yaml.safe_load(actual["runtime_config_yaml"].read_text(encoding="utf-8"))
    if not isinstance(runtime, dict):
        raise ValueError(
            f"SeaHub runtime config is not a mapping: {actual['runtime_config_yaml']}"
        )
    if runtime.get("experiments") != [str(experiment_id)]:
        raise ValueError(
            "SeaHub runtime config experiment does not match the selected shard: "
            f"{runtime.get('experiments')!r} vs {[str(experiment_id)]!r}."
        )
    dropin = runtime.get("dropin")
    if not isinstance(dropin, dict):
        raise ValueError("SeaHub runtime config has no dropin mapping.")
    for field in ("frame_inventory_csv", "plate_metadata_csv", "image_root"):
        configured = _resolved_manifest_path(dropin.get(field), field=f"dropin.{field}")
        _require_exact_path(
            actual=configured, expected=expected[field], field=f"dropin.{field}"
        )

    # Re-run the canonical strict gate with image_root=None.  This both opens every
    # declared image and guarantees that relative paths cannot be resolved by a
    # convenient root supplied only at validation time.
    output_flag = frame_inventory_csv.with_suffix(".csv.validated")
    validate_frame_inventory(
        frame_inventory_csv,
        output_flag,
        image_root=None,
        check_sources=True,
        validation_scope="merged",
    )
    return output_flag


__all__ = ["preflight_materialized_experiment"]
