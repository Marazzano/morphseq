from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.object_extraction.snip_processing.contract import (
    NULLABLE_COLUMNS_SNIP_MANIFEST,
    REQUIRED_COLUMNS_SNIP_MANIFEST,
)


def rel_to_root(path: Path, *, output_root: Path) -> str:
    path = Path(path)
    output_root = Path(output_root)
    try:
        rel = path.relative_to(output_root)
        return rel.as_posix()
    except Exception:
        return str(path)


def resolve_from_root(path_str: str, *, output_root: Path) -> Path:
    p = Path(str(path_str))
    return p if p.is_absolute() else (Path(output_root) / p)


def resolve_snip_inventory_image_paths(
    snip_inventory: pd.DataFrame,
    *,
    output_root: Path,
) -> pd.DataFrame:
    """Resolve ``processed_snip_path`` to an absolute path for report rendering.

    Snip inventories store processed paths relative to the data root. Reports should import this
    helper rather than reimplementing path resolution locally.
    """

    def _resolve(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        return str(resolve_from_root(str(value), output_root=output_root))

    resolved = snip_inventory[["snip_id", "processed_snip_path"]].copy()
    resolved["resolved_image_path"] = resolved["processed_snip_path"].map(_resolve)
    return resolved[["snip_id", "resolved_image_path"]]


def resolve_snip_inventory_image_and_mask_paths(
    snip_inventory: pd.DataFrame,
    *,
    output_root: Path,
) -> pd.DataFrame:
    """Resolve snip image and snip-space embryo-mask paths for report overlays.

    The snip-processing generator writes the cropped-and-rotated embryo mask as ``embryo_mask``
    beside each snip image under the per-well ``snips_dir`` tree. ``embryo_mask_snip_path`` is kept
    as a compatibility alias. This helper resolves whichever stored relative path is present
    against ``output_root`` so report code can draw overlays without knowing the output layout.
    """

    def _resolve(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        return str(resolve_from_root(str(value), output_root=output_root))

    mask_col = "embryo_mask" if "embryo_mask" in snip_inventory.columns else "embryo_mask_snip_path"
    resolved = snip_inventory[["snip_id", "image_id", "processed_snip_path", mask_col]].copy()
    resolved["resolved_image_path"] = resolved["processed_snip_path"].map(_resolve)
    resolved["resolved_mask_path"] = resolved[mask_col].map(_resolve)
    return resolved[["snip_id", "image_id", "resolved_image_path", "resolved_mask_path"]]


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def pipeline_version() -> str:
    """
    Best-effort pipeline version string for provenance.
    Falls back to "unknown" when git is unavailable.
    """
    try:
        here = Path(__file__).resolve()
        # repo_root = .../src/data_pipeline/object_extraction/snip_processing/io.py -> go up to repo root
        repo_root = here.parents[4]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha[:12]
    except Exception:
        return "unknown"


def validate_snip_manifest_df(df: pd.DataFrame) -> None:
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_SNIP_MANIFEST,
        "snip_manifest",
        nullable_columns=NULLABLE_COLUMNS_SNIP_MANIFEST,
    )

    # Conditional non-null requirements.
    if "is_valid" in df.columns:
        valid = df["is_valid"] == True  # noqa: E712
        if valid.any():
            must_have = [
                "processed_snip_path",
                "processed_file_size_bytes",
                "rotation_angle_rad",
                "rotation_angle_deg",
            ]
            for col in must_have:
                if df.loc[valid, col].isna().any():
                    n = int(df.loc[valid, col].isna().sum())
                    raise ValueError(f"Valid snips must have non-null {col}; found {n} null values.")
            if (df.loc[valid, "processed_file_size_bytes"].astype(float) <= 0).any():
                raise ValueError("Valid snips must have processed_file_size_bytes > 0.")

        invalid = df["is_valid"] == False  # noqa: E712
        if invalid.any():
            if df.loc[invalid, "error_message"].isna().any():
                raise ValueError("Invalid snips must have error_message populated.")

    if df["snip_id"].duplicated().any():
        dups = df.loc[df["snip_id"].duplicated(keep=False), ["snip_id"]].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate snip_id values in snip_manifest: {dups}")


@dataclass(frozen=True)
class SnipPaths:
    per_well_root: Path
    contracts_dir: Path
    processed_dir: Path
    raw_crops_dir: Path
    artifacts_dir: Path


def _snip_experiment_root(*, output_root: Path, experiment_id: str, well_id: str | None = None) -> Path:
    output_root = Path(output_root)
    exp_root = output_root / "processed_snips" / str(experiment_id)
    if well_id is None:
        return exp_root
    return exp_root / "per_well" / str(well_id)


def per_well_output_dirs(*, output_root: Path, experiment_id: str, well_id: str) -> SnipPaths:
    """Return the per-well snip shard layout.

    Per-well producer code should import this wrapper. It owns the on-disk fanout for
    ``processed_snips/{experiment_id}/per_well/{well_id}/`` and returns the four canonical
    subdirectories used by snip-processing jobs.
    """

    per_well_root = _snip_experiment_root(output_root=output_root, experiment_id=experiment_id, well_id=well_id)
    contracts_dir = per_well_root / "contracts"
    processed_dir = per_well_root / "processed"
    raw_crops_dir = per_well_root / "raw_crops"
    artifacts_dir = per_well_root / "artifacts"
    for p in (contracts_dir, processed_dir, raw_crops_dir, artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)
    return SnipPaths(
        per_well_root=per_well_root,
        contracts_dir=contracts_dir,
        processed_dir=processed_dir,
        raw_crops_dir=raw_crops_dir,
        artifacts_dir=artifacts_dir,
    )


def merged_output_dirs(*, output_root: Path, experiment_id: str) -> tuple[Path, Path]:
    """Return the merged experiment-level snip layout.

    Use this wrapper for merged views and report consumers. It resolves the shared
    ``processed_snips/{experiment_id}/`` root and exposes the experiment-level contracts and
    views directories.
    """

    exp_root = _snip_experiment_root(output_root=output_root, experiment_id=experiment_id)
    contracts_dir = exp_root / "contracts"
    views_dir = exp_root / "views"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)
    return contracts_dir, views_dir
