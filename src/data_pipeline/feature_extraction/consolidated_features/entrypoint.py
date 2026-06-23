"""consolidated_features entrypoint — load upstream shards via registry, merge, validate, write."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import consolidate_feature_tables
from .contract import validate_consolidated_features
from .inputs import load_feature_shards

# MVP source feature products to consolidate. mask_geometry seeds the spine + core columns; the
# rest contribute their non-spine feature columns. Order matters only for which table seeds.
# fraction_alive is intentionally excluded until its VIA/embryo mask resolution mismatch is
# resolved; this list must match rules/consolidated_features.smk::_CF_SOURCE_STEPS.
DEFAULT_FEATURE_STEPS = [
    "mask_geometry",
    "curvature_metrics",
    "pose_kinematics",
    "stage_predictions",
]


def run_consolidated_features(
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    feature_steps: list[str] | None = None,
) -> None:
    feature_steps = feature_steps or DEFAULT_FEATURE_STEPS
    tables = load_feature_shards(
        output_root=Path(output_root),
        experiment_id=experiment_id,
        well_id=well_id,
        feature_steps=feature_steps,
    )
    merged = consolidate_feature_tables(tables, key="snip_id")

    registry = pd.read_csv(physical_embryo_registry_csv)
    validate_consolidated_features(merged, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
