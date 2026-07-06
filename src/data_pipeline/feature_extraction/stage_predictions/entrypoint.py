"""stage_predictions entrypoint — thin filesystem adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import compute_stage_prediction_features
from .contract import validate_stage_prediction_features


def run_stage_predictions(
    *,
    snip_inventory_csv: Path,
    frame_inventory_csv: Path,
    plate_metadata_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
) -> None:
    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    plate_metadata = pd.read_csv(plate_metadata_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    df = compute_stage_prediction_features(snip_inventory, frame_inventory, plate_metadata)
    validate_stage_prediction_features(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
