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
    collection_provenance_json: Path,
    output_csv: Path,
) -> None:
    from data_pipeline.acquisition.metadata_ingest.collection_provenance import (
        read_collection_provenance,
    )

    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    plate_metadata = pd.read_csv(plate_metadata_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)
    # The DECLARED collection fact (always present — one per experiment). compute branches on its
    # is_collection: collection reads age by time_index, single is byte-identical to before.
    collection_provenance = read_collection_provenance(collection_provenance_json)

    df = compute_stage_prediction_features(
        snip_inventory,
        frame_inventory,
        plate_metadata,
        collection_provenance=collection_provenance,
    )
    validate_stage_prediction_features(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
