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
    acquisition_inventory_csv: Path | None = None,
) -> None:
    from data_pipeline.acquisition.metadata_ingest.collection_provenance import (
        read_collection_provenance,
    )

    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    plate_metadata = pd.read_csv(plate_metadata_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)
    # The DECLARED collection fact (always present — every experiment has one; a single experiment
    # is a collection of ONE source). compute branches on is_collection for the age's HOME, but keys
    # on source_ordinal either way.
    collection_provenance = read_collection_provenance(collection_provenance_json)

    # A COLLECTION needs the per-frame source, and the acquisition inventory OWNS that mapping.
    # frame_inventory deliberately carries no per-frame source label (a frame has exactly one
    # source, so labelling every frame would restate what the inventory already says), so this is a
    # join at the consume boundary. A single experiment does not need it.
    acquisition_inventory = (
        pd.read_csv(acquisition_inventory_csv) if acquisition_inventory_csv else None
    )

    df = compute_stage_prediction_features(
        snip_inventory,
        frame_inventory,
        plate_metadata,
        acquisition_inventory_df=acquisition_inventory,
        collection_provenance=collection_provenance,
    )
    validate_stage_prediction_features(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
