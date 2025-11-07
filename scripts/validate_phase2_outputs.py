#!/usr/bin/env python3
"""Validate Phase 1-2 outputs (metadata + image manifest)."""

import sys
import pandas as pd
from pathlib import Path

def validate_phase2(exp_id, microscope=None):
    """Validate metadata and manifest for experiment."""

    # Import schemas
    from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
    from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA
    from data_pipeline.schemas.scope_and_plate_metadata import REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA
    from data_pipeline.io.validation import validate_dataframe_schema

    metadata_dir = Path(f"experiment_metadata/{exp_id}")

    # Validate each metadata file
    plate = pd.read_csv(metadata_dir / "plate_metadata.csv")
    validate_dataframe_schema(plate, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")
    print(f"✓ plate_metadata.csv: {len(plate)} rows")

    scope = pd.read_csv(metadata_dir / "scope_metadata.csv")
    validate_dataframe_schema(scope, REQUIRED_COLUMNS_SCOPE_METADATA, "scope_metadata")
    print(f"✓ scope_metadata.csv: {len(scope)} rows")

    aligned = pd.read_csv(metadata_dir / "scope_and_plate_metadata.csv")
    validate_dataframe_schema(aligned, REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA, "scope_and_plate")
    print(f"✓ scope_and_plate_metadata.csv: {len(aligned)} rows")

    # Validate manifest exists
    manifest_path = metadata_dir / "experiment_image_manifest.json"
    assert manifest_path.exists(), f"Missing: {manifest_path}"
    print(f"✓ experiment_image_manifest.json exists")

    # Check channel normalization
    assert "BF" in aligned['channel_name'].values, "BF channel missing (normalization failed)"
    print(f"✓ Channel normalization validated")

    print(f"\n✓✓✓ Phase 2 validation PASSED for {exp_id} ✓✓✓")

if __name__ == "__main__":
    validate_phase2(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
