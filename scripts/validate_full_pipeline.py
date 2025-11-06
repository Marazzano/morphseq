#!/usr/bin/env python3
"""Validate end-to-end pipeline outputs."""

import sys
import pandas as pd
from pathlib import Path

def validate_full_pipeline(exp_id):
    """Validate all phases completed successfully."""

    phases = [
        ("Metadata", "experiment_metadata/{exp}/scope_and_plate_metadata.csv"),
        ("Segmentation", "segmentation/{exp}/segmentation_tracking.csv"),
        ("Snips", "processed_snips/{exp}/snip_manifest.csv"),
        ("Features", "computed_features/{exp}/consolidated_snip_features.csv"),
        ("QC", "quality_control/{exp}/consolidated_qc_flags.csv"),
        ("Analysis", "analysis_ready/{exp}/features_qc_embeddings.csv"),
    ]

    for phase_name, path_template in phases:
        path = Path(path_template.format(exp=exp_id))
        assert path.exists(), f"{phase_name} output missing: {path}"

        df = pd.read_csv(path)
        assert len(df) > 0, f"{phase_name} output is empty"
        print(f"✓ {phase_name}: {len(df)} rows")

    # Validate ID formats
    tracking = pd.read_csv(f"segmentation/{exp_id}/segmentation_tracking.csv")
    assert tracking['embryo_id'].str.match(r'.*_e\d+').all(), "Invalid embryo_id format"
    assert tracking['snip_id'].str.match(r'.*_e\d+_t\d+').all(), "Invalid snip_id format"
    print(f"✓ ID formats validated")

    # Validate use_embryo flag
    qc = pd.read_csv(f"quality_control/{exp_id}/consolidated_qc_flags.csv")
    assert 'use_embryo' in qc.columns, "use_embryo flag missing"
    print(f"✓ QC flags validated")

    print(f"\n✓✓✓ Full pipeline validation PASSED for {exp_id} ✓✓✓")

if __name__ == "__main__":
    validate_full_pipeline(sys.argv[1])
