from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.scope.shared.apply_position_to_well_mapping import (
    apply_position_to_well_mapping,
)


def test_apply_position_to_well_mapping_joins_identity_by_position(tmp_path):
    scope_csv = tmp_path / "scope.csv"
    mapping_csv = tmp_path / "position_well_mapping.csv"
    output_csv = tmp_path / "scope_metadata_mapped.csv"

    pd.DataFrame([
        {
            "experiment_id": "20250912",
            "raw_position_label": "2",
            "time_int": 0,
            "channel": "BF",
            "experiment_time_s": 0.0,
        }
    ]).to_csv(scope_csv, index=False)
    pd.DataFrame([
        {
            "experiment_id": "20250912",
            "position_index": 2,
            "well_index": "B01",
            "well_id": "20250912_B01",
            "mapping_method": "test",
        }
    ]).to_csv(mapping_csv, index=False)

    mapped = apply_position_to_well_mapping(
        scope_metadata_csv=scope_csv,
        mapping_csv=mapping_csv,
        output_csv=output_csv,
        experiment_id="20250912",
    )

    assert mapped.loc[0, "position_index"] == 2
    assert mapped.loc[0, "well_index"] == "B01"
    assert mapped.loc[0, "well_id"] == "20250912_B01"
    assert mapped.loc[0, "image_id"] == "20250912_B01_BF_t0000"
    assert output_csv.exists()
