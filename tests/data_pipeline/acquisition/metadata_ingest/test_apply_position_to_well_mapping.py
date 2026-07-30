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
            "time_index": 0,
            "channel_id": "BF",
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


def test_apply_position_to_well_mapping_preserves_preidentified_keyence_wells(tmp_path):
    scope_csv = tmp_path / "scope.csv"
    mapping_csv = tmp_path / "position_well_mapping.csv"
    output_csv = tmp_path / "scope_metadata_mapped.csv"
    experiment = "20260702_hotchem_24hpf_plate01"

    scope_rows = []
    mapping_rows = []
    position_index = 0
    for row in "ABCDEFGH":
        for col in range(1, 13):
            well_index = f"{row}{col:02d}"
            well_id = f"{experiment}_{well_index}"
            scope_rows.append(
                {
                    "experiment_id": experiment,
                    "raw_position_label": position_index // 3,
                    "position_index": position_index // 3,
                    "well_index": well_index,
                    "well_id": well_id,
                    "time_index": 0,
                    "channel_id": "BF",
                    "experiment_time_s": 0.0,
                }
            )
            for tile_index in range(3):
                mapping_rows.append(
                    {
                        "experiment_id": experiment,
                        "position_index": position_index + tile_index,
                        "well_index": well_index,
                        "well_id": well_id,
                        "mapping_method": "keyence_directory_structure",
                    }
                )
            position_index += 3

    pd.DataFrame(scope_rows).to_csv(scope_csv, index=False)
    pd.DataFrame(mapping_rows).to_csv(mapping_csv, index=False)

    mapped = apply_position_to_well_mapping(
        scope_metadata_csv=scope_csv,
        mapping_csv=mapping_csv,
        output_csv=output_csv,
        experiment_id=experiment,
    )

    assert mapped["well_id"].nunique() == 96
    assert mapped["well_index"].nunique() == 96
    assert mapped.loc[mapped["well_index"] == "H12", "well_id"].iloc[0] == f"{experiment}_H12"
    assert output_csv.exists()
