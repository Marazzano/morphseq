from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.scope.shared.validate_physical_well_mapping import (
    validate_physical_well_mapping,
)


def test_validate_physical_well_mapping_passes_for_complete_mapping(tmp_path) -> None:
    scope = pd.DataFrame(
        [
            {"experiment_id": "20240418", "raw_position_label": "0", "time_int": 0},
            {"experiment_id": "20240418", "raw_position_label": "1", "time_int": 0},
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "experiment_id": "20240418",
                "position_index": 0,
                "well_index": "A01",
                "well_id": "20240418_A01",
                "mapping_method": "test",
            },
            {
                "experiment_id": "20240418",
                "position_index": 1,
                "well_index": "A02",
                "well_id": "20240418_A02",
                "mapping_method": "test",
            },
        ]
    )

    scope_csv = tmp_path / "scope.csv"
    mapping_csv = tmp_path / "mapping.csv"
    scope.to_csv(scope_csv, index=False)
    mapping.to_csv(mapping_csv, index=False)

    diag = validate_physical_well_mapping(
        scope_metadata_csv=scope_csv,
        mapping_csv=mapping_csv,
        allow_unmapped_wells=False,
    )
    assert diag["n_scope_positions"] == 2


def test_validate_physical_well_mapping_fails_for_incomplete_mapping(tmp_path) -> None:
    scope = pd.DataFrame(
        [
            {"experiment_id": "20240418", "raw_position_label": "0", "time_int": 0},
            {"experiment_id": "20240418", "raw_position_label": "1", "time_int": 0},
        ]
    )
    mapping = pd.DataFrame([
        {
            "experiment_id": "20240418",
            "position_index": 0,
            "well_index": "A01",
            "well_id": "20240418_A01",
            "mapping_method": "test",
        }
    ])

    scope_csv = tmp_path / "scope.csv"
    mapping_csv = tmp_path / "mapping.csv"
    scope.to_csv(scope_csv, index=False)
    mapping.to_csv(mapping_csv, index=False)

    try:
        validate_physical_well_mapping(
            scope_metadata_csv=scope_csv,
            mapping_csv=mapping_csv,
            allow_unmapped_wells=False,
        )
    except ValueError as e:
        assert "Physical well mapping appears incomplete" in str(e)
    else:
        raise AssertionError("Expected validator to raise ValueError")


def test_validate_physical_well_mapping_rejects_series_number(tmp_path) -> None:
    scope = pd.DataFrame([{"experiment_id": "20240418", "raw_position_label": "0", "time_int": 0}])
    mapping = pd.DataFrame([
        {
            "experiment_id": "20240418",
            "position_index": 0,
            "series_number": 1,
            "well_index": "A01",
            "well_id": "20240418_A01",
            "mapping_method": "test",
        }
    ])

    scope_csv = tmp_path / "scope.csv"
    mapping_csv = tmp_path / "mapping.csv"
    scope.to_csv(scope_csv, index=False)
    mapping.to_csv(mapping_csv, index=False)

    try:
        validate_physical_well_mapping(
            scope_metadata_csv=scope_csv,
            mapping_csv=mapping_csv,
            allow_unmapped_wells=False,
        )
    except ValueError as e:
        assert "retired column" in str(e)
    else:
        raise AssertionError("Expected validator to reject series_number")
