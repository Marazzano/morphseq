"""Collection acquisition ingest — the derivation helpers.

The full ingest (find sources → per-scope read → union → write) is integration-tested on real
Keyence data; here we unit-test the pure ``derive_position_well_mapping`` (union → canonical
position→well mapping), which lets the collection reuse the NATIVE materializer path.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_acquisition_ingest.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_ingest import (
    derive_position_well_mapping,
)
from data_pipeline.acquisition.metadata_ingest.position_well_mapping.position_well_mapping_contract import (
    REQUIRED_POSITION_WELL_MAPPING_COLUMNS,
    validate_position_well_mapping,
)


def _unioned_like(experiment_id="chem_coll_plate01"):
    """A tiny unioned-inventory shape: two wells x two time blocks (merged snapshot)."""
    rows = []
    for pos, well in ((1, "A01"), (2, "A02")):
        for time_index in (0, 1):  # two source blocks — the merge
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "position_index": pos,
                    "well_index": well,
                    "well_id": f"{experiment_id}_{well}",
                    "time_index": time_index,
                    "channel_id": "BF",
                    "n_sources": 2,
                }
            )
    return pd.DataFrame(rows)


def test_derive_position_well_mapping_dedups_to_one_row_per_position():
    unioned = _unioned_like()
    mapping = derive_position_well_mapping(unioned)

    # ONE row per (position, well) — the two time blocks collapse (mapping is time-invariant).
    assert len(mapping) == 2
    assert list(mapping.columns) == list(REQUIRED_POSITION_WELL_MAPPING_COLUMNS)
    assert set(mapping["well_id"]) == {"chem_coll_plate01_A01", "chem_coll_plate01_A02"}
    assert set(mapping["mapping_method"]) == {"collection_union"}
    validate_position_well_mapping(mapping)  # satisfies the native-path contract


def test_derive_position_well_mapping_fails_loud_when_well_id_absent():
    unioned = _unioned_like().drop(columns=["well_id"])
    with pytest.raises(ValueError, match="missing"):
        derive_position_well_mapping(unioned)
