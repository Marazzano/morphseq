"""stage_predictions product tests."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.feature_extraction.stage_predictions.compute import (
    compute_stage_prediction_features,
)
from data_pipeline.feature_extraction.stage_predictions.contract import (
    STAGE_PREDICTION_TABLE_COLUMNS,
    validate_stage_prediction_features,
)
from tests.data_pipeline.feature_extraction._feature_fixtures import make_inputs, make_plate_metadata


def test_compute_kimmel_formula_and_validates():
    snip, _, inv, reg = make_inputs()
    plate = make_plate_metadata()
    df = compute_stage_prediction_features(snip, inv, plate)
    assert list(df.columns) == STAGE_PREDICTION_TABLE_COLUMNS
    # t=0 -> elapsed 0 -> predicted == start_age_hpf (11.0).
    first = df.sort_values("time_index").iloc[0]
    assert abs(first["predicted_stage_hpf"] - 11.0) < 1e-9
    # rate = 0.055*30 - 0.57 = 1.08 hpf/hr; t=1 -> 100s -> +0.03 hpf.
    second = df.sort_values("time_index").iloc[1]
    assert abs(second["predicted_stage_hpf"] - (11.0 + (100 / 3600) * 1.08)) < 1e-6
    validate_stage_prediction_features(df, physical_embryo_registry_df=reg, check_sources=True)


def test_missing_plate_row_fails_loud():
    snip, _, inv, _ = make_inputs()
    empty_plate = make_plate_metadata().iloc[0:0]
    with pytest.raises(ValueError, match="not in plate_metadata"):
        compute_stage_prediction_features(snip, inv, empty_plate)


# ── Collection age branch (CLASSIFY ONCE): start_age_hpf by time_index ─────────

def _collection_provenance():
    # Snapshot collection: source_ordinal 0 → 28 hpf, 1 → 52 hpf, 2 → 52 hpf. (make_inputs emits
    # time_indices 0,1,2; for an all-snapshot collection the merged time_index equals the source
    # ordinal, which is why this consumer can still key on it — see the KNOWN LIMITATION in
    # compute._start_age_hpf_for_snip.)
    return {
        "experiment_id": "chem28c_coll_plate01",
        "is_collection": True,
        "sources": ["a", "b", "c"],
        "start_age_by_source_ordinal": {"0": 28, "1": 52, "2": 52},
    }


def test_collection_reads_start_age_by_source_ordinal():
    snip, _, inv, _ = make_inputs()
    # plate_metadata still supplies temperature; a collection well need NOT carry start_age_hpf.
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    df = compute_stage_prediction_features(
        snip, inv, plate, collection_provenance=_collection_provenance()
    )
    by_t = df.set_index("time_index")["predicted_stage_hpf"]
    # t=0 → elapsed 0 → predicted == start_age_hpf for that timepoint (28.0), NOT the plate 11.0.
    assert abs(by_t.loc[0] - 28.0) < 1e-9
    # t=1 → age 52, elapsed 100s, rate 1.08 hpf/hr → 52 + (100/3600)*1.08.
    assert abs(by_t.loc[1] - (52.0 + (100 / 3600) * 1.08)) < 1e-6


def test_collection_missing_age_for_timepoint_fails_loud():
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    provenance = _collection_provenance()
    provenance["start_age_by_source_ordinal"] = {"0": 28, "1": 52}  # missing ordinal 2
    with pytest.raises(ValueError, match="no start_age_hpf for source_ordinal 2"):
        compute_stage_prediction_features(
            snip, inv, plate, collection_provenance=provenance
        )


def test_single_is_byte_identical_with_or_without_inert_classification():
    """Regression guard: a non-collection provenance payload must not change ANY value vs the
    pre-collection call (collection_provenance=None)."""
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata()
    inert = {
        "experiment_id": "20250912", "is_collection": False,
        "sources": [], "start_age_by_source_ordinal": {},
        "start_age_by_time_index": {},
    }
    baseline = compute_stage_prediction_features(snip, inv, plate)
    with_inert = compute_stage_prediction_features(
        snip, inv, plate, collection_provenance=inert
    )
    pd.testing.assert_frame_equal(baseline, with_inert)


def test_collection_falls_back_to_legacy_age_map():
    """An artifact written before the rename carries only start_age_by_time_index.

    The reader must still resolve ages from it (the keys were always source ordinals). Guards the
    compatibility path until TODO(collection-legacy-age-map) removes the field.
    """
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    legacy_only = {
        "experiment_id": "chem28c_coll_plate01",
        "is_collection": True,
        "sources": ["a", "b", "c"],
        "start_age_by_time_index": {"0": 28, "1": 52, "2": 52},
    }
    df = compute_stage_prediction_features(
        snip, inv, plate, collection_provenance=legacy_only
    )
    by_t = df.set_index("time_index")["predicted_stage_hpf"]
    assert abs(by_t.loc[0] - 28.0) < 1e-9
