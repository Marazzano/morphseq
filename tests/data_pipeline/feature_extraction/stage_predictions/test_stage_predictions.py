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


# ── Collection age branch: keyed on SOURCE ORDINAL, not merged time_index ─────────────
# The formula is start_age_hpf + elapsed_within_that_source * rate, so the age is a per-SOURCE
# fact. Keying on the merged time_index is only correct when every source contributes ONE frame;
# for a timelapse source it stages later frames with the NEXT source's declared age.


def _collection_provenance(n_sources=3):
    """Provenance for a collection: one declared age per SOURCE ORDINAL."""
    ages = {"0": 28, "1": 52, "2": 52}
    return {
        "experiment_id": "chem28c_coll_plate01",
        "is_collection": True,
        "sources": [
            {"file": f"src{i}", "raw_path": f"/r/src{i}", "declared_hpf": ages[str(i)],
             "source_ordinal": i, "time_index": i}
            for i in range(n_sources)
        ],
        "start_age_by_source_ordinal": {k: v for k, v in ages.items() if int(k) < n_sources},
    }


def _with_source_ordinals(snip, ordinals):
    """Stamp source_ordinal on snip rows (frame provenance carried through frame_inventory)."""
    out = snip.copy()
    out["source_ordinal"] = list(ordinals)
    return out


def test_collection_reads_start_age_by_source_ordinal():
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    # make_inputs emits time_index 0,1,2; as SNAPSHOT sources ordinal == time_index.
    snip = _with_source_ordinals(snip, [0, 1, 2])
    df = compute_stage_prediction_features(
        snip, inv, plate, collection_provenance=_collection_provenance()
    )
    by_t = df.set_index("time_index")["predicted_stage_hpf"]
    assert abs(by_t.loc[0] - 28.0) < 1e-9
    assert abs(by_t.loc[1] - (52.0 + (100 / 3600) * 1.08)) < 1e-6


def test_TIMELAPSE_source_stages_every_frame_from_ITS_OWN_age():
    """THE case merged-time_index keying got wrong.

    One source spans MANY merged time_index values. All three snips belong to source_ordinal 0, so
    all three must stage from age 28 — not from ordinals 1 and 2 (which would be a 24-hour error
    on frames 1 and 2).
    """
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    # ONE timelapse source owning merged time_index 0,1,2.
    snip = _with_source_ordinals(snip, [0, 0, 0])
    df = compute_stage_prediction_features(
        snip, inv, plate, collection_provenance=_collection_provenance(n_sources=1)
    )
    by_t = df.set_index("time_index")["predicted_stage_hpf"]
    # Every frame ages from 28 within its own source; elapsed drives the increase, not the ordinal.
    assert abs(by_t.loc[0] - 28.0) < 1e-9
    assert abs(by_t.loc[1] - (28.0 + (100 / 3600) * 1.08)) < 1e-6
    # Keying on time_index would have used age 52 here — a 24-hour error.
    assert by_t.loc[1] < 30.0


def test_collection_missing_age_for_source_fails_loud():
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    snip = _with_source_ordinals(snip, [0, 1, 2])
    provenance = _collection_provenance()
    provenance["start_age_by_source_ordinal"] = {"0": 28, "1": 52}  # missing ordinal 2
    with pytest.raises(ValueError, match="no start_age_hpf for source_ordinal 2"):
        compute_stage_prediction_features(
            snip, inv, plate, collection_provenance=provenance
        )


def test_collection_without_source_ordinal_fails_loud():
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata().drop(columns=["start_age_hpf"])
    with pytest.raises(ValueError, match="no 'source_ordinal'"):
        compute_stage_prediction_features(
            snip, inv, plate, collection_provenance=_collection_provenance()
        )


def test_single_is_byte_identical_with_or_without_provenance():
    """Regression guard: a non-collection provenance payload must not change ANY value."""
    snip, _, inv, _ = make_inputs()
    plate = make_plate_metadata()
    single = {
        "experiment_id": "20250912", "is_collection": False,
        "sources": [{"file": "20250912", "raw_path": "/r/20250912", "declared_hpf": None,
                     "source_ordinal": 0, "time_index": 0}],
        "start_age_by_source_ordinal": {},
    }
    baseline = compute_stage_prediction_features(snip, inv, plate)
    with_prov = compute_stage_prediction_features(
        snip, inv, plate, collection_provenance=single
    )
    pd.testing.assert_frame_equal(baseline, with_prov)
