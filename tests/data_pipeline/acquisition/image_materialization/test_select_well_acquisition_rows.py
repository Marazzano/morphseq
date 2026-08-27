"""Tests for select_well_acquisition_rows — the pure position→well join + per-well row slice."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.image_materialization.select_well_acquisition_rows import (
    _join_well_identity,
    select_well_acquisition_rows,
)
from data_pipeline.shared.identifiers.constructors import build_well_id

EXP = "20250912"


def _acquisition_inventory() -> pd.DataFrame:
    # Two positions × two time_indices; source_nd2_path rides along (the input the sequencer needs).
    rows = []
    for position_index in (1, 2):
        for time_index in (0, 1):
            rows.append({
                "experiment_id": EXP,
                "position_index": position_index,
                "time_index": time_index,
                "source_nd2_path": "/data/exp.nd2",
            })
    return pd.DataFrame(rows)


def _mapping() -> pd.DataFrame:
    # position 1 → B01, position 2 → C01 (ids built via the constructor, never minted by hand).
    return pd.DataFrame([
        {"experiment_id": EXP, "position_index": 1, "well_index": "B01",
         "well_id": build_well_id(EXP, "B01"), "mapping_method": "test"},
        {"experiment_id": EXP, "position_index": 2, "well_index": "C01",
         "well_id": build_well_id(EXP, "C01"), "mapping_method": "test"},
    ])


def test_selects_only_the_requested_wells_rows():
    well_id = build_well_id(EXP, "B01")
    out = select_well_acquisition_rows(
        _acquisition_inventory(), _mapping(), experiment_id=EXP, well_id=well_id,
    )
    assert (out["well_id"] == well_id).all()
    assert sorted(out["position_index"].unique().tolist()) == [1]   # B01 → position 1 only
    assert len(out) == 2                                            # two time_indices


def test_joined_rows_carry_well_identity_and_source_path():
    well_id = build_well_id(EXP, "C01")
    out = select_well_acquisition_rows(
        _acquisition_inventory(), _mapping(), experiment_id=EXP, well_id=well_id,
    )
    assert "well_index" in out.columns and (out["well_index"] == "C01").all()
    assert (out["source_nd2_path"] == "/data/exp.nd2").all()       # input pointer preserved


def test_other_experiments_rows_are_excluded():
    acq = _acquisition_inventory()
    other = acq.copy()
    other["experiment_id"] = "20990101"
    acq = pd.concat([acq, other], ignore_index=True)
    well_id = build_well_id(EXP, "B01")
    out = select_well_acquisition_rows(acq, _mapping(), experiment_id=EXP, well_id=well_id)
    assert (out["experiment_id"] == EXP).all()
    assert len(out) == 2


def test_empty_result_fails_loud_naming_the_well():
    well_id = build_well_id(EXP, "Z99")  # no mapping row for Z99
    with pytest.raises(ValueError, match=r"Z99|no acquisition inventory rows resolve"):
        select_well_acquisition_rows(
            _acquisition_inventory(), _mapping(), experiment_id=EXP, well_id=well_id,
        )


def test_does_not_mutate_inputs():
    acq = _acquisition_inventory()
    mapping = _mapping()
    acq_before = acq.copy()
    mapping_before = mapping.copy()
    select_well_acquisition_rows(
        acq, mapping, experiment_id=EXP, well_id=build_well_id(EXP, "B01"),
    )
    pd.testing.assert_frame_equal(acq, acq_before)
    pd.testing.assert_frame_equal(mapping, mapping_before)


# ── Collections: source_ordinal is part of the join key ────────────────────────────────────────

def _collection_mapping() -> pd.DataFrame:
    """A COLLECTION mapping: several sources merged under one experiment_id.

    Each source contributes its own position 0..N, so (experiment_id, position_index) repeats once
    per source. collection_position_mapping states source_ordinal is "the JOIN KEY".
    """
    rows = []
    for source_ordinal in (0, 1, 2):
        for position_index, well_index in ((0, "A01"), (1, "B01")):
            rows.append({
                "experiment_id": "20260624_coll_plate01",
                "position_index": position_index,
                "well_index": well_index,
                "well_id": build_well_id("20260624_coll_plate01", well_index),
                "source_ordinal": source_ordinal,
            })
    return pd.DataFrame(rows)


def _collection_inventory() -> pd.DataFrame:
    rows = []
    for source_ordinal in (0, 1, 2):
        for position_index in (0, 1):
            rows.append({
                "experiment_id": "20260624_coll_plate01",
                "position_index": position_index,
                "source_ordinal": source_ordinal,
                "time_index": source_ordinal,
                "channel_id": "BF",
            })
    return pd.DataFrame(rows)


def test_collection_join_does_not_multiply_rows():
    """The pbx smoke failure: 'Merge keys are not unique in right dataset'.

    A single-source experiment has one mapping row per position, so the two-column key is unique. A
    collection merges N sources under one experiment_id and each contributes position 0..N, so that
    pair repeats N times -- and a many_to_one join either raises or silently fans every inventory
    row into N copies. Measured on the real pbx collection: 288 mapping rows = 96 positions x 3
    sources, ALL duplicated on the two-column key, ZERO once source_ordinal joins it.
    """
    inventory = _collection_inventory()
    joined = _join_well_identity(
        inventory, _collection_mapping(), experiment_id="20260624_coll_plate01"
    )
    assert len(joined) == len(inventory), "the join multiplied rows across sources"
    assert joined["well_id"].notna().all(), "some rows failed to resolve a well"
    # Each source keeps its OWN position->well assignment rather than inheriting source 0's.
    for source_ordinal in (0, 1, 2):
        block = joined[joined["source_ordinal"] == source_ordinal]
        assert set(block["well_index"]) == {"A01", "B01"}


def test_collection_join_survives_a_csv_dtype_round_trip():
    """One side str, the other int -- yields ZERO matches rather than an error.

    The failure then surfaces as a misleading "no rows resolve to this well", pointing at the
    mapping's coverage instead of at a dtype. apply_position_to_well_mapping hit exactly this, which
    is why both sides are coerced.
    """
    mapping = _collection_mapping()
    mapping["source_ordinal"] = mapping["source_ordinal"].astype(str)
    joined = _join_well_identity(
        _collection_inventory(), mapping, experiment_id="20260624_coll_plate01"
    )
    assert joined["well_id"].notna().all()


def test_a_collection_mapping_against_a_sourceless_inventory_fails_loud():
    """Mismatched pipelines must not silently produce a fanned-out join."""
    inventory = _collection_inventory().drop(columns=["source_ordinal"])
    with pytest.raises(ValueError, match="source_ordinal"):
        _join_well_identity(
            inventory, _collection_mapping(), experiment_id="20260624_coll_plate01"
        )


def test_single_source_experiments_keep_the_two_column_join():
    """No source_ordinal anywhere -> the original keys, unchanged behavior."""
    joined = _join_well_identity(
        _acquisition_inventory(), _mapping(), experiment_id=EXP
    )
    assert joined["well_id"].notna().all()
