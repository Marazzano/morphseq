"""Tests for select_well_acquisition_rows — the pure position→well join + per-well row slice."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.image_materialization.select_well_acquisition_rows import (
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
