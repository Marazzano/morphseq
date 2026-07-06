"""death_detection contract tests — two tables, two grains, distinct spines."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.death_detection.contract import (
    DEATH_DETECTION_QC_TABLE_COLUMNS,
    DEATH_EVENT_TABLE_COLUMNS,
    validate_death_detection_qc,
    validate_death_event,
)
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"
PHYS = build_physical_embryo_id(WELL, 1)


def _qc_df():
    image_id = build_image_id(WELL, CHANNEL, 0)
    embryo_id = build_embryo_id(PHYS, image_id)
    snip_id = build_snip_id(embryo_id, image_id)
    df = pd.DataFrame(
        [
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": PHYS,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "viability_dead_flag": False,
                "persistence_dead_flag": True,
            }
        ],
        columns=DEATH_DETECTION_QC_TABLE_COLUMNS,
    )
    for f in ("viability_dead_flag", "persistence_dead_flag"):
        df[f] = df[f].astype(bool)
    return df


def _event_df():
    return pd.DataFrame(
        [
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": PHYS,
                "death_event_time_index": 3,
                "death_event_stage_hpf": 27.0,
            }
        ],
        columns=DEATH_EVENT_TABLE_COLUMNS,
    )


def test_qc_valid_passes():
    validate_death_detection_qc(_qc_df())


def test_qc_non_bool_flag_fails():
    df = _qc_df()
    df["viability_dead_flag"] = [1]
    with pytest.raises(ValueError, match="boolean dtype"):
        validate_death_detection_qc(df)


def test_event_valid_passes():
    validate_death_event(_event_df())


def test_event_with_embryo_id_fails():
    df = _event_df()
    df["embryo_id"] = build_embryo_id(PHYS, build_image_id(WELL, CHANNEL, 0))
    with pytest.raises(ValueError, match="must NOT carry embryo_id"):
        validate_death_event(df)


def test_event_null_annotation_fails():
    df = _event_df()
    df["death_event_stage_hpf"] = [None]
    with pytest.raises(ValueError, match="null/non-numeric"):
        validate_death_event(df)


def test_event_spine_has_no_embryo_id():
    assert "embryo_id" not in DEATH_EVENT_TABLE_COLUMNS
    assert "snip_id" not in DEATH_EVENT_TABLE_COLUMNS
