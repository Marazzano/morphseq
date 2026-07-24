"""stage_predictions product tests."""

from __future__ import annotations

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
    assert first["stage_prediction_status"] == "predicted"
    # rate = 0.055*30 - 0.57 = 1.08 hpf/hr; t=1 -> 100s -> +0.03 hpf.
    second = df.sort_values("time_index").iloc[1]
    assert abs(second["predicted_stage_hpf"] - (11.0 + (100 / 3600) * 1.08)) < 1e-6
    validate_stage_prediction_features(df, physical_embryo_registry_df=reg, check_sources=True)


def test_missing_plate_row_fails_loud():
    snip, _, inv, _ = make_inputs()
    empty_plate = make_plate_metadata().iloc[0:0]
    with pytest.raises(ValueError, match="not in plate_metadata"):
        compute_stage_prediction_features(snip, inv, empty_plate)


def test_unresolved_start_age_emits_explicit_nullable_prediction():
    snip, _, inv, reg = make_inputs()
    plate = make_plate_metadata()
    plate["start_age_hpf"] = None
    df = compute_stage_prediction_features(snip, inv, plate)
    assert df["predicted_stage_hpf"].isna().all()
    assert set(df["stage_prediction_status"]) == {"missing_start_age_hpf"}
    validate_stage_prediction_features(
        df, physical_embryo_registry_df=reg, check_sources=True
    )
