"""pose_kinematics product tests."""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.pose_kinematics.compute import (
    compute_pose_kinematics_features,
)
from data_pipeline.feature_extraction.pose_kinematics.contract import (
    POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS,
    validate_pose_kinematics_features,
)
from tests.data_pipeline.feature_extraction._feature_fixtures import make_inputs


def test_compute_one_row_per_snip_and_validates():
    snip, masks, inv, reg = make_inputs()
    df = compute_pose_kinematics_features(snip, masks, inv)
    assert len(df) == len(snip)
    assert list(df.columns) == POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS
    validate_pose_kinematics_features(df, physical_embryo_registry_df=reg, check_sources=True)


def test_first_frame_kinematics_null_then_nonzero():
    snip, masks, inv, _ = make_inputs(time_indices=(0, 1, 2))
    df = compute_pose_kinematics_features(snip, masks, inv).sort_values("time_index").reset_index(drop=True)
    # First frame in the track -> null kinematics; later frames -> finite displacement.
    assert pd.isna(df.iloc[0]["displacement_um"])
    assert df.iloc[1]["displacement_um"] > 0
    assert df.iloc[1]["delta_time_s"] == 100.0
