"""pose_kinematics_features contract — orientation/bbox + kinematics per snip.

Grain: one row per ``snip_id``. Pose columns (orientation, bbox) are required-finite. Kinematics
columns (displacement/speed/deltas) are NULLABLE: the first observation in each track has no prior
frame, so its kinematics are documented null by contract.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_SPINE_COLUMNS,
    validate_feature_table,
)

_FEATURE_COLUMNS: tuple[str, ...] = (
    "orientation_angle",
    "bbox_width_um",
    "bbox_height_um",
    "displacement_um",
    "speed_um_per_s",
    "delta_x_um",
    "delta_y_um",
    "delta_time_s",
)
# First-frame-per-track kinematics are null by contract.
_NULLABLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "displacement_um",
    "speed_um_per_s",
    "delta_x_um",
    "delta_y_um",
    "delta_time_s",
)

POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS: list[str] = list(SNIP_SPINE_COLUMNS + _FEATURE_COLUMNS)


def validate_pose_kinematics_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "pose_kinematics_features",
) -> None:
    """Fail loud unless ``df`` is a valid pose_kinematics_features table (spine first)."""
    validate_feature_table(
        df,
        required_columns=POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS,
        feature_columns=_FEATURE_COLUMNS,
        nullable_feature_columns=_NULLABLE_FEATURE_COLUMNS,
        scope_label=scope_label,
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
    )
