"""pose_kinematics feature product — orientation/bbox + per-track displacement/speed per snip."""

from .compute import compute_pose_kinematics_features
from .contract import (
    POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS,
    validate_pose_kinematics_features,
)

__all__ = [
    "POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS",
    "validate_pose_kinematics_features",
    "compute_pose_kinematics_features",
]
