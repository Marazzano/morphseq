"""pose_kinematics compute — pose per mask + kinematics within each track, RLE-native.

Pose (orientation, bbox, centroid) is per-mask. Kinematics (displacement, speed, deltas) are
temporal: group by track_id, order by time_index, and difference consecutive frames using frame
timing (elapsed_time_s) from frame_inventory. First frame per track -> null kinematics.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.mask_geometry_metrics import compute_mask_geometry
from data_pipeline.feature_extraction.pose_kinematics_metrics import (
    compute_kinematics,
    compute_pose_features,
)
from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_SPINE_COLUMNS,
    pixel_size_for_image,
)
from data_pipeline.segmentation.masks.mask_rle import decode_binary_mask_rle

from .contract import POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS

# Frame timing source on a frame_inventory row (target name first, legacy fallbacks).
_TIME_COLUMNS: tuple[str, ...] = ("elapsed_time_s", "experiment_time_s", "time_s")


def _elapsed_time_s(frame_inventory_by_image: pd.DataFrame, image_id: str, snip_id: str) -> float:
    row = frame_inventory_by_image.loc[image_id]
    for col in _TIME_COLUMNS:
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    raise ValueError(
        f"pose_kinematics: frame_inventory row for image_id {image_id!r} (snip {snip_id!r}) has no "
        f"frame timing. Expected one of {_TIME_COLUMNS}."
    )


def compute_pose_kinematics_features(
    snip_inventory_df: pd.DataFrame,
    frame_masks_df: pd.DataFrame,
    frame_inventory_df: pd.DataFrame,
    *,
    mask_decoder: Callable[[dict], np.ndarray] = decode_binary_mask_rle,
) -> pd.DataFrame:
    """Return one pose_kinematics_features row per snip (pose per mask, kinematics within track)."""
    frame_masks_by_mask = frame_masks_df.set_index("mask_id")
    frame_inventory_by_image = frame_inventory_df.set_index("image_id")

    # Pose + centroid + timing per snip, in track/time order so kinematics can difference neighbours.
    ordered = snip_inventory_df.sort_values(by=["track_id", "time_index"]).reset_index(drop=True)

    rows: list[dict] = []
    prev_by_track: dict[str, dict] = {}
    for _, snip in ordered.iterrows():
        snip_id = str(snip["snip_id"])
        mask_id = str(snip["mask_id"])
        image_id = str(snip["image_id"])
        track_id = str(snip["track_id"])

        mask_row = frame_masks_by_mask.loc[mask_id]
        rle = mask_row["mask_rle"]
        rle = json.loads(str(rle)) if isinstance(rle, str) else rle
        mask = mask_decoder(rle)
        pixel_size_um = pixel_size_for_image(frame_inventory_by_image, image_id, snip_id)

        geometry = compute_mask_geometry(mask, pixel_size_um)
        pose = compute_pose_features(mask, pixel_size_um)
        centroid = (geometry["centroid_x_um"], geometry["centroid_y_um"])
        current_time = _elapsed_time_s(frame_inventory_by_image, image_id, snip_id)

        prev = prev_by_track.get(track_id)
        if prev is None:
            kin = compute_kinematics(centroid, None, current_time, None)
        else:
            kin = compute_kinematics(centroid, prev["centroid"], current_time, prev["time"])

        row = {col: snip[col] for col in SNIP_SPINE_COLUMNS}
        row["orientation_angle"] = pose["orientation_angle"]
        # bbox naming follows the consolidated feature contract: width/height come from the PCA
        # length/width of the mask (long axis = height), matching the legacy batch.
        row["bbox_width_um"] = geometry["width_um"]
        row["bbox_height_um"] = geometry["length_um"]
        row["displacement_um"] = kin["displacement_um"]
        row["speed_um_per_s"] = kin["speed_um_per_s"]
        row["delta_x_um"] = kin["delta_x_um"]
        row["delta_y_um"] = kin["delta_y_um"]
        row["delta_time_s"] = kin["delta_time_s"]
        rows.append(row)

        prev_by_track[track_id] = {"centroid": centroid, "time": current_time}

    return pd.DataFrame(rows, columns=POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS)
