"""Deterministic fake SAM2 predictor for Session B contract testing.

The fake predictor generates rectangular bool masks without requiring GPU or real SAM2.
It satisfies the same `SegmentOneWell` type alias as the real adapter so integration
tests can call `run_sam2_video_for_wells` with it directly.

This module is NOT wired into production paths. It exists only to let Session B tests
prove the adapter/validator pipeline end-to-end before real SAM2 is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.segmentation.backends.sam2_video.adapt_sam2_output import (
    adapt_sam2_well_output,
)
from data_pipeline.segmentation.sam2_video.run_sam2_video import Sam2WellInput


class FakePredictor:
    """Deterministic rectangular-mask predictor.

    For each frame it produces `n_objects` non-overlapping horizontal rectangles,
    each covering `mask_fill_fraction` of the image area. Output is fully
    deterministic given the same frame dimensions and object count.
    """

    def __init__(self, n_objects: int = 1, mask_fill_fraction: float = 0.1) -> None:
        if n_objects < 1:
            raise ValueError("n_objects must be >= 1")
        if not (0 < mask_fill_fraction <= 1.0):
            raise ValueError("mask_fill_fraction must be in (0, 1]")
        self.n_objects = n_objects
        self.mask_fill_fraction = mask_fill_fraction

    def predict_frame(
        self,
        height: int,
        width: int,
        sam2_frame_index: int,
    ) -> dict[int, np.ndarray]:
        """Return {object_id: bool_mask} for one frame.

        Masks are non-overlapping horizontal bands, one per object. The band height
        is chosen so each covers approximately `mask_fill_fraction` of the image.
        """
        band_height = max(1, int(round(height * self.mask_fill_fraction)))
        result: dict[int, np.ndarray] = {}
        for obj_id in range(self.n_objects):
            y_start = (obj_id * band_height) % height
            y_end = min(y_start + band_height, height)
            mask = np.zeros((height, width), dtype=bool)
            mask[y_start:y_end, :] = True
            result[obj_id] = mask
        return result


def segment_one_well_fake(
    predictor: FakePredictor,
    well: Sam2WellInput,
) -> pd.DataFrame:
    """SegmentOneWell adapter for the fake predictor.

    Assigns sequential sam2_frame_index values by sorting model_frame_view on
    (time_index, image_id), matching the ordering that build_sam2_frame_view would use.
    Calls adapt_sam2_well_output with the fake predictor's raw output so the result
    satisfies the same contract as the real adapter.
    """
    ordered = (
        well.model_frame_view.copy()
        .sort_values(["time_index", "image_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    ordered["sam2_frame_index"] = ordered.index

    sam2_raw_output: dict[int, dict[int, np.ndarray]] = {}
    for _, frame_row in ordered.iterrows():
        sam2_idx = int(frame_row["sam2_frame_index"])
        height = int(frame_row["image_height_px"])
        width = int(frame_row["image_width_px"])
        sam2_raw_output[sam2_idx] = predictor.predict_frame(height, width, sam2_idx)

    return adapt_sam2_well_output(
        well.well_id,
        sam2_raw_output,
        ordered,
        model_id="fake_predictor:v1",
    )
