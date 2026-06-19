"""Batch runner shape for the SAM2 video backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_pipeline.segmentation.sam2_video.model_loader import (
    Sam2VideoModelConfig,
    load_sam2_video_model,
)


@dataclass(frozen=True)
class Sam2WellInput:
    """Inputs needed to segment one well after frame/detection contracts are loaded."""

    well_id: str
    model_frame_view: pd.DataFrame
    frame_detections: pd.DataFrame


@dataclass(frozen=True)
class Sam2WellResult:
    """Frame-mask result for one well."""

    well_id: str
    frame_masks: pd.DataFrame


SegmentOneWell = Callable[[Any, Sam2WellInput], pd.DataFrame]


def run_sam2_video_for_wells(
    wells: Iterable[Sam2WellInput],
    *,
    model_config: Sam2VideoModelConfig,
    segment_one_well: SegmentOneWell,
) -> list[Sam2WellResult]:
    """Run SAM2 over multiple wells while loading the model once.

    The orchestration layer should use `well_runner.run_well_ids_for_experiment(...)` and
    shard-path helpers to decide which wells feed this call. This function owns the backend
    invariant that the expensive SAM2 predictor is constructed once per run batch, not once
    per well.
    """
    well_inputs = list(wells)
    if not well_inputs:
        return []

    predictor = load_sam2_video_model(model_config)
    results: list[Sam2WellResult] = []
    for well in well_inputs:
        frame_masks = segment_one_well(predictor, well)
        results.append(Sam2WellResult(well_id=str(well.well_id), frame_masks=frame_masks))
    return results
