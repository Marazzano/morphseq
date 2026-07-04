"""GroundingDINO detection config — the seam where backend knobs live.

These are the GroundingDINO-specific filtering knobs (prompt + thresholds). The shared runner and
shared validator know nothing about them; they only see the resulting ``is_kept`` outcome. Defaults
match the live inline path
(``segmentation_and_tracking/pipelines/segmentation_and_tracking.py``).

Confidence gating happens exactly once, at ``box_threshold``. There used to be a second,
stricter ``confidence_threshold`` (0.45) re-applied after detection — a duplicate gate on the
same score that silently dropped real detections between 0.35 and 0.45 (13.4% of all detections
above ``box_threshold`` dataset-wide; confirmed via the F05 recovery investigation in the SAM3
exemplar review). Removed: ``box_threshold`` is the one confidence knob; ``filter_detections``
only does IoU/NMS dedup now.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingDinoDetectionConfig:
    """Filtering / inference knobs for the GroundingDINO detector backend."""

    text_prompt: str = "individual embryo"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = "cpu"
