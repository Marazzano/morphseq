"""GroundingDINO detection config — the seam where backend knobs live.

These are the GroundingDINO-specific filtering knobs (prompt + thresholds). The shared runner and
shared validator know nothing about them; they only see the resulting ``is_kept`` outcome. Defaults
match the live inline path
(``segmentation_and_tracking/pipelines/segmentation_and_tracking.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingDinoDetectionConfig:
    """Filtering / inference knobs for the GroundingDINO detector backend."""

    text_prompt: str = "individual embryo"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    device: str = "cpu"
