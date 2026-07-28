"""Detectron2 detection config — stub.

Placeholder config for the Detectron2 backend. The concrete Facebook model name / config / weights
are not yet chosen (see ``specs/detect-seg-track/working/open_questions.md`` "Backend Routing").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detectron2DetectionConfig:
    """Placeholder filtering / inference knobs for the Detectron2 backend (not yet implemented)."""

    confidence_threshold: float = 0.5
    device: str = "cpu"
