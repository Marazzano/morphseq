"""Detectron2 backend adapter — interface-conformant stub.

This exists only to prove the router is backend-agnostic: it conforms to the same ``detect_frame``
adapter signature as the GroundingDINO backend but raises ``NotImplementedError``. The real
implementation is deferred until the Facebook model name / config / weights are chosen (see
``specs/detect-seg-track/working/open_questions.md`` "Backend Routing").
"""

from __future__ import annotations

from pathlib import Path

from .config import Detectron2DetectionConfig

DETECTOR_BACKEND = "detectron2"


def detect_frame(
    model,
    image_path: str | Path,
    *,
    identity_row: dict,
    detector_model_id: str,
    config: Detectron2DetectionConfig | None = None,
) -> list[dict]:
    """Adapter-conformant stub. Same signature as the GroundingDINO backend; not yet implemented.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(
        "Detectron2 detection backend is not implemented yet "
        "(Facebook model name / config / weights not chosen). "
        "It conforms to the detect_frame adapter interface so the router stays backend-agnostic."
    )
