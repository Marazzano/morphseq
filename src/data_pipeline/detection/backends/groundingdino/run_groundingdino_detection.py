"""GroundingDINO backend adapter — native model output → shared frame_detections rows.

This is the seam where GroundingDINO config (``box_threshold``, ``text_threshold``, confidence / NMS
policy, prompt) becomes the shared ``is_kept`` outcome. Inference is REUSED verbatim from
``segmentation/grounded_sam2/gdino_detection.py`` (``detect_embryos`` / ``filter_detections``) — this
module does not re-implement model inference.

The critical contract behavior is **flag-not-drop**: the old ingestor dropped rejected and empty
frames. Here every raw candidate becomes a row (``is_kept`` marks whether it survived filtering), and
a frame with zero candidates gets a single ``_det_none`` placeholder row. That is what makes the
artifact prove every frame was processed.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.detection.frame_detections_contract import (
    detection_id as make_detection_id,
    no_candidate_detection_id,
)
from data_pipeline.segmentation.grounded_sam2.gdino_detection import (
    detect_embryos,
    filter_detections,
)

from .config import GroundingDinoDetectionConfig

DETECTOR_BACKEND = "groundingdino"


def _clamp(val: float, lo: float, hi: float) -> float:
    return float(min(max(val, lo), hi))


def _norm_to_abs_xyxy(box_xyxy_norm, *, width: int, height: int) -> list[float]:
    """Normalized [0,1] xyxy → absolute pixel xyxy, clamped to image bounds.

    Mirrors ``segmentation_and_tracking/ingestors/gdino_ingestor.py::_norm_to_abs_xyxy``.
    """
    x0, y0, x1, y1 = [float(v) for v in box_xyxy_norm]
    return [
        _clamp(x0 * width, 0.0, float(width)),
        _clamp(y0 * height, 0.0, float(height)),
        _clamp(x1 * width, 0.0, float(width)),
        _clamp(y1 * height, 0.0, float(height)),
    ]


def _det_key(det: dict) -> tuple:
    """A hashable identity for a raw detection dict, to test kept-membership."""
    box = tuple(round(float(v), 6) for v in det.get("box_xyxy", []))
    return (box, round(float(det.get("confidence", 0.0)), 6), str(det.get("phrase", "")))


def detect_frame(
    model,
    image_path: str | Path,
    *,
    identity_row: dict,
    detector_model_id: str,
    config: GroundingDinoDetectionConfig | None = None,
) -> list[dict]:
    """Detect candidates in one frame, returning shared-table row dicts (detection block only).

    ``identity_row`` supplies ``image_id``, ``image_width_px``, ``image_height_px`` (carried from the
    frame_inventory row; the router merges the full identity header). Returns one row per raw
    candidate (``is_kept`` flags filtering outcome) or a single ``_det_none`` placeholder when the
    detector produced no candidates.
    """
    config = config or GroundingDinoDetectionConfig()
    image_id = str(identity_row["image_id"])
    width = int(identity_row["image_width_px"])
    height = int(identity_row["image_height_px"])

    # REUSE inference verbatim. All raw candidates, then the kept subset.
    raw = detect_embryos(
        model=model,
        image_path=Path(image_path),
        device=str(config.device),
        text_prompt=config.text_prompt,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
    )
    kept = filter_detections(
        raw,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
    )

    # No candidates at all → single placeholder row, flag-not-drop.
    if not raw:
        return [_placeholder_row(image_id, detector_model_id)]

    kept_keys = {_det_key(d) for d in kept}

    # Deterministic candidate ordering (matches the live ingestor: desc confidence, then x0, y0).
    raw_sorted = sorted(
        raw,
        key=lambda d: (
            -float(d.get("confidence", 0.0)),
            float(d.get("box_xyxy", [0, 0, 0, 0])[0]),
            float(d.get("box_xyxy", [0, 0, 0, 0])[1]),
        ),
    )

    rows: list[dict] = []
    for idx, det in enumerate(raw_sorted):
        abs_box = _norm_to_abs_xyxy(det.get("box_xyxy", [0, 0, 0, 0]), width=width, height=height)
        phrase = det.get("phrase")
        rows.append(
            {
                "detection_id": make_detection_id(image_id, idx),
                "detector_backend": DETECTOR_BACKEND,
                "detector_model_id": str(detector_model_id),
                "class_label": str(phrase) if phrase is not None else None,
                "confidence": float(det.get("confidence", 0.0)),
                "bbox_x_min_px": abs_box[0],
                "bbox_y_min_px": abs_box[1],
                "bbox_x_max_px": abs_box[2],
                "bbox_y_max_px": abs_box[3],
                "bbox_format": "xyxy_px_abs",
                "is_kept": _det_key(det) in kept_keys,
            }
        )
    return rows


def _placeholder_row(image_id: str, detector_model_id: str) -> dict:
    """A no-candidate placeholder row: identity-only, value columns NA, is_kept False."""
    return {
        "detection_id": no_candidate_detection_id(image_id),
        "detector_backend": DETECTOR_BACKEND,
        "detector_model_id": str(detector_model_id),
        "class_label": None,
        "confidence": None,
        "bbox_x_min_px": None,
        "bbox_y_min_px": None,
        "bbox_x_max_px": None,
        "bbox_y_max_px": None,
        "bbox_format": None,
        "is_kept": False,
    }
