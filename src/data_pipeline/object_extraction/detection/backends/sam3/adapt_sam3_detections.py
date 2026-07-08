"""SAM3 detection adapter — normalized SAM3 output -> shared frame_detections rows.

This module is deliberately torch-free. The model backend normalizes native SAM3 responses into a
small list of box/score dictionaries; this pipeline backend turns those dictionaries into the shared
``frame_detections`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from data_pipeline.object_extraction.detection.frame_detections_contract import (
    REQUIRED_FRAME_DETECTIONS_COLUMNS,
    detection_id as make_detection_id,
    no_candidate_detection_id,
)

DETECTOR_BACKEND = "sam3"
DEFAULT_DETECTOR_MODEL_ID = "facebook/sam3.1"
DEFAULT_CLASS_LABEL = "zebrafish embryo"


def adapt_sam3_detections(
    detections: Iterable[Mapping[str, object]],
    *,
    identity_row: Mapping[str, object],
    detector_model_id: str = DEFAULT_DETECTOR_MODEL_ID,
    class_label: str = DEFAULT_CLASS_LABEL,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Return canonical ``frame_detections`` rows for one frame.

    ``detections`` is the normalized model-backend output. Each item may use ``box_xyxy``,
    ``bbox_xyxy``, or ``bbox`` for absolute-pixel xyxy coordinates; score may be named ``score`` or
    ``confidence``. Empty input emits a single ``_det_none`` placeholder row.
    """

    rows = detect_frame_from_normalized_response(
        detections,
        identity_row=identity_row,
        detector_model_id=detector_model_id,
        class_label=class_label,
    )
    df = pd.DataFrame(rows, columns=REQUIRED_FRAME_DETECTIONS_COLUMNS)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def detect_frame_from_normalized_response(
    detections: Iterable[Mapping[str, object]],
    *,
    identity_row: Mapping[str, object],
    detector_model_id: str = DEFAULT_DETECTOR_MODEL_ID,
    class_label: str = DEFAULT_CLASS_LABEL,
) -> list[dict[str, object]]:
    image_id = str(identity_row["image_id"])
    raw = list(detections)
    if not raw:
        return [_placeholder_row(identity_row, detector_model_id=detector_model_id)]

    rows: list[dict[str, object]] = []
    for idx, det in enumerate(_sort_detections(raw)):
        x0, y0, x1, y1 = _coerce_box(det)
        score = _coerce_score(det)
        rows.append({
            **dict(identity_row),
            "detection_id": make_detection_id(image_id, idx),
            "detector_backend": DETECTOR_BACKEND,
            "detector_model_id": str(det.get("detector_model_id", detector_model_id)),
            "class_label": str(det.get("class_label", det.get("label", class_label))),
            "confidence": score,
            "bbox_x_min_px": x0,
            "bbox_y_min_px": y0,
            "bbox_x_max_px": x1,
            "bbox_y_max_px": y1,
            "bbox_format": "xyxy_px_abs",
            "is_kept": bool(det.get("is_kept", True)),
        })
    return rows


def _placeholder_row(
    identity_row: Mapping[str, object],
    *,
    detector_model_id: str,
) -> dict[str, object]:
    image_id = str(identity_row["image_id"])
    return {
        **dict(identity_row),
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


def _sort_detections(detections: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        detections,
        key=lambda det: (
            -_coerce_score(det),
            _coerce_box(det)[0],
            _coerce_box(det)[1],
        ),
    )


def _coerce_score(det: Mapping[str, object]) -> float:
    value = det.get("score", det.get("confidence", 0.0))
    return float(value)


def _coerce_box(det: Mapping[str, object]) -> tuple[float, float, float, float]:
    for key in ("box_xyxy", "bbox_xyxy", "bbox"):
        if key in det:
            box = det[key]
            break
    else:
        raise ValueError(f"SAM3 detection missing xyxy box field: {sorted(det.keys())}")
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        raise ValueError(f"SAM3 detection box must be a 4-item xyxy sequence, got {box!r}")
    return tuple(float(v) for v in box)  # type: ignore[return-value]
