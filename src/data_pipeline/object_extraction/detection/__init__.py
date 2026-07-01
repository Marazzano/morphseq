"""Standalone detection stage — the microscope-agnostic ``frame_detections`` product.

Detection is a sibling of ``segmentation/`` (NOT under it). It answers, for each trusted frame, where
candidate embryos are and how confident the detector is. Backend-specific code diverges before
``frame_detections``; everything downstream consumes the shared contract through ``is_kept``.

Public surface:
  - contract constants + id helpers (``frame_detections_contract``)
  - layered validators (``validate_frame_detections``, ``validate_frame_detection_block``)
  - consumer view + CSV reader (``kept_frame_detections``, ``read_frame_detections_csv``)
  - backend-agnostic router (``run_frame_detection``, ``run_frame_detection_df``)

See ``specs/detect-seg-track/targets/detection_world.md``.
"""

from .frame_detections_contract import (
    ALLOWED_BBOX_FORMATS,
    BBOX_COLUMNS,
    CONFIDENCE_RANGE,
    NO_CANDIDATE_SUFFIX,
    REQUIRED_DETECTION_BLOCK,
    REQUIRED_FRAME_DETECTIONS_COLUMNS,
    detection_id,
    is_no_candidate_id,
    no_candidate_detection_id,
)
from .kept_frame_detections import kept_frame_detections, read_frame_detections_csv
from .run_frame_detection import (
    BACKENDS,
    run_frame_detection,
    run_frame_detection_df,
)
from .validate_frame_detections import (
    validate_frame_detection_block,
    validate_frame_detections,
)

__all__ = [
    # contract
    "REQUIRED_FRAME_DETECTIONS_COLUMNS",
    "REQUIRED_DETECTION_BLOCK",
    "BBOX_COLUMNS",
    "ALLOWED_BBOX_FORMATS",
    "CONFIDENCE_RANGE",
    "NO_CANDIDATE_SUFFIX",
    "detection_id",
    "no_candidate_detection_id",
    "is_no_candidate_id",
    # validators
    "validate_frame_detections",
    "validate_frame_detection_block",
    # consumer
    "kept_frame_detections",
    "read_frame_detections_csv",
    # router
    "run_frame_detection",
    "run_frame_detection_df",
    "BACKENDS",
]
