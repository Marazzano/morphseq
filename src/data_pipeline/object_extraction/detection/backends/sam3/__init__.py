"""SAM3 detection backend adapter surface."""

from .adapt_sam3_detections import (
    DETECTOR_BACKEND,
    DEFAULT_DETECTOR_MODEL_ID,
    adapt_sam3_detections,
    detect_frame_from_normalized_response,
)

__all__ = [
    "DETECTOR_BACKEND",
    "DEFAULT_DETECTOR_MODEL_ID",
    "adapt_sam3_detections",
    "detect_frame_from_normalized_response",
]
