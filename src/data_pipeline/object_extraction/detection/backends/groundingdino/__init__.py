"""GroundingDINO detection backend."""

from .config import GroundingDinoDetectionConfig
from .run_groundingdino_detection import DETECTOR_BACKEND, detect_frame

__all__ = ["GroundingDinoDetectionConfig", "DETECTOR_BACKEND", "detect_frame"]
