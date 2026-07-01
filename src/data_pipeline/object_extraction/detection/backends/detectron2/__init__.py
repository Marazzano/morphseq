"""Detectron2 detection backend (interface-conformant stub; not yet implemented)."""

from .config import Detectron2DetectionConfig
from .run_detectron2_detection import DETECTOR_BACKEND, detect_frame

__all__ = ["Detectron2DetectionConfig", "DETECTOR_BACKEND", "detect_frame"]
