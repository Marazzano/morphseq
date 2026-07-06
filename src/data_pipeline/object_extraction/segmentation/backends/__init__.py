"""Segmentation backend implementations + backend-selection config.

Backend selection (which detector/tracker a run uses) is re-exported here so callers
import it as ``data_pipeline.object_extraction.segmentation.backends.load_segmentation_backends_config``,
the same path they always used; the implementations live in ``sam2_video`` / ``unet_snip``.
"""

from .selection import (
    DetectorBackend,
    SegmentationBackendsConfig,
    SUPPORTED_DETECTOR_BACKENDS,
    SUPPORTED_TRACKER_BACKENDS,
    TrackerBackend,
    load_segmentation_backends_config,
)

__all__ = [
    "DetectorBackend",
    "SegmentationBackendsConfig",
    "SUPPORTED_DETECTOR_BACKENDS",
    "SUPPORTED_TRACKER_BACKENDS",
    "TrackerBackend",
    "load_segmentation_backends_config",
]
