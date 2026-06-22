"""data_pipeline.viz — contract-native overlay rendering for detections and segmentations.

Reads ``frame_detections`` and ``frame_masks`` contract DataFrames; emits MP4 overlay videos.
For legacy evaluation video rendering (GroundedSAM2 / COCO JSON input), see
``segmentation.video_generation``.
"""

from data_pipeline.viz.render_well import (
    render_combined_video,
    render_detection_video,
    render_segmentation_video,
)

__all__ = [
    "render_combined_video",
    "render_detection_video",
    "render_segmentation_video",
]
