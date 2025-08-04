"""
Detection and Segmentation Module (Module 2)
===========================================

This module provides object detection and segmentation capabilities:
- GroundingDINO detection with annotation management
- SAM2 video segmentation (planned)
- Quality control utilities (planned)
- Mask export utilities (planned)

Key classes:
- GroundedDinoAnnotations: Detection annotation management with metadata integration
"""

from .grounded_dino_utils import (
    GroundedDinoAnnotations,
    load_groundingdino_model,
    load_config,
    get_model_metadata,
    calculate_detection_iou,
    run_inference,
    visualize_detections,
    gdino_inference_with_visualization
)

__all__ = [
    'GroundedDinoAnnotations',
    'load_groundingdino_model', 
    'load_config',
    'get_model_metadata',
    'calculate_detection_iou',
    'run_inference',
    'visualize_detections',
    'gdino_inference_with_visualization'
]
