"""
Overlay manager for different annotation types.

Handles smart positioning and rendering of various overlay types:
- GDINO detection bounding boxes
- SAM2 segmentation masks  
- Embryo metadata text
- QC flags and indicators
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .video_config import VideoConfig, OVERLAY_COLORS, COLORBLIND_PALETTE

@dataclass
class OverlayConfig:
    """Configuration for specific overlay types."""
    color: Tuple[int, int, int]
    thickness: int = 2
    alpha: float = 0.8
    font_scale: float = 0.7
    position: str = "auto"  # "auto", "top_left", "bottom_left", etc.

class OverlayManager:
    """
    Manages different overlay types and smart positioning to avoid crowding.
    """
    
    def __init__(self, video_config: VideoConfig):
        self.config = video_config
        self._color_cycle = self._get_color_cycle()
        
        # Default overlay configurations
        self.overlay_configs = {
            'detection': OverlayConfig(
                color=OVERLAY_COLORS['gdino_detection'],
                thickness=self.config.BBOX_THICKNESS,
                alpha=self.config.BBOX_ALPHA
            ),
            'mask': OverlayConfig(
                color=OVERLAY_COLORS['sam2_mask'],
                thickness=self.config.MASK_OUTLINE_THICKNESS,
                alpha=self.config.MASK_ALPHA
            ),
            'metadata': OverlayConfig(
                color=OVERLAY_COLORS['embryo_metadata'],
                thickness=2,
                alpha=0.9,
                position="bottom_left"
            ),
            'qc_flags': OverlayConfig(
                color=OVERLAY_COLORS['qc_flag_good'],
                thickness=2,
                alpha=0.9,
                position="top_left"
            )
        }
        
    def add_overlay(self, 
                   frame: np.ndarray, 
                   overlay_data: Any, 
                   overlay_type: str) -> np.ndarray:
        """
        Add overlay to frame based on type and data.
        
        Args:
            frame: Input frame
            overlay_data: Data structure containing overlay info
            overlay_type: 'detection', 'mask', 'metadata', 'qc_flags'
        """
        if overlay_type == 'detection':
            return self._add_detection_overlay(frame, overlay_data)
        elif overlay_type == 'mask':
            return self._add_mask_overlay(frame, overlay_data)
        elif overlay_type == 'metadata':
            return self._add_metadata_overlay(frame, overlay_data)
        elif overlay_type == 'qc_flags':
            return self._add_qc_flags_overlay(frame, overlay_data)
        else:
            print(f"⚠️ Unknown overlay type: {overlay_type}")
            return frame
            
    def _add_detection_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Add GDINO detection bounding boxes.
        
        Expected format:
        [{"bbox": [x, y, w, h], "confidence": 0.95, "label": "embryo"}, ...]
        """
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            x, y, w, h = bbox
            confidence = detection.get('confidence', 0.0)
            label = detection.get('label', 'detection')
            
            # Get color for this detection
            color = self._get_detection_color(i)
            
            # Draw bounding box
            cv2.rectangle(
                frame, 
                (int(x), int(y)), 
                (int(x + w), int(y + h)), 
                color, 
                self.overlay_configs['detection'].thickness
            )
            
            # Add label with confidence
            label_text = f"{label} {confidence:.2f}"
            self._add_text_with_background(
                frame, 
                label_text, 
                (int(x), int(y) - 10),
                color,
                scale=0.6
            )
            
        return frame
        
    def _add_mask_overlay(self, frame: np.ndarray, masks: List[Dict]) -> np.ndarray:
        """
        Add SAM2 segmentation masks.
        
        Expected format:
        [{"mask": numpy_array, "embryo_id": "e01"}, ...]
        """
        for i, mask_data in enumerate(masks):
            mask = mask_data.get('mask')
            embryo_id = mask_data.get('embryo_id', f'e{i+1:02d}')
            
            if mask is None:
                continue
                
            # Get color for this mask
            color = self._get_detection_color(i)
            
            # Create colored mask overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            
            # Blend with original frame
            alpha = self.overlay_configs['mask'].alpha
            frame = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
            
            # Add mask outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            cv2.drawContours(
                frame, 
                contours, 
                -1, 
                color, 
                self.overlay_configs['mask'].thickness
            )
            
            # Add embryo ID label
            if contours:
                # Position label at centroid of largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self._add_text_with_background(
                        frame, 
                        embryo_id, 
                        (cx, cy),
                        color,
                        scale=0.8
                    )
                    
        return frame
        
    def _add_metadata_overlay(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Add embryo metadata text (phenotype, treatment, etc.).
        
        Expected format:
        {"phenotype": "normal", "treatment": "DMSO", "stage": "blastula"}
        """
        height, width = frame.shape[:2]
        
        # Position at bottom-left
        y_start = height - 50
        x_start = 20
        
        line_height = 25
        current_y = y_start
        
        for key, value in metadata.items():
            if value:  # Only show non-empty values
                text = f"{key.title()}: {value}"
                self._add_text_with_background(
                    frame,
                    text,
                    (x_start, current_y),
                    OVERLAY_COLORS['embryo_metadata'],
                    scale=0.6
                )
                current_y -= line_height
                
        return frame
        
    def _add_qc_flags_overlay(self, frame: np.ndarray, qc_flags: List[str]) -> np.ndarray:
        """
        Add QC flags at top-left.
        
        Expected format: ["BLUR", "LOW_CONTRAST"]
        """
        if not qc_flags:
            return frame
            
        height, width = frame.shape[:2]
        
        # Position at top-left
        y_start = 40
        x_start = 20
        
        line_height = 25
        current_y = y_start
        
        for flag in qc_flags:
            # Color based on flag severity
            if flag in ['BLUR', 'DARK', 'CORRUPT']:
                color = OVERLAY_COLORS['qc_flag_error']
            elif flag in ['LOW_CONTRAST', 'BRIGHT']:
                color = OVERLAY_COLORS['qc_flag_warning']
            else:
                color = OVERLAY_COLORS['qc_flag_good']
                
            self._add_text_with_background(
                frame,
                f"⚠️ {flag}",
                (x_start, current_y),
                color,
                scale=0.6
            )
            current_y += line_height
            
        return frame
        
    def _add_text_with_background(self, 
                                 frame: np.ndarray, 
                                 text: str, 
                                 position: Tuple[int, int],
                                 color: Tuple[int, int, int],
                                 scale: float = 0.7) -> None:
        """Add text with semi-transparent background for better readability."""
        x, y = position
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, 
            self.config.FONT, 
            scale, 
            self.config.FONT_THICKNESS
        )
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 3, y - text_height - 3),
            (x + text_width + 3, y + baseline + 3),
            (0, 0, 0),  # Black background
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            self.config.FONT,
            scale,
            color,
            self.config.FONT_THICKNESS
        )
        
    def _get_detection_color(self, index: int) -> Tuple[int, int, int]:
        """Get color for detection/mask based on index."""
        colors = list(COLORBLIND_PALETTE.values())
        return colors[index % len(colors)]
        
    def _get_color_cycle(self):
        """Generate cycling iterator of colorblind-friendly colors."""
        colors = list(COLORBLIND_PALETTE.values())
        while True:
            for color in colors:
                yield color
