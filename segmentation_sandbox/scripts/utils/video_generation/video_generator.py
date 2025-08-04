"""
Video generator for MorphSeq pipeline - foundation video with overlays.

Supports progressive enhancement:
1. Foundation video (Module 0): Basic image_id overlay
2. Enhanced videos (later modules): Foundation + additional overlays  
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from .video_config import VideoConfig, COLORBLIND_PALETTE, OVERLAY_COLORS
from .overlay_manager import OverlayManager

class VideoGenerator:
    """
    Creates foundation videos and enhanced videos with overlays.
    
    Philosophy: Create foundation video once, then overlay annotations for speed.
    """
    
    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.overlay_manager = OverlayManager(self.config)
        
    def create_foundation_video(self, 
                              jpeg_paths: List[Path], 
                              video_path: Path,
                              video_id: str,
                              verbose: bool = True) -> bool:
        """
        Create basic video with just image_id overlay (10% down from top-right).
        This is the foundation that other modules will enhance.
        """
        if not jpeg_paths:
            if verbose:
                print("❌ No JPEG files to create video from")
            return False
            
        # Get video dimensions from first frame
        first_frame = cv2.imread(str(jpeg_paths[0]))
        if first_frame is None:
            if verbose:
                print("❌ Could not read first frame for video creation")
            return False
            
        height, width = first_frame.shape[:2]
        
        if verbose:
            print(f"🎬 Creating foundation video {width}x{height} with {len(jpeg_paths)} frames...")
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.CODEC)
        video_writer = cv2.VideoWriter(str(video_path), fourcc, self.config.FPS, (width, height))
        
        if not video_writer.isOpened():
            if verbose:
                print(f"❌ Could not open video writer for {video_path}")
            return False
            
        frames_written = 0
        
        for jpeg_path in sorted(jpeg_paths):
            frame = cv2.imread(str(jpeg_path))
            if frame is None:
                continue
                
            # Add image_id overlay (10% down from top-right)
            frame_num = jpeg_path.stem
            image_id = f"{video_id}_t{frame_num}"
            
            frame_with_overlay = self._add_image_id_overlay(frame, image_id)
            
            video_writer.write(frame_with_overlay)
            frames_written += 1
            
        video_writer.release()
        
        if verbose:
            print(f"✅ Foundation video created: {video_path.name} ({frames_written} frames)")
            
        return True
        
    def create_enhanced_video(self,
                            foundation_video_path: Path,
                            output_video_path: Path, 
                            overlay_dict: Dict[str, Any],
                            overlay_type: str = "detection",
                            verbose: bool = True) -> bool:
        """
        Create enhanced video by adding overlays to foundation video.
        
        Args:
            foundation_video_path: Path to existing foundation video
            output_video_path: Where to save enhanced video
            overlay_dict: {image_id: overlay_data} mapping
            overlay_type: 'detection', 'mask', 'metadata', 'qc_flags'
        """
        if not foundation_video_path.exists():
            if verbose:
                print(f"❌ Foundation video not found: {foundation_video_path}")
            return False
            
        # Open foundation video for reading
        cap = cv2.VideoCapture(str(foundation_video_path))
        if not cap.isOpened():
            if verbose:
                print(f"❌ Could not open foundation video: {foundation_video_path}")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.CODEC)
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            if verbose:
                print(f"❌ Could not open output video writer: {output_video_path}")
            cap.release()
            return False
            
        if verbose:
            print(f"🎨 Creating enhanced video with {overlay_type} overlays...")
            
        frame_idx = 0
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Generate image_id for this frame (assuming standard naming)
            frame_num = str(frame_idx).zfill(4)
            # Extract video_id from foundation video name (remove .mp4)
            video_id = foundation_video_path.stem
            image_id = f"{video_id}_t{frame_num}"
            
            # Add overlays if available for this image_id
            if image_id in overlay_dict:
                frame = self.overlay_manager.add_overlay(
                    frame, 
                    overlay_dict[image_id], 
                    overlay_type
                )
                
            out.write(frame)
            frames_processed += 1
            frame_idx += 1
            
        cap.release()
        out.release()
        
        if verbose:
            overlays_applied = sum(1 for image_id in overlay_dict.keys() 
                                 if image_id.startswith(video_id))
            print(f"✅ Enhanced video created: {output_video_path.name}")
            print(f"   📊 {frames_processed} frames processed, {overlays_applied} frames with overlays")
            
        return True
        
    def _add_image_id_overlay(self, frame: np.ndarray, image_id: str) -> np.ndarray:
        """
        Add image_id text overlay at top-right, 10% down from top.
        Matches original implementation positioning.
        """
        height, width = frame.shape[:2]
        
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            image_id, 
            self.config.FONT, 
            self.config.FONT_SCALE, 
            self.config.FONT_THICKNESS
        )
        
        # Position: 10% down from top, right-aligned with margin
        text_x = width - text_width - self.config.IMAGE_ID_MARGIN_RIGHT
        text_y = int(height * self.config.IMAGE_ID_Y_PERCENTAGE)
        
        # Add semi-transparent background for better readability
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (text_x - 5, text_y - text_height - 5), 
            (text_x + text_width + 5, text_y + 5), 
            self.config.TEXT_BACKGROUND_COLOR, 
            -1
        )
        
        # Blend background
        frame = cv2.addWeighted(
            overlay, 
            self.config.TEXT_BACKGROUND_ALPHA, 
            frame, 
            1 - self.config.TEXT_BACKGROUND_ALPHA, 
            0
        )
        
        # Add text
        cv2.putText(
            frame, 
            image_id, 
            (text_x, text_y), 
            self.config.FONT, 
            self.config.FONT_SCALE, 
            self.config.TEXT_COLOR, 
            self.config.FONT_THICKNESS
        )
        
        return frame
        
    @staticmethod
    def extract_video_id_from_path(video_path: Path) -> str:
        """Extract video_id from video file path."""
        return video_path.stem
        
    @staticmethod
    def generate_image_id(video_id: str, frame_number: int) -> str:
        """Generate standard image_id from video_id and frame number."""
        return f"{video_id}_t{str(frame_number).zfill(4)}"
