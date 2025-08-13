#!/usr/bin/env python3
"""
SAM2 Video Processing Utilities for Embryo Segmentation (Refactored)
====================================================================

Refactored SAM2 integration with GroundedDINO annotations for embryo tracking.
This version uses the modular utilities from Module 0 and Module 1:

- Uses parsing_utils for consistent ID operations
- Uses EntityIDTracker for entity validation
- Uses ExperimentMetadata for metadata management
- Uses BaseFileHandler for atomic JSON operations
- Maintains snip_id format with '_s' prefix

Output Structure:
================

GroundedSam2Annotations.json format:
{
  "script_version": "sam2_utils.py",
  "creation_time": "YYYY-MM-DDThh:mm:ss",
  "last_updated": "YYYY-MM-DDThh:mm:ss",
  "entity_tracking": {...},  # Added by EntityIDTracker
  "snip_ids": ["20240411_A01_e01_s0000", "20240411_A01_e01_s0001", ...],  # Note '_s' prefix
  "experiments": {
    "20240411": {
      "images": {
        "20240411_A01_t0000": {  # Note 't' prefix in JSON
          "image_id": "20240411_A01_t0000",
          "frame_index": 0,
          "is_seed_frame": true,
          "embryos": {
            "20240411_A01_e01": {
              "embryo_id": "20240411_A01_e01",
              "snip_id": "20240411_A01_e01_s0000",  # CORRECTED: '_s' prefix format
              "segmentation": {...},
              "segmentation_format": "rle",
              "bbox": [x, y, x, y],
              "area": 1234.5,
              "mask_confidence": 0.85
            }
          }
        }
      }
    }
  }
}
"""

import os
import sys
import json
import yaml
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np
from collections import Counter, defaultdict, OrderedDict
import cv2
import tempfile
import shutil
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure the project root is in the path
SANDBOX_ROOT = Path(__file__).parent.parent.parent
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.append(str(SANDBOX_ROOT))

# Add SAM2 to path - using working path structure
SAM2_MODELS_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/models/sam2")
SAM2_ROOT = SAM2_MODELS_ROOT / "sam2"  # The actual sam2 directory

# Add the models directory to path
if str(SAM2_MODELS_ROOT) not in sys.path:
    sys.path.append(str(SAM2_MODELS_ROOT))

# REFACTORED: Import modular utilities from Module 0 and Module 1
from scripts.utils.parsing_utils import (
    parse_entity_id,
    extract_frame_number, 
    extract_experiment_id,
    extract_embryo_id,
    get_entity_type
)
from scripts.utils.entity_id_tracker import EntityIDTracker
from scripts.utils.base_file_handler import BaseFileHandler
from scripts.metadata.experiment_metadata import ExperimentMetadata


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sam2_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load SAM2 model using exact working approach - simple and clean."""
    print(f"🔧 Loading SAM2 model...")
    
    # Store current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to SAM2 directory (working approach)
        os.chdir(SAM2_ROOT)
        
        # Import SAM2 from the correct location
        from sam2.build_sam import build_sam2_video_predictor
        
        # Build the predictor
        predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        
        if device == "cuda":
            print(f"✅ SAM2 model loaded on GPU")
        else:
            print(f"✅ SAM2 model loaded on CPU")
            
        return predictor
        
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)


class GroundedSamAnnotations(BaseFileHandler):
    """
    SAM2 video processing manager that integrates with GroundedDINO annotations.
    
    REFACTORED to use:
    - BaseFileHandler for atomic JSON operations
    - ExperimentMetadata for metadata management  
    - EntityIDTracker for entity validation
    - parsing_utils for consistent ID operations
    
    Handles:
    - Loading high-quality GroundedDINO annotations
    - Video grouping and seed frame selection
    - SAM2 video segmentation and tracking
    - Structured output generation matching experiment metadata format
    - Entity validation and snip_id standardization
    """

    def __init__(self, 
                filepath: Union[str, Path],
                seed_annotations_path: Union[str, Path],
                experiment_metadata_path: Union[str, Path],
                sam2_config: Optional[str] = None,
                sam2_checkpoint: Optional[str] = None,
                device: str = "cuda",
                target_prompt: str = "individual embryo",
                segmentation_format: str = "rle",
                verbose: bool = True):
        """
        Initialize GroundedSamAnnotations with modular utilities.
        
        Args:
            filepath: Path where SAM2 results will be saved
            seed_annotations_path: Path to GroundedDINO annotations JSON
            experiment_metadata_path: Path to experiment_metadata.json file
            sam2_config: SAM2 model config path (optional, can be set later)
            sam2_checkpoint: SAM2 model checkpoint path (optional, can be set later)
            device: Device for SAM2 model ('cuda' or 'cpu')
            target_prompt: Prompt to use from annotations (default: 'individual embryo')
            segmentation_format: Output format ('rle' or 'polygon')
            verbose: Enable verbose output
        """
        # Initialize BaseFileHandler
        super().__init__(filepath, verbose=verbose)
        
        # Store configuration
        self.seed_annotations_path = Path(seed_annotations_path) if seed_annotations_path else None
        self.experiment_metadata_path = Path(experiment_metadata_path) if experiment_metadata_path else None
        self.target_prompt = target_prompt
        self.segmentation_format = segmentation_format
        self.device = device
        
        # SAM2 model paths (can be set later)
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.predictor = None
        
        if self.verbose:
            print(f"🎬 Initializing GroundedSamAnnotations...")
            print(f"   Target prompt: '{self.target_prompt}'")
            print(f"   Segmentation format: {self.segmentation_format}")
            print(f"   Output file: {self.filepath}")
        
        # EARLY VALIDATION
        validation_errors = []
        
        # Check seed annotations path
        if not self.seed_annotations_path:
            validation_errors.append("No seed annotations path provided")
        elif not self.seed_annotations_path.exists():
            validation_errors.append(f"Seed annotations file not found: {self.seed_annotations_path}")
        
        # Check experiment metadata path
        if not self.experiment_metadata_path:
            validation_errors.append(
                "Missing experiment metadata path. "
                "Please provide the path to your experiment_metadata.json as experiment_metadata_path."
            )
        elif not self.experiment_metadata_path.exists():
            validation_errors.append(f"Experiment metadata file not found: {self.experiment_metadata_path}")
        
        if validation_errors:
            raise ValueError("Validation errors:\n" + "\n".join(f"  - {error}" for error in validation_errors))
        
        # Load components
        self._load_components()
        
        # Load or initialize results
        self.results = self._load_or_initialize_results()
        
        if self.verbose:
            print(f"✅ GroundedSamAnnotations initialized successfully")

    def _load_components(self):
        """Load seed annotations and experiment metadata using modular utilities."""
        if self.verbose:
            print("📚 Loading components...")
        
        # Load seed annotations
        self.seed_annotations = self._load_seed_annotations()
        if not self.seed_annotations:
            raise ValueError("Failed to load seed annotations")
        
        # REFACTORED: Use ExperimentMetadata class instead of custom loading
        self.exp_metadata = None
        if self.experiment_metadata_path and self.experiment_metadata_path.exists():
            self.exp_metadata = ExperimentMetadata(self.experiment_metadata_path)
            
            # FIX: Set base data path for ExperimentMetadata
            # Use the parent directory of experiment_metadata.json as base path
            base_data_path = self.experiment_metadata_path.parent
            self.exp_metadata.set_base_data_path(base_data_path)
            
            self.experiment_metadata = self.exp_metadata.metadata  # For compatibility
        else:
            raise ValueError("Failed to load experiment metadata")
        
        if self.verbose:
            print(f"✅ Components loaded successfully")

    def _load_seed_annotations(self) -> Optional[Dict]:
        """Load seed annotations from GroundedDINO JSON file."""
        try:
            with open(self.seed_annotations_path, 'r') as f:
                annotations = json.load(f)
            
            # Verify high_quality_annotations exist
            if 'high_quality_annotations' not in annotations:
                if self.verbose:
                    print("⚠️ No high_quality_annotations found in seed file. Use GroundedDinoAnnotations.generate_high_quality_annotations() first.")
                return None
            
            return annotations
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load seed annotations: {e}")
            return None

    def _load_or_initialize_results(self) -> Dict:
        """Load existing results or initialize new structure."""
        if self.filepath.exists():
            try:
                existing_data = self.load_json()  # FIXED: use load_json instead of load
                if self.verbose:
                    print(f"📂 Loaded existing results from {self.filepath}")
                return existing_data
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to load existing file, initializing new: {e}")
        
        # Initialize new results structure matching your format
        initial_data = {
            "script_version": "sam2_utils.py (refactored)",
            "creation_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "gsam_id": self._generate_gsam_id(),
            "seed_annotations_path": str(self.seed_annotations_path),
            "experiment_metadata_path": str(self.experiment_metadata_path),
            "target_prompt": self.target_prompt,
            "segmentation_format": self.segmentation_format,
            "device": self.device,
            "snip_ids": [],
            "experiments": {}
        }
        
        if self.verbose:
            print(f"📝 Initialized new results structure")
        
        return initial_data

    def _generate_gsam_id(self) -> str:
        """Generate unique GSAM annotation ID."""
        return f"gsam_{random.randint(1000, 9999)}"

    def get_gsam_id(self) -> str:
        """Get the GSAM ID for this instance."""
        return self.results.get("gsam_id", "unknown")

    def save(self):
        """Save results with entity validation using embedded tracker approach."""
        # REFACTORED: Add entity validation using EntityIDTracker
        try:
            # Update embedded entity tracker (no separate files needed)
            self.results = EntityIDTracker.update_entity_tracker(
                self.results,
                pipeline_step="module_2_segmentation" 
            )
            
            # Skip hierarchy validation for SAM2 processing since it's not relevant
            # and can cause issues when processing partial datasets
            entities = EntityIDTracker.extract_entities(self.results)
            validation_result = EntityIDTracker.validate_hierarchy(entities, check_hierarchy=False)
            
            if validation_result.get('skipped'):
                if self.verbose:
                    print(f"ℹ️ {validation_result['skipped']}")
            elif not validation_result.get('valid', True):
                if self.verbose:
                    print(f"⚠️ Entity validation warnings: {validation_result.get('violations', [])}")
            
            # REFACTORED: Verify snip IDs use '_s' format
            for snip_id in entities.get("snips", []):
                if not snip_id.count("_s") == 1:
                    print(f"⚠️ Non-standard snip_id format: {snip_id}")
                    
            if self.verbose:
                entity_counts = EntityIDTracker.get_counts(entities)
                print(f"📋 Entity tracker updated: {entity_counts}")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Entity validation warning: {e}")
        
        # Update timestamp and save using BaseFileHandler (FIXED: use save_json method)
        self.results["last_updated"] = datetime.now().isoformat()
        self.save_json(self.results)
        self._unsaved_changes = False
        
        if self.verbose:
            print(f"💾 SAM2 results saved to {self.filepath}")

    def set_seed_annotations_path(self, seed_annotations_path: Union[str, Path]):
        """Update seed annotations path and reload."""
        self.seed_annotations_path = Path(seed_annotations_path)
        if not self.seed_annotations_path.exists():
            raise FileNotFoundError(f"Seed annotations file not found: {self.seed_annotations_path}")
        
        self.seed_annotations = self._load_seed_annotations()
        if self.verbose:
            print(f"✅ Updated seed annotations path: {self.seed_annotations_path}")

    def set_sam2_model_paths(self, config_path: str, checkpoint_path: str):
        """Set SAM2 model paths and load model."""
        self.sam2_config = config_path
        self.sam2_checkpoint = checkpoint_path
        self._load_sam2_model()

    def _load_sam2_model(self):
        """Load SAM2 model."""
        if not self.sam2_config or not self.sam2_checkpoint:
            raise ValueError("SAM2 config and checkpoint paths must be set before loading model")
        
        self.predictor = load_sam2_model(self.sam2_config, self.sam2_checkpoint, self.device)
        
        if self.verbose:
            print(f"✅ SAM2 model loaded successfully")

    def group_annotations_by_video(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Group high-quality annotations by video_id for processing."""
        if not self.seed_annotations or 'high_quality_annotations' not in self.seed_annotations:
            if self.verbose:
                print("❌ No high_quality_annotations available")
            return {}
        
        hq_annotations = self.seed_annotations['high_quality_annotations']
        
        # NEW: Handle the correct structure - experiments contain filtered annotations
        all_filtered_annotations = {}
        target_prompt_found = False
        
        for exp_id, exp_data in hq_annotations.items():
            if exp_data.get('prompt') == self.target_prompt:
                target_prompt_found = True
                if 'filtered' in exp_data:
                    all_filtered_annotations.update(exp_data['filtered'])
        
        if not target_prompt_found:
            if self.verbose:
                print(f"❌ Target prompt '{self.target_prompt}' not found in high_quality_annotations")
            return {}
        
        if not all_filtered_annotations:
            if self.verbose:
                print(f"❌ No filtered annotations found for prompt '{self.target_prompt}'")
            return {}
        
        # Group by video_id
        video_groups = defaultdict(lambda: defaultdict(list))
        
        for image_id, detections in all_filtered_annotations.items():
            # REFACTORED: Use parsing_utils for consistent video_id extraction
            try:
                # Parse the full image_id to get components  
                parsed = parse_entity_id(image_id)
                video_id = parsed.get('video_id')
                
                if video_id:
                    video_groups[video_id][image_id] = detections
                else:
                    if self.verbose:
                        print(f"⚠️ Could not extract video_id from {image_id}")
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to parse video_id from {image_id}: {e}")
                continue
        
        # Convert defaultdict to regular dict
        return {video_id: dict(images) for video_id, images in video_groups.items()}

    def get_processed_video_ids(self) -> List[str]:
        """Get list of video IDs that have been processed."""
        processed_videos = []
        
        for exp_data in self.results.get("experiments", {}).values():
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_data.get("sam2_success", False):
                    processed_videos.append(video_id)
        
        return processed_videos

    def get_missing_videos(self, video_ids: Optional[List[str]] = None,
                          experiment_ids: Optional[List[str]] = None) -> List[str]:
        """Get list of video IDs that need processing."""
        # Get available videos from annotations
        available_videos = list(self.group_annotations_by_video().keys())
        
        # Filter by requested videos/experiments
        if video_ids:
            available_videos = [v for v in available_videos if v in video_ids]
        
        if experiment_ids:
            available_videos = [v for v in available_videos 
                              if any(v.startswith(exp_id) for exp_id in experiment_ids)]
        
        # Get already processed videos
        processed_videos = self.get_processed_video_ids()
        
        # Return missing videos
        missing_videos = [v for v in available_videos if v not in processed_videos]
        
        return missing_videos

    def process_missing_annotations(self, 
                                  video_ids: Optional[List[str]] = None,
                                  experiment_ids: Optional[List[str]] = None,
                                  max_videos: Optional[int] = None,
                                  auto_save_interval: Optional[int] = 5,
                                  overwrite: bool = False) -> Dict:
        """Process missing annotations with SAM2 segmentation."""
        if not self.predictor:
            raise ValueError("SAM2 model not loaded. Call set_sam2_model_paths() first.")
        
        # Get videos to process
        if overwrite:
            video_groups = self.group_annotations_by_video()
            target_videos = list(video_groups.keys())
            
            if video_ids:
                target_videos = [v for v in target_videos if v in video_ids]
            if experiment_ids:
                target_videos = [v for v in target_videos 
                               if any(v.startswith(exp_id) for exp_id in experiment_ids)]
        else:
            target_videos = self.get_missing_videos(video_ids, experiment_ids)
        
        if max_videos:
            target_videos = target_videos[:max_videos]
        
        if self.verbose:
            print(f"🎯 Processing {len(target_videos)} videos for SAM2 segmentation")
        
        processing_stats = {"processed": 0, "errors": 0, "total": len(target_videos)}
        
        for i, video_id in enumerate(target_videos):
            try:
                if self.verbose:
                    print(f"\n📹 Processing video {i+1}/{len(target_videos)}: {video_id}")
                
                result = self.process_video(video_id)
                processing_stats["processed"] += 1
                
                # Auto-save periodically
                if auto_save_interval and (i + 1) % auto_save_interval == 0:
                    self.save()
                    if self.verbose:
                        print(f"💾 Auto-saved after {i+1} videos")
                
            except Exception as e:
                processing_stats["errors"] += 1
                if self.verbose:
                    print(f"❌ Error processing {video_id}: {e}")
                continue
        
        # Final save
        self.save()
        
        if self.verbose:
            print(f"\n✅ Processing complete: {processing_stats}")
        
        return processing_stats

    def process_video(self, video_id: str) -> Dict:
        """Process a single video with SAM2 segmentation."""
        if self.verbose:
            print(f"🎬 Processing video: {video_id}")
        
        # Get video annotations
        video_groups = self.group_annotations_by_video()
        if video_id not in video_groups:
            raise ValueError(f"Video {video_id} not found in annotations")
        
        video_annotations = video_groups[video_id]
        processing_start_time = datetime.now().isoformat()
        
        # Process video using helper function
        processing_stats = {"processed": 0, "errors": 0}
        
        try:
            sam2_results, video_metadata, seed_frame_info = process_single_video_from_annotations(
                video_id, video_annotations, self, self.predictor, 
                processing_stats, self.segmentation_format, self.verbose
            )
            
            # Extract experiment ID and well ID
            exp_id = extract_experiment_id(video_id)
            well_id = video_id.replace(f"{exp_id}_", "")
            
            # Initialize experiment structure if needed
            if exp_id not in self.results["experiments"]:
                self.results["experiments"][exp_id] = {
                    "experiment_id": exp_id,
                    "first_processed_time": processing_start_time,
                    "last_processed_time": processing_start_time,
                    "videos": {}
                }
            else:
                # Ensure experiment has the new structure (backward compatibility)
                exp_data = self.results["experiments"][exp_id]
                if "videos" not in exp_data:
                    exp_data["videos"] = {}
                if "experiment_id" not in exp_data:
                    exp_data["experiment_id"] = exp_id
                if "first_processed_time" not in exp_data:
                    exp_data["first_processed_time"] = processing_start_time
                    
                # Update last processed time
                exp_data["last_processed_time"] = processing_start_time
            
            # Create video-level structure
            video_structure = {
                "video_id": video_id,
                "well_id": well_id,
                "seed_frame_info": seed_frame_info,
                "embryo_ids": seed_frame_info["embryo_ids"],
                "num_embryos": seed_frame_info["num_embryos"],
                "frames_processed": len(sam2_results),
                "sam2_success": True,
                "processing_timestamp": processing_start_time,
                "requires_bidirectional_propagation": seed_frame_info.get("requires_bidirectional_propagation", False),
                "images": sam2_results
            }
            
            # Store video structure
            self.results["experiments"][exp_id]["videos"][video_id] = video_structure
            
            # Update snip_ids list
            for image_data in sam2_results.values():
                for embryo_data in image_data.get("embryos", {}).values():
                    snip_id = embryo_data.get("snip_id")
                    if snip_id and snip_id not in self.results["snip_ids"]:
                        self.results["snip_ids"].append(snip_id)
            
            if self.verbose:
                print(f"✅ Video {video_id} processed successfully")
            
            return sam2_results
            
        except Exception as e:
            # Handle failed processing
            exp_id = extract_experiment_id(video_id)
            well_id = video_id.replace(f"{exp_id}_", "")
            
            if exp_id not in self.results["experiments"]:
                self.results["experiments"][exp_id] = {
                    "experiment_id": exp_id,
                    "first_processed_time": processing_start_time,
                    "last_processed_time": processing_start_time,
                    "videos": {}
                }
            else:
                # Ensure experiment has the new structure (backward compatibility)
                exp_data = self.results["experiments"][exp_id]
                if "videos" not in exp_data:
                    exp_data["videos"] = {}
                if "experiment_id" not in exp_data:
                    exp_data["experiment_id"] = exp_id
                if "first_processed_time" not in exp_data:
                    exp_data["first_processed_time"] = processing_start_time
                    
                # Update last processed time
                exp_data["last_processed_time"] = processing_start_time
            
            # Create failed video structure
            self.results["experiments"][exp_id]["videos"][video_id] = {
                "video_id": video_id,
                "well_id": well_id,
                "sam2_success": False,
                "processing_timestamp": processing_start_time,
                "error_message": str(e),
                "images": {}
            }
            
            if self.verbose:
                print(f"❌ Video {video_id} processing failed: {e}")
            
            raise

    def get_summary(self) -> Dict:
        """Get processing summary statistics."""
        total_snips = len(self.results.get("snip_ids", []))
        total_experiments = len(self.results.get("experiments", {}))
        
        # Count videos and images from new structure
        total_videos = 0
        total_images = 0
        successful_videos = 0
        failed_videos = 0
        
        for exp_data in self.results.get("experiments", {}).values():
            videos = exp_data.get("videos", {})
            total_videos += len(videos)
            
            for video_data in videos.values():
                if video_data.get("sam2_success", False):
                    successful_videos += 1
                else:
                    failed_videos += 1
                total_images += len(video_data.get("images", {}))
        
        return {
            "total_experiments": total_experiments,
            "total_videos": total_videos,
            "total_images": total_images,
            "total_snips": total_snips,
            "successful_videos": successful_videos,
            "failed_videos": failed_videos,
            "segmentation_format": self.segmentation_format,
            "target_prompt": self.target_prompt
        }

    def print_summary(self):
        """Print processing summary."""
        summary = self.get_summary()
        print(f"\n📊 GroundedSamAnnotations Summary:")
        print(f"   Experiments: {summary['total_experiments']}")
        print(f"   Videos: {summary['total_videos']} (✅ {summary['successful_videos']} success, ❌ {summary['failed_videos']} failed)")
        print(f"   Images: {summary['total_images']}")
        print(f"   Snips: {summary['total_snips']}")
        print(f"   Format: {summary['segmentation_format']}")
        print(f"   Prompt: '{summary['target_prompt']}')")

    def print_summary(self):
        """Print processing summary."""
        summary = self.get_summary()
        print(f"\n📊 GroundedSamAnnotations Summary:")
        print(f"   Experiments: {summary['total_experiments']}")
        print(f"   Images: {summary['total_images']}")
        print(f"   Snips: {summary['total_snips']}")
        print(f"   Format: {summary['segmentation_format']}")
        print(f"   Prompt: '{summary['target_prompt']}'")

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (f"GroundedSamAnnotations(experiments={summary['total_experiments']}, "
                f"videos={summary['total_videos']}, images={summary['total_images']}, "
                f"snips={summary['total_snips']})")


# REFACTORED: Fix create_snip_id to use standard format with '_s' prefix
def create_snip_id(embryo_id: str, image_id: str) -> str:
    """Create snip_id using standard format with '_s' prefix."""
    frame_number = extract_frame_number(image_id)
    return f"{embryo_id}_s{frame_number:04d}"


def extract_frame_suffix(image_id: str) -> str:
    """Extract frame suffix from image_id using parsing_utils."""
    return f"{extract_frame_number(image_id):04d}"


def convert_sam2_mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Convert SAM2 binary mask to RLE format for compact storage."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError("pycocotools required for RLE encoding. Install with: pip install pycocotools")

    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    binary_mask_fortran = np.asfortranarray(binary_mask)
    rle_result = mask_utils.encode(binary_mask_fortran)

    # Standard COCO RLE: dict with 'counts' as bytes, or list of dicts for multi-object
    if isinstance(rle_result, dict):
        # Single mask: ensure 'counts' is a string
        counts = rle_result['counts']
        if isinstance(counts, bytes):
            counts = counts.decode('utf-8')
        rle_result['counts'] = counts
        return rle_result
    elif isinstance(rle_result, list):
        # Multi-object: join all counts as strings (rare for binary mask)
        # For single-object binary mask, this should not occur, but handle for robustness
        result_dict = {'counts': [], 'size': binary_mask.shape}
        for rle in rle_result:
            counts = rle['counts']
            if isinstance(counts, bytes):
                counts = counts.decode('utf-8')
            result_dict['counts'].append(counts)
        # Optionally, join all counts into a single string (not standard, but for compactness)
        result_dict['counts'] = ''.join(result_dict['counts'])
        return result_dict
    else:
        # Fallback for unexpected types
        return {'counts': str(rle_result), 'size': binary_mask.shape}


def convert_sam2_mask_to_polygon(binary_mask: np.ndarray) -> List[List[float]]:
    """Convert SAM2 binary mask to polygon format."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Valid polygon needs at least 3 points
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    
    return polygons


def extract_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Extract bounding box from binary mask in normalized xyxy format."""
    # Ensure binary mask is 2D
    if binary_mask.ndim > 2:
        # If mask has multiple dimensions, take the first channel or squeeze
        binary_mask = binary_mask.squeeze()
        if binary_mask.ndim > 2:
            binary_mask = binary_mask[0]  # Take first channel
    
    y_indices, x_indices = np.where(binary_mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    
    h, w = binary_mask.shape
    bbox_xyxy = [
        x_min / w,  # x_min normalized
        y_min / h,  # y_min normalized  
        x_max / w,  # x_max normalized
        y_max / h   # y_max normalized
    ]
    
    return bbox_xyxy


def run_sam2_propagation(predictor, video_dir: Path, seed_frame_idx: int, 
                        seed_detections: List[Dict], embryo_ids: List[str],
                        image_ids: List[str], segmentation_format: str = 'rle',
                        verbose: bool = True) -> Dict:
    """
    Run SAM2 propagation from seed frame using the actual processed images directory.
    FIXED: Updated to use corrected bbox format.
    """
    if verbose:
        print(f"🎬 Running SAM2 propagation from seed frame {seed_frame_idx}")
    
    # Create temporary directory with properly named symlinks for SAM2
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        # Create symlinks with sequential naming (SAM2 expects this)
        for i, image_id in enumerate(image_ids):
            # REFACTORED: Use consistent frame number extraction
            frame_num = extract_frame_number(image_id)
            src_path = video_dir / f"{frame_num:04d}.jpg"  # Disk format (no 't' prefix)
            dst_path = temp_dir / f"{i:05d}.jpg"
            
            if src_path.exists():
                dst_path.symlink_to(src_path)
            else:
                if verbose:
                    print(f"⚠️ Image not found: {src_path}")
        
        # Initialize video with seed frame
        inference_state = predictor.init_state(video_path=str(temp_dir))
        
        # Add objects from seed detections
        for i, (detection, embryo_id) in enumerate(zip(seed_detections, embryo_ids)):
            # Handle different bbox field names from GroundedDINO
            bbox = detection.get('box_xyxy') or detection.get('bbox_xyxy') or detection.get('bbox')
            if bbox is None:
                if verbose:
                    print(f"⚠️ No bbox found in detection: {list(detection.keys())}")
                continue
            
            # Add positive click at bbox center
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            points = np.array([[center_x, center_y]], dtype=np.float32)
            labels = np.array([1], np.int32)  # Positive click
            
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=seed_frame_idx,
                obj_id=i,
                points=points,
                labels=labels,
            )
        
        # Propagate through video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    if verbose:
        print(f"✅ SAM2 propagation complete for {len(video_segments)} frames")
    
    return video_segments


def run_bidirectional_propagation(predictor, video_dir: Path, seed_frame_idx: int,
                                 seed_detections: List[Dict], embryo_ids: List[str],
                                 image_ids: List[str], segmentation_format: str = 'rle',
                                 verbose: bool = True) -> Dict:
    """
    Run bidirectional SAM2 propagation when seed frame is not the first frame.
    FIXED: Properly maintains frame ordering in final results.
    """
    if verbose:
        print(f"🔄 Running bidirectional SAM2 propagation from seed frame {seed_frame_idx}")
    
    # Forward propagation (seed to end) - use original video directory
    if verbose:
        print(f"➡️ Forward propagation: frames {seed_frame_idx} to {len(image_ids)-1}")
    forward_results = run_sam2_propagation(predictor, video_dir, seed_frame_idx, 
                                          seed_detections, embryo_ids, image_ids, 
                                          segmentation_format, verbose=verbose)
    
    # Backward propagation (seed to beginning) - only if there are frames before seed
    backward_results = {}
    if seed_frame_idx > 0:
        if verbose:
            print(f"⬅️ Backward propagation: frames 0 to {seed_frame_idx}")
        
        # Create reversed image list for backward propagation
        reversed_image_ids = image_ids[:seed_frame_idx+1][::-1]
        reversed_seed_idx = 0  # Seed is now at index 0 in reversed list
        
        backward_results = run_sam2_propagation(predictor, video_dir, reversed_seed_idx,
                                               seed_detections, embryo_ids, reversed_image_ids,
                                               segmentation_format, verbose=verbose)
    
    # FIXED: Properly combine results maintaining original frame order
    if verbose:
        print(f"🔄 Combining bidirectional results...")
    
    # Create combined results in original frame order using frame_idx as keys
    combined_results = {}
    
    # Process all frames in strict original temporal order
    for frame_idx, image_id in enumerate(image_ids):
        if frame_idx == seed_frame_idx:
            # Use forward results for seed frame
            if frame_idx in forward_results:
                combined_results[frame_idx] = forward_results[frame_idx]
        elif frame_idx > seed_frame_idx:
            # Use forward results for frames after seed
            if frame_idx in forward_results:
                combined_results[frame_idx] = forward_results[frame_idx]
        else:
            # Use backward results for frames before seed (need to map indices)
            backward_frame_idx = seed_frame_idx - frame_idx
            if backward_frame_idx in backward_results:
                combined_results[frame_idx] = backward_results[backward_frame_idx]
    
    if verbose:
        print(f"✅ Bidirectional propagation complete: {len(combined_results)} frames")
    
    return combined_results


def process_single_video_from_annotations(video_id: str, video_annotations: Dict, grounded_sam_instance,
                                         predictor, processing_stats: Dict, segmentation_format: str = 'rle',
                                         verbose: bool = True) -> Tuple[Dict, Dict, Dict]:
    """
    Process a single video with SAM2 segmentation using class experiment metadata.
    
    Args:
        video_id: Video identifier
        video_annotations: Annotations for this video (image_id -> detections)
        grounded_sam_instance: GroundedSamAnnotations instance (contains experiment metadata)
        predictor: SAM2 video predictor
        processing_stats: Dictionary to update with processing statistics
        segmentation_format: 'rle' (recommended) or 'polygon' for segmentation storage
        verbose: Enable verbose output
        
    Returns:
        Tuple of (sam2_results, video_metadata, seed_frame_info)
    """
    if verbose:
        print(f"🎬 Processing single video: {video_id}")
    
    try:
        # REFACTORED: Use ExperimentMetadata method instead of custom lookup
        video_info = grounded_sam_instance.exp_metadata.get_video_metadata(video_id)
        if not video_info:
            raise ValueError(f"Video {video_id} not found in experiment metadata")
        
        # Get video directory and image order from metadata
        video_dir = grounded_sam_instance.exp_metadata.get_video_directory_path(video_id)
        image_ids_ordered = video_info['image_ids']  # Already in temporal order
        
        if verbose:
            print(f"📁 Video directory: {video_dir}")
            print(f"🖼️ Total frames: {len(image_ids_ordered)}")
        
        # Find seed frame and detections
        seed_image_id, seed_detections_dict = find_seed_frame_from_video_annotations(
            video_annotations, video_id
        )
        
        seed_frame_idx = image_ids_ordered.index(seed_image_id)
        seed_detections = seed_detections_dict['detections']
        
        if verbose:
            print(f"🌱 Seed frame: {seed_image_id} (index {seed_frame_idx})")
            print(f"🎯 Seed detections: {len(seed_detections)}")
        
        # Assign embryo IDs
        embryo_ids = assign_embryo_ids(video_id, len(seed_detections))
        
        # Run SAM2 propagation
        if seed_frame_idx == 0:
            # Simple forward propagation
            video_segments = run_sam2_propagation(
                predictor, video_dir, seed_frame_idx, seed_detections, 
                embryo_ids, image_ids_ordered, segmentation_format, verbose
            )
        else:
            # Bidirectional propagation
            video_segments = run_bidirectional_propagation(
                predictor, video_dir, seed_frame_idx, seed_detections,
                embryo_ids, image_ids_ordered, segmentation_format, verbose
            )
        
        # Convert to structured format
        sam2_results = {}
        
        for frame_idx, image_id in enumerate(image_ids_ordered):
            if frame_idx not in video_segments:
                continue
                
            image_data = {
                "image_id": image_id,
                "frame_index": frame_idx,
                "is_seed_frame": (frame_idx == seed_frame_idx),
                "embryos": {}
            }
            
            frame_masks = video_segments[frame_idx]
            
            for obj_id, binary_mask in frame_masks.items():
                if obj_id >= len(embryo_ids):
                    continue
                    
                embryo_id = embryo_ids[obj_id]
                
                # REFACTORED: Use standardized snip_id creation
                snip_id = create_snip_id(embryo_id, image_id)
                
                # Convert mask to requested format
                if segmentation_format == 'rle':
                    segmentation = convert_sam2_mask_to_rle(binary_mask.astype(np.uint8))
                else:
                    segmentation = convert_sam2_mask_to_polygon(binary_mask.astype(np.uint8))
                
                # Extract bbox and area
                bbox = extract_bbox_from_mask(binary_mask)
                area = float(np.sum(binary_mask > 0))
                
                embryo_data = {
                    "embryo_id": embryo_id,
                    "snip_id": snip_id,
                    "segmentation": segmentation,
                    "segmentation_format": segmentation_format,
                    "bbox": bbox,
                    "area": area,
                    "mask_confidence": 0.85  # SAM2 default confidence
                }
                
                image_data["embryos"][embryo_id] = embryo_data
            
            sam2_results[image_id] = image_data
        
        # Create seed frame info structure
        seed_frame_info = {
            "video_id": video_id,
            "seed_frame": seed_image_id,
            "num_embryos": len(seed_detections),
            "detections": seed_detections,
            "is_first_frame": (seed_frame_idx == 0),
            "all_frames": image_ids_ordered,
            "seed_frame_index": seed_frame_idx,
            "embryo_ids": embryo_ids,
            "requires_bidirectional_propagation": (seed_frame_idx > 0)
        }
        
        processing_stats["processed"] += 1
        
        if verbose:
            print(f"✅ Video {video_id} processed: {len(sam2_results)} frames")
        
        return sam2_results, video_info, seed_frame_info
        
    except Exception as e:
        processing_stats["errors"] += 1
        if verbose:
            print(f"❌ Error processing video {video_id}: {e}")
        raise


def find_seed_frame_from_video_annotations(video_annotations: Dict[str, List[Dict]], video_id: str) -> Tuple[str, Dict]:
    """Find the best seed frame from video annotations, preferring first frame to avoid bidirectional propagation."""
    
    # First, check if the first frame (t0000) exists and has detections
    first_frame_id = f"{video_id}_t0000"
    if first_frame_id in video_annotations and video_annotations[first_frame_id]:
        if len(video_annotations[first_frame_id]) > 0:
            print(f"➡️ Forward propagation from first frame - avoiding bidirectional propagation")
            return first_frame_id, {"detections": video_annotations[first_frame_id]}
    
    # Sort frames by temporal order (extract frame numbers)
    sorted_frames = []
    for image_id, detections in video_annotations.items():
        if detections:  # Only consider frames with detections
            frame_num = extract_frame_number(image_id)
            avg_confidence = sum(det.get('confidence', 0) for det in detections) / len(detections)
            detection_count = len(detections)
            score = avg_confidence * detection_count
            sorted_frames.append((frame_num, image_id, detections, score))
    
    if not sorted_frames:
        raise ValueError(f"No valid seed frame found for video {video_id}")
    
    # Sort by frame number (temporal order)
    sorted_frames.sort(key=lambda x: x[0])
    
    # Look for good quality detections in the first 20% of frames to avoid bidirectional propagation
    total_frames = len(sorted_frames)
    early_frame_cutoff = max(1, int(total_frames * 0.2))  # First 20% of frames
    
    early_frames = sorted_frames[:early_frame_cutoff]
    
    # If we have good detections in early frames, use the best one from there
    if early_frames:
        best_early_frame = max(early_frames, key=lambda x: x[3])  # Best score in early frames
        best_score_overall = max(sorted_frames, key=lambda x: x[3])[3]  # Best score in entire video
        
        # Use early frame if it's at least 80% as good as the best frame overall
        if best_early_frame[3] >= 0.8 * best_score_overall:
            _, best_image_id, best_detections, _ = best_early_frame
            return best_image_id, {"detections": best_detections}
    
    # Fallback: use the frame with highest confidence overall
    _, best_image_id, best_detections, _ = max(sorted_frames, key=lambda x: x[3])
    return best_image_id, {"detections": best_detections}


def assign_embryo_ids(video_id: str, num_embryos: int) -> List[str]:
    """Assign embryo IDs for detected embryos in a video."""
    embryo_ids = []
    for i in range(num_embryos):
        embryo_id = f"{video_id}_e{i+1:02d}"
        embryo_ids.append(embryo_id)
    
    return embryo_ids
