#!/usr/bin/env python3
"""
Pipeline Script 5: SAM2 Quality Control Analysis

Analyze SAM2 segmentation results for quality issues and add flags
to the GSAM JSON structure.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from time import time

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from pycocotools import mask as mask_utils
    from scipy import ndimage
    from skimage.measure import regionprops, label
    _HAS_IMAGE_LIBS = True
except ImportError:
    _HAS_IMAGE_LIBS = False

# Pipeline imports
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.base_file_handler import BaseFileHandler

def ensure_json_serializable(obj):
    """
    Recursively convert numpy types and other non-serializable objects to JSON-safe types.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


class GSAMQualityControl(BaseFileHandler):
    """
    Quality Control for SAM2 annotations. Inherits from BaseFileHandler
    for consistent file operations with the rest of the pipeline.
    """
    
    def __init__(self, gsam_path: str, verbose: bool = True, progress: bool = True):
        """
        Initialize QC with path to GSAM annotations.
        
        Args:
            gsam_path: Path to grounded_sam_annotations.json
            verbose: Whether to print progress
            progress: Whether to show progress bars
        """
        # Initialize BaseFileHandler
        super().__init__(gsam_path, verbose=verbose)
        
        self.progress = progress
        
        # Load GSAM data
        self.gsam_data = self.load_json()
        
        # Initialize top-level flags structure
        if "flags" not in self.gsam_data:
            self.gsam_data["flags"] = {
                "qc_meta": {},
                "qc_history": [],
                "flag_overview": {},
                "by_experiment": {},
                "by_video": {},
                "by_image": {},
                "by_snip": {}
            }
        
        # Initialize tracking sets for processed entities
        self._initialize_entity_tracking()
        
        if self.verbose:
            print(f"🔍 Loaded GSAM annotations from {self.filepath}")
            print(f"📊 Found {len(self.processed_snip_ids)} already processed snips")
        
        if self.verbose and self.progress:
            if _HAS_TQDM:
                print("⏳ Progress bars enabled (tqdm).")
            else:
                print("⏳ tqdm not installed; using basic percentage progress.")
    
    def _initialize_entity_tracking(self):
        """Entity tracking for incremental processing."""
        
        if self.verbose:
            print("🔄 Initializing entity tracking...")
            start_time = time()
        
        # Get previously processed IDs from flags.qc_meta (or empty sets if first run)
        qc_meta = self.gsam_data["flags"].setdefault("qc_meta", {})
        self.processed_experiment_ids = set(qc_meta.get("processed_experiment_ids", []))
        self.processed_video_ids = set(qc_meta.get("processed_video_ids", []))
        self.processed_image_ids = set(qc_meta.get("processed_image_ids", []))
        self.processed_snip_ids = set(qc_meta.get("processed_snip_ids", []))
        
        # Find all current entities by iterating through the data structure directly
        all_experiment_ids = set()
        all_video_ids = set()
        all_image_ids = set()
        all_snip_ids = set()

        experiments_items = self.gsam_data.get("experiments", {}).items()
        for exp_id, exp_data in experiments_items:
            all_experiment_ids.add(exp_id)
            for video_id, video_data in exp_data.get("videos", {}).items():
                all_video_ids.add(video_id)
                for image_id, image_data in video_data.get("images", {}).items():
                    all_image_ids.add(image_id)
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id:
                            all_snip_ids.add(snip_id)
        
        # New entities = all entities - processed entities
        self.new_experiment_ids = all_experiment_ids - self.processed_experiment_ids
        self.new_video_ids = all_video_ids - self.processed_video_ids
        self.new_image_ids = all_image_ids - self.processed_image_ids
        self.new_snip_ids = all_snip_ids - self.processed_snip_ids
        
        if self.verbose:
            elapsed = time() - start_time
            print(f"✅ Entity tracking initialized in {elapsed:.2f}s")
            print(f"   New entities: {len(self.new_experiment_ids)} experiments, "
                  f"{len(self.new_video_ids)} videos, {len(self.new_image_ids)} images, "
                  f"{len(self.new_snip_ids)} snips")
            
            # Add diagnostic call if no snips found
            if len(all_snip_ids) == 0:
                print(f"❌ No snips found in the data structure!")
                print(f"   Running diagnosis...")
                self.diagnose_data_structure()

    def diagnose_data_structure(self, max_samples=3):
        """Diagnose the GSAM data structure to understand why no snips are found."""
        print("\n🔍 GSAM Data Structure Diagnosis")
        print("=" * 50)
        
        experiments = self.gsam_data.get("experiments", {})
        print(f"Total experiments: {len(experiments)}")
        
        if not experiments:
            print("❌ No experiments found in data!")
            print("Available top-level keys:", list(self.gsam_data.keys()))
            return
        
        sample_count = 0
        for exp_id, exp_data in experiments.items():
            if sample_count >= max_samples:
                break
                
            print(f"\n📁 Experiment: {exp_id}")
            videos = exp_data.get("videos", {})
            print(f"   Videos: {len(videos)}")
            
            if not videos:
                print("   ❌ No videos found in this experiment!")
                print(f"   Available keys in experiment: {list(exp_data.keys())}")
                continue
            
            video_sample_count = 0
            for video_id, video_data in videos.items():
                if video_sample_count >= 2:  # Limit video samples
                    break
                    
                print(f"   📹 Video: {video_id}")
                images = video_data.get("images", {})
                print(f"      Images: {len(images)}")
                
                if not images:
                    print("      ❌ No images found in this video!")
                    print(f"      Available keys in video: {list(video_data.keys())}")
                    continue
                
                image_sample_count = 0
                for image_id, image_data in images.items():
                    if image_sample_count >= 2:  # Limit image samples
                        break
                        
                    print(f"      🖼️  Image: {image_id}")
                    embryos = image_data.get("embryos", {})
                    print(f"         Embryos: {len(embryos)}")
                    
                    if not embryos:
                        print("         ❌ No embryos found in this image!")
                        print(f"         Available keys in image: {list(image_data.keys())}")
                        continue
                    
                    embryo_sample_count = 0
                    for embryo_id, embryo_data in embryos.items():
                        if embryo_sample_count >= 2:  # Limit embryo samples
                            break
                            
                        snip_id = embryo_data.get("snip_id")
                        print(f"         🧬 Embryo: {embryo_id}")
                        print(f"            snip_id: {snip_id}")
                        print(f"            Available keys: {list(embryo_data.keys())}")
                        
                        if not snip_id:
                            print("            ❌ No snip_id found!")
                        
                        embryo_sample_count += 1
                    
                    image_sample_count += 1
                
                video_sample_count += 1
            
            sample_count += 1
        
        print("\n" + "=" * 50)

    def _mark_entities_checked(self, entities):
        """Mark entities as processed by adding them to the stored sets."""
        ts = datetime.now().isoformat()
        
        # Update our tracking sets
        self.processed_experiment_ids.update(entities.get("experiment_ids", []))
        self.processed_video_ids.update(entities.get("video_ids", []))
        self.processed_image_ids.update(entities.get("image_ids", []))
        self.processed_snip_ids.update(entities.get("snip_ids", []))
        
        # Save to flags.qc_meta for persistence
        qc_meta = self.gsam_data["flags"]["qc_meta"]
        qc_meta["processed_experiment_ids"] = sorted(self.processed_experiment_ids)
        qc_meta["processed_video_ids"] = sorted(self.processed_video_ids)
        qc_meta["processed_image_ids"] = sorted(self.processed_image_ids)
        qc_meta["processed_snip_ids"] = sorted(self.processed_snip_ids)
        qc_meta["last_updated"] = ts

    def _add_flag(self, flag_type: str, flag_data: dict, entity_type: str, entity_id: str):
        """
        Add a flag to the top-level flags structure.
        
        Args:
            flag_type: Type of flag (e.g., 'HIGH_SEGMENTATION_VAR_SNIP')
            flag_data: Flag data dictionary
            entity_type: 'experiment', 'video', 'image', or 'snip'
            entity_id: ID of the entity being flagged
        """
        flags_section = self.gsam_data["flags"]
        
        # Add to appropriate entity section
        entity_key = f"by_{entity_type}"
        if entity_key not in flags_section:
            flags_section[entity_key] = {}
        
        if entity_id not in flags_section[entity_key]:
            flags_section[entity_key][entity_id] = {}
        
        if flag_type not in flags_section[entity_key][entity_id]:
            flags_section[entity_key][entity_id][flag_type] = []
        
        flags_section[entity_key][entity_id][flag_type].append(flag_data)

    def _progress_iter(self, iterable, desc: str, total: Optional[int] = None):
        """Wrap an iterable with a progress indicator compatible with tmux."""
        if not self.progress:
            return iterable
        if _HAS_TQDM:
            return tqdm(iterable, desc=desc, total=total, ncols=80)
        else:
            # Simple fallback progress
            items = list(iterable)
            total = len(items)
            for i, item in enumerate(items):
                if i % max(1, total // 10) == 0:
                    pct = (i / total) * 100
                    print(f"\r{desc}: {pct:.1f}%", end="", flush=True)
                yield item
            print(f"\r{desc}: 100.0%")
            return
    
    def get_new_entities_to_process(self) -> Dict[str, List[str]]:
        """Get entities that need QC processing (only new ones)."""
        return {
            "experiment_ids": list(self.new_experiment_ids),
            "video_ids": list(self.new_video_ids),
            "image_ids": list(self.new_image_ids),
            "snip_ids": list(self.new_snip_ids)
        }
    
    def get_all_entities_to_process(self) -> Dict[str, List[str]]:
        """Get all entities for full reprocessing."""
        all_experiment_ids = set()
        all_video_ids = set()
        all_image_ids = set()
        all_snip_ids = set()

        experiments_items = self.gsam_data.get("experiments", {}).items()
        for exp_id, exp_data in experiments_items:
            all_experiment_ids.add(exp_id)
            for video_id, video_data in exp_data.get("videos", {}).items():
                all_video_ids.add(video_id)
                for image_id, image_data in video_data.get("images", {}).items():
                    all_image_ids.add(image_id)
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id:
                            all_snip_ids.add(snip_id)
                            
        return {
            "experiment_ids": list(all_experiment_ids),
            "video_ids": list(all_video_ids),
            "image_ids": list(all_image_ids),
            "snip_ids": list(all_snip_ids)
        }
    
    def _should_process_experiment(self, exp_id: str, target_entities: Dict[str, List[str]]) -> bool:
        """Check if experiment should be processed."""
        exp_targets = target_entities.get("experiment_ids", [])
        return len(exp_targets) == 0 or exp_id in exp_targets
    
    def _should_process_video(self, video_id: str, target_entities: Dict[str, List[str]]) -> bool:
        """Check if video should be processed."""
        video_targets = target_entities.get("video_ids", [])
        return len(video_targets) == 0 or video_id in video_targets
    
    def _should_process_image(self, image_id: str, target_entities: Dict[str, List[str]]) -> bool:
        """Check if image should be processed."""
        image_targets = target_entities.get("image_ids", [])
        return len(image_targets) == 0 or image_id in image_targets
    
    def _should_process_snip(self, snip_id: str, target_entities: Dict[str, List[str]]) -> bool:
        """Check if snip should be processed."""
        snip_targets = target_entities.get("snip_ids", [])
        return len(snip_targets) == 0 or snip_id in snip_targets
    
    def run_all_checks(self,
                      author: str = "auto_qc",
                      process_all: bool = False,
                      target_entities: Optional[Dict[str, List[str]]] = None,
                      force_reprocess: bool = False,
                      save_in_place: bool = True):
        """
        Run all QC checks and save flags to GSAM file.

        Args:
            author: Author identifier for the QC run
            process_all: If True, process all entities. If False, only process new entities
            target_entities: Specific entities to process (overrides process_all)
            force_reprocess: If True, process target entities even if already processed
            save_in_place: If True (default) QC modifications are written back to the
                           GSAM JSON on disk.  If False, the GSAM data is mutated in
                           memory only (caller can decide when / whether to save).
        """
        start_overall = time()
        
        # Determine which entities to process
        if target_entities:
            entities_to_process = target_entities
        elif process_all:
            entities_to_process = self.get_all_entities_to_process()
        else:
            entities_to_process = self.get_new_entities_to_process()
        
        if self.verbose:
            print(f"\n🔍 Starting QC analysis with author='{author}'")
            print(f"📊 Entities to process: {len(entities_to_process['experiment_ids'])} experiments, "
                  f"{len(entities_to_process['video_ids'])} videos, "
                  f"{len(entities_to_process['image_ids'])} images, "
                  f"{len(entities_to_process['snip_ids'])} snips")
            
        # Record QC run in history
        qc_run = {
            "author": author,
            "timestamp": datetime.now().isoformat(),
            "entities_processed": entities_to_process,
            "entity_counts": {k: len(v) for k, v in entities_to_process.items()}
        }
        self.gsam_data["flags"]["qc_history"].append(qc_run)
        
        # Mark entities checked before processing (prevents re-processing on failure)
        self._mark_entities_checked(entities_to_process)

        if self.verbose:
            print("🚀 Running quality checks...")

        # Run all quality checks
        self.check_segmentation_variability(author, entities_to_process)
        self.check_mask_on_edge(author, entities_to_process)
        self.check_overlapping_masks(author, entities_to_process)
        self.check_large_masks(author, entities_to_process)
        self.check_small_masks(author, entities_to_process)
        self.check_detection_failure(author, entities_to_process)
        
        if _HAS_IMAGE_LIBS:
            self.check_discontinuous_masks(author, entities_to_process)
        else:
            if self.verbose:
                print("⚠️ Skipping discontinuous mask check (scipy/skimage not available)")
        
        elapsed = time() - start_overall
        if self.verbose:
            print(f"✅ All QC checks completed in {elapsed:.2f}s")
        
        # Always build in‑memory overview
        self.generate_overview(entities_to_process)

        if save_in_place:
            self._save_qc_summary(author)
        else:
            if self.verbose:
                print("💾 QC results updated in memory (not saved to disk)")
    
    def check_segmentation_variability(self, author: str, entities: Dict[str, List[str]], n_frames_check: int = 2):
        """
        Flag embryos with high area variance across frames (>15% CV).
        Flags both at embryo level (HIGH_SEGMENTATION_VAR_EMBRYO) and 
        at individual snip level (HIGH_SEGMENTATION_VAR_SNIP).
        """
        if self.verbose:
            print("🔍 Checking segmentation variability...")
        
        t0 = time()
        cv_threshold = 0.15  # 15% coefficient of variation
        
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="SegVar", total=len(experiments_items)):
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                continue

            videos = exp_data.get("videos", {})

            for video_id, video_data in videos.items():
                
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue

                images = video_data.get("images", {})
                image_ids = sorted(images.keys())
                
                embryo_frames = defaultdict(dict)  # embryo_id -> {image_id: area}
                embryo_areas = defaultdict(list)   # embryo_id -> [area, ...]

                # First pass: collect areas from all frames
                for idx, image_id in enumerate(image_ids):
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                        
                    image_data = images[image_id]
                    embryos = image_data.get("embryos", {})
                    
                    for embryo_id, embryo_data in embryos.items():
                        snip_id = embryo_data.get("snip_id")
                        
                        if not snip_id:
                            continue
                            
                        # Check if snip should be processed
                        if not self._should_process_snip(snip_id, entities):
                            continue
                            
                        # Calculate area from segmentation
                        area = embryo_data.get("area")
                        if area is None:
                            segmentation = embryo_data.get("segmentation")
                            segmentation_format = embryo_data.get("segmentation_format", "unknown")
                            if segmentation and segmentation_format == "rle":
                                try:
                                    mask = mask_utils.decode(segmentation)
                                    area = float(np.sum(mask))
                                except Exception as e:
                                    continue
                        
                        if area is not None:
                            embryo_frames[embryo_id][image_id] = area
                            embryo_areas[embryo_id].append(area)

                # Embryo-level variability flag
                for embryo_id, areas in embryo_areas.items():
                    if len(areas) >= 3:
                        mean_area = np.mean(areas)
                        std_area = np.std(areas)
                        cv = std_area / mean_area if mean_area > 0 else 0
                        if cv > cv_threshold:
                            flag_data = {
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "embryo_id": embryo_id,
                                "issue": "HIGH_SEGMENTATION_VAR_EMBRYO",
                                "coefficient_of_variation": float(round(cv, 3)),
                                "frame_count": int(len(areas)),
                                "mean_area": float(round(mean_area, 1)),
                                "std_area": float(round(std_area, 1)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            # Use first image_id where this embryo appears for reference
                            ref_image_id = next(img_id for img_id in image_ids 
                                              if embryo_id in video_data["images"][img_id].get("embryos", {}))
                            self._add_flag("HIGH_SEGMENTATION_VAR_EMBRYO", flag_data, "image", ref_image_id)

                # Snip-level variability flag
                for i, image_id in enumerate(image_ids):
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                        
                    image_data = video_data["images"][image_id]
                    embryos_in_image = image_data.get("embryos", {})
                    
                    for embryo_id, embryo_data in embryos_in_image.items():
                        snip_id = embryo_data.get("snip_id")
                        
                        if not snip_id:
                            continue
                            
                        # Check if snip should be processed
                        if not self._should_process_snip(snip_id, entities):
                            continue

                        current_area = embryo_data.get("area")
                        if current_area is None:
                            segmentation = embryo_data.get("segmentation")
                            segmentation_format = embryo_data.get("segmentation_format", "unknown")
                            if segmentation and segmentation_format == "rle":
                                try:
                                    mask = mask_utils.decode(segmentation)
                                    current_area = float(np.sum(mask))
                                except Exception as e:
                                    continue
                        
                        if current_area is None:
                            continue

                        # Get areas from nearby frames
                        before_areas = []
                        after_areas = []

                        for j in range(max(0, i - n_frames_check), i):
                            frame_id = image_ids[j]
                            if frame_id in embryo_frames[embryo_id]:
                                before_areas.append(embryo_frames[embryo_id][frame_id])

                        for j in range(i + 1, min(len(image_ids), i + n_frames_check + 1)):
                            frame_id = image_ids[j]
                            if frame_id in embryo_frames[embryo_id]:
                                after_areas.append(embryo_frames[embryo_id][frame_id])
                        
                        avg_before = np.mean(before_areas) if before_areas else None
                        avg_after = np.mean(after_areas) if after_areas else None

                        diff_before_pct = abs(current_area - avg_before) / avg_before if avg_before else None
                        diff_after_pct = abs(current_area - avg_after) / avg_after if avg_after else None

                        # Explicit boolean conversion to avoid numpy.bool_ serialization issues
                        flag_before = bool(diff_before_pct > 0.20) if diff_before_pct is not None else False
                        flag_after = bool(diff_after_pct > 0.20) if diff_after_pct is not None else False

                        if flag_before or flag_after:
                            flag_data = {
                                "snip_id": snip_id,
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "image_id": image_id,
                                "embryo_id": embryo_id,
                                "current_area": float(round(current_area, 1)),
                                "avg_before": float(round(avg_before, 1)) if avg_before else None,
                                "avg_after": float(round(avg_after, 1)) if avg_after else None,
                                "diff_before_pct": float(round(diff_before_pct, 3)) if diff_before_pct is not None else None,
                                "diff_after_pct": float(round(diff_after_pct, 3)) if diff_after_pct is not None else None,
                                "flagged_before": flag_before,
                                "flagged_after": flag_after,
                                "frames_checked_before": int(len(before_areas)),
                                "frames_checked_after": int(len(after_areas)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            self._add_flag("HIGH_SEGMENTATION_VAR_SNIP", flag_data, "snip", snip_id)
        
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Segmentation variability check completed in {elapsed:.2f}s")
    
    def check_mask_on_edge(self, author: str, entities: Dict[str, List[str]]):
        """Flag masks that touch image edges (within 5 pixels)."""
        if self.verbose:
            print("🔍 Checking masks on image edges...")
        
        t0 = time()
        margin_pixels = 5  # safety margin in pixels
        flag_count = 0

        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in experiments_items:
            print(f"DEBUG EDGE: Processing experiment {exp_id}")
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                print(f"DEBUG EDGE: Skipping experiment {exp_id}")
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                        
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if not snip_id:
                            continue
                            
                        # Check if snip should be processed
                        if not self._should_process_snip(snip_id, entities):
                            continue
                        
                        segmentation = embryo_data.get("segmentation")
                        segmentation_format = embryo_data.get("segmentation_format", "unknown")
                        print(f"DEBUG EDGE MASK: Processing snip {snip_id}, segmentation exists: {segmentation is not None}")
                        if segmentation:
                            print(f"DEBUG EDGE MASK: Segmentation format: {segmentation_format}")
                        
                        if segmentation and segmentation_format == "rle":
                            print(f"DEBUG EDGE MASK: About to decode RLE for snip {snip_id}")
                            try:
                                mask = mask_utils.decode(segmentation)
                                height, width = mask.shape
                                print(f"DEBUG EDGE MASK: Decoded mask shape: {height}x{width} for snip {snip_id}")
                                
                                # Check if mask touches edges
                                touches_edges = {
                                    "top": bool(np.any(mask[:margin_pixels, :])),
                                    "bottom": bool(np.any(mask[-margin_pixels:, :])),
                                    "left": bool(np.any(mask[:, :margin_pixels])),
                                    "right": bool(np.any(mask[:, -margin_pixels:]))
                                }
                                
                                print(f"DEBUG EDGE MASK: Edge touches for snip {snip_id}: {touches_edges}")
                                
                                if any(touches_edges.values()):
                                    print(f"DEBUG EDGE: *** FLAGGING snip {snip_id} for edge touching: {touches_edges} ***")
                                    flag_data = {
                                        "snip_id": snip_id,
                                        "image_id": image_id,
                                        "embryo_id": embryo_id,
                                        "touches_edges": touches_edges,
                                        "margin_pixels": margin_pixels,
                                        "image_shape": [height, width],
                                        "author": author,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    self._add_flag("MASK_ON_EDGE", flag_data, "snip", snip_id)
                                    flag_count += 1
                                    print(f"DEBUG: Flag count is now {flag_count}")
                                else:
                                    print(f"DEBUG: snip {snip_id} does NOT touch edges: {touches_edges}")
                                    
                            except Exception as e:
                                print(f"DEBUG EDGE MASK: Exception processing snip {snip_id}: {e}")
                                continue
                        else:
                            print(f"DEBUG EDGE MASK: Skipping snip {snip_id} - no RLE segmentation")
        
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Edge detection check completed in {elapsed:.2f}s ({flag_count} masks flagged)")
    
    def check_detection_failure(self, author: str, entities: Dict[str, List[str]]):
        """Flag images where expected embryos are missing (fewer than expected)."""
        if self.verbose:
            print("🔍 Checking for detection failures...")
        
        t0 = time()
        expected_min_embryos = 3  # Minimum expected embryos per image
        flag_count = 0

        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in experiments_items:
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                    
                    embryos = image_data.get("embryos", {})
                    embryo_count = len(embryos)
                    
                    if embryo_count < expected_min_embryos:
                        flag_data = {
                            "image_id": image_id,
                            "video_id": video_id,
                            "embryo_count": embryo_count,
                            "expected_min": expected_min_embryos,
                            "author": author,
                            "timestamp": datetime.now().isoformat()
                        }
                        self._add_flag("DETECTION_FAILURE", flag_data, "image", image_id)
                        flag_count += 1
        
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Detection failure check completed in {elapsed:.2f}s ({flag_count} images flagged)")
    
    def check_overlapping_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag images where embryo masks overlap (IoU > 0.1)."""
        if self.verbose:
            print("🔍 Checking for overlapping masks...")
        
        t0 = time()
        iou_threshold = 0.1  # 10% overlap threshold
        flag_count = 0

        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Overlap", total=len(experiments_items)):
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                    
                    embryos = list(image_data.get("embryos", {}).items())
                    if len(embryos) < 2:
                        continue
                    
                    # Check all pairs of embryos
                    for i in range(len(embryos)):
                        for j in range(i + 1, len(embryos)):
                            embryo_id1, embryo_data1 = embryos[i]
                            embryo_id2, embryo_data2 = embryos[j]
                            
                            seg1 = embryo_data1.get("segmentation")
                            seg2 = embryo_data2.get("segmentation")
                            format1 = embryo_data1.get("segmentation_format", "unknown")
                            format2 = embryo_data2.get("segmentation_format", "unknown")
                            
                            if (seg1 and seg2 and 
                                format1 == "rle" and format2 == "rle"):
                                try:
                                    mask1 = mask_utils.decode(seg1)
                                    mask2 = mask_utils.decode(seg2)
                                    
                                    # Calculate IoU
                                    intersection = np.sum(mask1 & mask2)
                                    union = np.sum(mask1 | mask2)
                                    
                                    if union > 0:
                                        iou = intersection / union
                                        
                                        if iou > iou_threshold:
                                            flag_data = {
                                                "image_id": image_id,
                                                "video_id": video_id,
                                                "embryo_id1": embryo_id1,
                                                "embryo_id2": embryo_id2,
                                                "snip_id1": embryo_data1.get("snip_id"),
                                                "snip_id2": embryo_data2.get("snip_id"),
                                                "iou": float(round(iou, 4)),
                                                "intersection_pixels": int(intersection),
                                                "union_pixels": int(union),
                                                "threshold_used": iou_threshold,
                                                "author": author,
                                                "timestamp": datetime.now().isoformat()
                                            }
                                            self._add_flag("OVERLAPPING_MASKS", flag_data, "image", image_id)
                                            flag_count += 1
                                            
                                except Exception:
                                    continue
        
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Overlapping masks check completed in {elapsed:.2f}s ({flag_count} images flagged)")
    
    def check_large_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag unusually large masks as percentage of total image area."""
        if self.verbose:
            print("🔍 Checking for unusually large masks...")
        
        t0 = time()
        pct_threshold = 0.15  # 15% of frame area
        flag_count = 0

        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Large", total=len(experiments_items)):
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                        
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if not snip_id:
                            continue
                            
                        # Check if snip should be processed
                        if not self._should_process_snip(snip_id, entities):
                            continue
                        
                        segmentation = embryo_data.get("segmentation")
                        segmentation_format = embryo_data.get("segmentation_format", "unknown")
                        if segmentation and segmentation_format == "rle":
                            try:
                                mask = mask_utils.decode(segmentation)
                                mask_area = np.sum(mask)
                                total_area = mask.shape[0] * mask.shape[1]
                                pct_area = mask_area / total_area
                                
                                if pct_area > pct_threshold:
                                    flag_data = {
                                        "snip_id": snip_id,
                                        "embryo_id": embryo_id,
                                        "image_id": image_id,
                                        "mask_area": int(mask_area),
                                        "total_area": int(total_area),
                                        "area_percentage": float(pct_area),
                                        "threshold_used": pct_threshold,
                                        "image_shape": list(mask.shape),
                                        "author": author,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    self._add_flag("LARGE_MASK", flag_data, "snip", snip_id)
                                    flag_count += 1
                                    
                            except Exception:
                                continue
        
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Large mask check completed in {elapsed:.2f}s ({flag_count} masks flagged)")

    def check_small_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag unusually small masks as percentage of total image area."""
        if self.verbose:
            print("🔍 Checking for unusually small masks...")
        
        t0 = time()
        pct_threshold = 0.001  # 0.1% of frame area (very small)
        flag_count = 0

        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Small", total=len(experiments_items)):
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                        
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if not snip_id:
                            continue
                            
                        # Check if snip should be processed
                        if not self._should_process_snip(snip_id, entities):
                            continue
                        
                        segmentation = embryo_data.get("segmentation")
                        segmentation_format = embryo_data.get("segmentation_format", "unknown")
                        if segmentation and segmentation_format == "rle":
                            try:
                                mask = mask_utils.decode(segmentation)
                                mask_area = np.sum(mask)
                                total_area = mask.shape[0] * mask.shape[1]
                                pct_area = mask_area / total_area
                                
                                if pct_area < pct_threshold and mask_area > 0:  # Don't flag empty masks
                                    flag_data = {
                                        "snip_id": snip_id,
                                        "embryo_id": embryo_id,
                                        "image_id": image_id,
                                        "mask_area": int(mask_area),
                                        "total_area": int(total_area),
                                        "area_percentage": float(pct_area),
                                        "threshold_used": pct_threshold,
                                        "image_shape": list(mask.shape),
                                        "author": author,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    self._add_flag("SMALL_MASK", flag_data, "snip", snip_id)
                                    flag_count += 1
                                    
                            except Exception:
                                continue
        
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Small mask check completed in {elapsed:.2f}s ({flag_count} masks flagged)")

    def check_discontinuous_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag masks whose segmentation contains multiple disconnected components."""
        if not _HAS_IMAGE_LIBS:
            if self.verbose:
                print("⚠️ Skipping discontinuous mask check (scipy/skimage not available)")
            return
            
        if self.verbose:
            print("🔍 Checking for discontinuous masks...")
        
        t0 = time()
        flag_count = 0
        
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Discontinuous", total=len(experiments_items)):
            
            # Check if experiment should be processed
            if not self._should_process_experiment(exp_id, entities):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Check if video should be processed
                if not self._should_process_video(video_id, entities):
                    continue
                
                for image_id, image_data in video_data.get("images", {}).items():
                    # Check if image should be processed
                    if not self._should_process_image(image_id, entities):
                        continue
                        
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if not snip_id:
                            continue
                            
                        # Check if snip should be processed
                        if not self._should_process_snip(snip_id, entities):
                            continue
                        
                        segmentation = embryo_data.get("segmentation")
                        segmentation_format = embryo_data.get("segmentation_format", "unknown")
                        if segmentation and segmentation_format == "rle":
                            try:
                                mask = mask_utils.decode(segmentation)
                                labeled_mask = label(mask)
                                num_components = int(np.max(labeled_mask))
                                
                                if num_components > 1:
                                    component_areas = [int(np.sum(labeled_mask == comp_id)) 
                                                       for comp_id in range(1, num_components + 1)]
                                    
                                    flag_data = {
                                        "snip_id": snip_id,
                                        "embryo_id": embryo_id,
                                        "image_id": image_id,
                                        "num_components": num_components,
                                        "component_areas": component_areas,
                                        "total_area": int(np.sum(mask)),
                                        "largest_component_area": int(max(component_areas)),
                                        "smallest_component_area": int(min(component_areas)),
                                        "author": author,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    self._add_flag("DISCONTINUOUS_MASK", flag_data, "snip", snip_id)
                                    flag_count += 1

                            except Exception:
                                continue
                                
        if self.verbose:
            elapsed = time() - t0
            print(f"   ✓ Discontinuous mask check completed in {elapsed:.2f}s ({flag_count} masks flagged)")
    
    def _save_qc_summary(self, author: str):
        """Save QC summary and write updated GSAM file."""
        # Count flags from newly processed entities only
        flag_counts = self._count_flags_from_new_entities()
        
        # Create QC run record
        qc_run = {
            "timestamp": datetime.now().isoformat(),
            "author": author,
            "entities_processed": {
                "experiments": int(len(self.new_experiment_ids)),
                "videos": int(len(self.new_video_ids)),
                "images": int(len(self.new_image_ids)),
                "snips": int(len(self.new_snip_ids))
            },
            "flags_added": int(sum(flag_counts.values())),
            "flag_breakdown": {k: int(v) for k, v in flag_counts.items()}
        }
        
        # Add to QC history
        self.gsam_data["flags"]["qc_history"].append(qc_run)
        
        # Update overall QC summary
        self.gsam_data["flags"]["qc_summary"] = {
            "last_updated": datetime.now().isoformat(),
            "total_qc_runs": int(len(self.gsam_data["flags"]["qc_history"])),
            "total_entities_processed": {
                "experiments": int(len(self.processed_experiment_ids | self.new_experiment_ids)),
                "videos": int(len(self.processed_video_ids | self.new_video_ids)),
                "images": int(len(self.processed_image_ids | self.new_image_ids)),
                "snips": int(len(self.processed_snip_ids | self.new_snip_ids))
            }
        }
        
        # Ensure JSON serializable before saving
        self.gsam_data = ensure_json_serializable(self.gsam_data)
        
        # Save to file using BaseFileHandler method
        self.save_json(self.gsam_data, create_backup=True)
            
        if self.verbose:
            print(f"💾 Saved QC results: {qc_run['flags_added']} new flags added")
            for category, count in qc_run['flag_breakdown'].items():
                if count > 0:
                    print(f"   {category}: {count} flags")
    
    def _count_flags_from_new_entities(self) -> Dict[str, int]:
        """Count flags added in this run from newly processed entities only."""
        flag_counts = defaultdict(int)
        flags_section = self.gsam_data["flags"]
        
        # Count flags from all entity types, but only for newly processed entities
        for entity_type in ["by_experiment", "by_video", "by_image", "by_snip"]:
            if entity_type not in flags_section:
                continue
                
            for entity_id, entity_flags in flags_section[entity_type].items():
                # Check if this entity was processed in this run
                entity_is_new = False
                if entity_type == "by_experiment" and entity_id in self.new_experiment_ids:
                    entity_is_new = True
                elif entity_type == "by_video" and entity_id in self.new_video_ids:
                    entity_is_new = True
                elif entity_type == "by_image" and entity_id in self.new_image_ids:
                    entity_is_new = True
                elif entity_type == "by_snip" and entity_id in self.new_snip_ids:
                    entity_is_new = True
                
                # Only count flags for newly processed entities
                if entity_is_new:
                    for flag_type, flag_instances in entity_flags.items():
                        flag_counts[flag_type] += len(flag_instances)
        
        return dict(flag_counts)
   
    def generate_overview(self, entities: Dict[str, List[str]]):
        """Generate summary overview of all flags."""
        flag_counts = self._count_flags_in_hierarchy()
        
        overview = {
            "total_flags": sum(flag_counts.values()),
            "flag_breakdown": flag_counts,
            "entities_checked": {k: len(v) for k, v in entities.items()},
            "last_updated": datetime.now().isoformat()
        }
        
        self.gsam_data["flags"]["flag_overview"] = overview
    
    def _count_flags_in_hierarchy(self) -> Dict[str, int]:
        """Count all flags by type across the hierarchy."""
        flag_counts = defaultdict(int)
        flags_section = self.gsam_data["flags"]
        
        for entity_key in ["by_experiment", "by_video", "by_image", "by_snip"]:
            entity_flags = flags_section.get(entity_key, {})
            for entity_id, entity_flag_data in entity_flags.items():
                for flag_type, flag_list in entity_flag_data.items():
                    count = len(flag_list)
                    flag_counts[flag_type] += count
        
        return dict(flag_counts)
    
    def get_flags_summary(self) -> Dict:
        """Get summary of all QC flags."""
        overview = self.gsam_data["flags"].get("flag_overview", {})
        qc_history = self.gsam_data["flags"].get("qc_history", [])
        
        # Get flag breakdown - it's already a dict of {flag_type: count}
        flag_breakdown = overview.get("flag_breakdown", {})
        total_flags = overview.get("total_flags", 0)

        return {
            "total_flags": total_flags,
            "flag_categories": flag_breakdown,
            "entities_with_flags": {
                "experiments": len(self.gsam_data["flags"].get("by_experiment", {})),
                "videos": len(self.gsam_data["flags"].get("by_video", {})),
                "images": len(self.gsam_data["flags"].get("by_image", {})),
                "snips": len(self.gsam_data["flags"].get("by_snip", {}))
            },
            "last_run": qc_history[-1] if qc_history else None,
            "total_runs": len(qc_history)
        }

    def print_summary(self):
        """Print a summary of QC results."""
        summary = self.get_flags_summary()
        print(f"\n🏁 QC Summary")
        print(f"{'=' * 40}")
        print(f"Total flags: {summary['total_flags']}")
        print(f"\nFlags by category:")
        for category, count in summary['flag_categories'].items():
            if count > 0:
                print(f"  {category}: {count}")
        
        print(f"\nEntities with flags:")
        for entity_type, count in summary['entities_with_flags'].items():
            print(f"  {entity_type}: {count}")
        
        if summary['total_flags'] == 0:
            print("\n✅ No quality issues detected!")
        
        # Last run info
        last_run = summary.get("last_run")
        if last_run:
            timestamp = last_run.get("timestamp", "Unknown")
            author = last_run.get("author", "Unknown")
            print(f"\n🕒 Last Run: {timestamp} by {author}")
        
        print("="*40)

    def get_flags_for_entity(self, entity_type: str, entity_id: str) -> Dict:
        """Get all flags for a specific entity."""
        entity_key = f"by_{entity_type}"
        flags_section = self.gsam_data["flags"]
        
        return flags_section.get(entity_key, {}).get(entity_id, {})

    def get_flags_by_type(self, flag_type: str) -> List[Dict]:
        """Get all flags of a specific type across all entities."""
        all_flags = []
        flags_section = self.gsam_data["flags"]
        
        for entity_key in ["by_experiment", "by_video", "by_image", "by_snip"]:
            entity_flags = flags_section.get(entity_key, {})
            for entity_id, entity_flag_data in entity_flags.items():
                flag_list = entity_flag_data.get(flag_type, [])
                all_flags.extend(flag_list)
        
        return all_flags


def main():
    parser = argparse.ArgumentParser(
        description="Run quality control analysis on SAM2 segmentation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run QC on new entities only (incremental)
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json
  
  # Force reprocess all entities
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json --process-all
  
  # QC specific experiments only
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json \\
    --experiments "20240506,20250703_chem3_28C_T00_1325"
  
  # Custom author and output path
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json \\
    --output qc_results.json --author "researcher_name"
  
  # Dry run (analyze but don't save changes)
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to grounded_sam_annotations.json file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", 
        help="Output path (default: modify input file in-place)"
    )
    parser.add_argument(
        "--author", 
        default="pipeline_qc", 
        help="QC run author identifier (default: pipeline_qc)"
    )
    parser.add_argument(
        "--process-all", 
        action="store_true", 
        help="Process all entities instead of just new ones"
    )
    parser.add_argument(
        "--experiments", 
        help="Comma-separated experiment IDs to target"
    )
    parser.add_argument(
        "--videos", 
        help="Comma-separated video IDs to target"
    )
    parser.add_argument(
        "--images", 
        help="Comma-separated image IDs to target"
    )
    parser.add_argument(
        "--snips", 
        help="Comma-separated snip IDs to target"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=True,
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "--no-progress", 
        action="store_true", 
        help="Disable progress bars"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Run analysis without saving results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    if args.verbose:
        print(f"🔍 Starting QC analysis on: {input_path}")
        if args.dry_run:
            print("🧪 DRY RUN MODE - no changes will be saved")
    
    try:
        # Initialize QC
        qc = GSAMQualityControl(
            gsam_path=str(input_path),
            verbose=args.verbose,
            progress=not args.no_progress
        )
        
        # Prepare target entities (if specified)
        target_entities = None
        if any([args.experiments, args.videos, args.images, args.snips]):
            target_entities = {
                "experiment_ids": args.experiments.split(",") if args.experiments else [],
                "video_ids": args.videos.split(",") if args.videos else [],
                "image_ids": args.images.split(",") if args.images else [],
                "snip_ids": args.snips.split(",") if args.snips else []
            }
            
            if args.verbose:
                print("🎯 Targeting specific entities:")
                for entity_type, entity_list in target_entities.items():
                    if entity_list:
                        print(f"   {entity_type}: {len(entity_list)} entities")
        
        # Run QC analysis
        qc.run_all_checks(
            author=args.author,
            process_all=args.process_all,
            target_entities=target_entities,
            save_in_place=not args.dry_run and args.output is None
        )
        
        # Save to custom output if specified
        if args.output and not args.dry_run:
            output_path = Path(args.output)
            qc.save_json(qc.gsam_data, filepath=str(output_path), create_backup=True)
            if args.verbose:
                print(f"💾 Results saved to: {output_path}")
        
        # Print summary
        qc.print_summary()
        
        if args.verbose:
            print("✅ QC analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ QC analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
