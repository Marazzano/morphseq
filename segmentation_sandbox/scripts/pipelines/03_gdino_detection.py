#!/usr/bin/env python3
"""
Stage 3: GroundedDINO Detection with Quality Filtering (Modern Module 2 Implementation)
======================================================================================

This script uses the new Module 2 GroundingDINO implementation to generate annotations
with high-quality filtering using our modular pipeline utilities.

Features:
- Uses Module 2 GroundedDinoAnnotations class
- Integrates with ExperimentMetadata for efficient image discovery
- Generates high-quality annotations with confidence and IoU filtering
- Entity tracking and validation using Module 0/1 utilities
- Atomic saves with backup functionality
"""

import os
import sys
import json
import yaml
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
SANDBOX_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(SANDBOX_ROOT))

# Import Module 2 utilities
from scripts.detection_segmentation.grounded_dino_utils import (
    load_config, load_groundingdino_model, GroundedDinoAnnotations
)
from scripts.metadata.experiment_metadata import ExperimentMetadata


def main():
    parser = argparse.ArgumentParser(description="Generate GroundedDINO detections with quality filtering (Module 2)")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--annotations", required=True, help="Path to output annotations JSON")
    
    # Quality filtering arguments
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                       help="Confidence threshold for filtering (default: 0.45)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for duplicate removal (default: 0.5)")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip quality filtering step")
    
    # Model configuration
    parser.add_argument("--weights", default=None,
                       help="Path to model weights (overrides config)")
    parser.add_argument("--prompt", default="individual embryo",
                       help="Detection prompt (default: individual embryo)")
    
    # Processing options
    parser.add_argument("--experiment-ids", nargs="*", default=None,
                       help="Specific experiment IDs to process")
    parser.add_argument("--video-ids", nargs="*", default=None,
                       help="Specific video IDs to process")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (for testing)")
    parser.add_argument("--auto-save-interval", type=int, default=100,
                       help="Auto-save every N images")
    
    args = parser.parse_args()

    print("🚀 Starting GroundingDINO Detection with Quality Filtering (Module 2)")
    print("=" * 70)

    # Load config and setup device
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    # Override weights if specified
    if args.weights:
        config["models"]["groundingdino"]["weights"] = args.weights
        print(f"🔄 Using custom weights: {Path(args.weights).name}")

    # Load experiment metadata
    metadata_manager = ExperimentMetadata(args.metadata, verbose=True)
    print(f"📊 Loaded metadata: {metadata_manager}")

    # Get target images
    if args.experiment_ids:
        print(f"🎯 Processing specific experiments: {args.experiment_ids}")
        target_images = []
        for exp_id in args.experiment_ids:
            exp_images = metadata_manager.list_images(experiment_id=exp_id)
            target_images.extend(exp_images)
            print(f"   {exp_id}: {len(exp_images)} images")
    elif args.video_ids:
        print(f"🎯 Processing specific videos: {args.video_ids}")
        target_images = []
        for video_id in args.video_ids:
            video_images = []
            # Find experiment for this video
            for exp_id in metadata_manager.list_experiments():
                if video_id in metadata_manager.list_videos(exp_id):
                    video_images = metadata_manager.list_images(exp_id, video_id)
                    break
            target_images.extend(video_images)
            print(f"   {video_id}: {len(video_images)} images")
    else:
        target_images = metadata_manager.list_images()
        print(f"📊 Processing all images: {len(target_images)}")

    if args.max_images and len(target_images) > args.max_images:
        target_images = target_images[:args.max_images]
        print(f"🔢 Limited to first {args.max_images} images for testing")

    # =================================================================================
    # PHASE 1: GroundingDINO Detection
    # =================================================================================
    print("\n" + "="*60)
    print(f"🔍 PHASE 1: GroundingDINO Detection ('{args.prompt}')")
    print("="*60)

    # Initialize annotations manager with metadata integration
    annotations = GroundedDinoAnnotations(args.annotations, verbose=True, metadata_path=args.metadata)
    print(f"💾 Annotations will be saved to: {args.annotations}")

    try:
        # Load model
        model = load_groundingdino_model(config, device=device)
        print("✅ Model loaded successfully")

        # Process missing annotations
        print(f"🔄 Processing annotations for prompt: '{args.prompt}'")
        results = annotations.process_missing_annotations(
            model=model,
            prompts=args.prompt,
            experiment_ids=args.experiment_ids,
            video_ids=args.video_ids,
            image_ids=target_images if not args.experiment_ids and not args.video_ids else None,
            auto_save_interval=args.auto_save_interval,
            store_image_source=False,
            show_anno=False,
            overwrite=False,
            consider_different_if_different_weights=True,
        )
        
        print(f"✅ Processed {len(results)} images")
        annotations.save()
        print("✅ Phase 1 complete: Detection finished")

    except Exception as e:
        print(f"❌ Error in Phase 1 (detection): {e}")
        import traceback
        traceback.print_exc()
        return

    # Skip filtering if requested
    if args.skip_filtering:
        print("\n⏭️ Skipping quality filtering as requested.")
        annotations.print_summary()
        print("✅ GroundingDINO detection generation completed!")
        return

    # =================================================================================
    # PHASE 2: High-Quality Filtering
    # =================================================================================
    print("\n" + "="*60)
    print("🎯 PHASE 2: High-Quality Filtering")
    print("="*60)
    
    try:
        # Get all image IDs that have annotations for the prompt
        annotated_images = annotations.get_annotated_image_ids(prompt=args.prompt)
        print(f"Found {len(annotated_images)} images with '{args.prompt}' annotations")
        
        if annotated_images:
            print(f"🎯 Generating high-quality annotations...")
            print(f"   Confidence threshold: {args.confidence_threshold}")
            print(f"   IoU threshold: {args.iou_threshold}")
            
            result = annotations.generate_high_quality_annotations(
                image_ids=annotated_images,
                prompt=args.prompt,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                overwrite=True,
                save_to_self=True
            )
            
            # Print filtering results
            stats = result["statistics"]
            print(f"📊 Filtering Summary:")
            print(f"   Original detections: {stats['original_detections']}")
            print(f"   Confidence removed: {stats['confidence_removed']}")
            print(f"   IoU removed: {stats['iou_removed']}")
            print(f"   Final detections: {stats['final_detections']}")
            print(f"   Retention rate: {stats['retention_rate']:.1%}")
            print(f"   Final images: {stats['final_images']}")
            print(f"   Experiments processed: {stats['experiments_processed']}")
        else:
            print("⚠️ No annotations found for filtering")
        
        annotations.save()
        print("✅ Phase 2 complete: Quality filtering finished")
        
    except Exception as e:
        print(f"❌ Error in Phase 2 (filtering): {e}")
        import traceback
        traceback.print_exc()

    # =================================================================================
    # FINAL SUMMARY
    # =================================================================================
    print("\n" + "="*60)
    print("📊 FINAL PIPELINE SUMMARY")
    print("="*60)
    
    annotations.print_summary()
    
    print(f"\n📁 OUTPUT:")
    print(f"   Annotations file: {args.annotations}")
    print(f"   Prompt processed: '{args.prompt}'")
    print(f"   Quality filtering: {'Applied' if not args.skip_filtering else 'Skipped'}")
    
    if not args.skip_filtering:
        hq_data = annotations.annotations.get("high_quality_annotations", {})
        if hq_data:
            total_hq_images = sum(len(exp_data.get("filtered", {})) for exp_data in hq_data.values())
            total_hq_detections = sum(
                sum(len(dets) for dets in exp_data.get("filtered", {}).values()) 
                for exp_data in hq_data.values()
            )
            print(f"   High-quality results: {len(hq_data)} experiments, {total_hq_images} images, {total_hq_detections} detections")
    
    print(f"\n✅ ANNOTATION GENERATION COMPLETE!")
    print(f"🎯 Ready for downstream processing (SAM2, etc.)")


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/pipelines/03_gdino_detection.py \
#   --config configs/pipeline_config.yaml \
#   --metadata data/raw_data_organized/experiment_metadata.json \
#   --annotations data/annotation_and_masks/gdino_annotations/gdino_annotations_modern.json \
#   --confidence-threshold 0.45 \
#   --iou-threshold 0.5 \
#   --experiment-ids 20250612_30hpf_ctrl_atf6 \
#   --max-images 50
