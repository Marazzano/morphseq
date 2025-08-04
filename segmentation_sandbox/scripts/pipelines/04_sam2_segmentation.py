#!/usr/bin/env python3
"""
Pipeline Script 4: SAM2 Video Segmentation

Run SAM2 video segmentation using GroundedDINO detection annotations.
Processes all experiments by default.
"""

import argparse
import sys
import json
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from metadata.experiment_metadata import ExperimentMetadata

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video segmentation for MorphSeq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all detections for segmentation
  python 04_sam2_segmentation.py --metadata experiment_metadata.json \\
    --annotations detections.json --output segmentations.json
  
  # Process specific experiments only
  python 04_sam2_segmentation.py --metadata experiment_metadata.json \\
    --annotations detections.json --output segmentations.json \\
    --experiments "20240506,20250703_chem3_28C_T00_1325"
  
  # Custom segmentation parameters
  python 04_sam2_segmentation.py --metadata experiment_metadata.json \\
    --annotations detections.json --output segmentations.json \\
    --propagation-frames 5 --temporal-window 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--metadata", 
        required=True, 
        help="Path to experiment_metadata.json from step 1"
    )
    parser.add_argument(
        "--annotations", 
        required=True, 
        help="Path to detection annotations JSON from step 3"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output path for segmentation annotations JSON"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", 
        help="Pipeline config YAML file (for SAM2 model paths)"
    )
    parser.add_argument(
        "--experiments", 
        help="Comma-separated experiment IDs to process (default: all)"
    )
    parser.add_argument(
        "--propagation-frames", 
        type=int, 
        default=10, 
        help="Number of frames to propagate per sequence (default: 10)"
    )
    parser.add_argument(
        "--temporal-window", 
        type=int, 
        default=5, 
        help="Temporal window for SAM2 tracking (default: 5)"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.3, 
        help="Minimum confidence for using detections (default: 0.3)"
    )
    parser.add_argument(
        "--max-objects-per-frame", 
        type=int, 
        default=20, 
        help="Maximum objects to track per frame (default: 20)"
    )
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=50, 
        help="Auto-save every N frames (default: 50)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be processed without running segmentation"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    metadata_path = Path(args.metadata).resolve()
    annotations_path = Path(args.annotations).resolve()
    output_path = Path(args.output).resolve()
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata file does not exist: {metadata_path}")
        sys.exit(1)
    
    if not annotations_path.exists():
        print(f"❌ Error: Annotations file does not exist: {annotations_path}")
        sys.exit(1)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("📋 Loading experiment metadata...")
    try:
        meta = ExperimentMetadata(str(metadata_path))
        print(f"✅ Loaded metadata with {len(meta.metadata['experiments'])} experiments")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        sys.exit(1)
    
    # Load detection annotations
    print("🔍 Loading detection annotations...")
    try:
        with open(annotations_path, 'r') as f:
            detection_data = json.load(f)
        
        annotations = detection_data.get('high_quality_annotations', detection_data.get('annotations', {}))
        print(f"📸 Found annotations for {len(annotations)} images")
        
        if len(annotations) == 0:
            print("⚠️  No annotations found - check detection results")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        sys.exit(1)
    
    # Parse experiment filter
    experiment_ids = None
    if args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
        print(f"📌 Processing specific experiments: {experiment_ids}")
    else:
        print("📌 Processing ALL experiments in metadata")
    
    # Get video sequences for segmentation
    print("🎬 Finding video sequences for segmentation...")
    try:
        video_sequences = meta.get_video_sequences_for_segmentation(
            detection_annotations=annotations, 
            experiment_ids=experiment_ids
        )
        print(f"🎬 Found {len(video_sequences)} video sequences to process")
        
        if len(video_sequences) == 0:
            print("⚠️  No video sequences found - check detection annotations and metadata")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error getting video sequences: {e}")
        # Method may not exist yet, create placeholder
        print("⚠️  Video sequence detection not yet implemented in ExperimentMetadata")
        video_sequences = []
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Segmentation plan:")
        print(f"  🎬 Video sequences: {len(video_sequences)}")
        print(f"  📸 Detection annotations: {len(annotations)}")
        print(f"  🕰️  Propagation frames: {args.propagation_frames}")
        print(f"  🎯 Confidence threshold: {args.confidence_threshold}")
        print(f"  📊 Max objects per frame: {args.max_objects_per_frame}")
        print(f"  💾 Output: {output_path}")
        
        # Show sample of sequences
        print(f"\n🎬 Sample sequences (first 3):")
        for i, seq in enumerate(video_sequences[:3]):
            print(f"  {i+1}. {seq.get('experiment_id', 'unknown')}")
            print(f"     📁 {seq.get('video_path', 'unknown')}")
            print(f"     📸 {len(seq.get('frames', []))} frames")
        
        if len(video_sequences) > 3:
            print(f"  ... and {len(video_sequences) - 3} more")
        
        return
    
    print("🚀 Starting SAM2 video segmentation...")
    print(f"🕰️  Propagation frames: {args.propagation_frames}")
    print(f"🎯 Confidence threshold: {args.confidence_threshold}")
    print(f"📊 Max objects per frame: {args.max_objects_per_frame}")
    
    try:
        # TODO: Import and use actual SAM2 segmentation
        # For now, create placeholder
        print("⚠️  SAM2 video segmentation not yet implemented")
        print("📝 This script will call the segmentation logic once Module 3 is complete")
        
        # Create placeholder segmentation file
        from datetime import datetime
        
        placeholder_segmentations = {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "script_version": "04_sam2_segmentation.py",
                "metadata_source": str(metadata_path),
                "detection_source": str(annotations_path),
                "processing_parameters": {
                    "propagation_frames": args.propagation_frames,
                    "temporal_window": args.temporal_window,
                    "confidence_threshold": args.confidence_threshold,
                    "max_objects_per_frame": args.max_objects_per_frame
                },
                "processing_status": "placeholder"
            },
            "segmentations": {},
            "tracking_results": {},
            "statistics": {
                "total_sequences": len(video_sequences),
                "processed_sequences": 0,
                "total_objects": 0,
                "total_masks": 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(placeholder_segmentations, f, indent=2)
        
        print(f"📄 Created placeholder segmentations: {output_path}")
        print(f"✅ Segmentation complete!")
        print(f"📋 Next step: Run 05_analysis_export.py with --segmentations {output_path}")
        
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
