#!/usr/bin/env python3
"""
Export embryo masks from SAM2 annotations as labeled images
Also transfers QC flags from SAM annotations to EmbryoMetadata using GSAM ID matching
"""

import argparse
from pathlib import Path
import sys
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
print("SCRIPT_DIR:", SCRIPT_DIR)
UTILS_DIR = SCRIPT_DIR / "utils"
sys.path.insert(0, str(UTILS_DIR))
print("sys.path:", sys.path)

from mask_export_utils import EmbryoMaskExporter
from embryo_metada_dev_instruction.embryo_metadata_refactored import EmbryoMetadata

def transfer_qc_flags_via_gsam_id(sam_annotations_path: Path, embryo_metadata_path: Path, verbose: bool = True):
    """
    Transfer QC flags from SAM annotations to EmbryoMetadata using GSAM ID matching.
    
    Args:
        sam_annotations_path: Path to SAM annotations file (GroundedSamAnnotation)
        embryo_metadata_path: Path to embryo metadata file
        verbose: Enable verbose output
        
    Returns:
        Number of flags transferred
    """
    # Load SAM annotations
    with open(sam_annotations_path, 'r') as f:
        sam_data = json.load(f)
    
    # Check for QC flags in both old and new GSAM QC formats
    qc_overview = sam_data.get("GEN_flag_overview", {})
    qc_meta = sam_data.get("qc_meta", {})
    flags_section = sam_data.get("flags", {})

    if not (qc_overview or qc_meta or flags_section):
        if verbose:
            print("ℹ️  No QC analysis found in SAM annotations (checked GEN_flag_overview, qc_meta, and flags)")
        return 0
    
    # Get GSAM ID from SAM annotations metadata
    sam_gsam_id = sam_data.get("file_info", {}).get("gsam_annotation_id")
    if not sam_gsam_id:
        # Try alternative locations
        sam_gsam_id = sam_data.get("qc_meta", {}).get("gsam_annotation_id")
    
    if not sam_gsam_id:
        if verbose:
            print("ℹ️  No GSAM ID found in SAM annotations")
        return 0
    
    # Load EmbryoMetadata
    try:
        embryo_metadata = EmbryoMetadata(
            sam_annotation_path=str(sam_annotations_path),
            embryo_metadata_path=str(embryo_metadata_path),
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"❌ Failed to load EmbryoMetadata: {e}")
        return 0
    
    # Check GSAM ID match
    metadata_gsam_id = embryo_metadata.data.get("file_info", {}).get("gsam_annotation_id")
    
    if sam_gsam_id != metadata_gsam_id:
        if verbose:
            print(f"⚠️  GSAM ID mismatch: SAM={sam_gsam_id}, Metadata={metadata_gsam_id}")
        return 0
    
    if verbose:
        print(f"✅ GSAM ID match confirmed: {sam_gsam_id}")
    
    flags_transferred = 0

    # --- New GSAM QC format: flags section ---
    if flags_section:
        # Experiment-level
        for exp_id, exp_flags in flags_section.get("by_experiment", {}).items():
            for flag_type, flag_list in exp_flags.items():
                for flag_data in flag_list:
                    author = flag_data.get("author", "qc_transfer")
                    embryo_metadata.add_experiment_flag(exp_id, flag_type, flag_data, author)
                    flags_transferred += 1
        # Video-level
        for vid_id, vid_flags in flags_section.get("by_video", {}).items():
            for flag_type, flag_list in vid_flags.items():
                for flag_data in flag_list:
                    author = flag_data.get("author", "qc_transfer")
                    embryo_metadata.add_video_flag(vid_id, flag_type, flag_data, author)
                    flags_transferred += 1
        # Image-level
        for img_id, img_flags in flags_section.get("by_image", {}).items():
            for flag_type, flag_list in img_flags.items():
                for flag_data in flag_list:
                    author = flag_data.get("author", "qc_transfer")
                    embryo_metadata.add_image_flag(img_id, flag_type, flag_data, author)
                    flags_transferred += 1
        # Snip-level
        for snip_id, snip_flags in flags_section.get("by_snip", {}).items():
            for flag_type, flag_list in snip_flags.items():
                for flag_data in flag_list:
                    author = flag_data.get("author", "qc_transfer")
                    embryo_metadata.add_snip_flag(snip_id, flag_type, flag_data, author)
                    flags_transferred += 1
        # Embryo-level (if present)
        for emb_id, emb_flags in flags_section.get("by_embryo", {}).items():
            for flag_type, flag_list in emb_flags.items():
                for flag_data in flag_list:
                    author = flag_data.get("author", "qc_transfer")
                    embryo_metadata.add_snip_flag(emb_id, flag_type, flag_data, author)
                    flags_transferred += 1

    # --- Old GSAM QC format: hierarchical flags ---
    else:
        # Process experiment-level flags
        for exp_id, exp_data in sam_data.get("experiments", {}).items():
            exp_flags = exp_data.get("flags", {})
            for flag_type, flag_list in exp_flags.items():
                for flag_data in flag_list:
                    author = flag_data.get("author", "qc_transfer")
                    embryo_metadata.add_experiment_flag(exp_id, flag_type, flag_data, author)
                    flags_transferred += 1
        # Process video-level flags
        for exp_id, exp_data in sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                video_flags = video_data.get("flags", {})
                for flag_type, flag_list in video_flags.items():
                    for flag_data in flag_list:
                        author = flag_data.get("author", "qc_transfer")
                        embryo_metadata.add_video_flag(video_id, flag_type, flag_data, author)
                        flags_transferred += 1
        # Process image-level flags
        for exp_id, exp_data in sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    image_flags = image_data.get("flags", {})
                    for flag_type, flag_list in image_flags.items():
                        for flag_data in flag_list:
                            author = flag_data.get("author", "qc_transfer")
                            embryo_metadata.add_image_flag(image_id, flag_type, flag_data, author)
                            flags_transferred += 1
        # Process snip-level flags (stored in embryo data)
        for exp_id, exp_data in sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        embryo_flags = embryo_data.get("flags", {})
                        for flag_type, flag_list in embryo_flags.items():
                            for flag_data in flag_list:
                                author = flag_data.get("author", "qc_transfer")
                                if snip_id:
                                    embryo_metadata.add_snip_flag(snip_id, flag_type, flag_data, author)
                                    flags_transferred += 1
    
    # Save updated metadata
    try:
        embryo_metadata.save()
        if verbose:
            print(f"✅ Transferred {flags_transferred} QC flags from SAM annotations to EmbryoMetadata")
    except Exception as e:
        if verbose:
            print(f"❌ Failed to save updated metadata: {e}")
        return 0
    
    return flags_transferred

def main():
    parser = argparse.ArgumentParser(description="Export embryo masks as labeled images and transfer QC flags")
    parser.add_argument("--annotations", required=True, help="Path to grounded_sam_annotations.json")
    parser.add_argument("--embryo_metadata", required=True, help="Path to embryo_metadata.json")
    parser.add_argument("--output", required=True, help="Output directory for mask images")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], 
                       help="Output format (default: jpg, recommended: png for label masks)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🎭 Embryo Mask Export & QC Flag Transfer")
    print("=" * 50)
    
    if args.format == "jpg":
        print("⚠️  Warning: Using JPEG format for label masks may introduce compression artifacts.")
        print("   Consider using --format png for exact pixel value preservation.")
    
    # Step 1: Export masks
    print("\n📦 Step 1: Exporting embryo masks...")
    exporter = EmbryoMaskExporter(
        sam2_annotations_path=args.annotations,
        output_base_dir=args.output,
        verbose=args.verbose,
        output_format=args.format
    )
    
    export_paths = exporter.export_all_masks(max_workers=args.workers)
    print(f"✅ Exported {len(export_paths)} mask images")
    
    # Step 2: Transfer QC flags via GSAM ID matching
    print("\n🔄 Step 2: Transferring QC flags via GSAM ID...")
    try:
        flags_transferred = transfer_qc_flags_via_gsam_id(
            sam_annotations_path=Path(args.annotations),
            embryo_metadata_path=Path(args.embryo_metadata),
            verbose=args.verbose
        )
        
        if flags_transferred > 0:
            print(f"✅ Successfully transferred {flags_transferred} QC flags")
        else:
            print("ℹ️  No QC flags to transfer")
            
    except Exception as e:
        print(f"❌ QC flag transfer failed: {e}")
    
    print(f"\n📁 Output directory: {args.output}")
    print(f"📊 Updated metadata: {args.embryo_metadata}")

if __name__ == "__main__":
    main()


# Example usage:
# python scripts/05_export_embryo_masks.py \
#   --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json \
#   --embryo_metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/embryo_metadata/embryo_metadata_finetuned.json \
#   --output  /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/jpg_masks/grounded_sam_finetuned \
#   --format jpg \
#   --workers 16 \
#   --verbose
