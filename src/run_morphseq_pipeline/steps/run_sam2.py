#!/usr/bin/env python3
"""
SAM2 Pipeline Wrapper for MorphSeq CLI Integration

This module provides Python-based orchestration of the segmentation_sandbox SAM2 pipeline,
eliminating the need for manual bash script execution. It runs the complete SAM2 workflow
from video preparation through mask export and metadata CSV generation.

CRITICAL PATH STRUCTURE:
- Repository Root: /net/trapnell/vol1/home/mdcolon/proj/morphseq
- Data Root: morphseq_playground/ (for testing; in production will be nlammers data directory) 
- SAM2 Scripts: <repo_root>/segmentation_sandbox/scripts/ (executable scripts)
- SAM2 Data Output: <data_root>/sam2_pipeline_files/ (all SAM2 generated data)

Pipeline Stages:
1. 01_prepare_videos.py - Video preparation and metadata initialization
2. 03_gdino_detection.py - GroundingDINO object detection  
3. 04_sam2_video_processing.py - SAM2 video segmentation
4. 05_sam2_qc_analysis.py - Quality control analysis
5. 06_export_masks.py - Mask export to PNG files
6. export_sam2_metadata_to_csv.py - Per-experiment CSV generation

Data Flow:
- Input: Stitched images from Build01 at <data_root>/built_image_data/stitched_FF_images/
- Output: SAM2 metadata CSV at <data_root>/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv
- Environment: Sets MORPHSEQ_SANDBOX_MASKS_DIR=<data_root>/sam2_pipeline_files/exported_masks
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import yaml
import shutil


def run_sam2(
    root: str | Path,
    exp: str,
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.5, 
    target_prompt: str = "individual embryo",
    workers: int = 8,
    device: str = "cuda",
    segmentation_format: str = "rle",
    save_interval: int = 10,
    dry_run: bool = False,
    verbose: bool = False,
    **kwargs
) -> Path:
    """Run SAM2 segmentation pipeline for a single experiment.
    
    Args:
        root: Data root directory (e.g., 'morphseq_playground')
        exp: Experiment name (e.g., '20250529_24hpf_ctrl_atf6')
        confidence_threshold: GroundingDINO confidence threshold (default: 0.45)
        iou_threshold: GroundingDINO IoU threshold (default: 0.5)
        target_prompt: SAM2 detection prompt (default: "individual embryo")
        workers: Number of parallel workers (default: 8)
        device: Device for SAM2 model - "cuda" or "cpu" (default: "cuda")
        segmentation_format: Output format - "rle" or "polygon" (default: "rle")
        save_interval: Auto-save interval for processing (default: 10)
        dry_run: Show commands without executing (default: False)
        verbose: Enable verbose logging (default: False)
        **kwargs: Additional parameters passed to pipeline scripts
        
    Returns:
        Path to generated SAM2 metadata CSV file
        
    Raises:
        FileNotFoundError: If stitched images or sandbox scripts not found
        RuntimeError: If any pipeline stage fails
    """
    root = Path(root)
    
    # Validate data inputs
    stitched_dir = root / "built_image_data" / "stitched_FF_images" / exp
    if not stitched_dir.exists():
        raise FileNotFoundError(f"Stitched images not found: {stitched_dir}")
    
    # Find repository root (contains segmentation_sandbox/)
    repo_root = _find_repo_root(root)
    sandbox_dir = repo_root / "segmentation_sandbox"
    if not sandbox_dir.exists():
        raise FileNotFoundError(f"Segmentation sandbox not found at {sandbox_dir}")
    
    scripts_dir = sandbox_dir / "scripts" / "pipelines"
    utils_dir = sandbox_dir / "scripts" / "utils"
    
    # Set up SAM2 data structure in data root
    sam2_root = root / "sam2_pipeline_files"
    sam2_root.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable for mask paths (data root, not repo root)
    env = os.environ.copy()
    env["MORPHSEQ_SANDBOX_MASKS_DIR"] = str(sam2_root / "exported_masks")
    
    print(f"🚀 Starting SAM2 pipeline for experiment: {exp}")
    print(f"📁 Data root: {root}")
    print(f"📁 Repo root: {repo_root}")
    print(f"🛠️ Scripts: {scripts_dir}")
    print(f"💾 SAM2 data: {sam2_root}")
    
    if dry_run:
        print("🔍 DRY RUN: Would execute SAM2 pipeline with these settings:")
        print(f"  • Confidence threshold: {confidence_threshold}")
        print(f"  • IoU threshold: {iou_threshold}")
        print(f"  • Target prompt: '{target_prompt}'")
        print(f"  • Workers: {workers}")
        print(f"  • Device: {device}")
        print(f"  • Segmentation format: {segmentation_format}")
        return sam2_root / "sam2_expr_files" / f"sam2_metadata_{exp}.csv"
    
    try:
        # Create temporary config file for this run with model paths
        config_data = {
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "target_prompt": target_prompt,
            "workers": workers,
            "experiment": exp,
            "data_root": str(root),
            "device": device,
            "segmentation_format": segmentation_format,
            "save_interval": save_interval,
            "models": {
                "groundingdino": {
                    "config": "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "weights": "/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/Open-GroundingDino/finetune_output/finetune_output_run_nick_masks_20250308/checkpoint_best_regular.pth"
                },
                "sam2": {
                    "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
                    "checkpoint": "../checkpoints/sam2.1_hiera_large.pt"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        # Stage 1: Prepare Videos
        print("📹 Stage 1: Preparing videos and metadata...")
        stitched_images_dir = root / "built_image_data" / "stitched_FF_images"
        stage1_args = [
            "--directory_with_experiments", str(stitched_images_dir.absolute()),
            "--output_parent_dir", str(sam2_root.absolute()),
            "--experiments_to_process", exp,
            "--workers", str(workers)
        ]
        if verbose:
            stage1_args.append("--verbose")
            
        result = _run_pipeline_script(
            script_path=scripts_dir / "01_prepare_videos.py",
            args=stage1_args,
            env=env,
            cwd=sandbox_dir
        )
        
        # Stage 2: GroundingDINO Detection  
        print("🔍 Stage 2: Running GroundingDINO detection...")
        metadata_path = sam2_root / "raw_data_organized" / "experiment_metadata.json"
        annotations_path = sam2_root / "detections" / "gdino_detections.json"
        
        # Ensure directories exist
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        
        stage2_args = [
            "--config", config_path,
            "--metadata", str(metadata_path.absolute()),
            "--annotations", str(annotations_path.absolute()),
            "--confidence-threshold", str(confidence_threshold),
            "--iou-threshold", str(iou_threshold),
            "--prompt", target_prompt
        ]
        if verbose:
            stage2_args.append("--verbose")
        
        result = _run_pipeline_script(
            script_path=scripts_dir / "03_gdino_detection.py", 
            args=stage2_args,
            env=env,
            cwd=sandbox_dir
        )
        
        # Stage 3: SAM2 Video Processing
        print("🎯 Stage 3: SAM2 video segmentation...")
        sam2_output_path = sam2_root / "segmentation" / "grounded_sam_segmentations.json"
        sam2_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sam2_args = [
            "--config", config_path,
            "--metadata", str(metadata_path.absolute()), 
            "--annotations", str(annotations_path.absolute()),
            "--output", str(sam2_output_path.absolute()),
            "--target-prompt", target_prompt,
            "--segmentation-format", segmentation_format,
            "--device", device,
            "--save-interval", str(save_interval)
        ]
        if verbose:
            sam2_args.append("--verbose")
            
        result = _run_pipeline_script(
            script_path=scripts_dir / "04_sam2_video_processing.py",
            args=sam2_args,
            env=env,
            cwd=sandbox_dir
        )
        
        # Stage 4: QC Analysis
        print("📊 Stage 4: Quality control analysis...")
        qc_args = [
            "--input", str(sam2_output_path.absolute()),
            "--experiments", exp
        ]
        if verbose:
            qc_args.append("--verbose")
            
        result = _run_pipeline_script(
            script_path=scripts_dir / "05_sam2_qc_analysis.py",
            args=qc_args,
            env=env,
            cwd=sandbox_dir
        )
        
        # Stage 5: Export Masks
        print("💾 Stage 5: Exporting masks...")
        masks_output_dir = sam2_root / "exported_masks"
        masks_output_dir.mkdir(parents=True, exist_ok=True)
        
        export_args = [
            "--sam2-annotations", str(sam2_output_path.absolute()),
            "--output", str(masks_output_dir.absolute()),
            "--entities-to-process", exp
        ]
        if verbose:
            export_args.append("--verbose")
        
        result = _run_pipeline_script(
            script_path=scripts_dir / "06_export_masks.py",
            args=export_args,
            env=env,
            cwd=sandbox_dir
        )
        
        # Stage 6: Export Metadata CSV
        print("📋 Stage 6: Generating metadata CSV...")
        csv_output_dir = sam2_root / "sam2_expr_files"
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        csv_output_path = csv_output_dir / f"sam2_metadata_{exp}.csv"
        
        csv_args = [
            str(sam2_output_path.absolute()),
            "-o", str(csv_output_path.absolute()),
            "--experiment-filter", exp,
            "--masks-dir", str((masks_output_dir / exp / "masks").absolute())
        ]
        if verbose:
            csv_args.append("-v")
        
        result = _run_pipeline_script(
            script_path=utils_dir / "export_sam2_metadata_to_csv.py",
            args=csv_args,
            env=env,
            cwd=sandbox_dir
        )
        
        # Cleanup
        os.unlink(config_path)
        
        # Validate output
        if not csv_output_path.exists():
            raise RuntimeError(f"SAM2 metadata CSV not generated: {csv_output_path}")
        
        print(f"✅ SAM2 pipeline completed successfully!")
        print(f"📄 Generated CSV: {csv_output_path}")
        print(f"🎭 Exported masks: {masks_output_dir}")
        
        return csv_output_path
        
    except Exception as e:
        print(f"❌ SAM2 pipeline failed: {e}")
        # Cleanup config file on error
        if 'config_path' in locals():
            try:
                os.unlink(config_path)
            except:
                pass
        raise RuntimeError(f"SAM2 pipeline failed: {e}") from e


def _find_repo_root(data_root: Path) -> Path:
    """Find repository root containing segmentation_sandbox.
    
    Logic:
    1. If data_root contains segmentation_sandbox/, it's the repo root
    2. Otherwise, look in parent directories  
    3. For testing with morphseq_playground/, repo root is parent
    
    Args:
        data_root: Data directory path
        
    Returns:
        Repository root path
        
    Raises:
        FileNotFoundError: If repo root cannot be found
    """
    # Case 1: data_root is repo root (contains segmentation_sandbox)
    if (data_root / "segmentation_sandbox").exists():
        return data_root
    
    # Case 2: data_root is subdirectory of repo (e.g., morphseq_playground/)
    current = data_root.parent
    while current != current.parent:  # Not at filesystem root
        if (current / "segmentation_sandbox").exists():
            return current
        current = current.parent
    
    # Case 3: Look in known locations for testing
    known_locations = [
        Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq"),
        Path.cwd(),
        Path.cwd().parent
    ]
    
    for location in known_locations:
        if (location / "segmentation_sandbox").exists():
            return location
    
    raise FileNotFoundError(
        f"Repository root with segmentation_sandbox/ not found. "
        f"Searched from {data_root} and known locations: {known_locations}"
    )


def _run_pipeline_script(
    script_path: Path,
    args: list[str],
    env: dict,
    cwd: Path
) -> subprocess.CompletedProcess:
    """Run a pipeline script with error handling.
    
    Args:
        script_path: Path to Python script
        args: Command line arguments
        env: Environment variables
        cwd: Working directory (should be sandbox_dir for proper imports)
        
    Returns:
        Completed subprocess result
        
    Raises:
        RuntimeError: If script fails
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")
    
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"🔧 Running: {script_path.name}")
    print(f"📂 Working directory: {cwd}")
    print(f"⚙️ Arguments: {' '.join(args)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print stdout for progress tracking
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"📝 {line}")
            
        return result
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Pipeline script failed: {script_path.name}\n"
        error_msg += f"Return code: {e.returncode}\n"
        if e.stdout:
            error_msg += f"STDOUT:\n{e.stdout}\n"
        if e.stderr:
            error_msg += f"STDERR:\n{e.stderr}\n"
        raise RuntimeError(error_msg) from e


def run_sam2_batch(
    root: str | Path,
    experiments: Optional[list[str]] = None,
    **sam2_kwargs
) -> Dict[str, Path]:
    """Run SAM2 pipeline for multiple experiments in batch mode.
    
    Args:
        root: Data root directory
        experiments: List of experiment names (if None, auto-discover)
        **sam2_kwargs: Arguments passed to run_sam2()
        
    Returns:
        Dictionary mapping experiment names to CSV file paths
    """
    root = Path(root)
    
    if experiments is None:
        # Auto-discover experiments from stitched images directory
        stitched_root = root / "built_image_data" / "stitched_FF_images"
        if not stitched_root.exists():
            raise FileNotFoundError(f"Stitched images directory not found: {stitched_root}")
        
        experiments = [d.name for d in stitched_root.iterdir() if d.is_dir()]
        
    if not experiments:
        raise ValueError("No experiments found for SAM2 processing")
    
    print(f"🔄 Running SAM2 pipeline for {len(experiments)} experiments")
    print(f"📋 Experiments: {', '.join(experiments)}")
    
    results = {}
    failed = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"{'='*60}")
        print(f"🔄 Processing experiment {i}/{len(experiments)}: {exp}")
        print(f"{'='*60}")
        
        try:
            csv_path = run_sam2(root=root, exp=exp, **sam2_kwargs)
            results[exp] = csv_path
            print(f"✅ Completed: {exp} → {csv_path}")
        except Exception as e:
            print(f"❌ Failed: {exp} → {e}")
            failed.append((exp, str(e)))
            
    print(f"🏁 Batch processing complete!")
    print(f"✅ Successful: {len(results)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print(f"💥 Failed experiments:")
        for exp, error in failed:
            print(f"  • {exp}: {error}")
    
    return results


def main():
    """Command-line interface for SAM2 pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run SAM2 segmentation pipeline for MorphSeq experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python run_sam2.py --data-root morphseq_playground --exp 20250529_24hpf_ctrl_atf6
  
  # Batch mode (auto-discover experiments)
  python run_sam2.py --data-root morphseq_playground --batch
  
  # Custom parameters
  python run_sam2.py --data-root morphseq_playground --exp 20250529_24hpf_ctrl_atf6 
    --confidence-threshold 0.35 --workers 12 --verbose
  
  # Dry run
  python run_sam2.py --data-root morphseq_playground --exp 20250529_24hpf_ctrl_atf6 --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument("--data-root", required=True,
                        help="Data root directory (contains built_image_data/)")
    
    # Experiment selection (mutually exclusive)
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument("--exp", help="Single experiment name")
    exp_group.add_argument("--batch", action="store_true", 
                          help="Process all experiments (auto-discover)")
    exp_group.add_argument("--experiments", nargs="+",
                          help="List of specific experiments to process")
    
    # SAM2 parameters
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                        help="GroundingDINO confidence threshold (default: 0.45)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="GroundingDINO IoU threshold (default: 0.5)")
    parser.add_argument("--target-prompt", default="individual embryo",
                        help="SAM2 detection prompt (default: 'individual embryo')")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Device for SAM2 model (default: cuda)")
    parser.add_argument("--segmentation-format", choices=["rle", "polygon"], default="rle",
                        help="Segmentation output format (default: rle)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Auto-save interval for processing (default: 10)")
    
    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Prepare kwargs for run_sam2
    sam2_kwargs = {
        "confidence_threshold": args.confidence_threshold,
        "iou_threshold": args.iou_threshold,
        "target_prompt": args.target_prompt,
        "workers": args.workers,
        "device": args.device,
        "segmentation_format": args.segmentation_format,
        "save_interval": args.save_interval,
        "dry_run": args.dry_run,
        "verbose": args.verbose
    }
    
    try:
        if args.exp:
            # Single experiment mode
            csv_path = run_sam2(root=args.data_root, exp=args.exp, **sam2_kwargs)
            print(f"🎉 SAM2 pipeline completed!")
            print(f"📄 Generated CSV: {csv_path}")
            
        elif args.batch:
            # Batch mode (auto-discover)
            results = run_sam2_batch(root=args.data_root, **sam2_kwargs)
            print(f"🎉 Batch processing completed!")
            print(f"📄 Generated {len(results)} CSV files")
            
        elif args.experiments:
            # Batch mode (specific experiments)
            results = run_sam2_batch(root=args.data_root, experiments=args.experiments, **sam2_kwargs)
            print(f"🎉 Batch processing completed!")
            print(f"📄 Generated {len(results)} CSV files")
            
    except Exception as e:
        print(f"❌ SAM2 pipeline failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())


def run_sam2_batch(
    root: str | Path,
    experiments: Optional[list[str]] = None,
    **sam2_kwargs
) -> Dict[str, Path]:
    """Run SAM2 pipeline for multiple experiments in batch mode.
    
    Args:
        root: Data root directory
        experiments: List of experiment names (if None, auto-discover)
        **sam2_kwargs: Arguments passed to run_sam2()
        
    Returns:
        Dictionary mapping experiment names to CSV file paths
    """
    root = Path(root)
    
    if experiments is None:
        # Auto-discover experiments from stitched images directory
        stitched_root = root / "built_image_data" / "stitched_FF_images"
        if not stitched_root.exists():
            raise FileNotFoundError(f"Stitched images directory not found: {stitched_root}")
        
        experiments = [d.name for d in stitched_root.iterdir() if d.is_dir()]
        
    if not experiments:
        raise ValueError("No experiments found for SAM2 processing")
    
    print(f"🔄 Running SAM2 pipeline for {len(experiments)} experiments")
    print(f"📋 Experiments: {', '.join(experiments)}")
    
    results = {}
    failed = []
    
    for exp in experiments:
        try:
            print(f"\n{'='*60}")
            print(f"🧪 Processing experiment: {exp}")
            print(f"{'='*60}")
            
            csv_path = run_sam2(root=root, exp=exp, **sam2_kwargs)
            results[exp] = csv_path
            print(f"✅ {exp}: SUCCESS → {csv_path}")
            
        except Exception as e:
            failed.append((exp, str(e)))
            print(f"❌ {exp}: FAILED → {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"📊 BATCH RESULTS: {len(results)} succeeded, {len(failed)} failed")
    print(f"{'='*60}")
    
    if results:
        print("✅ Successful experiments:")
        for exp, csv_path in results.items():
            print(f"  - {exp}: {csv_path}")
    
    if failed:
        print("❌ Failed experiments:")
        for exp, error in failed:
            print(f"  - {exp}: {error}")
    
    return results