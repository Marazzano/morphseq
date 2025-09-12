#!/usr/bin/env python3
"""
Build03 CLI Wrapper: Thin wrapper around build03A_process_images SAM2 functions.

This is a concise CLI that discovers         # Use the parameterized pipeline function
        stats_df = run_build03_pipeline(
            experiment_name=exp,
            sam2_csv_path=sam2_csv,
            output_file_path=output_csv,
            root_dir=root,
            verbose=verbose
        )ls the consolidated 
build03A functions (segment_wells_sam2_csv → compile_embryo_stats_sam2),
and writes the per-experiment output CSV.

Usage:
  python run_build03.py --data-root <root> --exp <experiment_id>

Output:
  metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the consolidated build03A functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.build.build03A_process_images import segment_wells_sam2_csv, compile_embryo_stats_sam2


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Run Build03 for a single experiment using SAM2 pipeline outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True, help="Data root directory")
    p.add_argument("--exp", required=True, help="Experiment ID")
    p.add_argument("--sam2-csv", help="Override SAM2 CSV path")
    p.add_argument("--output-dir", help="Override output directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def run_build03_pipeline(experiment_name, sam2_csv_path, output_file_path, root_dir=None, verbose=False):
    """
    Thin wrapper that calls segment_wells_sam2_csv -> compile_embryo_stats_sam2
    and writes the embryo metadata to the specified output path.
    
    Args:
        experiment_name: Name of the experiment to process
        sam2_csv_path: Path to the SAM2 CSV file
        output_file_path: Full path where the output CSV should be written
        root_dir: Root directory (optional, for compatibility)
        verbose: Enable verbose logging
        
    Returns:
        pandas.DataFrame: The compiled embryo statistics
    """
    from pathlib import Path
    
    # Ensure output directory exists
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"🏃 Build03 pipeline for experiment: {experiment_name}")
        print(f"📄 SAM2 CSV: {sam2_csv_path}")
        print(f"📁 Output file: {output_file_path}")
    
    try:
        # Step 1: Segment wells from SAM2 CSV files
        if verbose:
            print("📊 Step 1: Loading SAM2 data...")
        tracked_df = segment_wells_sam2_csv(root_dir, exp_name=experiment_name, sam2_csv_path=sam2_csv_path)
        
        if verbose:
            print(f"📈 Loaded {len(tracked_df)} embryo records")
        
        # Step 2: Compile embryo statistics with QC flags
        if verbose:
            print("🔄 Step 2: Computing embryo statistics and QC...")
        stats_df = compile_embryo_stats_sam2(root_dir, tracked_df)
        
        # Step 3: Write output CSV
        if verbose:
            print(f"💾 Step 3: Writing results to {output_file_path}")
        stats_df.to_csv(output_file_path, index=False)
        
        if verbose:
            total_embryos = len(stats_df)
            usable_embryos = (stats_df["use_embryo_flag"] == "true").sum() if "use_embryo_flag" in stats_df.columns else 0
            print(f"✅ Build03 pipeline completed!")
            print(f"   📈 Total embryos: {total_embryos}")
            print(f"   ✔️  Usable embryos: {usable_embryos}")
        
        return stats_df
        
    except Exception as e:
        print(f"❌ Build03 pipeline failed: {e}")
        raise


def _discover_sam2_csv(root: Path, exp: str) -> Path:
    """Discover the SAM2 CSV file for the experiment."""
    # Try per-experiment location first
    per_exp_csv = root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp}.csv"
    if per_exp_csv.exists():
        return per_exp_csv
    
    # Fallback to root location
    root_csv = root / f"sam2_metadata_{exp}.csv"
    if root_csv.exists():
        return root_csv
    
    raise FileNotFoundError(f"SAM2 CSV not found for experiment {exp}")


def main() -> int:
    """Main entry point."""
    args = _parse_args()
    
    root = Path(args.data_root)
    exp = args.exp
    verbose = args.verbose
    
    # Discover SAM2 CSV
    sam2_csv = Path(args.sam2_csv) if args.sam2_csv else _discover_sam2_csv(root, exp)
    
    # Set output path according to per-experiment structure
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = root / "metadata" / "build03" / "per_experiment"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"expr_embryo_metadata_{exp}.csv"
    
    if verbose:
        print(f"🏃 Build03 wrapper for experiment: {exp}")
        print(f"📁 Data root: {root}")
        print(f"📄 SAM2 CSV: {sam2_csv}")
        print(f"� Output CSV: {output_csv}")
    
    # Check if output exists and handle overwrite
    if output_csv.exists() and not args.overwrite:
        print(f"ℹ️  Output exists, use --overwrite to replace: {output_csv}")
        return 0
    
    try:
        # Step 1: Segment wells using SAM2 CSV
        if verbose:
            print("� Step 1: Loading SAM2 data...")
        tracked_df = segment_wells_sam2_csv(root, exp_name=exp, sam2_csv_path=sam2_csv)
        
        if verbose:
            print(f"📊 Loaded {len(tracked_df)} embryo records")
        
        # Step 2: Compile embryo statistics with full QC
        if verbose:
            print("🔄 Step 2: Computing embryo statistics and QC...")
        stats_df = compile_embryo_stats_sam2(root, tracked_df)
        
        # Step 3: Write output CSV
        if verbose:
            print(f"💾 Step 3: Writing results to {output_csv}")
        stats_df.to_csv(output_csv, index=False)
        
        # Summary
        total_embryos = len(stats_df)
        usable_embryos = (stats_df["use_embryo_flag"] == "true").sum() if "use_embryo_flag" in stats_df.columns else 0
        
        print(f"✅ Build03 complete for {exp}")
        print(f"   📈 Total embryos: {total_embryos}")
        print(f"   ✔️  Usable embryos: {usable_embryos}")
        print(f"   📁 Output: {output_csv}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Build03 failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
