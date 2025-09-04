#!/usr/bin/env python3
"""
Standalone Build06 script that bypasses heavy CLI imports.

Usage:
    python run_build06_standalone.py \
      --morphseq-repo-root /path/to/morphseq/repo \
      --data-root /path/to/data \
      --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Standalone Build06 runner")
    parser.add_argument("--root", required=True, help="Data root directory (contains metadata, models, latents)")
    parser.add_argument("--external-data-root", help="Optional external data root for legacy mode (defaults to MORPHSEQ_DATA_ROOT env var)")
    parser.add_argument("--model-name", default="20241107_ds_sweep01_optimum", help="Model name for embedding generation")
    parser.add_argument("--latents-tag", default=None, help="Optional name for latent output dir (defaults to --model-name)")
    parser.add_argument("--experiments", nargs="*", help="Explicit experiment list (defaults to inference from df02)")
    parser.add_argument("--generate-missing-latents", action="store_true", help="Generate missing latent files using analysis_utils")
    parser.add_argument("--use-repo-snips", action="store_true", help="Generate latents from repo's bf_embryo_snips instead of data-root")
    parser.add_argument("--export-analysis-copies", action="store_true", help="Export per-experiment df03 copies to data root")
    parser.add_argument("--train-run", help="Training run name for optional join")
    parser.add_argument("--write-train-output", action="store_true", help="Write training metadata with embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    
    args = parser.parse_args()
    
    # Resolve external data root from environment if not provided
    external_data_root = args.external_data_root
    if external_data_root is None:
        external_data_root = os.environ.get("MORPHSEQ_DATA_ROOT")
        # external_data_root can be None - that's fine for unified mode
    
    print("Loading Build06 service...")
    
    # Import only what we need
    try:
        from src.run_morphseq_pipeline.services.gen_embeddings import build_df03_with_embeddings
    except ImportError as e:
        print(f"ERROR: Could not import Build06 service: {e}")
        return 1
    
    print("Running Build06...")
    
    try:
        result_path = build_df03_with_embeddings(
            root=args.root,
            data_root=external_data_root,
            model_name=args.model_name,
            latents_tag=args.latents_tag,
            experiments=args.experiments,
            generate_missing=args.generate_missing_latents,
            use_repo_snips=args.use_repo_snips,
            export_analysis=args.export_analysis_copies,
            train_name=args.train_run,
            write_train_output=args.write_train_output,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        
        if not args.dry_run:
            print(f"\n✅ Build06 completed successfully!")
            print(f"Result: {result_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Build06 failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
