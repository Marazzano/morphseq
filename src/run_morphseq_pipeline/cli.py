#!/usr/bin/env python3
"""
Centralized MorphSeq pipeline runner.

Usage examples:
    python -m src.run_morphseq_pipeline.cli build03 --data-root /data/morphseq --exp 20250612_30hpf_ctrl_atf6 \
        --sam2-csv /data/morphseq/sam2_metadata_20250612_30hpf_ctrl_atf6.csv --by-embryo 5 --frames-per-embryo 3

    python -m src.run_morphseq_pipeline.cli build04 --data-root /data/morphseq
    python -m src.run_morphseq_pipeline.cli build05 --data-root /data/morphseq --train-name train_ff_20250612

Build01/02 orchestration is provided via thin wrappers to existing build scripts.
SAM2 segmentation is now integrated as a first-class CLI citizen with the 'sam2' subcommand.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import json

from .steps.run_build03 import run_build03
from .steps.run_build04 import run_build04
from .steps.run_build05 import run_build05
from .steps.run_build01 import run_build01
from .steps.run_build_combine_metadata import run_combine_metadata
from .steps.run_build02 import run_build02
from .steps.run_sam2 import run_sam2, run_sam2_batch
from .validation import run_validation
from .steps.run_embed import run_embed
from .steps.run_build06 import run_build06
from .services.gen_embeddings import build_df03_with_embeddings
# Import centralized embedding generation utilities
from ..analyze.gen_embeddings import ensure_embeddings_for_experiments


MISCROSCOPE_CHOICES = ["Keyence", "YX1"]

def resolve_root(args) -> str:
    """Resolve the data root path from CLI args, with test suffix support.

    Prefers `--data-root`; falls back to legacy `--root` for compatibility.
    WARNING: --test-suffix creates a subdirectory under the resolved root for isolation.
    """
    root_value = getattr(args, 'data_root', None) or getattr(args, 'root', None)
    if root_value is None:
        raise SystemExit("ERROR: --data-root not provided (and legacy --root missing)")
    root = Path(root_value)
    if hasattr(args, 'test_suffix') and args.test_suffix:
        # Create subdirectory under root for isolation
        root = (root / args.test_suffix)
        root.mkdir(parents=True, exist_ok=True)
        print(f"📁 Using test root: {root}")
    return str(root)


def _add_common_root_and_exp(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data-root", required=True, help="Project root (contains built_image_data/, metadata/, training_data/)")
    ap.add_argument("--test-suffix", help="Append suffix to root for test isolation (e.g., test_sam2_20250830). WARNING: Creates directory outside root path, may cause permission errors.")
    ap.add_argument("--exp", required=False, help="Experiment name (e.g., 20250612_30hpf_ctrl_atf6)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="morphseq-runner", description="Centralized MorphSeq pipeline runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build01
    p01 = sub.add_parser("build01", help="Compile+stitch FF images; write built metadata CSV")
    _add_common_root_and_exp(p01)
    p01.add_argument("--microscope", choices= MISCROSCOPE_CHOICES, required=True)
    p01.add_argument("--metadata-only", action="store_true", help="Skip image processing; write metadata only")
    p01.add_argument("--overwrite", action="store_true")

    # combine (master well metadata)
    pc = sub.add_parser("combine-metadata", help="Create master well metadata (experiment + built + well xlsx)")
    pc.add_argument("--root", required=True)

    # build02
    p02 = sub.add_parser("build02", help="Legacy segmentation (optional if using SAM2)")
    p02.add_argument("--data-root", required=True)
    p02.add_argument("--mode", choices=["legacy", "skip"], default="legacy")
    p02.add_argument("--model-name", default="mask_v1_0050", help="Segmentation model name (legacy)")
    p02.add_argument("--n-classes", type=int, default=2)
    p02.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (0=single-threaded)")
    p02.add_argument("--overwrite", action="store_true")

    # sam2
    p_sam2 = sub.add_parser("sam2", help="Run SAM2 segmentation pipeline")
    _add_common_root_and_exp(p_sam2)
    p_sam2.add_argument("--confidence-threshold", type=float, default=0.45, 
                       help="GroundingDINO confidence threshold (default: 0.45)")
    p_sam2.add_argument("--iou-threshold", type=float, default=0.5,
                       help="GroundingDINO IoU threshold (default: 0.5)")
    p_sam2.add_argument("--target-prompt", default="individual embryo",
                       help="SAM2 detection prompt (default: 'individual embryo')")
    p_sam2.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    p_sam2.add_argument("--batch", action="store_true",
                       help="Batch mode: process all experiments in data-root (ignores --exp)")

    # build03
    p03 = sub.add_parser("build03", help="Build03A using SAM2 bridge CSV or legacy tracked metadata")
    _add_common_root_and_exp(p03)
    p03.add_argument("--sam2-csv", help="Path to sam2_metadata_{exp}.csv (if absent, uses legacy segment_wells)")
    p03.add_argument("--by-embryo", type=int, help="Sample this many embryos")
    p03.add_argument("--frames-per-embryo", type=int, help="Sample this many frames per embryo")
    p03.add_argument("--max-samples", type=int, help="Cap total rows")
    p03.add_argument("--n-workers", type=int, default=1)
    p03.add_argument("--df01-out", help="Path to write embryo_metadata_df01.csv",
                    default="metadata/combined_metadata_files/embryo_metadata_df01.csv")

    # build04
    p04 = sub.add_parser("build04", help="QC + stage inference")
    p04.add_argument("--data-root", required=True)
    # Accept --exp for interface consistency with other steps; currently unused by build04
    p04.add_argument("--exp", required=False, help="Experiment name (accepted for consistency; ignored by build04)")
    p04.add_argument("--dead-lead-time", type=int, default=2)
    p04.add_argument("--pert-key", help="Path to perturbation_name_key.csv (overrides default datroot/metadata path)")
    p04.add_argument("--no-auto-augment-pert-key", dest="auto_augment_pert_key", action="store_false",
                     help="Disable auto-adding missing perturbations to key with unknown defaults")
    p04.add_argument("--write-augmented-key", action="store_true",
                     help="Write back augmented key to the provided path (non-destructive append)")

    # build05
    p05 = sub.add_parser("build05", help="Create training snips/folders from df02 + snips")
    p05.add_argument("--data-root", required=True)
    p05.add_argument("--train-name", required=True)
    p05.add_argument("--label-var", default=None)
    p05.add_argument("--rs-factor", type=float, default=1.0)
    p05.add_argument("--overwrite", action="store_true")

    # e2e
    pe2e = sub.add_parser("e2e", help="Run Build01→Build02→SAM2→Build03→Build04→Build05")
    _add_common_root_and_exp(pe2e)
    pe2e.add_argument("--sam2-csv", help="Path to SAM2 CSV (overrides auto-discovery)")
    pe2e.add_argument("--by-embryo", type=int)
    pe2e.add_argument("--frames-per-embryo", type=int)
    pe2e.add_argument("--max-samples", type=int)
    pe2e.add_argument("--n-workers", type=int, default=1)
    pe2e.add_argument("--train-name", required=True)
    
    # Pipeline step control
    pe2e.add_argument("--skip-build01", action="store_true", help="Skip Build01 (image stitching)")
    pe2e.add_argument("--skip-build02", action="store_true", help="Skip Build02 (QC masks)")
    pe2e.add_argument("--run-sam2", action="store_true", help="Run SAM2 segmentation pipeline")
    pe2e.add_argument("--skip-build03", action="store_true", help="Skip Build03 (embryo processing)")
    pe2e.add_argument("--skip-build04", action="store_true", help="Skip Build04 (QC + staging)")
    pe2e.add_argument("--skip-build05", action="store_true", help="Skip Build05 (training snips)")
    
    # SAM2 parameters (only used if --run-sam2 is set)
    pe2e.add_argument("--sam2-confidence", type=float, default=0.45, 
                     help="SAM2 GroundingDINO confidence threshold (default: 0.45)")
    pe2e.add_argument("--sam2-iou", type=float, default=0.5,
                     help="SAM2 GroundingDINO IoU threshold (default: 0.5)")
    pe2e.add_argument("--sam2-prompt", default="individual embryo",
                     help="SAM2 detection prompt (default: 'individual embryo')")
    pe2e.add_argument("--sam2-workers", type=int, default=8,
                     help="SAM2 parallel workers (default: 8)")
    
    # Build01 parameters (only used if not --skip-build01)
    pe2e.add_argument("--microscope", choices= MISCROSCOPE_CHOICES, 
                     help="Microscope type for Build01 (required if not skipping Build01)")
    pe2e.add_argument("--metadata-only", action="store_true", 
                     help="Build01: skip image processing, write metadata only")
    
    # Build02 parameters (only used if not --skip-build02)
    pe2e.add_argument("--build02-num-workers", type=int, default=0,
                     help="Build02: Number of DataLoader workers (default: 0=single-threaded)")
    
    pe2e.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing outputs")

    # validate
    pv = sub.add_parser("validate", help="Run validation gates (schema, units, paths)")
    pv.add_argument("--data-root", required=True)
    pv.add_argument("--exp", required=False)
    pv.add_argument("--df01", default="metadata/combined_metadata_files/embryo_metadata_df01.csv")
    pv.add_argument("--checks", default="schema,units,paths")

    # embed
    pem = sub.add_parser("embed", help="Generate morphological embeddings for training snips")
    pem.add_argument("--data-root", required=True)
    pem.add_argument("--train-name", required=True)
    pem.add_argument("--model-dir", required=False, help="Path to model or its parent (for real embeddings)")
    pem.add_argument("--out-csv", required=False)
    pem.add_argument("--batch-size", type=int, default=64)
    pem.add_argument("--simulate", action="store_true")
    pem.add_argument("--latent-dim", type=int, default=16)
    pem.add_argument("--seed", type=int, default=0)

    # build06 (standardize embeddings + df03 merge)
    p06 = sub.add_parser("build06", help="Generate df03 with quality-filtered embeddings (skips Build05)")
    p06.add_argument("--morphseq-repo-root", required=True, help="MorphSeq repository root directory")
    p06.add_argument("--data-root", required=True, 
                     help="Data root directory containing models/ and metadata/ (REQUIRED for model access)")
    p06.add_argument("--model-name", default="20241107_ds_sweep01_optimum", 
                     help="Model name for embedding generation")
    
    # Standardized experiment selection (following segmentation_sandbox patterns)
    p06.add_argument("--experiments", help="Comma-separated experiment IDs (default: auto-discover from df02)")
    p06.add_argument("--entities_to_process", dest="experiments", 
                     help="[Alias] Comma-separated experiment IDs")
    
    # Processing mode controls
    p06.add_argument("--process-missing", action="store_true", default=True,
                     help="Process only experiments missing from df03 (default)")
    p06.add_argument("--generate-missing-latents", action="store_true", default=True,
                     help="Generate missing latent files [REDUNDANT with --process-missing, kept for CLI standardization]")
    p06.add_argument("--py39-env", default="/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster",
                     help="Python 3.9 environment path for legacy model compatibility")
    p06.add_argument("--overwrite", action="store_true",
                     help="Force reprocess - REQUIRES --experiments specification (use 'all' for everything)")
    
    # Optional outputs
    p06.add_argument("--export-analysis-copies", action="store_true",
                     help="Export per-experiment df03 copies to data root")
    p06.add_argument("--train-run", help="Training run name for optional join")
    p06.add_argument("--write-train-output", action="store_true",
                     help="Write training metadata with embeddings")
    p06.add_argument("--dry-run", action="store_true",
                     help="Print planned actions without executing")

    # status (read-only tracking view)
    p_status = sub.add_parser("status", help="Show pipeline status for experiments and global files")
    p_status.add_argument("--data-root", required=True)
    p_status.add_argument("--experiments", help="Comma-separated experiment IDs to show (default: all)")
    p_status.add_argument("--verbose", action="store_true", help="Show detailed status (e.g., QC 3/5)")
    p_status.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    p_status.add_argument("--model-name", default="20241107_ds_sweep01_optimum",
                          help="Model name for latent check (default: 20241107_ds_sweep01_optimum)")

    # pipeline (orchestrated execution)
    p_pipe = sub.add_parser("pipeline", help="Orchestrated pipeline execution")
    p_pipe.add_argument("--data-root", required=True)
    p_pipe.add_argument("action", choices=["e2e", "sam2", "build03", "build04", "build06"])
    p_pipe.add_argument("--experiments", help="Comma-separated experiment IDs")
    p_pipe.add_argument("--later-than", type=int, help="Process experiments after YYYYMMDD")
    p_pipe.add_argument("--force", action="store_true", help="Force rerun even if not needed")
    p_pipe.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    p_pipe.add_argument("--model-name", default="20241107_ds_sweep01_optimum",
                        help="Model name for embedding generation")
    # SAM2 parameters
    p_pipe.add_argument("--sam2-workers", type=int, default=8, help="SAM2 parallel workers")
    p_pipe.add_argument("--sam2-confidence", type=float, default=0.45, help="SAM2 confidence threshold")
    p_pipe.add_argument("--sam2-iou", type=float, default=0.5, help="SAM2 IoU threshold")
    # Build03 parameters
    p_pipe.add_argument("--by-embryo", type=int, help="Sample this many embryos")
    p_pipe.add_argument("--frames-per-embryo", type=int, help="Sample this many frames per embryo")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "build01":
        run_build01(root=resolve_root(args), exp=args.exp, microscope=args.microscope,
                    metadata_only=args.metadata_only, overwrite=args.overwrite)

    elif args.cmd == "combine-metadata":
        run_combine_metadata(root=resolve_root(args))

    elif args.cmd == "build02":
        run_build02(root=resolve_root(args), mode=args.mode, model_name=args.model_name,
                    n_classes=args.n_classes, num_workers=args.num_workers, overwrite=args.overwrite)

    elif args.cmd == "sam2":
        root = resolve_root(args)
        if args.batch:
            # Batch mode: process all experiments
            print("🔄 Running SAM2 in batch mode")
            results = run_sam2_batch(
                root=root,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                target_prompt=args.target_prompt,
                workers=args.workers
            )
            print(f"✅ Batch SAM2 completed: {len(results)} experiments processed")
        else:
            # Single experiment mode
            if not args.exp:
                raise SystemExit("--exp is required for sam2 (or use --batch for all experiments)")
            print(f"🚀 Running SAM2 for experiment: {args.exp}")
            csv_path = run_sam2(
                root=root,
                exp=args.exp,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                target_prompt=args.target_prompt,
                workers=args.workers
            )
            print(f"✅ SAM2 completed: {csv_path}")

    elif args.cmd == "build03":
        if not args.exp:
            raise SystemExit("--exp is required for build03")
        run_build03(
            root=resolve_root(args),
            exp=args.exp,
            sam2_csv=args.sam2_csv,
            by_embryo=args.by_embryo,
            frames_per_embryo=args.frames_per_embryo,
            max_samples=args.max_samples,
            n_workers=args.n_workers,
            df01_out=args.df01_out,
        )

    elif args.cmd == "build04":
        run_build04(
            root=resolve_root(args),
            dead_lead_time=args.dead_lead_time,
            pert_key_path=args.pert_key,
            auto_augment_pert_key=getattr(args, 'auto_augment_pert_key', True),
            write_augmented_key=args.write_augmented_key,
        )

    elif args.cmd == "build05":
        run_build05(root=resolve_root(args), train_name=args.train_name,
                    label_var=args.label_var, rs_factor=args.rs_factor,
                    overwrite=args.overwrite)

    elif args.cmd == "e2e":
        if not args.exp:
            raise SystemExit("--exp is required for e2e")
        
        root = resolve_root(args)
        
        print("🚀 Starting End-to-End MorphSeq Pipeline")
        print(f"📁 Data root: {root}")
        print(f"🧪 Experiment: {args.exp}")
        print(f"🏷️ Training name: {args.train_name}")
        
        # Build01: Image stitching  
        if not args.skip_build01:
            if not args.microscope:
                raise SystemExit("--microscope is required for Build01 (or use --skip-build01)")
            print("\n" + "="*60)
            print("📹 STEP 1: Build01 - Image stitching and metadata")
            print("="*60)
            run_build01(
                root=root, 
                exp=args.exp, 
                microscope=args.microscope,
                metadata_only=args.metadata_only, 
                overwrite=args.overwrite
            )
        else:
            print("\n⏭️  Skipping Build01 (image stitching)")
            
        # Build02: Complete QC mask generation
        if not args.skip_build02:
            print("\n" + "="*60)
            print("🎭 STEP 2: Build02 - Complete QC mask suite (5 UNets)")
            print("="*60)
            run_build02(
                root=root, 
                mode="legacy",  # Run all 5 models
                num_workers=args.build02_num_workers,
                overwrite=args.overwrite
            )
        else:
            print("\n⏭️  Skipping Build02 (QC masks)")
            
        # SAM2: Superior embryo segmentation
        if args.run_sam2:
            print("\n" + "="*60)
            print("🎯 STEP 3: SAM2 - Superior embryo segmentation")
            print("="*60)
            csv_path = run_sam2(
                root=root,
                exp=args.exp,
                confidence_threshold=args.sam2_confidence,
                iou_threshold=args.sam2_iou,
                target_prompt=args.sam2_prompt,
                workers=args.sam2_workers
            )
            print(f"✅ SAM2 completed: {csv_path}")
        else:
            print("\n⏭️  Skipping SAM2 segmentation")
            
        # Build03: Hybrid mask processing 
        if not args.skip_build03:
            print("\n" + "="*60)
            print("🔬 STEP 4: Build03 - Embryo processing (hybrid masks)")
            print("="*60)
            run_build03(
                root=root, 
                exp=args.exp, 
                sam2_csv=args.sam2_csv,  # Will auto-discover if None
                by_embryo=args.by_embryo, 
                frames_per_embryo=args.frames_per_embryo,
                max_samples=args.max_samples, 
                n_workers=args.n_workers
            )
        else:
            print("\n⏭️  Skipping Build03 (embryo processing)")
            
        # Build04: QC and stage inference
        if not args.skip_build04:
            print("\n" + "="*60) 
            print("📊 STEP 5: Build04 - QC analysis and stage inference")
            print("="*60)
            run_build04(root=root)
        else:
            print("\n⏭️  Skipping Build04 (QC + staging)")
            
        # Build05: Training data preparation
        if not args.skip_build05:
            print("\n" + "="*60)
            print("🎓 STEP 6: Build05 - Training data preparation")
            print("="*60)
            run_build05(
                root=root, 
                train_name=args.train_name,
                overwrite=args.overwrite
            )
        else:
            print("\n⏭️  Skipping Build05 (training snips)")
            
        print("\n" + "="*60)
        print("🎉 End-to-End Pipeline Complete!")
        print("="*60)
        print(f"📁 Results available in: {root}")
        if args.run_sam2:
            print("🎯 Pipeline used SAM2 for superior embryo segmentation")
        print(f"🏷️ Training data: {args.train_name}")

    elif args.cmd == "validate":
        run_validation(root=resolve_root(args), exp=args.exp, df01=args.df01, checks=args.checks)

    elif args.cmd == "embed":
        run_embed(
            root=resolve_root(args),
            train_name=args.train_name,
            model_dir=args.model_dir,
            out_csv=args.out_csv,
            batch_size=args.batch_size,
            simulate=args.simulate,
            latent_dim=args.latent_dim,
            seed=args.seed,
        )

    elif args.cmd == "build06":
        # Resolve data_root from environment if not provided
        data_root = args.data_root
        if data_root is None:
            data_root = os.environ.get("MORPHSEQ_DATA_ROOT")
            if data_root is None:
                print("ERROR: --data-root not provided and MORPHSEQ_DATA_ROOT environment variable not set")
                return 1
        
        # Convert to absolute path for proper model resolution
        data_root = os.path.abspath(data_root)
        
        # Parse experiments from comma-separated string
        experiments = None
        if args.experiments:
            if args.experiments.lower() == "all":
                experiments = "all"  # Special case for explicit overwrite all
            else:
                experiments = [exp.strip() for exp in args.experiments.split(',') if exp.strip()]
        
        # Validate overwrite semantics for safety
        if args.overwrite:
            if not args.experiments:
                print("ERROR: --overwrite requires explicit --experiments specification")
                print("Safe usage:")
                print("  --overwrite --experiments 'exp1,exp2'  # Overwrite specific experiments")
                print("  --overwrite --experiments 'all'        # Overwrite ALL experiments (explicit)")
                return 1
            
            if experiments == "all":
                print("⚠️  WARNING: OVERWRITE ALL mode - will reprocess ALL experiments")
                print("⚠️  WARNING: This will regenerate the entire df03 file")
        
        print(f"🔬 Build06: Enhanced df03 generation (skipping Build05)")
        print(f"📂 Repo root: {args.morphseq_repo_root}")
        print(f"📊 Data root: {data_root}")
        print(f"🤖 Model: {args.model_name}")
        
        # Generate missing embeddings using Python 3.9 subprocess if needed
        if args.generate_missing_latents and experiments:
            # Handle experiments format for embedding generation
            if experiments == "all":
                print("🧬 Ensuring embeddings exist for ALL experiments (will auto-discover)...")
                exp_list = None  # Let build_df03_with_embeddings discover experiments
            else:
                print(f"🧬 Ensuring embeddings exist for {len(experiments)} experiments...")
                exp_list = experiments
            
            # Only run embedding generation if we have specific experiments
            # For "all", let build_df03_with_embeddings handle discovery and generation
            if exp_list:
                success = ensure_embeddings_for_experiments(
                    data_root=data_root,
                    experiments=exp_list,
                    model_name=args.model_name,
                    py39_env_path=args.py39_env,
                    overwrite=args.overwrite,
                    process_missing=args.process_missing,
                    verbose=False
                )
                
                if not success:
                    print("❌ Failed to generate required embeddings")
                    return 1
                print("✅ Embedding generation completed")
        
        # Enable environment switching by default for legacy models
        os.environ["MSEQ_ENABLE_ENV_SWITCH"] = "1"
        
        build_df03_with_embeddings(
            root=args.morphseq_repo_root,
            data_root=data_root,
            model_name=args.model_name,
            experiments=experiments,
            generate_missing=args.generate_missing_latents,
            export_analysis=args.export_analysis_copies,
            train_name=args.train_run,
            write_train_output=args.write_train_output,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    elif args.cmd == "status":
        # Lazy import to avoid heavy imports for other commands
        try:
            from src.build.pipeline_objects import ExperimentManager
        except Exception as e:
            print(f"ERROR: Failed to import ExperimentManager: {e}")
            return 1

        root = resolve_root(args)
        try:
            manager = ExperimentManager(root)
        except Exception as e:
            print(f"ERROR: Failed to initialize ExperimentManager: {e}")
            return 1

        # Filter experiments, if requested
        if args.experiments:
            allow = {e.strip() for e in args.experiments.split(',') if e.strip()}
            exps = {k: v for k, v in manager.experiments.items() if k in allow}
        else:
            exps = manager.experiments

        # Assemble status data
        model_name = args.model_name
        data = {
            "root": root,
            "global": {
                "df01_exists": manager.df01_path.exists(),
                "df02_exists": manager.df02_path.exists(),
                "df03_exists": manager.df03_path.exists(),
                "needs_build04": manager.needs_build04,
                "needs_build06": manager.needs_build06,
                "df01_path": str(manager.df01_path),
                "df02_path": str(manager.df02_path),
                "df03_path": str(manager.df03_path),
            },
            "experiments": {}
        }

        for date, exp in sorted(exps.items()):
            # Each field in try/except to keep reporting robust
            def safe(fn, fallback=None):
                try:
                    return fn()
                except Exception:
                    return fallback

            qc_present, qc_total = safe(exp.qc_mask_status, (0, 5))
            status = {
                "microscope": safe(lambda: exp.microscope, None),
                "ff": bool(exp.flags.get("ff", False)),
                "qc_all": qc_present == qc_total and qc_total > 0,
                "qc_present": qc_present,
                "qc_total": qc_total,
                "sam2_csv": safe(lambda: exp.sam2_csv_path.exists(), False),
                "needs_build03": safe(lambda: exp.needs_build03, False),
                "has_latents": safe(lambda: exp.has_latents(model_name), False),
            }
            data["experiments"][date] = status

        if args.format == "json":
            print(json.dumps(data, indent=2))
        else:
            # Human readable table-ish view
            print("\n" + "=" * 80)
            print("MORPHSEQ PIPELINE STATUS REPORT")
            print("=" * 80)
            print(f"Data root: {root}")
            g = data["global"]
            print(f"Global: df01={g['df01_exists']} df02={g['df02_exists']} df03={g['df03_exists']} | "
                  f"needs_build04={g['needs_build04']} needs_build06={g['needs_build06']}")

            for date, st in data["experiments"].items():
                bits = []
                bits.append("FF✅" if st["ff"] else "FF❌")
                if args.verbose:
                    bits.append(f"QC {st['qc_present']}/{st['qc_total']}")
                else:
                    bits.append("QC✅" if st["qc_all"] else "QC❌")
                bits.append("SAM2✅" if st["sam2_csv"] else "SAM2❌")
                bits.append("B03🔁" if st["needs_build03"] else "B03✔️")
                bits.append("LAT✅" if st["has_latents"] else "LAT❌")
                mic = st["microscope"] or "?"
                print(f"{date} [{mic}]: " + " ".join(bits))

    elif args.cmd == "pipeline":
        # Lazy import to avoid heavy imports for other commands
        try:
            from src.build.pipeline_objects import ExperimentManager
        except Exception as e:
            print(f"ERROR: Failed to import ExperimentManager: {e}")
            return 1

        root = resolve_root(args)
        try:
            manager = ExperimentManager(root)
        except Exception as e:
            print(f"ERROR: Failed to initialize ExperimentManager: {e}")
            return 1

        # Select experiments based on filters
        if args.experiments:
            exp_list = [e.strip() for e in args.experiments.split(',') if e.strip()]
            selected = [manager.experiments[e] for e in exp_list if e in manager.experiments]
            if len(selected) != len(exp_list):
                missing = [e for e in exp_list if e not in manager.experiments]
                print(f"WARNING: Missing experiments: {missing}")
        elif args.later_than:
            selected = []
            for exp in manager.experiments.values():
                try:
                    exp_date = int(exp.date[:8])  # Extract YYYYMMDD
                    if exp_date > args.later_than:
                        selected.append(exp)
                except (ValueError, IndexError):
                    continue
        else:
            selected = list(manager.experiments.values())

        if args.dry_run:
            print("🔍 DRY RUN - No changes will be made")
        
        print(f"\n{'='*60}")
        print(f"MORPHSEQ PIPELINE ORCHESTRATION")
        print(f"{'='*60}")
        print(f"📁 Data root: {root}")
        print(f"🎯 Action: {args.action}")
        print(f"🧪 Selected experiments: {len(selected)}")
        if args.force:
            print("⚡ Force mode: Will rerun steps even if not needed")

        # Execute based on action
        if args.action == "e2e":
            # Per-experiment steps first
            for exp in selected:
                print(f"\n{'='*40}")
                print(f"Processing {exp.date}")
                print(f"{'='*40}")
                
                # Show the complete pipeline flow in order
                print("  📋 Pipeline Steps:")
                
                # Step 1: Raw data (Build01)
                if exp.flags.get("ff", False):
                    print("    1️⃣ ✅ Raw data → FF images (Build01)")
                else:
                    print("    1️⃣ ❌ Raw data → FF images (Build01) - Missing")
                
                # Step 2: QC Masks (Build02) 
                qc_present, qc_total = exp.qc_mask_status()
                if qc_present == qc_total and qc_total > 0:
                    print(f"    2️⃣ ✅ QC mask generation (Build02) - {qc_present}/{qc_total}")
                else:
                    print(f"    2️⃣ ❌ QC mask generation (Build02) - {qc_present}/{qc_total}")
                
                # Step 3: SAM2 segmentation
                if args.force or exp.needs_sam2:
                    if args.dry_run:
                        print("    3️⃣ 🔄 SAM2 segmentation - would run")
                    else:
                        print("    3️⃣ 🔄 Running SAM2 segmentation...")
                        exp.run_sam2(
                            workers=args.sam2_workers,
                            confidence_threshold=args.sam2_confidence,
                            iou_threshold=args.sam2_iou
                        )
                else:
                    print("    3️⃣ ✅ SAM2 segmentation complete")
                
                # Step 4: Build03 (embryo processing)
                if args.force or exp.needs_build03:
                    if args.dry_run:
                        print("    4️⃣ 🔄 Embryo processing (Build03) - would run")
                    else:
                        print("    4️⃣ 🔄 Running embryo processing (Build03)...")
                        exp.run_build03(
                            by_embryo=args.by_embryo,
                            frames_per_embryo=args.frames_per_embryo
                        )
                else:
                    print("    4️⃣ ✅ Embryo processing (Build03) complete")
                
                # Step 5: Latent embeddings
                if args.force or not exp.has_latents(args.model_name):
                    if args.dry_run:
                        print("    5️⃣ 🔄 Latent embeddings - would generate")
                    else:
                        print("    5️⃣ 🔄 Generating latent embeddings...")
                        exp.generate_latents(model_name=args.model_name)
                else:
                    print("    5️⃣ ✅ Latent embeddings exist")
            
            # Global steps after all experiments
            print(f"\n{'='*40}")
            print("Global Pipeline Steps")
            print(f"{'='*40}")
            print("  📋 Global Steps:")
            
            # Step 6: Build04 (df01 → df02 QC)
            if args.force or manager.needs_build04:
                if args.dry_run:
                    print("    6️⃣ 🔄 Global QC & staging (Build04: df01→df02) - would run")
                else:
                    print("    6️⃣ 🔄 Running global QC & staging (Build04: df01→df02)...")
                    manager.run_build04()
            else:
                print("    6️⃣ ✅ Global QC & staging (Build04: df01→df02) complete")
                
            # Step 7: Build06 (df02 + latents → df03)
            if args.force or manager.needs_build06:
                # Show which specific experiments need merging
                experiments_needing_merge = []
                for exp in selected:
                    if exp.needs_build06_merge(args.model_name):
                        experiments_needing_merge.append(exp.date)
                
                if args.dry_run:
                    if experiments_needing_merge:
                        print(f"    7️⃣ 🔄 Final merge (Build06: df02+latents→df03) - would run")
                        print(f"        📋 Experiments needing merge: {len(experiments_needing_merge)}")
                        if args.verbose:
                            for exp_date in experiments_needing_merge[:3]:  # Show first 3
                                print(f"            • {exp_date}")
                            if len(experiments_needing_merge) > 3:
                                print(f"            • ... and {len(experiments_needing_merge) - 3} more")
                    else:
                        print("    7️⃣ 🔄 Final merge (Build06: df02+latents→df03) - would run (other experiments)")
                else:
                    print("    7️⃣ 🔄 Running final merge (Build06: df02+latents→df03)...")
                    if experiments_needing_merge:
                        print(f"        📋 {len(experiments_needing_merge)} experiments need merging")
                    manager.run_build06(model_name=args.model_name)
            else:
                print("    7️⃣ ✅ Final merge (Build06: df02+latents→df03) complete")

        elif args.action == "sam2":
            for exp in selected:
                if args.force or exp.needs_sam2:
                    if args.dry_run:
                        print(f"🔄 SAM2 needed for {exp.date} - would run")
                    else:
                        print(f"🔄 Running SAM2 for {exp.date}...")
                        exp.run_sam2(
                            workers=args.sam2_workers,
                            confidence_threshold=args.sam2_confidence,
                            iou_threshold=args.sam2_iou
                        )
                else:
                    print(f"✅ SAM2 already complete for {exp.date}")

        elif args.action == "build03":
            for exp in selected:
                if args.force or exp.needs_build03:
                    if args.dry_run:
                        print(f"🔄 Build03 needed for {exp.date} - would run")
                    else:
                        print(f"🔄 Running Build03 for {exp.date}...")
                        exp.run_build03(
                            by_embryo=args.by_embryo,
                            frames_per_embryo=args.frames_per_embryo
                        )
                else:
                    print(f"✅ Build03 already complete for {exp.date}")

        elif args.action == "build04":
            if args.force or manager.needs_build04:
                if args.dry_run:
                    print("🔄 Build04 needed - would run global QC")
                else:
                    print("🔄 Running Build04 (global QC)...")
                    manager.run_build04()
            else:
                print("✅ Build04 already complete")

        elif args.action == "build06":
            if args.force or manager.needs_build06:
                if args.dry_run:
                    print("🔄 Build06 needed - would run global merge")
                else:
                    print("🔄 Running Build06 (global merge)...")
                    manager.run_build06(model_name=args.model_name)
            else:
                print("✅ Build06 already complete")

        print(f"\n🎉 Pipeline {args.action} completed!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
