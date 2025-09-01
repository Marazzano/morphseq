#!/usr/bin/env python3
"""
Centralized MorphSeq pipeline runner.

Usage examples:
    python -m src.run_morphseq_pipeline.cli build03 --root /data/morphseq --exp 20250612_30hpf_ctrl_atf6 \
        --sam2-csv /data/morphseq/sam2_metadata_20250612_30hpf_ctrl_atf6.csv --by-embryo 5 --frames-per-embryo 3

    python -m src.run_morphseq_pipeline.cli build04 --root /data/morphseq
    python -m src.run_morphseq_pipeline.cli build05 --root /data/morphseq --train-name train_ff_20250612

Build01/02 orchestration is provided via thin wrappers to existing build scripts.
SAM2 segmentation orchestration is intentionally not included here; provide --sam2-csv for Build03.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

from .steps.run_build03 import run_build03
from .steps.run_build04 import run_build04
from .steps.run_build05 import run_build05
from .steps.run_build01 import run_build01
from .steps.run_build_combine_metadata import run_combine_metadata
from .steps.run_build02 import run_build02
from .validation import run_validation
from .steps.run_embed import run_embed
from .steps.run_build06 import run_build06


def resolve_root(args) -> str:
    """Resolve the root path, optionally appending test suffix for isolation.
    
    WARNING: --test-suffix creates directory OUTSIDE the root path, which may cause
    permission errors. For /net/trapnell/.../morphseq, user may only have write access
    INSIDE morphseq/, not in parent directory. Consider omitting --test-suffix if
    getting PermissionError during directory creation.
    """
    root = Path(args.root)
    if hasattr(args, 'test_suffix') and args.test_suffix:
        # Create subdirectory instead of sibling directory
        root = root / args.test_suffix
        root.mkdir(parents=True, exist_ok=True)
        print(f"📁 Using test root: {root}")
    return str(root)


def _add_common_root_and_exp(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--root", required=True, help="Project root (contains built_image_data/, metadata/, training_data/)")
    ap.add_argument("--test-suffix", help="Append suffix to root for test isolation (e.g., test_sam2_20250830). WARNING: Creates directory outside root path, may cause permission errors.")
    ap.add_argument("--exp", required=False, help="Experiment name (e.g., 20250612_30hpf_ctrl_atf6)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="morphseq-runner", description="Centralized MorphSeq pipeline runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build01
    p01 = sub.add_parser("build01", help="Compile+stitch FF images; write built metadata CSV")
    _add_common_root_and_exp(p01)
    p01.add_argument("--microscope", choices=["keyence", "yx1"], required=True)
    p01.add_argument("--metadata-only", action="store_true", help="Skip image processing; write metadata only")
    p01.add_argument("--overwrite", action="store_true")

    # combine (master well metadata)
    pc = sub.add_parser("combine-metadata", help="Create master well metadata (experiment + built + well xlsx)")
    pc.add_argument("--root", required=True)

    # build02
    p02 = sub.add_parser("build02", help="Legacy segmentation (optional if using SAM2)")
    p02.add_argument("--root", required=True)
    p02.add_argument("--mode", choices=["legacy", "skip"], default="skip")
    p02.add_argument("--model-name", default="mask_v1_0050", help="Segmentation model name (legacy)")
    p02.add_argument("--n-classes", type=int, default=2)
    p02.add_argument("--overwrite", action="store_true")

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
    p04.add_argument("--root", required=True)
    # Accept --exp for interface consistency with other steps; currently unused by build04
    p04.add_argument("--exp", required=False, help="Experiment name (accepted for consistency; ignored by build04)")
    p04.add_argument("--dead-lead-time", type=int, default=2)

    # build05
    p05 = sub.add_parser("build05", help="Create training snips/folders from df02 + snips")
    p05.add_argument("--root", required=True)
    p05.add_argument("--train-name", required=True)
    p05.add_argument("--label-var", default=None)
    p05.add_argument("--rs-factor", type=float, default=1.0)
    p05.add_argument("--overwrite", action="store_true")

    # e2e
    pe2e = sub.add_parser("e2e", help="Run Build03→Build04→Build05")
    _add_common_root_and_exp(pe2e)
    pe2e.add_argument("--sam2-csv")
    pe2e.add_argument("--by-embryo", type=int)
    pe2e.add_argument("--frames-per-embryo", type=int)
    pe2e.add_argument("--max-samples", type=int)
    pe2e.add_argument("--n-workers", type=int, default=1)
    pe2e.add_argument("--train-name", required=True)
    pe2e.add_argument("--skip-build03", action="store_true")
    pe2e.add_argument("--skip-build04", action="store_true")
    pe2e.add_argument("--skip-build05", action="store_true")

    # validate
    pv = sub.add_parser("validate", help="Run validation gates (schema, units, paths)")
    pv.add_argument("--root", required=True)
    pv.add_argument("--exp", required=False)
    pv.add_argument("--df01", default="metadata/combined_metadata_files/embryo_metadata_df01.csv")
    pv.add_argument("--checks", default="schema,units,paths")

    # embed
    pem = sub.add_parser("embed", help="Generate morphological embeddings for training snips")
    pem.add_argument("--root", required=True)
    pem.add_argument("--train-name", required=True)
    pem.add_argument("--model-dir", required=False, help="Path to model or its parent (for real embeddings)")
    pem.add_argument("--out-csv", required=False)
    pem.add_argument("--batch-size", type=int, default=64)
    pem.add_argument("--simulate", action="store_true")
    pem.add_argument("--latent-dim", type=int, default=16)
    pem.add_argument("--seed", type=int, default=0)

    # build06 (standardize embeddings + df03 merge)
    p06 = sub.add_parser("build06", help="Generate embeddings and merge into df02 → df03")
    p06.add_argument("--root", required=True)
    p06.add_argument("--train-name", required=True)
    p06.add_argument("--simulate", action="store_true")
    p06.add_argument("--latent-dim", type=int, default=16)
    p06.add_argument("--seed", type=int, default=0)
    p06.add_argument("--model-dir", required=False)
    p06.add_argument("--batch-size", type=int, default=64)
    p06.add_argument("--no-join-df02", action="store_true", help="Do not write df03 merge")

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
                    n_classes=args.n_classes, overwrite=args.overwrite)

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
        run_build04(root=resolve_root(args), dead_lead_time=args.dead_lead_time)

    elif args.cmd == "build05":
        run_build05(root=resolve_root(args), train_name=args.train_name,
                    label_var=args.label_var, rs_factor=args.rs_factor,
                    overwrite=args.overwrite)

    elif args.cmd == "e2e":
        if not args.exp:
            raise SystemExit("--exp is required for e2e")
        root = resolve_root(args)
        if not args.skip_build03:
            run_build03(root=root, exp=args.exp, sam2_csv=args.sam2_csv,
                        by_embryo=args.by_embryo, frames_per_embryo=args.frames_per_embryo,
                        max_samples=args.max_samples, n_workers=args.n_workers)
        if not args.skip_build04:
            run_build04(root=root)
        if not args.skip_build05:
            run_build05(root=root, train_name=args.train_name)

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
        run_build06(
            root=resolve_root(args),
            train_name=args.train_name,
            simulate=args.simulate,
            latent_dim=args.latent_dim,
            seed=args.seed,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            join_df02=not args.no_join_df02,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
