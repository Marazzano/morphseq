"""Command-line entrypoint for SeaHub reconciliation and drop-in bundling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .integration import (
    SeaHubIntegrationConfig,
    build_seahub_dropin_bundle,
    materialize_planned_experiment,
)
from .reconciliation import (
    apply_inclusion_policy,
    load_collection_metadata,
    reconcile_seahub_metadata,
)


def _config_from_args(args: argparse.Namespace) -> SeaHubIntegrationConfig:
    return SeaHubIntegrationConfig(
        operational_date=args.operational_date,
        micrometers_per_pixel=args.micrometers_per_pixel,
        jpeg_quality=args.jpeg_quality,
        canvas_rounding_px=args.canvas_rounding_px,
        canvas_fill_value=args.canvas_fill_value,
        overwrite_images=args.overwrite_images,
    )


def _write_reconciliation(
    *,
    image_reconciliation_csv: Path,
    collection_metadata_xlsx: Path,
    output_dir: Path,
    fuzzy_threshold: float,
) -> pd.DataFrame:
    source = pd.read_csv(image_reconciliation_csv)
    metadata = load_collection_metadata(collection_metadata_xlsx)
    reconciled = reconcile_seahub_metadata(
        source, metadata, fuzzy_threshold=fuzzy_threshold
    )
    policy = apply_inclusion_policy(reconciled)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.to_csv(output_dir / "reconciled_fovs.csv", index=False)
    policy[~policy["include_for_seahub"].astype(bool)].to_csv(
        output_dir / "dropped_fovs.csv", index=False
    )
    return policy


def cmd_reconcile(args: argparse.Namespace) -> None:
    policy = _write_reconciliation(
        image_reconciliation_csv=args.image_reconciliation_csv,
        collection_metadata_xlsx=args.collection_metadata_xlsx,
        output_dir=args.output_dir,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    counts = policy["include_for_seahub"].value_counts(dropna=False).to_dict()
    print(f"SeaHub reconciliation complete: {counts}")


def cmd_build_bundle(args: argparse.Namespace) -> None:
    reconciled = pd.read_csv(args.reconciled_fovs_csv)
    detections = pd.read_csv(args.detection_manifest_csv)
    result = build_seahub_dropin_bundle(
        reconciled,
        detections,
        output_root=args.output_root,
        config=_config_from_args(args),
        materialize_images=not args.plan_only,
    )
    print(
        "SeaHub bundle complete: "
        f"{len(result.well_provenance)} embryos, "
        f"{len(result.experiment_manifest)} operational shards, "
        f"{len(result.detection_failures)} detection failures, "
        f"{len(result.dropped_fovs)} policy exclusions."
    )


def cmd_all(args: argparse.Namespace) -> None:
    reconciliation_dir = args.output_root / "integration"
    reconciled = _write_reconciliation(
        image_reconciliation_csv=args.image_reconciliation_csv,
        collection_metadata_xlsx=args.collection_metadata_xlsx,
        output_dir=reconciliation_dir,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    detections = pd.read_csv(args.detection_manifest_csv)
    result = build_seahub_dropin_bundle(
        reconciled,
        detections,
        output_root=args.output_root,
        config=_config_from_args(args),
        materialize_images=not args.plan_only,
    )
    print(
        "SeaHub integration complete: "
        f"{len(result.well_provenance)} embryos across "
        f"{len(result.experiment_manifest)} operational shards."
    )


def cmd_materialize_shard(args: argparse.Namespace) -> None:
    flag = materialize_planned_experiment(
        bundle_root=args.bundle_root,
        experiment_id=args.experiment_id,
        overwrite_images=args.overwrite_images,
    )
    print(f"SeaHub shard materialized and validated: {flag}")


def _add_bundle_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detection-manifest-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--operational-date", default="20260723")
    parser.add_argument("--micrometers-per-pixel", type=float, default=7.8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--canvas-rounding-px", type=int, default=32)
    parser.add_argument("--canvas-fill-value", type=int, default=0)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Write identities/manifests without opening or writing images. "
            "The resulting frame inventories are intentionally not source-validated."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument(
        "--image-reconciliation-csv", type=Path, required=True
    )
    reconcile.add_argument(
        "--collection-metadata-xlsx", type=Path, required=True
    )
    reconcile.add_argument("--output-dir", type=Path, required=True)
    reconcile.add_argument("--fuzzy-threshold", type=float, default=0.90)
    reconcile.set_defaults(func=cmd_reconcile)

    bundle = subparsers.add_parser("build-bundle")
    bundle.add_argument("--reconciled-fovs-csv", type=Path, required=True)
    _add_bundle_options(bundle)
    bundle.set_defaults(func=cmd_build_bundle)

    all_steps = subparsers.add_parser("all")
    all_steps.add_argument(
        "--image-reconciliation-csv", type=Path, required=True
    )
    all_steps.add_argument(
        "--collection-metadata-xlsx", type=Path, required=True
    )
    all_steps.add_argument("--fuzzy-threshold", type=float, default=0.90)
    _add_bundle_options(all_steps)
    all_steps.set_defaults(func=cmd_all)

    materialize = subparsers.add_parser("materialize-shard")
    materialize.add_argument("--bundle-root", type=Path, required=True)
    materialize.add_argument("--experiment-id", required=True)
    materialize.add_argument("--overwrite-images", action="store_true")
    materialize.set_defaults(func=cmd_materialize_shard)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
