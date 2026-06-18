"""Task entrypoints used by Snakemake rules."""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from data_pipeline.metadata_ingest.experiment_identity import resolve_experiment_id
from data_pipeline.metadata_ingest.plate.plate_processing import process_plate_layout
from data_pipeline.metadata_ingest.scope.keyence.extract_scope_metadata import extract_keyence_scope_metadata
from data_pipeline.metadata_ingest.scope.yx1.extract_yx1_scope_metadata import extract_yx1_scope_metadata
from data_pipeline.metadata_ingest.scope.keyence.map_keyence_positions_to_wells import map_positions_to_wells_keyence
from data_pipeline.metadata_ingest.scope.yx1.map_yx1_positions_to_wells import map_positions_to_wells_yx1
from data_pipeline.metadata_ingest.scope.shared.apply_position_to_well_mapping import (
    apply_position_to_well_mapping,
)
from data_pipeline.metadata_ingest.position_well_mapping import validate_position_well_mapping
from data_pipeline.metadata_ingest.stitched_index.materialize_stitched_images import materialize_stitched_images
from data_pipeline.metadata_ingest.well_discovery.discover_wells_from_scope_metadata import (
    discover_wells_from_scope_metadata,
)
from data_pipeline.metadata_ingest.frame_inventory import (
    build_frame_inventory_for_well,
    merge_frame_inventory_shards,
    validate_frame_inventory,
)
from data_pipeline.segmentation_and_tracking.pipelines.segmentation_and_tracking import run_segmentation_and_tracking


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_selected_wells(csv: str | None) -> list[str]:
    if not csv:
        return []
    return [part.strip() for part in csv.split(",") if part.strip()]


def cmd_normalize_plate(args: argparse.Namespace) -> None:
    process_plate_layout(
        input_file=args.input_file,
        experiment_id=args.experiment,
        output_csv=args.output_csv,
    )


def cmd_extract_scope(args: argparse.Namespace) -> None:
    if args.microscope == "YX1":
        experiment_id = resolve_experiment_id(args.raw_images_dir, args.microscope, explicit_experiment_id=args.experiment)
        extract_yx1_scope_metadata(
            raw_data_dir=args.raw_images_dir,
            output_csv=args.output_csv,
            experiment_id=experiment_id,
            acquisition_inventory_csv=getattr(args, "acquisition_inventory_csv", None),
        )
    elif args.microscope == "Keyence":
        experiment_id = resolve_experiment_id(args.raw_images_parent, args.microscope, explicit_experiment_id=args.experiment)
        extract_keyence_scope_metadata(
            raw_data_dir=args.raw_images_parent,
            experiment_id=experiment_id,
            output_csv=args.output_csv,
        )
    else:
        raise ValueError(f"Unsupported microscope: {args.microscope}")


def cmd_map_positions(args: argparse.Namespace) -> None:
    if args.microscope == "YX1":
        if not args.ref_xy_csv:
            raise ValueError(
                "--ref-xy-csv is required for YX1 mapping. "
                "Set scope_metadata.yx1.ref_xy_csv in config.yaml."
            )
        map_positions_to_wells_yx1(
            scope_metadata_csv=args.scope_csv,
            output_mapping_csv=args.output_mapping_csv,
            output_provenance_json=args.output_provenance_json,
            experiment_id=args.experiment,
            ref_xy_csv=args.ref_xy_csv,
            max_distance_um=args.max_distance_um,
            allow_unmapped_wells=_parse_bool(args.allow_unmapped_wells),
            row_y_tol_um=float(args.row_y_tol_um),
            col_x_tol_um=float(args.col_x_tol_um),
            dx_cv_tol=float(args.dx_cv_tol),
            dy_cv_tol=float(args.dy_cv_tol),
        )
    elif args.microscope == "Keyence":
        experiment_id = resolve_experiment_id(args.raw_images_parent, args.microscope, explicit_experiment_id=args.experiment)
        map_positions_to_wells_keyence(
            raw_data_dir=args.raw_images_parent,
            scope_metadata_csv=args.scope_csv,
            output_mapping_csv=args.output_mapping_csv,
            output_provenance_json=args.output_provenance_json,
            experiment_id=experiment_id,
        )
    else:
        raise ValueError(f"Unsupported microscope: {args.microscope}")


def cmd_apply_position_to_well_mapping(args: argparse.Namespace) -> None:
    apply_position_to_well_mapping(
        scope_metadata_csv=args.scope_csv,
        mapping_csv=args.mapping_csv,
        output_csv=args.output_csv,
        experiment_id=args.experiment,
        selected_wells=_parse_selected_wells(args.selected_wells),
    )


def cmd_materialize_stitched(args: argparse.Namespace) -> None:
    materialize_stitched_images(
        experiment=args.experiment,
        microscope=args.microscope,
        raw_images_dir=args.raw_images_dir,
        scope_csv=args.scope_csv,
        mapping_csv=args.mapping_csv,
        output_root=args.output_root,
        output_stitched_index_csv=args.output_stitched_index_csv,
        selected_wells=_parse_selected_wells(args.selected_wells),
        overwrite=_parse_bool(args.overwrite),
        output_image_extension=args.output_image_extension,
        device_preference=args.device_preference,
        keyence_projection_method=args.keyence_projection_method,
        keyence_ff_filter_res_um=args.keyence_ff_filter_res_um,
        done_flag=args.done_flag,
    )


def cmd_build_frame_inventory_for_well(args: argparse.Namespace) -> None:
    build_frame_inventory_for_well(
        frame_contract_csv=args.frame_contract_csv,
        experiment_id=args.experiment,
        well_id=args.well_id,
        output_csv=args.output_csv,
    )


def cmd_validate_frame_inventory(args: argparse.Namespace) -> None:
    validate_frame_inventory(input_csv=args.input_csv, output_flag=args.output_flag)


def cmd_merge_frame_inventory(args: argparse.Namespace) -> None:
    merge_frame_inventory_shards(input_csvs=args.inputs, output_csv=args.output_csv)


def cmd_discover_wells(args: argparse.Namespace) -> None:
    discover_wells_from_scope_metadata(
        mapped_csv=Path(args.mapped_csv),
        output_wells=Path(args.output_wells),
    )


def cmd_materialize_yx1_well_candidate(args: argparse.Namespace) -> None:
    import pandas as pd
    from data_pipeline.image_materialization.scope.yx1.materialize_well_yx1 import (
        materialize_yx1_well,
    )

    acq_df = pd.read_csv(args.acquisition_inventory_csv)
    acq_df["experiment_id"] = acq_df["experiment_id"].astype(str)

    mapping_df = pd.read_csv(args.position_well_mapping_csv)
    mapping_df["experiment_id"] = mapping_df["experiment_id"].astype(str)
    validate_position_well_mapping(mapping_df, scope_label=str(args.position_well_mapping_csv))
    mapping_df = mapping_df[mapping_df["experiment_id"] == str(args.experiment)].copy()
    acq_df = acq_df.merge(
        mapping_df[["experiment_id", "position_index", "well_index", "well_id"]],
        on=["experiment_id", "position_index"],
        how="left",
        validate="many_to_one",
    )

    well_rows = acq_df[
        (acq_df["experiment_id"].astype(str) == str(args.experiment))
        & (acq_df["well_id"].astype(str) == str(args.well_id))
    ]
    if well_rows.empty:
        raise ValueError(
            f"No acquisition inventory rows found for experiment={args.experiment!r}, "
            f"well_id={args.well_id!r} after joining position_well_mapping."
        )
    inv_df = materialize_yx1_well(
        experiment_id=args.experiment,
        well_id=args.well_id,
        well_index=args.well_index,
        well_acquisition_inventory_df=well_rows,
        nd2_path=Path(args.nd2_path),
        built_image_data_dir=Path(args.built_image_data_dir),
        device=getattr(args, "device", "cuda"),
    )
    out_csv = Path(args.frame_inventory_csv)
    done = Path(args.done_flag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(out_csv, index=False)
    done.touch()


def cmd_segmentation_and_tracking(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config_yaml).read_text()) or {}
    run_segmentation_and_tracking(
        frame_contract_csv=args.frame_contract_csv,
        experiment_id=args.experiment,
        well_id=args.well_id,
        output_root=args.output_root,
        pipeline_config=cfg,
        device=args.device,
        run_id=args.run_id,
        verbose=_parse_bool(args.verbose),
    )


def cmd_snip_processing(args: argparse.Namespace) -> None:
    from data_pipeline.snip_processing.pipelines.snip_processing import run_snip_processing_well

    cfg = yaml.safe_load(Path(args.config_yaml).read_text()) or {}

    output_root = Path(args.output_root)
    exp = str(args.experiment)
    well = str(args.well_id)

    frame_contract_csv = Path(args.frame_contract_csv)
    segmentation_tracking_csv = Path(args.segmentation_tracking_csv)

    run_snip_processing_well(
        output_root=output_root,
        experiment_id=exp,
        well_id=well,
        frame_contract_csv=frame_contract_csv,
        segmentation_tracking_csv=segmentation_tracking_csv,
        pipeline_config=cfg,
        verbose=_parse_bool(args.verbose),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_norm = sub.add_parser("ingest-plate-metadata", aliases=["normalize-plate"])
    p_norm.add_argument("--input-file", type=Path, required=True)
    p_norm.add_argument("--experiment", required=True)
    p_norm.add_argument("--output-csv", type=Path, required=True)
    p_norm.set_defaults(func=cmd_normalize_plate)

    p_scope = sub.add_parser("ingest-scope-metadata", aliases=["extract-scope"])
    p_scope.add_argument("--raw-images-dir", type=Path, required=True)
    p_scope.add_argument("--experiment", required=True)
    p_scope.add_argument("--microscope", choices=["YX1", "Keyence"], required=True)
    p_scope.add_argument("--output-csv", type=Path, required=True)
    p_scope.add_argument(
        "--acquisition-inventory-csv",
        type=Path,
        default=None,
        help="Optional output path for acquisition_inventory__{scope}.csv (YX1: record-only).",
    )
    p_scope.set_defaults(func=cmd_extract_scope)

    p_map = sub.add_parser("map-positions-to-wells")
    p_map.add_argument("--experiment", required=True)
    p_map.add_argument("--microscope", choices=["YX1", "Keyence"], required=True)
    p_map.add_argument("--scope-csv", type=Path, required=True)
    p_map.add_argument("--output-mapping-csv", type=Path, required=True)
    p_map.add_argument("--output-provenance-json", type=Path, required=True)
    p_map.add_argument("--ref-xy-csv", type=Path, default=None)
    p_map.add_argument("--max-distance-um", type=float, default=4500.0)
    p_map.add_argument("--allow-unmapped-wells", default="false")
    p_map.add_argument("--row-y-tol-um", type=float, default=1200.0)
    p_map.add_argument("--col-x-tol-um", type=float, default=1200.0)
    p_map.add_argument("--dx-cv-tol", type=float, default=0.15)
    p_map.add_argument("--dy-cv-tol", type=float, default=0.15)
    p_map.set_defaults(func=cmd_map_positions)

    p_apply = sub.add_parser("apply-position-to-well-mapping")
    p_apply.add_argument("--experiment", required=True)
    p_apply.add_argument("--scope-csv", type=Path, required=True)
    p_apply.add_argument("--mapping-csv", type=Path, required=True)
    p_apply.add_argument("--output-csv", type=Path, required=True)
    p_apply.add_argument("--selected-wells", default="")
    p_apply.set_defaults(func=cmd_apply_position_to_well_mapping)

    p_mat = sub.add_parser("materialize-stitched")
    p_mat.add_argument("--experiment", required=True)
    p_mat.add_argument("--microscope", choices=["YX1", "Keyence"], required=True)
    p_mat.add_argument("--raw-images-dir", type=Path, required=True)
    p_mat.add_argument("--scope-csv", type=Path, required=True)
    p_mat.add_argument("--mapping-csv", type=Path, required=False)
    p_mat.add_argument("--output-root", type=Path, required=True)
    p_mat.add_argument("--output-stitched-index-csv", type=Path, required=True)
    p_mat.add_argument("--selected-wells", default="")
    p_mat.add_argument("--output-image-extension", default="jpg")
    p_mat.add_argument("--device-preference", default="cuda")
    p_mat.add_argument("--keyence-projection-method", default="log")
    p_mat.add_argument("--keyence-ff-filter-res-um", type=float, default=3.0)
    p_mat.add_argument("--overwrite", default="false")
    p_mat.add_argument("--done-flag", type=Path, required=False)
    p_mat.set_defaults(func=cmd_materialize_stitched)

    p_discover = sub.add_parser("discover-wells")
    p_discover.add_argument("--mapped-csv", type=Path, required=True)
    p_discover.add_argument("--output-wells", type=Path, required=True)
    p_discover.set_defaults(func=cmd_discover_wells)

    p_fi_build = sub.add_parser("build-frame-inventory-for-well")
    p_fi_build.add_argument("--frame-contract-csv", type=Path, required=True)
    p_fi_build.add_argument("--experiment", required=True)
    p_fi_build.add_argument("--well-id", required=True)
    p_fi_build.add_argument("--output-csv", type=Path, required=True)
    p_fi_build.set_defaults(func=cmd_build_frame_inventory_for_well)

    p_fi_validate = sub.add_parser("validate-frame-inventory")
    p_fi_validate.add_argument("--input-csv", type=Path, required=True)
    p_fi_validate.add_argument("--output-flag", type=Path, required=True)
    p_fi_validate.set_defaults(func=cmd_validate_frame_inventory)

    p_fi_merge = sub.add_parser("merge-frame-inventory")
    p_fi_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_fi_merge.add_argument("--output-csv", type=Path, required=True)
    p_fi_merge.set_defaults(func=cmd_merge_frame_inventory)

    p_yx1 = sub.add_parser("materialize-yx1-well-candidate")
    p_yx1.add_argument("--experiment", required=True)
    p_yx1.add_argument("--well-id", required=True)
    p_yx1.add_argument("--well-index", required=True)
    p_yx1.add_argument("--acquisition-inventory-csv", type=Path, required=True)
    p_yx1.add_argument("--position-well-mapping-csv", type=Path, required=True)
    p_yx1.add_argument("--nd2-path", type=Path, required=True)
    p_yx1.add_argument("--built-image-data-dir", type=Path, required=True)
    p_yx1.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_yx1.add_argument("--done-flag", type=Path, required=True)
    p_yx1.add_argument("--device", default="cuda")
    p_yx1.set_defaults(func=cmd_materialize_yx1_well_candidate)

    p_sat = sub.add_parser("segmentation-and-tracking")
    p_sat.add_argument("--frame-contract-csv", type=Path, required=True)
    p_sat.add_argument("--experiment", required=True)
    p_sat.add_argument("--well-id", required=True)
    p_sat.add_argument("--output-root", type=Path, required=True)
    p_sat.add_argument("--config-yaml", type=Path, required=True)
    p_sat.add_argument("--device", default="cuda")
    p_sat.add_argument("--run-id", default=None)
    p_sat.add_argument("--verbose", default="false")
    p_sat.set_defaults(func=cmd_segmentation_and_tracking)

    p_snip = sub.add_parser("snip-processing")
    p_snip.add_argument("--experiment", required=True)
    p_snip.add_argument("--well-id", required=True)
    p_snip.add_argument("--output-root", type=Path, required=True)
    p_snip.add_argument("--frame-contract-csv", type=Path, required=True)
    p_snip.add_argument("--segmentation-tracking-csv", type=Path, required=True)
    p_snip.add_argument("--config-yaml", type=Path, required=True)
    p_snip.add_argument("--verbose", default="false")
    p_snip.set_defaults(func=cmd_snip_processing)

    return parser


def _normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    # Keyence extraction/mapping helpers expect raw_data_root parent and experiment_id.
    if hasattr(args, "raw_images_dir") and isinstance(args.raw_images_dir, Path):
        raw_images_dir = args.raw_images_dir
        if getattr(args, "microscope", None) == "Keyence":
            args.raw_images_parent = raw_images_dir.parent
        else:
            args.raw_images_parent = raw_images_dir
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = _normalize_paths(args)
    args.func(args)


if __name__ == "__main__":
    main()
