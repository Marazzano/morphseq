"""Task entrypoints used by Snakemake rules."""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from data_pipeline.acquisition.metadata_ingest.experiment_identity import resolve_experiment_id
from data_pipeline.acquisition.metadata_ingest.plate.dropin_plate_metadata import (
    ingest_dropin_plate_metadata,
)
from data_pipeline.acquisition.metadata_ingest.plate.plate_processing import process_plate_layout
from data_pipeline.acquisition.metadata_ingest.plate.validate_plate_metadata import validate_plate_metadata_csv
from data_pipeline.acquisition.metadata_ingest.scope.keyence.extract_scope_metadata import extract_keyence_scope_metadata
from data_pipeline.acquisition.metadata_ingest.scope.yx1.extract_yx1_scope_metadata import extract_yx1_scope_metadata
from data_pipeline.acquisition.metadata_ingest.scope.keyence.map_keyence_positions_to_wells import map_positions_to_wells_keyence
from data_pipeline.acquisition.metadata_ingest.scope.yx1.map_yx1_positions_to_wells import map_positions_to_wells_yx1
from data_pipeline.acquisition.metadata_ingest.scope.shared.apply_position_to_well_mapping import (
    apply_position_to_well_mapping,
)
from data_pipeline.acquisition.metadata_ingest.position_well_mapping import validate_position_well_mapping
from data_pipeline.acquisition.metadata_ingest.well_discovery.discover_wells_from_scope_metadata import (
    discover_wells_from_scope_metadata,
)
from data_pipeline.acquisition.metadata_ingest.frame_inventory import (
    merge_frame_inventory_shards,
    validate_frame_inventory,
)
from data_pipeline.object_extraction.snip_processing.defaults import (
    DEFAULT_BLEND_RADIUS_UM,
    DEFAULT_TARGET_PIXEL_SIZE_UM,
)


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
    validate_plate_metadata_csv(input_csv=args.output_csv, output_flag=args.output_flag)


def cmd_ingest_dropin_plate(args: argparse.Namespace) -> None:
    ingest_dropin_plate_metadata(
        input_csv=args.input_csv,
        experiment_id=args.experiment,
        output_csv=args.output_csv,
        output_flag=args.output_flag,
    )


def cmd_extract_scope(args: argparse.Namespace) -> None:
    # The inventory stores full absolute source paths; readers re-anchor them under input_root at
    # consume time, so ingest needs no input_root.
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
            acquisition_inventory_csv=getattr(args, "acquisition_inventory_csv", None),
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
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_materialize_stitched(args: argparse.Namespace) -> None:
    # Function-local: materialize_stitched_images.py imports torch at module level. This is a
    # DELIBERATE GPU compute dependency, not incidental — LoG focus-stacking (log_focus.py) runs
    # a real conv2d over the Z-stack and is intentionally torch-accelerated; that isn't going away.
    # Deferred purely so `tasks.py` (imported once, module-wide, by every Snakemake rule) stays
    # importable in envs that don't need this specific command — orchestration/contract code has
    # no business requiring torch just to dispatch. This command itself still runs on the model
    # side once the pipeline/backend split (spec Phases 2-3) gives it a proper backend env; it does
    # not become torch-free.
    from data_pipeline.acquisition.metadata_ingest.stitched_index.materialize_stitched_images import (
        materialize_stitched_images,
    )

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


def cmd_validate_frame_inventory(args: argparse.Namespace) -> None:
    validate_frame_inventory(
        input_csv=args.input_csv,
        output_flag=args.output_flag,
        image_root=args.image_root,
        check_sources=_parse_bool(args.check_sources),
        validation_scope=args.validation_scope,
    )


def cmd_merge_frame_inventory(args: argparse.Namespace) -> None:
    merge_frame_inventory_shards(input_csvs=args.inputs, output_csv=args.output_csv)


def cmd_discover_wells(args: argparse.Namespace) -> None:
    discover_wells_from_scope_metadata(
        mapped_csv=Path(args.mapped_csv),
        output_wells=Path(args.output_wells),
    )


def cmd_discover_wells_from_handoff(args: argparse.Namespace) -> None:
    from data_pipeline.acquisition.metadata_ingest.well_discovery.discover_wells_from_handoff import (
        discover_wells_from_handoff,
    )

    discover_wells_from_handoff(
        manifest_csv=Path(args.manifest_csv),
        output_wells=Path(args.output_wells),
    )


def cmd_split_dropin_inventory(args: argparse.Namespace) -> None:
    # Per-well producer (race-free): writes EXACTLY the declared shard for --well-id.
    from data_pipeline.acquisition.metadata_ingest.well_discovery.split_dropin_inventory import (
        select_dropin_well_shard,
    )

    select_dropin_well_shard(
        manifest_csv=Path(args.manifest_csv),
        well_id=str(args.well_id),
        output_csv=Path(args.output_csv),
        image_root=Path(args.image_root),
    )


def cmd_scaffold_dropin_inventory(args: argparse.Namespace) -> None:
    from data_pipeline.acquisition.metadata_ingest.frame_inventory.scaffold_dropin_inventory import (
        scaffold_dropin_inventory,
    )

    scaffold_dropin_inventory(
        image_dir=Path(args.image_dir),
        output_csv=Path(args.output_csv),
        image_root=Path(args.image_root) if args.image_root else None,
    )


def cmd_materialize_well(args: argparse.Namespace) -> None:
    """CLI adapter: read inputs, delegate the domain work, write outputs.

    Thin dispatcher — it reads CSVs, normalizes CLI/state args, and delegates: the position→well
    join + per-well row selection is ``select_well_acquisition_rows`` (a pure domain step); the
    materialization workflow is ``run_materialize_well`` (the sequencer). This function holds no
    dataframe algebra and no scope/plan knowledge.
    """
    from data_pipeline.acquisition.image_materialization.run_materialize_well import run_materialize_well

    well_rows, well_index = _selected_well_acquisition_rows_for_materialization(args)

    config = None
    if getattr(args, "config_yaml", None):
        config = yaml.safe_load(Path(args.config_yaml).read_text()) or {}

    # A non-positive smoke cap means "no cap" (Snakemake passes 0 when the knob is unset).
    smoke_cap = getattr(args, "smoke_max_time_indices", None)
    if smoke_cap is not None and smoke_cap <= 0:
        smoke_cap = None

    inv_df = run_materialize_well(
        experiment_id=args.experiment,
        well_id=args.well_id,
        well_index=well_index,
        scope_name=args.scope,
        well_acquisition_inventory_df=well_rows,  # carries source_nd2_path
        built_image_data_dir=Path(args.built_image_data_dir),
        config=config,
        device=getattr(args, "device", "cuda"),
        candidate=_parse_bool(getattr(args, "candidate", "false")),
        smoke_max_time_indices=smoke_cap,
    )
    out_csv = Path(args.frame_inventory_csv)
    done = Path(args.done_flag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(out_csv, index=False)
    done.touch()


def _selected_well_acquisition_rows_for_materialization(args: argparse.Namespace):
    """Read acquisition + mapping inputs and return the selected well rows plus well_index."""
    import pandas as pd
    from data_pipeline.acquisition.image_materialization.select_well_acquisition_rows import (
        select_well_acquisition_rows,
    )
    from data_pipeline.shared.identifiers.parsers import split_well_id

    acq_df = pd.read_csv(args.acquisition_inventory_csv)
    mapping_df = pd.read_csv(args.position_well_mapping_csv)
    validate_position_well_mapping(mapping_df, scope_label=str(args.position_well_mapping_csv))

    well_rows = select_well_acquisition_rows(
        acq_df,
        mapping_df,
        experiment_id=str(args.experiment),
        well_id=str(args.well_id),
    )
    _, well_index = split_well_id(str(args.well_id))
    if getattr(args, "well_index", None) and str(args.well_index) != well_index:
        raise ValueError(
            f"--well-index={args.well_index!r} disagrees with well_id {args.well_id!r} "
            f"(parses to well_index={well_index!r})."
        )
    return well_rows, well_index


def cmd_write_resolved_product_plan_for_well(args: argparse.Namespace) -> None:
    """Write one resolved product plan JSON for one ``(well_id, product_key)``."""
    from data_pipeline.acquisition.image_materialization.resolved_product_plans import (
        write_resolved_product_plan_for_well,
    )

    config = None
    if getattr(args, "config_yaml", None):
        config = yaml.safe_load(Path(args.config_yaml).read_text()) or {}

    write_resolved_product_plan_for_well(
        experiment_id=str(args.experiment),
        well_id=str(args.well_id),
        scope_name=str(args.scope),
        config=config,
        product_key=str(args.product_key),
        output_json=Path(args.output_json),
    )


def cmd_build_keyence_stitch_map(args: argparse.Namespace) -> None:
    """Build the experiment-grain Keyence stitch map (master_params JSON)."""
    import pandas as pd
    from data_pipeline.acquisition.image_materialization.scope.keyence.build_keyence_stitch_map import (
        build_keyence_stitch_map,
    )

    build_keyence_stitch_map(
        acquisition_inventory_df=pd.read_csv(args.acquisition_inventory_csv),
        n_samples=int(getattr(args, "n_samples", 50)),
        out_path=Path(args.output_json),
        input_root=(
            Path(args.input_root) if getattr(args, "input_root", None) else None
        ),
    )


def cmd_materialize_image_product_for_well(args: argparse.Namespace) -> None:
    """Materialize one resolved image product and write its product frame-inventory shard."""
    from data_pipeline.acquisition.image_materialization.run_materialize_well import (
        run_materialize_image_product_for_well,
    )

    well_rows, well_index = _selected_well_acquisition_rows_for_materialization(args)

    config = None
    if getattr(args, "config_yaml", None):
        config = yaml.safe_load(Path(args.config_yaml).read_text()) or {}

    smoke_cap = getattr(args, "smoke_max_time_indices", None)
    if smoke_cap is not None and smoke_cap <= 0:
        smoke_cap = None

    master_params_raw = getattr(args, "master_params_path", None)
    master_params_path = Path(master_params_raw) if master_params_raw else None

    inv_df = run_materialize_image_product_for_well(
        experiment_id=str(args.experiment),
        well_id=str(args.well_id),
        well_index=well_index,
        scope_name=str(args.scope),
        well_acquisition_inventory_df=well_rows,
        built_image_data_dir=Path(args.built_image_data_dir),
        resolved_product_plan_json=Path(args.resolved_product_plan_json),
        product_key=str(args.product_key),
        config=config,
        device=getattr(args, "device", "cuda"),
        candidate=_parse_bool(getattr(args, "candidate", "false")),
        smoke_max_time_indices=smoke_cap,
        master_params_path=master_params_path,
        input_root=(
            Path(args.input_root) if getattr(args, "input_root", None) else None
        ),
    )
    out_csv = Path(args.frame_inventory_product_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(out_csv, index=False)


def cmd_discover_product_shards_for_well(args: argparse.Namespace) -> None:
    """Discover validated product frame-inventory shards for one well."""
    from data_pipeline.acquisition.image_materialization.product_shard_assembly import (
        discover_product_shards_for_well,
    )

    discover_product_shards_for_well(
        experiment_id=str(args.experiment),
        well_id=str(args.well_id),
        frame_inventory_products_dir=Path(args.frame_inventory_products_dir),
        output_csv=Path(args.output_csv),
    )


def cmd_assemble_well_frame_inventory(args: argparse.Namespace) -> None:
    """Assemble validated product shards into the canonical per-well frame_inventory."""
    from data_pipeline.acquisition.image_materialization.product_shard_assembly import (
        assemble_well_frame_inventory,
    )

    assemble_well_frame_inventory(
        discovered_product_shards_csv=Path(args.discovered_product_shards_csv),
        output_csv=Path(args.output_csv),
    )


def cmd_frame_detections(args: argparse.Namespace) -> None:
    from data_pipeline.object_extraction.detection import run_frame_detection
    from data_pipeline.object_extraction.detection.backends.groundingdino.config import GroundingDinoDetectionConfig
    from data_pipeline.models.groundingdino import load_groundingdino_model

    model = load_groundingdino_model(
        repo_dir=Path(args.gdino_repo_dir),
        config_path=Path(args.gdino_config),
        weights_path=Path(args.gdino_weights),
        device=args.device,
    )
    config = GroundingDinoDetectionConfig(device=args.device)
    run_frame_detection(
        frame_inventory_csv=args.frame_inventory_csv,
        output_csv=args.output_csv,
        backend="groundingdino",
        model=model,
        detector_model_id="SwinT_OGC",
        config=config,
    )


def cmd_validate_snip_inventory(args: argparse.Namespace) -> None:
    import pandas as pd

    from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
        validate_snip_inventory_contract,
    )

    df = pd.read_csv(args.input_csv)
    validate_snip_inventory_contract(df, scope_label=str(args.input_csv))
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_validate_frame_masks(args: argparse.Namespace) -> None:
    """Validate a per-well frame_masks shard against its frame_inventory and write the .validated sentinel.

    Thin dispatcher: the full contract (validate_frame_masks) cross-checks each mask row against
    frame identity, so it needs BOTH the frame_masks shard and the matching frame_inventory shard.
    """
    import pandas as pd

    from data_pipeline.object_extraction.segmentation.validate_frame_masks import validate_frame_masks

    validate_frame_masks(
        pd.read_csv(args.input_csv),
        pd.read_csv(args.frame_inventory_csv),
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_validate_frame_detections(args: argparse.Namespace) -> None:
    """Validate a per-well frame_detections shard against its frame_inventory and write the .validated sentinel.

    Thin dispatcher mirroring cmd_validate_frame_masks: the composed validator cross-checks each
    detection row's frame identity against the trusted frame_inventory, so it needs BOTH shards.
    Writing the sentinel is what makes the shard eligible for merge_frame_detections
    (collect_well_shard_paths only picks up shards carrying a .validated sentinel).
    """
    import pandas as pd

    from data_pipeline.object_extraction.detection.validate_frame_detections import validate_frame_detections

    validate_frame_detections(
        pd.read_csv(args.input_csv),
        pd.read_csv(args.frame_inventory_csv),
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_snip_processing(args: argparse.Namespace) -> None:
    from data_pipeline.object_extraction.snip_processing.entrypoints.run_snip_processing import run_snip_processing

    run_snip_processing(
        frame_masks_csv=args.frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
        snips_dir=args.snips_dir,
        output_root=args.output_root,
        target_pixel_size_um=args.target_pixel_size_um,
        output_height_px=args.output_height_px,
        output_width_px=args.output_width_px,
        background_noise_scale=args.background_noise_scale,
        blend_radius_um=args.blend_radius_um,
        apply_clahe=_parse_bool(args.apply_clahe),
    )


def cmd_mask_geometry(args: argparse.Namespace) -> None:
    """Compute the per-well mask_geometry feature shard. Thin dispatcher; logic lives in the product."""
    from data_pipeline.feature_extraction.mask_geometry.entrypoint import run_mask_geometry

    run_mask_geometry(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_validate_mask_geometry(args: argparse.Namespace) -> None:
    """Validate a per-well mask_geometry shard (spine + features, registry as verifier) and write .validated."""
    import pandas as pd

    from data_pipeline.feature_extraction.mask_geometry.contract import (
        validate_mask_geometry_features,
    )

    validate_mask_geometry_features(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_mask_geometry_report(args: argparse.Namespace) -> None:
    """Build the mask_geometry TERMINAL report (feature histogram grid + area quartile gallery)."""
    from data_pipeline.feature_extraction.mask_geometry.report import build_mask_geometry_report

    build_mask_geometry_report(
        mask_geometry_csv=args.mask_geometry_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        output_root=args.output_root,
        output_geometry_feature_grid_png=args.output_geometry_feature_grid_png,
        output_area_um2_quartile_gallery_png=args.output_area_um2_quartile_gallery_png,
    )


def cmd_curvature_metrics(args: argparse.Namespace) -> None:
    from data_pipeline.feature_extraction.curvature_metrics.entrypoint import run_curvature_metrics

    run_curvature_metrics(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_validate_curvature_metrics(args: argparse.Namespace) -> None:
    import pandas as pd

    from data_pipeline.feature_extraction.curvature_metrics.contract import (
        validate_curvature_features,
    )

    validate_curvature_features(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_curvature_metrics_report(args: argparse.Namespace) -> None:
    """Build the curvature_metrics TERMINAL report (feature histogram grid + spine-overlay gallery)."""
    from data_pipeline.feature_extraction.curvature_metrics.report import build_curvature_metrics_report

    build_curvature_metrics_report(
        curvature_metrics_csv=args.curvature_metrics_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        output_root=args.output_root,
        output_feature_grid_png=args.output_feature_grid_png,
        output_gallery_png=args.output_gallery_png,
    )


def cmd_pose_kinematics(args: argparse.Namespace) -> None:
    from data_pipeline.feature_extraction.pose_kinematics.entrypoint import run_pose_kinematics

    run_pose_kinematics(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_validate_pose_kinematics(args: argparse.Namespace) -> None:
    import pandas as pd

    from data_pipeline.feature_extraction.pose_kinematics.contract import (
        validate_pose_kinematics_features,
    )

    validate_pose_kinematics_features(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_stage_predictions(args: argparse.Namespace) -> None:
    from data_pipeline.feature_extraction.stage_predictions.entrypoint import run_stage_predictions

    run_stage_predictions(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        plate_metadata_csv=args.plate_metadata_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_validate_stage_predictions(args: argparse.Namespace) -> None:
    import pandas as pd

    from data_pipeline.feature_extraction.stage_predictions.contract import (
        validate_stage_prediction_features,
    )

    validate_stage_prediction_features(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_fraction_alive(args: argparse.Namespace) -> None:
    from data_pipeline.feature_extraction.fraction_alive.entrypoint import run_fraction_alive
    from data_pipeline.object_extraction.snip_processing.snip_frame_shape import resolve_snip_frame_shape

    config = yaml.safe_load(Path(args.config_yaml).read_text()) or {} if args.config_yaml else {}

    run_fraction_alive(
        snip_inventory_csv=args.snip_inventory_csv,
        snip_auxiliary_masks_csv=args.snip_auxiliary_masks_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
        snip_frame_shape=resolve_snip_frame_shape(config),
        output_root=args.output_root,
        missing_via_policy=args.missing_via_policy,
    )


def cmd_validate_fraction_alive(args: argparse.Namespace) -> None:
    import pandas as pd

    from data_pipeline.feature_extraction.fraction_alive.contract import (
        validate_fraction_alive_features,
    )

    validate_fraction_alive_features(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_snip_auxiliary_masks(args: argparse.Namespace) -> None:
    """Run per-snip UNet auxiliary-mask inference for one well. Thin dispatcher."""
    import yaml

    from data_pipeline.object_extraction.segmentation.backends.unet_snip.entrypoint import (
        run_snip_auxiliary_masks,
    )
    from data_pipeline.object_extraction.snip_processing.snip_frame_shape import resolve_snip_frame_shape

    config_yaml = Path(args.config_yaml)
    config = yaml.safe_load(config_yaml.read_text()) or {}
    unet_snip_config = config.get("unet_snip") or config.get("auxiliary_masks", {}).get("unet_snip", {})
    # models_root is ENVIRONMENT, not data layout: aux-mask checkpoints may live anywhere on a given
    # machine (env.yaml.paths.models_root, passed as --models-root) — NOT under the data/output tree.
    # The env root is authoritative and replaces any config models_root. The per-family checkpoint
    # key (unet_snip.models.<family>.checkpoint) is the only path the science config owns.
    unet_snip_config = dict(unet_snip_config)
    if args.models_root:
        unet_snip_config["models_root"] = str(args.models_root)

    run_snip_auxiliary_masks(
        snip_inventory_csv=args.snip_inventory_csv,
        output_root=args.output_root,
        output_csv=args.output_csv,
        unet_snip_config=unet_snip_config,
        snip_frame_shape=resolve_snip_frame_shape(config),
    )


def cmd_validate_snip_auxiliary_masks(args: argparse.Namespace) -> None:
    """Validate a per-well snip_auxiliary_masks shard (contract + cross-check) and write .validated."""
    import pandas as pd

    from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
        validate_snip_auxiliary_masks,
        validate_snip_auxiliary_masks_against_snip_inventory,
    )

    df = pd.read_csv(args.input_csv)
    df["is_valid_auxiliary_mask"] = df["is_valid_auxiliary_mask"].astype(bool)
    validate_snip_auxiliary_masks(df)
    validate_snip_auxiliary_masks_against_snip_inventory(df, pd.read_csv(args.snip_inventory_csv))
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_surface_area_qc(args: argparse.Namespace) -> None:
    """Compute the per-well surface_area_qc shard. Thin dispatcher; logic lives in the product."""
    from data_pipeline.quality_control.surface_area_qc.entrypoint import run_surface_area_qc

    run_surface_area_qc(
        mask_geometry_csv=args.mask_geometry_csv,
        stage_predictions_csv=args.stage_predictions_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_validate_surface_area_qc(args: argparse.Namespace) -> None:
    """Validate a per-well surface_area_qc shard (spine + flag, registry as verifier) and write .validated."""
    import pandas as pd

    from data_pipeline.quality_control.surface_area_qc.contract import validate_surface_area_qc

    validate_surface_area_qc(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_surface_area_qc_report(args: argparse.Namespace) -> None:
    """Build the surface_area_qc TERMINAL report (area-vs-stage scatter + stage-banded gallery)."""
    from data_pipeline.quality_control.surface_area_qc.report import build_surface_area_qc_report

    build_surface_area_qc_report(
        surface_area_qc_csv=args.surface_area_qc_csv,
        mask_geometry_csv=args.mask_geometry_csv,
        stage_predictions_csv=args.stage_predictions_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        output_root=args.output_root,
        output_vs_stage_png=args.output_vs_stage_png,
        output_gallery_png=args.output_gallery_png,
    )


def cmd_mask_quality_qc(args: argparse.Namespace) -> None:
    """Compute the per-well mask_quality_qc shard. Thin dispatcher; logic lives in the product."""
    from data_pipeline.quality_control.mask_quality_qc.entrypoint import run_mask_quality_qc

    run_mask_quality_qc(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_validate_mask_quality_qc(args: argparse.Namespace) -> None:
    """Validate a per-well mask_quality_qc shard (spine + flags, registry as verifier) and write .validated."""
    import pandas as pd

    from data_pipeline.quality_control.mask_quality_qc.contract import validate_mask_quality_qc

    validate_mask_quality_qc(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_mask_quality_qc_report(args: argparse.Namespace) -> None:
    """Build the mask_quality_qc TERMINAL report (histograms + per-flag galleries with overlays)."""
    from data_pipeline.quality_control.mask_quality_qc.report import build_mask_quality_qc_report

    build_mask_quality_qc_report(
        mask_quality_qc_csv=args.mask_quality_qc_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        output_root=args.output_root,
        output_edge_flag_histogram_png=args.output_edge_flag_histogram_png,
        output_edge_flag_gallery_png=args.output_edge_flag_gallery_png,
        output_discontinuous_mask_flag_histogram_png=args.output_discontinuous_mask_flag_histogram_png,
        output_discontinuous_mask_flag_gallery_png=args.output_discontinuous_mask_flag_gallery_png,
        output_overlapping_mask_flag_histogram_png=args.output_overlapping_mask_flag_histogram_png,
        output_overlapping_mask_flag_gallery_png=args.output_overlapping_mask_flag_gallery_png,
    )


def cmd_focus_qc(args: argparse.Namespace) -> None:
    """Compute the per-well focus_qc shard. Thin dispatcher; logic lives in the product."""
    from data_pipeline.quality_control.focus_qc.entrypoint import run_focus_qc

    run_focus_qc(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_motion_blur_qc(args: argparse.Namespace) -> None:
    """Compute the per-well motion_blur_qc shard. Thin dispatcher; logic lives in the product."""
    from data_pipeline.quality_control.motion_blur_qc.entrypoint import run_motion_blur_qc

    run_motion_blur_qc(
        snip_inventory_csv=args.snip_inventory_csv,
        frame_masks_csv=args.frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
    )


def cmd_motion_blur_qc_report(args: argparse.Namespace) -> None:
    """Build the motion_blur_qc TERMINAL report (histogram + cutoff-relative gallery)."""
    from data_pipeline.quality_control.motion_blur_qc.report import build_motion_blur_qc_report

    build_motion_blur_qc_report(
        motion_blur_qc_csv=args.motion_blur_qc_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        output_root=args.output_root,
        output_histogram_png=args.output_histogram_png,
        output_gallery_png=args.output_gallery_png,
    )


def cmd_validate_focus_qc(args: argparse.Namespace) -> None:
    """Validate a per-well focus_qc shard (spine + metric + flag, registry as verifier) and write .validated."""
    import pandas as pd

    from data_pipeline.quality_control.focus_qc.contract import validate_focus_qc

    validate_focus_qc(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_validate_motion_blur_qc(args: argparse.Namespace) -> None:
    """Validate a per-well motion_blur_qc shard (spine + metrics + flag, registry as verifier)."""
    import pandas as pd

    from data_pipeline.quality_control.motion_blur_qc.contract import validate_motion_blur_qc

    validate_motion_blur_qc(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_death_detection(args: argparse.Namespace) -> None:
    """Compute BOTH death_detection outputs (per-snip QC + per-animal death_event). Thin dispatcher."""
    from data_pipeline.quality_control.death_detection.entrypoint import run_death_detection

    run_death_detection(
        fraction_alive_csv=args.fraction_alive_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        plate_metadata_csv=args.plate_metadata_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_qc_csv=args.output_qc_csv,
        output_death_event_csv=args.output_death_event_csv,
    )


def cmd_death_detection_report(args: argparse.Namespace) -> None:
    """Build the death_detection TERMINAL report (survival curve + mortality curtain + death-time hist)."""
    from data_pipeline.quality_control.death_detection.report import build_death_detection_report

    build_death_detection_report(
        death_detection_qc_csv=args.death_detection_qc_csv,
        fraction_alive_csv=args.fraction_alive_csv,
        output_experiment_png=args.output_experiment_png,
        output_curtain_png=args.output_curtain_png,
        output_death_time_png=args.output_death_time_png,
        output_well_survival_png=args.output_well_survival_png,
    )


def cmd_analysis_ready(args: argparse.Namespace) -> None:
    """Build the analysis_ready merged-level fan-in join (OPTIONAL downstream product)."""
    from data_pipeline.analysis_ready.entrypoint import run_analysis_ready

    run_analysis_ready(
        curvature_csv=args.curvature_metrics_csv,
        stage_predictions_csv=args.stage_predictions_csv,
        mask_geometry_csv=args.mask_geometry_csv,
        pose_kinematics_csv=args.pose_kinematics_csv,
        fraction_alive_csv=args.fraction_alive_csv,
        latents_parquet=args.latents_parquet,
        snip_qc_parquet=args.snip_qc_parquet,
        plate_metadata_csv=args.plate_metadata_csv,
        output_parquet=args.output_parquet,
    )


def cmd_analysis_ready_report(args: argparse.Namespace) -> None:
    """Build the analysis_ready TERMINAL report (latent projection + survival-over-stage panels)."""
    from data_pipeline.analysis_ready.report import build_analysis_ready_report

    build_analysis_ready_report(
        analysis_ready_parquet=args.analysis_ready_parquet,
        death_event_csv=args.death_event_csv,
        snip_inventory_csv=args.snip_inventory_csv,
        output_root=args.output_root,
        output_latent_pca_qc_state_png=args.output_latent_pca_qc_state_png,
        output_latent_pca_stage_png=args.output_latent_pca_stage_png,
        output_latent_pca_genotype_png=args.output_latent_pca_genotype_png,
        output_post_qc_area_um2_gallery_png=args.output_post_qc_area_um2_gallery_png,
        output_post_qc_baseline_deviation_gallery_png=args.output_post_qc_baseline_deviation_gallery_png,
        output_survival_over_stage_png=args.output_survival_over_stage_png,
        output_genotype_survival_panel_png=args.output_genotype_survival_panel_png,
        output_well_survival_over_stage_all_png=args.output_well_survival_over_stage_all_png,
        output_well_survival_over_stage_by_genotype_png=args.output_well_survival_over_stage_by_genotype_png,
    )


def cmd_validate_death_detection_qc(args: argparse.Namespace) -> None:
    """Validate a per-well death_detection_qc shard (snip grain + flags, registry verifier)."""
    import pandas as pd

    from data_pipeline.quality_control.death_detection.contract import validate_death_detection_qc

    validate_death_detection_qc(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_validate_death_event(args: argparse.Namespace) -> None:
    """Validate a per-well death_event shard (physical-embryo grain, no embryo_id, registry verifier)."""
    import pandas as pd

    from data_pipeline.quality_control.death_detection.contract import validate_death_event

    validate_death_event(
        pd.read_csv(args.input_csv),
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_write_snip_qc_resolved_sources(args: argparse.Namespace) -> None:
    """Serialize the snip_qc resolver output for one well as a tracked JSON artifact.

    Thin dispatcher: parses exclusion_flags from JSON, calls the resolver (pure — no disk
    reads), and writes both exclusion_flags and resolved sources to the output JSON so the
    build rule receives the exact same plan the DAG was declared with.
    """
    import json

    from data_pipeline.quality_control.snip_qc.flag_input_resolver import (
        resolve_snip_qc_flag_sources,
    )

    exclusion_flags: tuple[str, ...] = tuple(json.loads(args.exclusion_flags_json))
    resolved = resolve_snip_qc_flag_sources(
        exclusion_flags,
        output_root=args.output_root,
        experiment_id=args.experiment,
        well_id=args.well_id,
    )
    payload = {
        "exclusion_flags": list(exclusion_flags),
        "resolved_sources": [src.to_dict() for src in resolved],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def cmd_snip_qc(args: argparse.Namespace) -> None:
    """Build the per-well snip_qc verdict. Thin dispatcher; logic lives in the product."""
    import json

    from data_pipeline.quality_control.snip_qc.entrypoint import run_snip_qc
    from data_pipeline.quality_control.snip_qc.flag_input_resolver import ResolvedFlagSource

    payload = json.loads(Path(args.resolved_sources_json_path).read_text())
    exclusion_flags: tuple[str, ...] = tuple(payload["exclusion_flags"])
    resolved_sources = tuple(
        ResolvedFlagSource.from_dict(d) for d in payload["resolved_sources"]
    )

    run_snip_qc(
        snip_inventory_csv=args.snip_inventory_csv,
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        output_csv=args.output_csv,
        resolved_sources=resolved_sources,
        exclusion_flags=exclusion_flags,
    )


def cmd_snip_qc_report(args: argparse.Namespace) -> None:
    """Build the snip_qc TERMINAL report (exclusion reasons over time, all + not-dead side by side)."""
    from data_pipeline.quality_control.snip_qc.report import build_snip_qc_report

    build_snip_qc_report(
        snip_qc_path=args.snip_qc_path,
        output_exclusion_reasons_png=args.output_exclusion_reasons_png,
    )


def cmd_validate_snip_qc(args: argparse.Namespace) -> None:
    """Validate a per-well snip_qc verdict shard (spine + verdict, registry verifier) and write .validated."""
    import pandas as pd

    from data_pipeline.quality_control.snip_qc.contract import validate_snip_qc

    input_path = args.input_csv
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    validate_snip_qc(
        df,
        physical_embryo_registry_df=pd.read_csv(args.physical_embryo_registry_csv),
        check_sources=True,
    )  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_frame_masks(args: argparse.Namespace) -> None:
    """Run SAM2 video segmentation for one well. Thin dispatcher."""
    import json
    import tempfile

    import numpy as np
    import pandas as pd
    from PIL import Image

    from data_pipeline.models.sam2 import load_sam2_video_predictor
    from data_pipeline.object_extraction.segmentation.backends.sam2_video.adapt_sam2_output import (
        adapt_sam2_well_output,
    )
    from data_pipeline.object_extraction.segmentation.backends.sam2_video.prompt_detections import (
        select_segmentation_frame_view,
        validate_sam2_prompts,
    )
    from data_pipeline.object_extraction.segmentation.validate_frame_masks import validate_frame_masks

    def _to_rgb_jpeg(src: Path, dst: Path) -> None:
        Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)

    frame_inventory = pd.read_csv(args.frame_inventory_csv)
    frame_detections = pd.read_csv(args.frame_detections_csv)
    well_id = str(frame_inventory["well_id"].iloc[0])
    model_inventory = select_segmentation_frame_view(frame_inventory, frame_detections)

    # Build prompt detections from kept frame_detections rows.
    # prompt_detection_id = detection_id; bbox columns are already in the right vocabulary.
    kept = frame_detections[frame_detections["is_kept"].astype(bool)].copy()
    prompt_detections = kept.rename(columns={"detection_id": "prompt_detection_id"})[
        [
            "prompt_detection_id", "image_id", "time_index",
            "bbox_x_min_px", "bbox_y_min_px", "bbox_x_max_px", "bbox_y_max_px",
            "is_kept",
        ]
    ]
    validate_sam2_prompts(prompt_detections, model_inventory)

    predictor = load_sam2_video_predictor(
        sam2_models_root=Path(args.sam2_models_root),
        config_path=Path(args.sam2_config),
        checkpoint_path=Path(args.sam2_checkpoint),
        device=args.device,
    )

    ordered = (
        model_inventory.sort_values(["time_index", "image_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    model_frame_view = ordered.copy()
    model_frame_view["sam2_frame_index"] = model_frame_view.index

    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmpdir:
        rgb_dir = Path(tmpdir)
        for _, row in ordered.iterrows():
            dst = rgb_dir / f"{int(row['time_index']):05d}.jpg"
            _to_rgb_jpeg(Path(str(row["image_path"])), dst)

        # Seed frame: earliest time_index that has kept detections.
        seed_time = int(prompt_detections["time_index"].min())
        seed_sam2_idx = int(
            model_frame_view[model_frame_view["time_index"] == seed_time]["sam2_frame_index"].iloc[0]
        )

        inference_state = predictor.init_state(video_path=str(rgb_dir))
        seed_prompts = prompt_detections[prompt_detections["time_index"] == seed_time].reset_index(drop=True)
        for i, (_, row) in enumerate(seed_prompts.iterrows()):
            box = np.array([
                row["bbox_x_min_px"], row["bbox_y_min_px"],
                row["bbox_x_max_px"], row["bbox_y_max_px"],
            ], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=seed_sam2_idx,
                obj_id=i,
                box=box,
            )

        sam2_raw_output: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks: dict[int, np.ndarray] = {}
            for obj_id, logit in zip(obj_ids, mask_logits):
                arr = logit.squeeze().cpu().numpy() if hasattr(logit, "cpu") else np.squeeze(np.asarray(logit))
                masks[int(obj_id)] = (arr > 0).astype(bool)
            sam2_raw_output[int(frame_idx)] = masks

    frame_masks = adapt_sam2_well_output(
        well_id,
        sam2_raw_output,
        model_frame_view,
        prompt_detections,
        model_id=str(args.sam2_model_id),
    )
    validate_frame_masks(frame_masks, model_inventory)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_masks.to_csv(args.output_csv, index=False)

    prompt_detections.to_csv(args.prompt_seeds_csv, index=False)


def cmd_ingest_precomputed_frame_masks(args: argparse.Namespace) -> None:
    """Materialize one authoritative precomputed frame-mask shard without model inference."""

    from data_pipeline.object_extraction.segmentation.precomputed_frame_masks import (
        write_precomputed_frame_masks_for_well,
    )

    write_precomputed_frame_masks_for_well(
        precomputed_frame_masks_csv=args.precomputed_frame_masks_csv,
        frame_inventory_csv=args.frame_inventory_csv,
        frame_detections_csv=args.frame_detections_csv,
        well_id=str(args.well_id),
        output_csv=args.output_csv,
        detection_audit_csv=args.prompt_seeds_csv,
        require_detection_audit=_parse_bool(args.require_detection_audit),
    )


def cmd_build_physical_embryo_registry(args: argparse.Namespace) -> None:
    """Mint the per-well physical_embryo_registry shard from a per-well frame_masks shard.

    Thin dispatcher: read frame_masks CSV -> Stage-2 builder (which validates before returning)
    -> write the registry CSV. No domain logic here.
    """
    import pandas as pd

    from data_pipeline.object_extraction.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
        build_physical_embryo_registry,
    )

    registry = build_physical_embryo_registry(pd.read_csv(args.frame_masks_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.output_csv, index=False)


def cmd_validate_physical_embryo_registry(args: argparse.Namespace) -> None:
    """Validate a physical_embryo_registry CSV (per-well or merged) and write its .validated sentinel."""
    import pandas as pd

    from data_pipeline.object_extraction.segmentation.physical_embryo_registry.validate_physical_embryo_registry import (
        validate_physical_embryo_registry,
    )

    validate_physical_embryo_registry(pd.read_csv(args.input_csv))  # raises on failure
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("")


def cmd_merge_physical_embryo_registry(args: argparse.Namespace) -> None:
    """Concat per-well registry shards into the experiment-level table, re-validating GLOBAL uniqueness.

    Uses the Stage-2 merge (not the generic concat helper) because the registry product owns the
    global physical_embryo_id uniqueness law — its post-concat validator is what turns the
    by-construction invariant into an enforced promise.
    """
    import pandas as pd

    from data_pipeline.object_extraction.segmentation.physical_embryo_registry.build_physical_embryo_registry import (
        merge_physical_embryo_registry,
    )

    merged = merge_physical_embryo_registry([pd.read_csv(p) for p in args.inputs])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)


def cmd_physical_embryo_registry_report(args: argparse.Namespace) -> None:
    """Build the physical_embryo_registry TERMINAL report (embryos-per-well dist + plate heatmap +
    embryos-per-well over time death proxy)."""
    from data_pipeline.object_extraction.segmentation.physical_embryo_registry.report import (
        build_physical_embryo_registry_report,
    )

    build_physical_embryo_registry_report(
        physical_embryo_registry_csv=args.physical_embryo_registry_csv,
        frame_masks_csv=args.frame_masks_csv,
        output_embryos_per_well_png=args.output_embryos_per_well_png,
        output_embryos_per_well_plate_png=args.output_embryos_per_well_plate_png,
        output_embryos_per_well_over_time_png=args.output_embryos_per_well_over_time_png,
    )


def cmd_stage_rollup_report(args: argparse.Namespace) -> None:
    """Build one stage's TERMINAL rollup page (all that stage's per-step report PNGs, one
    HTML+PDF). Self-resolves the PNG inputs from the registry (see viz/stage_report.py), so the
    rule only passes the stage, data root, experiment, and the output HTML path."""
    from data_pipeline.viz.stage_report import build_stage_rollup_report

    build_stage_rollup_report(
        args.data_root,
        args.stage,
        args.experiment,
        output_html=args.output_html,
    )


def cmd_validate_latent_embeddings(args: argparse.Namespace) -> None:
    """Validate a latents parquet (per-well or merged) and write its .validated sentinel."""
    import pandas as pd

    from data_pipeline.feature_extraction.legacy_embeddings.contract import (
        validate_latent_embeddings,
    )

    validate_latent_embeddings(pd.read_parquet(args.input_parquet), source=str(args.input_parquet))
    args.output_flag.parent.mkdir(parents=True, exist_ok=True)
    args.output_flag.write_text("ok\n")


def cmd_merge_latent_embeddings(args: argparse.Namespace) -> None:
    """Concat per-well latents shards into the experiment-level parquet, then validate the result.

    pd.concat unions columns, so a shard with a different latent_dim would surface as NaN; the
    latents validator (z_mu_* non-null) catches that — fail loud rather than emit a ragged table.
    """
    import pandas as pd

    from data_pipeline.feature_extraction.legacy_embeddings.contract import (
        validate_latent_embeddings,
    )

    inputs = list(args.inputs or [])
    if not inputs:
        if args.data_root is None or not args.experiment_id:
            raise ValueError(
                "merge-latent-embeddings requires either --inputs or both "
                "--data-root and --experiment-id"
            )
        from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
            collect_well_shard_paths,
        )

        inputs = collect_well_shard_paths(
            args.data_root,
            "latent_embeddings",
            "latents",
            args.experiment_id,
        )
    if not inputs:
        raise ValueError("merge-latent-embeddings found no validated per-well shards")

    merged = pd.concat([pd.read_parquet(p) for p in inputs], ignore_index=True)
    validate_latent_embeddings(merged, source=str(args.output_parquet))
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_parquet, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_norm = sub.add_parser("ingest-plate-metadata", aliases=["normalize-plate"])
    p_norm.add_argument("--input-file", type=Path, required=True)
    p_norm.add_argument("--experiment", required=True)
    p_norm.add_argument("--output-csv", type=Path, required=True)
    p_norm.add_argument("--output-flag", type=Path, required=True)
    p_norm.set_defaults(func=cmd_normalize_plate)

    p_dropin_plate = sub.add_parser("ingest-dropin-plate-metadata")
    p_dropin_plate.add_argument("--input-csv", type=Path, required=True)
    p_dropin_plate.add_argument("--experiment", required=True)
    p_dropin_plate.add_argument("--output-csv", type=Path, required=True)
    p_dropin_plate.add_argument("--output-flag", type=Path, required=True)
    p_dropin_plate.set_defaults(func=cmd_ingest_dropin_plate)

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
    p_map.add_argument("--raw-images-parent", type=Path, default=None,
                       help="Parent of the raw-images dir (i.e. .../raw_image_data/Keyence). "
                            "Required for Keyence; not used for YX1.")
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
    p_apply.add_argument("--output-flag", type=Path, required=True)
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

    p_fi_validate = sub.add_parser("validate-frame-inventory")
    p_fi_validate.add_argument("--input-csv", type=Path, required=True)
    p_fi_validate.add_argument("--output-flag", type=Path, required=True)
    # Strict-gate mode flags — every call site passes these EXPLICITLY (no silent default).
    p_fi_validate.add_argument("--image-root", type=Path, default=None)
    p_fi_validate.add_argument("--check-sources", default="false")
    p_fi_validate.add_argument(
        "--validation-scope", choices=["per_well", "merged"], default="per_well"
    )
    p_fi_validate.set_defaults(func=cmd_validate_frame_inventory)

    p_fi_merge = sub.add_parser("merge-frame-inventory")
    p_fi_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_fi_merge.add_argument("--output-csv", type=Path, required=True)
    p_fi_merge.set_defaults(func=cmd_merge_frame_inventory)

    # --- external drop-in entrance (config/CLI ingress; not a registry artifact) ---
    p_dwh = sub.add_parser("discover-wells-from-handoff")
    p_dwh.add_argument("--manifest-csv", type=Path, required=True)
    p_dwh.add_argument("--output-wells", type=Path, required=True)
    p_dwh.set_defaults(func=cmd_discover_wells_from_handoff)

    # Per-well producer (race-free): writes EXACTLY one well's shard for --well-id.
    p_split = sub.add_parser("split-dropin-inventory")
    p_split.add_argument("--manifest-csv", type=Path, required=True)
    p_split.add_argument("--well-id", required=True)
    p_split.add_argument("--output-csv", type=Path, required=True)
    p_split.add_argument("--image-root", type=Path, required=True)
    p_split.set_defaults(func=cmd_split_dropin_inventory)

    p_scaffold = sub.add_parser("scaffold-dropin-inventory")
    p_scaffold.add_argument("--image-dir", type=Path, required=True)
    p_scaffold.add_argument("--output-csv", type=Path, required=True)
    p_scaffold.add_argument("--image-root", type=Path, default=None)
    p_scaffold.set_defaults(func=cmd_scaffold_dropin_inventory)

    p_mw = sub.add_parser(
        "materialize-well", aliases=["materialize-yx1-well-candidate"]
    )
    p_mw.add_argument("--experiment", required=True)
    p_mw.add_argument("--well-id", required=True)
    # --well-index is optional: derived from --well-id via the identity parser. Pass it only to
    # cross-check (a mismatch fails loud).
    p_mw.add_argument("--well-index", default=None)
    p_mw.add_argument("--scope", default="yx1")
    p_mw.add_argument("--acquisition-inventory-csv", type=Path, required=True)
    p_mw.add_argument("--position-well-mapping-csv", type=Path, required=True)
    p_mw.add_argument("--built-image-data-dir", type=Path, required=True)
    p_mw.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_mw.add_argument("--done-flag", type=Path, required=True)
    p_mw.add_argument("--config-yaml", type=Path, default=None)
    p_mw.add_argument("--candidate", default="false")
    p_mw.add_argument("--smoke-max-time-indices", type=int, default=None)
    p_mw.add_argument("--device", default="cuda")
    p_mw.set_defaults(func=cmd_materialize_well)

    p_rpp = sub.add_parser("write-resolved-product-plan-for-well")
    p_rpp.add_argument("--experiment", required=True)
    p_rpp.add_argument("--well-id", required=True)
    p_rpp.add_argument("--scope", default="yx1")
    p_rpp.add_argument("--product-key", required=True)
    p_rpp.add_argument("--output-json", type=Path, required=True)
    p_rpp.add_argument("--config-yaml", type=Path, default=None)
    p_rpp.set_defaults(func=cmd_write_resolved_product_plan_for_well)

    p_mip = sub.add_parser("materialize-image-product-for-well")
    p_mip.add_argument("--experiment", required=True)
    p_mip.add_argument("--well-id", required=True)
    p_mip.add_argument("--well-index", default=None)
    p_mip.add_argument("--scope", default="yx1")
    p_mip.add_argument("--product-key", required=True)
    p_mip.add_argument("--resolved-product-plan-json", type=Path, required=True)
    p_mip.add_argument("--acquisition-inventory-csv", type=Path, required=True)
    p_mip.add_argument("--position-well-mapping-csv", type=Path, required=True)
    p_mip.add_argument("--built-image-data-dir", type=Path, required=True)
    p_mip.add_argument("--frame-inventory-product-csv", type=Path, required=True)
    p_mip.add_argument("--config-yaml", type=Path, default=None)
    p_mip.add_argument("--candidate", default="false")
    p_mip.add_argument("--smoke-max-time-indices", type=int, default=None)
    p_mip.add_argument("--device", default="cuda")
    p_mip.add_argument("--master-params-path", type=Path, default=None)
    # input_root — resolves inventory source paths stored RELATIVE to it. Optional: legacy
    # absolute-path inventories resolve without it.
    p_mip.add_argument("--input-root", type=Path, default=None)
    p_mip.set_defaults(func=cmd_materialize_image_product_for_well)

    p_bksm = sub.add_parser("build-keyence-stitch-map")
    p_bksm.add_argument("--acquisition-inventory-csv", type=Path, required=True)
    p_bksm.add_argument("--output-json", type=Path, required=True)
    p_bksm.add_argument("--n-samples", type=int, default=50)
    # input_root — resolves inventory source paths stored RELATIVE to it (optional; see above).
    p_bksm.add_argument("--input-root", type=Path, default=None)
    p_bksm.set_defaults(func=cmd_build_keyence_stitch_map)

    p_dps = sub.add_parser("discover-product-shards-for-well")
    p_dps.add_argument("--experiment", required=True)
    p_dps.add_argument("--well-id", required=True)
    p_dps.add_argument("--frame-inventory-products-dir", type=Path, required=True)
    p_dps.add_argument("--output-csv", type=Path, required=True)
    p_dps.set_defaults(func=cmd_discover_product_shards_for_well)

    p_awfi = sub.add_parser("assemble-well-frame-inventory")
    p_awfi.add_argument("--discovered-product-shards-csv", type=Path, required=True)
    p_awfi.add_argument("--output-csv", type=Path, required=True)
    p_awfi.set_defaults(func=cmd_assemble_well_frame_inventory)

    p_fd = sub.add_parser("frame-detections")
    p_fd.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_fd.add_argument("--output-csv", type=Path, required=True)
    p_fd.add_argument("--gdino-repo-dir", type=Path, required=True)
    p_fd.add_argument("--gdino-config", type=Path, required=True)
    p_fd.add_argument("--gdino-weights", type=Path, required=True)
    p_fd.add_argument("--device", default="cuda")
    p_fd.set_defaults(func=cmd_frame_detections)

    p_fd_validate = sub.add_parser("validate-frame-detections")
    p_fd_validate.add_argument("--input-csv", type=Path, required=True)
    p_fd_validate.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_fd_validate.add_argument("--output-flag", type=Path, required=True)
    p_fd_validate.set_defaults(func=cmd_validate_frame_detections)

    p_si_validate = sub.add_parser("validate-snip-inventory")
    p_si_validate.add_argument("--input-csv", type=Path, required=True)
    p_si_validate.add_argument("--output-flag", type=Path, required=True)
    p_si_validate.set_defaults(func=cmd_validate_snip_inventory)

    p_sp = sub.add_parser("snip-processing")
    p_sp.add_argument("--frame-masks-csv", type=Path, required=True)
    p_sp.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_sp.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_sp.add_argument("--output-csv", type=Path, required=True)
    p_sp.add_argument("--snips-dir", type=Path, required=True)
    p_sp.add_argument("--output-root", type=Path, required=True)
    p_sp.add_argument(
        "--target-pixel-size-um", type=float, default=DEFAULT_TARGET_PIXEL_SIZE_UM
    )
    p_sp.add_argument("--output-height-px", type=int, default=576)
    p_sp.add_argument("--output-width-px", type=int, default=256)
    p_sp.add_argument("--background-noise-scale", type=float, default=0.1)
    p_sp.add_argument("--blend-radius-um", type=float, default=DEFAULT_BLEND_RADIUS_UM)
    p_sp.add_argument(
        "--apply-clahe",
        default="true",
        help="Apply legacy CLAHE before background blending (true/false; default true).",
    )
    p_sp.set_defaults(func=cmd_snip_processing)

    p_mg = sub.add_parser("mask-geometry")
    p_mg.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_mg.add_argument("--frame-masks-csv", type=Path, required=True)
    p_mg.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_mg.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_mg.add_argument("--output-csv", type=Path, required=True)
    p_mg.set_defaults(func=cmd_mask_geometry)

    p_mg_validate = sub.add_parser("validate-mask-geometry")
    p_mg_validate.add_argument("--input-csv", type=Path, required=True)
    p_mg_validate.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_mg_validate.add_argument("--output-flag", type=Path, required=True)
    p_mg_validate.set_defaults(func=cmd_validate_mask_geometry)

    p_mg_report = sub.add_parser("mask-geometry-report")
    p_mg_report.add_argument("--mask-geometry-csv", type=Path, required=True)
    p_mg_report.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_mg_report.add_argument("--output-root", type=Path, required=True)
    p_mg_report.add_argument("--output-geometry-feature-grid-png", type=Path, required=True)
    p_mg_report.add_argument("--output-area-um2-quartile-gallery-png", type=Path, required=True)
    p_mg_report.set_defaults(func=cmd_mask_geometry_report)

    # ── feature products that share the mask-derived shape (snip_inventory + frame_masks + ...) ──
    for verb, fn in (("curvature-metrics", cmd_curvature_metrics), ("pose-kinematics", cmd_pose_kinematics)):
        p = sub.add_parser(verb)
        p.add_argument("--snip-inventory-csv", type=Path, required=True)
        p.add_argument("--frame-masks-csv", type=Path, required=True)
        p.add_argument("--frame-inventory-csv", type=Path, required=True)
        p.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
        p.add_argument("--output-csv", type=Path, required=True)
        p.set_defaults(func=fn)

    for verb, fn in (
        ("validate-curvature-metrics", cmd_validate_curvature_metrics),
        ("validate-pose-kinematics", cmd_validate_pose_kinematics),
        ("validate-stage-predictions", cmd_validate_stage_predictions),
        ("validate-fraction-alive", cmd_validate_fraction_alive),
        ("validate-surface-area-qc", cmd_validate_surface_area_qc),
        ("validate-mask-quality-qc", cmd_validate_mask_quality_qc),
        ("validate-focus-qc", cmd_validate_focus_qc),
        ("validate-motion-blur-qc", cmd_validate_motion_blur_qc),
    ):
        p = sub.add_parser(verb)
        p.add_argument("--input-csv", type=Path, required=True)
        p.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
        p.add_argument("--output-flag", type=Path, required=True)
        p.set_defaults(func=fn)

    p_cm_report = sub.add_parser("curvature-metrics-report")
    p_cm_report.add_argument("--curvature-metrics-csv", type=Path, required=True)
    p_cm_report.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_cm_report.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_cm_report.add_argument("--output-root", type=Path, required=True)
    p_cm_report.add_argument("--output-feature-grid-png", type=Path, required=True)
    p_cm_report.add_argument("--output-gallery-png", type=Path, required=True)
    p_cm_report.set_defaults(func=cmd_curvature_metrics_report)

    p_stage = sub.add_parser("stage-predictions")
    p_stage.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_stage.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_stage.add_argument("--plate-metadata-csv", type=Path, required=True)
    p_stage.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_stage.add_argument("--output-csv", type=Path, required=True)
    p_stage.set_defaults(func=cmd_stage_predictions)

    p_fa = sub.add_parser("fraction-alive")
    p_fa.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_fa.add_argument("--snip-auxiliary-masks-csv", type=Path, required=True)
    p_fa.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_fa.add_argument("--output-csv", type=Path, required=True)
    p_fa.add_argument("--output-root", type=Path, default=None)
    p_fa.add_argument("--config-yaml", type=Path, default=None)
    p_fa.add_argument("--missing-via-policy", default="fail", choices=["fail", "null"])
    p_fa.set_defaults(func=cmd_fraction_alive)

    p_sam = sub.add_parser("snip-auxiliary-masks")
    p_sam.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_sam.add_argument("--output-root", type=Path, required=True)
    p_sam.add_argument("--output-csv", type=Path, required=True)
    p_sam.add_argument("--config-yaml", type=Path, required=True)
    p_sam.add_argument("--models-root", type=Path, default=None)
    p_sam.set_defaults(func=cmd_snip_auxiliary_masks)

    p_sam_validate = sub.add_parser("validate-snip-auxiliary-masks")
    p_sam_validate.add_argument("--input-csv", type=Path, required=True)
    p_sam_validate.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_sam_validate.add_argument("--output-flag", type=Path, required=True)
    p_sam_validate.set_defaults(func=cmd_validate_snip_auxiliary_masks)

    p_saqc = sub.add_parser("surface-area-qc")
    p_saqc.add_argument("--mask-geometry-csv", type=Path, required=True)
    p_saqc.add_argument("--stage-predictions-csv", type=Path, required=True)
    p_saqc.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_saqc.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_saqc.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_saqc.add_argument("--output-csv", type=Path, required=True)
    p_saqc.set_defaults(func=cmd_surface_area_qc)

    p_saqc_report = sub.add_parser("surface-area-qc-report")
    p_saqc_report.add_argument("--surface-area-qc-csv", type=Path, required=True)
    p_saqc_report.add_argument("--mask-geometry-csv", type=Path, required=True)
    p_saqc_report.add_argument("--stage-predictions-csv", type=Path, required=True)
    p_saqc_report.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_saqc_report.add_argument("--output-root", type=Path, required=True)
    p_saqc_report.add_argument("--output-vs-stage-png", type=Path, required=True)
    p_saqc_report.add_argument("--output-gallery-png", type=Path, required=True)
    p_saqc_report.set_defaults(func=cmd_surface_area_qc_report)

    p_mqqc_report = sub.add_parser("mask-quality-qc-report")
    p_mqqc_report.add_argument("--mask-quality-qc-csv", type=Path, required=True)
    p_mqqc_report.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_mqqc_report.add_argument("--frame-masks-csv", type=Path, required=True)
    p_mqqc_report.add_argument("--output-root", type=Path, required=True)
    p_mqqc_report.add_argument("--output-edge-flag-histogram-png", type=Path, required=True)
    p_mqqc_report.add_argument("--output-edge-flag-gallery-png", type=Path, required=True)
    p_mqqc_report.add_argument("--output-discontinuous-mask-flag-histogram-png", type=Path, required=True)
    p_mqqc_report.add_argument("--output-discontinuous-mask-flag-gallery-png", type=Path, required=True)
    p_mqqc_report.add_argument("--output-overlapping-mask-flag-histogram-png", type=Path, required=True)
    p_mqqc_report.add_argument("--output-overlapping-mask-flag-gallery-png", type=Path, required=True)
    p_mqqc_report.set_defaults(func=cmd_mask_quality_qc_report)

    p_mqqc = sub.add_parser("mask-quality-qc")
    p_mqqc.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_mqqc.add_argument("--frame-masks-csv", type=Path, required=True)
    p_mqqc.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_mqqc.add_argument("--output-csv", type=Path, required=True)
    p_mqqc.set_defaults(func=cmd_mask_quality_qc)

    p_fqc = sub.add_parser("focus-qc")
    p_fqc.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_fqc.add_argument("--frame-masks-csv", type=Path, required=True)
    p_fqc.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_fqc.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_fqc.add_argument("--output-csv", type=Path, required=True)
    p_fqc.set_defaults(func=cmd_focus_qc)

    p_mbqc_report = sub.add_parser("motion-blur-qc-report")
    p_mbqc_report.add_argument("--motion-blur-qc-csv", type=Path, required=True)
    p_mbqc_report.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_mbqc_report.add_argument("--output-root", type=Path, required=True)
    p_mbqc_report.add_argument("--output-histogram-png", type=Path, required=True)
    p_mbqc_report.add_argument("--output-gallery-png", type=Path, required=True)
    p_mbqc_report.set_defaults(func=cmd_motion_blur_qc_report)

    p_mbqc = sub.add_parser("motion-blur-qc")
    p_mbqc.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_mbqc.add_argument("--frame-masks-csv", type=Path, required=True)
    p_mbqc.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_mbqc.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_mbqc.add_argument("--output-csv", type=Path, required=True)
    p_mbqc.set_defaults(func=cmd_motion_blur_qc)

    p_dd = sub.add_parser("death-detection")
    p_dd.add_argument("--fraction-alive-csv", type=Path, required=True)
    p_dd.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_dd.add_argument("--plate-metadata-csv", type=Path, required=True)
    p_dd.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_dd.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_dd.add_argument("--output-qc-csv", type=Path, required=True)
    p_dd.add_argument("--output-death-event-csv", type=Path, required=True)
    p_dd.set_defaults(func=cmd_death_detection)

    p_dd_report = sub.add_parser("death-detection-report")
    p_dd_report.add_argument("--death-detection-qc-csv", type=Path, required=True)
    p_dd_report.add_argument("--fraction-alive-csv", type=Path, required=True)
    p_dd_report.add_argument("--output-experiment-png", type=Path, required=True)
    p_dd_report.add_argument("--output-curtain-png", type=Path, required=True)
    p_dd_report.add_argument("--output-death-time-png", type=Path, required=True)
    p_dd_report.add_argument("--output-well-survival-png", type=Path, required=True)
    p_dd_report.set_defaults(func=cmd_death_detection_report)

    p_ar = sub.add_parser("analysis-ready")
    p_ar.add_argument("--curvature-metrics-csv", type=Path, required=True)
    p_ar.add_argument("--stage-predictions-csv", type=Path, required=True)
    p_ar.add_argument("--mask-geometry-csv", type=Path, required=True)
    p_ar.add_argument("--pose-kinematics-csv", type=Path, required=True)
    p_ar.add_argument("--fraction-alive-csv", type=Path, required=True)
    p_ar.add_argument("--latents-parquet", type=Path, required=True)
    p_ar.add_argument("--snip-qc-parquet", type=Path, required=True)
    p_ar.add_argument("--plate-metadata-csv", type=Path, required=True)
    p_ar.add_argument("--output-parquet", type=Path, required=True)
    p_ar.set_defaults(func=cmd_analysis_ready)

    p_ar_report = sub.add_parser("analysis-ready-report")
    p_ar_report.add_argument("--analysis-ready-parquet", type=Path, required=True)
    p_ar_report.add_argument("--death-event-csv", type=Path, required=True)
    p_ar_report.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_ar_report.add_argument("--output-root", type=Path, required=True)
    p_ar_report.add_argument("--output-latent-pca-qc-state-png", type=Path, required=True)
    p_ar_report.add_argument("--output-latent-pca-stage-png", type=Path, required=True)
    p_ar_report.add_argument("--output-latent-pca-genotype-png", type=Path, required=True)
    p_ar_report.add_argument("--output-post-qc-area-um2-gallery-png", type=Path, required=True)
    p_ar_report.add_argument("--output-post-qc-baseline-deviation-gallery-png", type=Path, required=True)
    p_ar_report.add_argument("--output-survival-over-stage-png", type=Path, required=True)
    p_ar_report.add_argument("--output-genotype-survival-panel-png", type=Path, required=True)
    p_ar_report.add_argument("--output-well-survival-over-stage-all-png", type=Path, required=True)
    p_ar_report.add_argument("--output-well-survival-over-stage-by-genotype-png", type=Path, required=True)
    p_ar_report.set_defaults(func=cmd_analysis_ready_report)

    for verb, fn in (
        ("validate-death-detection-qc", cmd_validate_death_detection_qc),
        ("validate-death-event", cmd_validate_death_event),
        ("validate-snip-qc", cmd_validate_snip_qc),
    ):
        p = sub.add_parser(verb)
        p.add_argument("--input-csv", type=Path, required=True)
        p.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
        p.add_argument("--output-flag", type=Path, required=True)
        p.set_defaults(func=fn)

    p_snipqc_write = sub.add_parser("write-snip-qc-resolved-sources")
    p_snipqc_write.add_argument("--output-root", type=Path, required=True)
    p_snipqc_write.add_argument("--experiment", required=True)
    p_snipqc_write.add_argument("--well-id", required=True)
    p_snipqc_write.add_argument("--exclusion-flags-json", required=True)
    p_snipqc_write.add_argument("--output-json", type=Path, required=True)
    p_snipqc_write.set_defaults(func=cmd_write_snip_qc_resolved_sources)

    p_snipqc = sub.add_parser("snip-qc")
    p_snipqc.add_argument("--resolved-sources-json-path", type=Path, required=True)
    p_snipqc.add_argument("--snip-inventory-csv", type=Path, required=True)
    p_snipqc.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_snipqc.add_argument("--output-csv", type=Path, required=True)
    p_snipqc.set_defaults(func=cmd_snip_qc)

    p_snipqc_report = sub.add_parser("snip-qc-report")
    p_snipqc_report.add_argument("--snip-qc-path", type=Path, required=True)
    p_snipqc_report.add_argument("--output-exclusion-reasons-png", type=Path, required=True)
    p_snipqc_report.set_defaults(func=cmd_snip_qc_report)

    p_fm = sub.add_parser("frame-masks")
    p_fm.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_fm.add_argument("--frame-detections-csv", type=Path, required=True)
    p_fm.add_argument("--output-csv", type=Path, required=True)
    p_fm.add_argument("--prompt-seeds-csv", type=Path, required=True)
    # SAM2 model paths — pass relative paths exactly as stored in config.yaml / env.yaml;
    # load_sam2_video_predictor resolves them against the models root (see models/sam2.py).
    p_fm.add_argument("--sam2-models-root", type=Path, required=True)
    p_fm.add_argument("--sam2-config", type=Path, required=True)
    p_fm.add_argument("--sam2-checkpoint", type=Path, required=True)
    p_fm.add_argument("--sam2-model-id", default="sam2_video")
    p_fm.add_argument("--device", default="cuda")
    p_fm.set_defaults(func=cmd_frame_masks)

    p_fm_precomputed = sub.add_parser("ingest-precomputed-frame-masks")
    p_fm_precomputed.add_argument(
        "--precomputed-frame-masks-csv", type=Path, required=True
    )
    p_fm_precomputed.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_fm_precomputed.add_argument("--frame-detections-csv", type=Path)
    p_fm_precomputed.add_argument("--well-id", required=True)
    p_fm_precomputed.add_argument("--output-csv", type=Path, required=True)
    p_fm_precomputed.add_argument("--prompt-seeds-csv", type=Path, required=True)
    p_fm_precomputed.add_argument("--require-detection-audit", default="true")
    p_fm_precomputed.set_defaults(func=cmd_ingest_precomputed_frame_masks)

    p_fm_validate = sub.add_parser("validate-frame-masks")
    p_fm_validate.add_argument("--input-csv", type=Path, required=True)
    p_fm_validate.add_argument("--frame-inventory-csv", type=Path, required=True)
    p_fm_validate.add_argument("--output-flag", type=Path, required=True)
    p_fm_validate.set_defaults(func=cmd_validate_frame_masks)

    p_per_build = sub.add_parser("build-physical-embryo-registry")
    p_per_build.add_argument("--frame-masks-csv", type=Path, required=True)
    p_per_build.add_argument("--output-csv", type=Path, required=True)
    p_per_build.set_defaults(func=cmd_build_physical_embryo_registry)

    p_per_validate = sub.add_parser("validate-physical-embryo-registry")
    p_per_validate.add_argument("--input-csv", type=Path, required=True)
    p_per_validate.add_argument("--output-flag", type=Path, required=True)
    p_per_validate.set_defaults(func=cmd_validate_physical_embryo_registry)

    p_per_merge = sub.add_parser("merge-physical-embryo-registry")
    p_per_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_per_merge.add_argument("--output-csv", type=Path, required=True)
    p_per_merge.set_defaults(func=cmd_merge_physical_embryo_registry)

    p_per_report = sub.add_parser("physical-embryo-registry-report")
    p_per_report.add_argument("--physical-embryo-registry-csv", type=Path, required=True)
    p_per_report.add_argument("--frame-masks-csv", type=Path, required=True)
    p_per_report.add_argument("--output-embryos-per-well-png", type=Path, required=True)
    p_per_report.add_argument("--output-embryos-per-well-plate-png", type=Path, required=True)
    p_per_report.add_argument("--output-embryos-per-well-over-time-png", type=Path, required=True)
    p_per_report.set_defaults(func=cmd_physical_embryo_registry_report)

    p_stage_rollup = sub.add_parser("stage-rollup-report")
    p_stage_rollup.add_argument("--stage", required=True)
    p_stage_rollup.add_argument("--data-root", type=Path, required=True)
    p_stage_rollup.add_argument("--experiment", required=True)
    p_stage_rollup.add_argument("--output-html", type=Path, required=True)
    p_stage_rollup.set_defaults(func=cmd_stage_rollup_report)

    p_le_validate = sub.add_parser("validate-latent-embeddings")
    p_le_validate.add_argument("--input-parquet", type=Path, required=True)
    p_le_validate.add_argument("--output-flag", type=Path, required=True)
    p_le_validate.set_defaults(func=cmd_validate_latent_embeddings)

    p_le_merge = sub.add_parser("merge-latent-embeddings")
    p_le_merge.add_argument("--inputs", type=Path, nargs="+")
    p_le_merge.add_argument("--data-root", type=Path)
    p_le_merge.add_argument("--experiment-id")
    p_le_merge.add_argument("--output-parquet", type=Path, required=True)
    p_le_merge.set_defaults(func=cmd_merge_latent_embeddings)

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
