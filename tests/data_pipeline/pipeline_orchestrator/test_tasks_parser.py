"""Parser coverage for pipeline_orchestrator.tasks."""

from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import pandas as pd

from data_pipeline.pipeline_orchestrator import tasks
from data_pipeline.acquisition.image_materialization.resolved_product_plans import (
    write_resolved_product_plan_for_well,
)


def test_materialize_well_parses_local_well_label():
    parser = tasks.build_parser()

    # legacy alias still parses to the same command (back-compat for existing smoke invocations).
    args = parser.parse_args([
        "materialize-yx1-well-candidate",
        "--experiment",
        "20250912",
        "--well-id",
        "20250912_B01",
        "--well-index",
        "B01",
        "--acquisition-inventory-csv",
        "acquisition.csv",
        "--position-well-mapping-csv",
        "position_well_mapping.csv",
        "--built-image-data-dir",
        "built_image_data",
        "--frame-inventory-csv",
        "out/frame_inventory.csv",
        "--done-flag",
        "out/frame_inventory.done",
        "--device",
        "cpu",
    ])

    assert args.func is tasks.cmd_materialize_well
    assert args.scope == "yx1"  # default
    assert args.well_index == "B01"
    assert args.acquisition_inventory_csv == Path("acquisition.csv")
    assert args.position_well_mapping_csv == Path("position_well_mapping.csv")
    assert args.frame_inventory_csv == Path("out/frame_inventory.csv")


def test_materialize_well_joins_position_mapping(tmp_path):
    acquisition_csv = tmp_path / "acquisition.csv"
    mapping_csv = tmp_path / "position_well_mapping.csv"
    frame_inventory_csv = tmp_path / "out" / "frame_inventory.csv"
    done_flag = tmp_path / "out" / "frame_inventory.done"

    pd.DataFrame([
        {
            "experiment_id": "20250912",
            "position_index": 2,
            "time_index": 0,
            "source_nd2_path": "experiment.nd2",
            "micrometers_per_pixel": 0.65,
            "image_width_px": 8,
            "image_height_px": 8,
        }
    ]).to_csv(acquisition_csv, index=False)
    pd.DataFrame([
        {
            "experiment_id": "20250912",
            "position_index": 2,
            "well_index": "B01",
            "well_id": "20250912_B01",
            "mapping_method": "test",
        }
    ]).to_csv(mapping_csv, index=False)

    def _fake_materialize_yx1_well(**kwargs):
        well_rows = kwargs["well_acquisition_inventory_df"]
        assert list(well_rows["position_index"]) == [2]
        assert list(well_rows["well_index"]) == ["B01"]
        assert list(well_rows["well_id"]) == ["20250912_B01"]
        # The sequencer resolved the plan and handed the backend a resolved (identity) product.
        resolved = kwargs["resolved_plan"]
        assert resolved.products[0].xy_composition == "identity"
        return pd.DataFrame([{"experiment_id": "20250912", "well_index": "B01"}])

    args = Namespace(
        experiment="20250912",
        well_id="20250912_B01",
        well_index="B01",
        scope="yx1",
        acquisition_inventory_csv=acquisition_csv,
        position_well_mapping_csv=mapping_csv,
        built_image_data_dir=tmp_path / "built_image_data",
        frame_inventory_csv=frame_inventory_csv,
        done_flag=done_flag,
        config_yaml=None,
        candidate="true",
        smoke_max_time_indices=None,
        device="cpu",
    )

    with patch(
        "data_pipeline.acquisition.image_materialization.scope.yx1"
        ".materialize_well_yx1.materialize_yx1_well",
        side_effect=_fake_materialize_yx1_well,
    ):
        tasks.cmd_materialize_well(args)

    assert frame_inventory_csv.exists()
    assert done_flag.exists()


def test_write_resolved_product_plan_for_well_command(tmp_path):
    out_json = tmp_path / "BF__z_stack_resolved_product_plan.json"
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        "image_materialization:\n"
        "  products:\n"
        "    - {channel_id: BF, image_product_type: z_stack}\n",
        encoding="utf-8",
    )

    args = Namespace(
        experiment="20250912",
        well_id="20250912_B01",
        scope="yx1",
        product_key="BF__z_stack",
        output_json=out_json,
        config_yaml=config_yaml,
    )
    tasks.cmd_write_resolved_product_plan_for_well(args)

    text = out_json.read_text(encoding="utf-8")
    assert '"product_key": "BF__z_stack"' in text
    assert '"image_product_type": "z_stack"' in text


def test_materialize_image_product_for_well_joins_position_mapping(tmp_path):
    acquisition_csv = tmp_path / "acquisition.csv"
    mapping_csv = tmp_path / "position_well_mapping.csv"
    product_csv = tmp_path / "out" / "20250912_B01_BF__projection__focus_stack_frame_inventory.csv"
    plan_json = tmp_path / "BF__projection__focus_stack_resolved_product_plan.json"

    pd.DataFrame([
        {
            "experiment_id": "20250912",
            "position_index": 2,
            "time_index": 0,
            "source_nd2_path": "experiment.nd2",
            "micrometers_per_pixel": 0.65,
            "image_width_px": 8,
            "image_height_px": 8,
        }
    ]).to_csv(acquisition_csv, index=False)
    pd.DataFrame([
        {
            "experiment_id": "20250912",
            "position_index": 2,
            "well_index": "B01",
            "well_id": "20250912_B01",
            "mapping_method": "test",
        }
    ]).to_csv(mapping_csv, index=False)
    write_resolved_product_plan_for_well(
        experiment_id="20250912",
        well_id="20250912_B01",
        scope_name="yx1",
        config=None,
        product_key="BF__projection__focus_stack",
        output_json=plan_json,
    )

    def _fake_materialize_yx1_product_for_well(**kwargs):
        well_rows = kwargs["well_acquisition_inventory_df"]
        assert list(well_rows["position_index"]) == [2]
        assert kwargs["resolved_product"].image_product_type == "projection"
        return pd.DataFrame([{"experiment_id": "20250912", "well_index": "B01"}])

    args = Namespace(
        experiment="20250912",
        well_id="20250912_B01",
        well_index="B01",
        scope="yx1",
        product_key="BF__projection__focus_stack",
        resolved_product_plan_json=plan_json,
        acquisition_inventory_csv=acquisition_csv,
        position_well_mapping_csv=mapping_csv,
        built_image_data_dir=tmp_path / "built_image_data",
        frame_inventory_product_csv=product_csv,
        candidate="false",
        smoke_max_time_indices=None,
        device="cpu",
    )

    with patch(
        "data_pipeline.acquisition.image_materialization.scope.yx1"
        ".materialize_well_yx1.materialize_yx1_product_for_well",
        side_effect=_fake_materialize_yx1_product_for_well,
    ):
        tasks.cmd_materialize_image_product_for_well(args)

    assert product_csv.exists()


def test_product_shard_discovery_command_parses():
    parser = tasks.build_parser()

    args = parser.parse_args([
        "discover-product-shards-for-well",
        "--experiment",
        "20250912",
        "--well-id",
        "20250912_B01",
        "--frame-inventory-products-dir",
        "frame_inventory_products/per_well/20250912_B01",
        "--output-csv",
        "discovered_product_shards/per_well/20250912_B01/20250912_B01_discovered_product_shards.csv",
    ])

    assert args.func is tasks.cmd_discover_product_shards_for_well
    assert args.frame_inventory_products_dir == Path("frame_inventory_products/per_well/20250912_B01")
    assert args.output_csv == Path(
        "discovered_product_shards/per_well/20250912_B01/20250912_B01_discovered_product_shards.csv"
    )


def test_assemble_well_frame_inventory_command_parses():
    parser = tasks.build_parser()

    args = parser.parse_args([
        "assemble-well-frame-inventory",
        "--discovered-product-shards-csv",
        "discovered_product_shards/per_well/20250912_B01/20250912_B01_discovered_product_shards.csv",
        "--output-csv",
        "frame_inventory/per_well/20250912_B01/20250912_B01_frame_inventory.csv",
    ])

    assert args.func is tasks.cmd_assemble_well_frame_inventory
    assert args.discovered_product_shards_csv == Path(
        "discovered_product_shards/per_well/20250912_B01/20250912_B01_discovered_product_shards.csv"
    )
    assert args.output_csv == Path(
        "frame_inventory/per_well/20250912_B01/20250912_B01_frame_inventory.csv"
    )


def test_snip_auxiliary_masks_models_root_is_environment(tmp_path):
    """REGRESSION (Tier-1 through-line, 2026-06-26): model route is ENVIRONMENT, not data layout.

    The env --models-root (env.yaml.paths.models_root) is authoritative and REPLACES any config
    unet_snip.models_root; the per-family ``checkpoint`` key (carrying e.g. "segmentation/<family>")
    is the only model path the science config owns. The original bug clobbered the config models_root
    AND dropped the family segment, yielding <env_root>/<family> (checkpoint-not-found) instead of
    <env_root>/segmentation/<family>. This pins env-authoritative resolution without any
    data-relative-path rebasing onto output_root (explicitly rejected).
    """
    import yaml

    config = {
        "unet_snip": {
            "models_root": "models",  # placeholder — env --models-root must REPLACE this
            "device": "cpu",
            "models": {
                "bubble": {"checkpoint": "segmentation/bubble_v0_0100"},
            },
        },
    }
    config_yaml = tmp_path / "merged_config.yaml"
    config_yaml.write_text(yaml.safe_dump(config))

    env_models_root = tmp_path / "env" / "weights_anywhere"  # NOT under output_root

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    args = Namespace(
        config_yaml=config_yaml,
        snip_inventory_csv=tmp_path / "snip_inventory.csv",
        output_root=tmp_path / "data_out",
        output_csv=tmp_path / "out.csv",
        models_root=env_models_root,
    )

    with patch(
        "data_pipeline.segmentation.backends.unet_snip.entrypoint.run_snip_auxiliary_masks",
        new=_capture,
    ), patch(
        "data_pipeline.snip_processing.snip_frame_shape.resolve_snip_frame_shape",
        return_value=(288, 128),
    ):
        tasks.cmd_snip_auxiliary_masks(args)

    passed = captured["unet_snip_config"]
    # The env root REPLACED the config "models" placeholder (env-authoritative) ...
    assert passed["models_root"] == str(env_models_root)
    # ... and is NOT rebased under output_root (no data-relative-path pollution).
    assert str(args.output_root) not in passed["models_root"]
    # The family segment still lives in the checkpoint key, so the resolved path is
    # <env_root>/segmentation/bubble_v0_0100 — the very path the bug failed to build.
    from pathlib import Path as _P
    resolved = _P(passed["models_root"]) / passed["models"]["bubble"]["checkpoint"]
    assert resolved == env_models_root / "segmentation" / "bubble_v0_0100"
