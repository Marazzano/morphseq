"""Parser coverage for pipeline_orchestrator.tasks."""

from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import pandas as pd

from data_pipeline.pipeline_orchestrator import tasks


def test_materialize_yx1_well_candidate_parses_local_well_label():
    parser = tasks.build_parser()

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
        "--nd2-path",
        "experiment.nd2",
        "--built-image-data-dir",
        "built_image_data",
        "--frame-inventory-csv",
        "out/frame_inventory.csv",
        "--done-flag",
        "out/frame_inventory.done",
        "--device",
        "cpu",
    ])

    assert args.func is tasks.cmd_materialize_yx1_well_candidate
    assert args.well_index == "B01"
    assert args.acquisition_inventory_csv == Path("acquisition.csv")
    assert args.position_well_mapping_csv == Path("position_well_mapping.csv")
    assert args.frame_inventory_csv == Path("out/frame_inventory.csv")


def test_materialize_yx1_well_candidate_joins_position_mapping(tmp_path):
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
        return pd.DataFrame([{"experiment_id": "20250912", "well_index": "B01"}])

    args = Namespace(
        experiment="20250912",
        well_id="20250912_B01",
        well_index="B01",
        acquisition_inventory_csv=acquisition_csv,
        position_well_mapping_csv=mapping_csv,
        nd2_path=tmp_path / "experiment.nd2",
        built_image_data_dir=tmp_path / "built_image_data",
        frame_inventory_csv=frame_inventory_csv,
        done_flag=done_flag,
        device="cpu",
    )

    with patch(
        "data_pipeline.image_materialization.stitched.scope.yx1"
        ".materialize_yx1_stitched_images.materialize_yx1_well",
        side_effect=_fake_materialize_yx1_well,
    ):
        tasks.cmd_materialize_yx1_well_candidate(args)

    assert frame_inventory_csv.exists()
    assert done_flag.exists()
