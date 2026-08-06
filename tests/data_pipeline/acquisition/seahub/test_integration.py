from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from data_pipeline.acquisition.seahub.integration import (
    SeaHubIntegrationConfig,
    assign_operational_identity,
    build_embryo_ingest,
    build_seahub_dropin_bundle,
    materialize_planned_experiment,
)


def _fov(source_path: Path, *, source_fov_id: str = "abc123") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": source_fov_id,
                "image_path": str(source_path),
                "relative_path": f"GENE1/images/{source_path.name}",
                "filename": source_path.name,
                "experiment_id": "GENE1",
                "image_role": "eight_embryo_fov",
                "excluded_path": False,
                "read_error": pd.NA,
                "stage_hpf": 24.0,
                "stage_source_label": "24hpf",
                "stage_match_method": "stage_exact",
                "perturbation_parsed": "ctrl",
                "perturbation_key": "ctrl",
                "perturbation_domain": "genetic",
                "fov_label": "A",
                "metadata_match_status": "unmatched_condition",
                "metadata_match_score": 0.4,
            }
        ]
    )


def _detections(*, source_fov_id: str = "abc123", count: int = 8) -> pd.DataFrame:
    rows = []
    for position in range(1, count + 1):
        offset = position - 1
        x1 = (offset % 4) * 40
        y1 = (offset // 4) * 40
        rows.append(
            {
                "image_id": source_fov_id,
                "embryo_position": position,
                "crop_x1_px": x1,
                "crop_y1_px": y1,
                "crop_x2_px": x1 + 30,
                "crop_y2_px": y1 + 30,
                "segmentation_qc_status": "pass",
                "selected_detection_count": count,
            }
        )
    return pd.DataFrame(rows)


def test_detection_count_failure_never_emits_partial_embryos(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    embryos, failures = build_embryo_ingest(
        _fov(source), _detections(count=7)
    )
    assert embryos.empty
    assert failures["failure_reason"].tolist() == ["detection_count_not_8"]
    assert failures["detection_count"].tolist() == [7]


def test_operational_shards_are_deterministic_one_embryo_wells():
    ingest = pd.DataFrame(
        [
            {
                "source_experiment_id": "GENE1",
                "source_fov_id": f"fov-{index // 8:03d}",
                "source_embryo_id": f"embryo-{index:03d}",
                "embryo_position": index % 8 + 1,
                "stage_hpf": 24.0,
                "perturbation_key": "ctrl",
                "fov_label": f"{index // 8:03d}",
            }
            for index in range(97)
        ]
    )
    assigned = assign_operational_identity(
        ingest, config=SeaHubIntegrationConfig()
    )
    assert assigned["well_id"].nunique() == 97
    assert assigned.iloc[0]["well_index"] == "A01"
    assert assigned.iloc[95]["well_index"] == "H12"
    assert assigned.iloc[96]["well_index"] == "A01"
    assert assigned.iloc[0]["experiment_id"].endswith("shard001")
    assert assigned.iloc[96]["experiment_id"].endswith("shard002")
    assert "physical_embryo_id" not in assigned.columns


def test_bundle_materializes_valid_single_z_dropin(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "bundle",
        config=SeaHubIntegrationConfig(),
    )

    assert len(result.well_provenance) == 8
    assert result.detection_failures.empty
    assert result.canvas_width_px == 32
    assert result.canvas_height_px == 32
    experiment = result.experiment_manifest.iloc[0]
    frame = pd.read_csv(experiment["frame_inventory_csv"])
    plate = pd.read_csv(experiment["plate_metadata_csv"])
    assert set(frame["image_kind"]) == {"single_z"}
    assert set(frame["source_scope"]) == {"seahub"}
    assert frame["z_position"].isna().all()
    assert set(frame["image_micrometers_per_pixel"]) == {7.8}
    assert set(frame["calibration_status"]) == {"placeholder"}
    assert plate["well_id"].nunique() == 8
    assert set(plate["embryos_per_well"]) == {1}
    assert plate["reconciliation_failure_passed_through"].all()
    assert set(plate["operational_integration_date"]) == {20260723}

    first_image = Path(experiment["image_root"]) / frame.iloc[0]["image_path"]
    with Image.open(first_image) as image:
        assert image.mode == "L"
        assert image.size == (32, 32)

    config = yaml.safe_load(Path(experiment["runtime_config_yaml"]).read_text())
    assert config["front_end"]["mode"] == "dropin"
    assert config["microscope"] == "SeaHub"
    assert config["dropin"]["plate_metadata_csv"] == experiment[
        "plate_metadata_csv"
    ]


def test_plan_once_then_materialize_one_cluster_shard(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "bundle",
        materialize_images=False,
    )
    experiment = result.experiment_manifest.iloc[0]
    frame_csv = Path(experiment["frame_inventory_csv"])
    assert not frame_csv.with_suffix(".csv.validated").exists()
    flag = materialize_planned_experiment(
        bundle_root=tmp_path / "bundle",
        experiment_id=experiment["experiment_id"],
    )
    assert flag.exists()
    frame = pd.read_csv(frame_csv)
    assert (
        Path(experiment["image_root"]) / frame.iloc[0]["image_path"]
    ).exists()
