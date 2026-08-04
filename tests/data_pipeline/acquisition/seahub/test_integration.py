from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from data_pipeline.acquisition.seahub.integration import (
    SeaHubIntegrationConfig,
    assign_operational_identity,
    build_embryo_ingest,
    build_seahub_dropin_bundle,
    materialize_planned_experiment,
)
from data_pipeline.acquisition.seahub.cli import (
    _read_scale_mask_manifest,
    build_parser,
)
from data_pipeline.acquisition.seahub.production_safety import (
    preflight_materialized_experiment,
)
from data_pipeline.acquisition.seahub.scale_calibration import (
    calibrate_reconciled_source_fovs,
)
from data_pipeline.object_extraction.segmentation.masks.mask_rle import (
    decode_binary_mask_rle,
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


def _scale_calibration(source_path: Path) -> pd.DataFrame:
    masks = pd.DataFrame(
        {
            "image_id": ["abc123"] * 8,
            "embryo_position": list(range(1, 9)),
            "mask_score": [0.95] * 8,
            "mask_area_px": [30_000.0] * 8,
        }
    )
    return calibrate_reconciled_source_fovs(_fov(source_path), masks)


def _persisted_source_masks(tmp_path: Path) -> pd.DataFrame:
    mask_dir = tmp_path / "source_masks"
    mask_dir.mkdir()
    rows = []
    for detection in _detections().to_dict("records"):
        position = int(detection["embryo_position"])
        x1 = int(detection["crop_x1_px"]) + 7
        y1 = int(detection["crop_y1_px"]) + 8
        x2 = x1 + 14
        y2 = y1 + 12
        mask = np.zeros((80, 160), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask_path = (mask_dir / f"abc123__p{position:02d}.png").resolve()
        Image.fromarray(mask).save(mask_path)
        rows.append(
            {
                "source_fov_id": "abc123",
                "embryo_position": position,
                "mask_path": str(mask_path),
                "mask_score": 0.97,
                "mask_area_px": int((mask > 0).sum()),
                "raw_mask_area_px": int((mask > 0).sum()),
                "component_count_raw": 1,
                "component_selection_method": "prompt_center",
                "removed_component_area_px": 0,
                "holes_filled_px": 0,
                "cleaned_to_prompt_area_ratio": 0.2,
                "mask_bbox_x1_px": x1,
                "mask_bbox_y1_px": y1,
                "mask_bbox_x2_px": x2,
                "mask_bbox_y2_px": y2,
            }
        )
    return pd.DataFrame.from_records(rows)


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
    assert frame["image_path"].map(lambda value: Path(value).is_absolute()).all()
    assert frame["image_path"].map(lambda value: Path(value).is_file()).all()
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
    assert config["frame_detections"]["use_model_server"] is True
    assert config["frame_masks"]["use_model_server"] is True
    assert config["unet_snip"]["use_model_server"] is True


def test_bundle_broadcasts_one_mask_inferred_scale_to_all_fov_embryos(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    calibration = _scale_calibration(source)

    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "calibrated_bundle",
        fov_scale_calibration=calibration,
    )

    assert len(result.fov_scale_calibration) == 1
    expected_scale = calibration.iloc[0]["image_micrometers_per_pixel"]
    experiment = result.experiment_manifest.iloc[0]
    frame = pd.read_csv(experiment["frame_inventory_csv"])
    plate = pd.read_csv(experiment["plate_metadata_csv"])
    persisted = pd.read_csv(
        tmp_path
        / "calibrated_bundle"
        / "integration"
        / "fov_scale_calibration.csv"
    )

    assert len(frame) == 8
    assert frame["image_micrometers_per_pixel"].nunique() == 1
    assert frame["image_micrometers_per_pixel"].iloc[0] == pytest.approx(
        expected_scale
    )
    assert set(frame["calibration_status"]) == {"placeholder"}
    assert set(frame["scale_estimation_status"]) == {"mask_area_regularized"}
    assert plate["micrometers_per_pixel"].tolist() == pytest.approx(
        [expected_scale] * 8
    )
    assert persisted.iloc[0]["source_fov_id"] == "abc123"
    assert persisted.iloc[0]["n_valid_masks"] == 8


def test_bundle_materializes_authoritative_source_masks_and_seahub_runtime(tmp_path):
    source = tmp_path / "fov.jpg"
    gradient = np.tile(np.arange(160, dtype=np.uint8), (80, 1))
    Image.fromarray(gradient).save(source)
    masks = _persisted_source_masks(tmp_path)

    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "masked_bundle",
        source_mask_manifest=masks,
    )

    experiment = result.experiment_manifest.iloc[0]
    frame_masks_path = Path(experiment["precomputed_frame_masks_csv"])
    frame_masks = pd.read_csv(frame_masks_path)
    assert len(frame_masks) == 8
    assert frame_masks["image_id"].nunique() == 8
    assert set(frame_masks["segmentation_backend"]) == {
        "sam2_image_precomputed"
    }
    decoded = decode_binary_mask_rle(json.loads(frame_masks.iloc[0]["mask_rle"]))
    assert decoded.sum() == 14 * 12

    runtime = yaml.safe_load(Path(experiment["runtime_config_yaml"]).read_text())
    assert runtime["frame_masks"]["mode"] == "precomputed"
    assert runtime["frame_masks"]["precomputed_csv"] == str(
        frame_masks_path.resolve()
    )
    assert runtime["frame_masks"]["require_detection_audit"] is True
    assert runtime["snip_processing"]["apply_clahe"] is False
    assert preflight_materialized_experiment(
        bundle_root=tmp_path / "masked_bundle",
        experiment_id=experiment["experiment_id"],
    ).is_file()


def test_bundle_rejects_source_mask_with_hole(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("L", (160, 80), 100).save(source)
    masks = _persisted_source_masks(tmp_path)
    bad_path = Path(masks.loc[0, "mask_path"])
    with Image.open(bad_path) as image:
        bad_mask = np.asarray(image).copy()
    bad_mask[12:15, 10:13] = 0
    Image.fromarray(bad_mask).save(bad_path)
    masks.loc[0, "mask_area_px"] = int((bad_mask > 0).sum())

    with pytest.raises(ValueError, match="still contains holes"):
        build_seahub_dropin_bundle(
            _fov(source),
            _detections(),
            output_root=tmp_path / "hole_bundle",
            source_mask_manifest=masks,
        )


def test_bundle_requires_complete_source_mask_coverage(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("L", (160, 80), 100).save(source)
    masks = _persisted_source_masks(tmp_path).iloc[:-1].copy()

    with pytest.raises(ValueError, match="does not cover every detected"):
        build_seahub_dropin_bundle(
            _fov(source),
            _detections(),
            output_root=tmp_path / "incomplete_mask_bundle",
            source_mask_manifest=masks,
            materialize_images=False,
        )


def test_bundle_rejects_source_mask_manifest_blowout_ratio(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("L", (160, 80), 100).save(source)
    masks = _persisted_source_masks(tmp_path)
    masks.loc[0, "cleaned_to_prompt_area_ratio"] = 0.95

    with pytest.raises(ValueError, match="cleaned_to_prompt_area_ratio"):
        build_seahub_dropin_bundle(
            _fov(source),
            _detections(),
            output_root=tmp_path / "blowout_ratio_bundle",
            source_mask_manifest=masks,
            materialize_images=False,
        )


def test_bundle_rejects_source_mask_assigned_to_wrong_detector_crop(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("L", (160, 80), 100).save(source)
    masks = _persisted_source_masks(tmp_path)
    swap_columns = [
        "mask_path",
        "mask_bbox_x1_px",
        "mask_bbox_y1_px",
        "mask_bbox_x2_px",
        "mask_bbox_y2_px",
    ]
    first_values = masks.loc[0, swap_columns].copy()
    masks.loc[0, swap_columns] = masks.loc[7, swap_columns].to_numpy()
    masks.loc[7, swap_columns] = first_values.to_numpy()

    with pytest.raises(ValueError, match="mislabeled/wrong-neighbor"):
        build_seahub_dropin_bundle(
            _fov(source),
            _detections(),
            output_root=tmp_path / "wrong_neighbor_bundle",
            source_mask_manifest=masks,
        )


def test_preflight_rejects_duplicate_authoritative_mask_rows(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("L", (160, 80), 100).save(source)
    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "duplicate_preflight_bundle",
        source_mask_manifest=_persisted_source_masks(tmp_path),
    )
    experiment = result.experiment_manifest.iloc[0]
    masks_path = Path(experiment["precomputed_frame_masks_csv"])
    masks = pd.read_csv(masks_path)
    pd.concat([masks, masks.iloc[[0]]], ignore_index=True).to_csv(
        masks_path, index=False
    )

    with pytest.raises(ValueError, match="mask_id values must be unique"):
        preflight_materialized_experiment(
            bundle_root=tmp_path / "duplicate_preflight_bundle",
            experiment_id=experiment["experiment_id"],
        )


def test_bundle_rejects_stale_scale_stage(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    calibration = _scale_calibration(source)
    calibration["stage_hpf"] = 48.0

    with pytest.raises(ValueError, match="stage_hpf disagrees"):
        build_seahub_dropin_bundle(
            _fov(source),
            _detections(),
            output_root=tmp_path / "bad_scale_bundle",
            fov_scale_calibration=calibration,
            materialize_images=False,
        )


def test_production_bundle_cli_requires_scale_mask_manifest():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build-bundle",
                "--reconciled-fovs-csv",
                "reconciled.csv",
                "--detection-manifest-csv",
                "detections.csv",
                "--output-root",
                "bundle",
            ]
        )

    args = parser.parse_args(
        [
            "build-bundle",
            "--reconciled-fovs-csv",
            "reconciled.csv",
            "--detection-manifest-csv",
            "detections.csv",
            "--scale-mask-manifest-csv",
            "scale_masks.csv",
            "--output-root",
            "bundle",
        ]
    )
    assert args.scale_mask_manifest_csv == Path("scale_masks.csv")


def test_scale_mask_reader_rejects_partial_checkpoint(tmp_path):
    manifest_path = tmp_path / "sam2_mask_areas.csv"
    pd.DataFrame(
        {
            "source_fov_id": ["fov-1"],
            "mask_area_px": [1000],
            "checkpoint_complete": [False],
            "processed_fov_count": [1],
            "total_fov_count": [2],
        }
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="partial checkpoint"):
        _read_scale_mask_manifest(manifest_path)


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
    planned_frame = pd.read_csv(frame_csv)
    assert planned_frame["image_path"].map(
        lambda value: Path(value).is_absolute()
    ).all()
    assert not planned_frame["image_path"].map(
        lambda value: Path(value).exists()
    ).any()
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
    assert preflight_materialized_experiment(
        bundle_root=tmp_path / "bundle",
        experiment_id=experiment["experiment_id"],
    ).is_file()


def test_plan_then_materialize_persists_authoritative_source_masks(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("L", (160, 80), 120).save(source)
    masks = _persisted_source_masks(tmp_path)
    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "masked_plan",
        source_mask_manifest=masks,
        materialize_images=False,
    )
    experiment = result.experiment_manifest.iloc[0]
    frame_masks_csv = Path(experiment["precomputed_frame_masks_csv"])
    assert not frame_masks_csv.exists()

    materialize_planned_experiment(
        bundle_root=tmp_path / "masked_plan",
        experiment_id=experiment["experiment_id"],
    )

    written_masks = pd.read_csv(frame_masks_csv)
    assert len(written_masks) == 8
    assert written_masks["well_id"].nunique() == 8
    assert preflight_materialized_experiment(
        bundle_root=tmp_path / "masked_plan",
        experiment_id=experiment["experiment_id"],
    ).is_file()


def test_bundle_refuses_nonempty_output_root(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    output_root = tmp_path / "reused_bundle"
    output_root.mkdir()
    (output_root / "stale.csv").write_text("old run\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be fresh and empty"):
        build_seahub_dropin_bundle(
            _fov(source),
            _detections(),
            output_root=output_root,
            materialize_images=False,
        )


def test_production_preflight_rejects_relative_image_path(tmp_path):
    source = tmp_path / "fov.jpg"
    Image.new("RGB", (160, 80), (230, 230, 230)).save(source)
    result = build_seahub_dropin_bundle(
        _fov(source),
        _detections(),
        output_root=tmp_path / "bundle",
    )
    experiment = result.experiment_manifest.iloc[0]
    frame_csv = Path(experiment["frame_inventory_csv"])
    frame = pd.read_csv(frame_csv)
    absolute = Path(frame.loc[0, "image_path"])
    frame.loc[0, "image_path"] = str(
        absolute.relative_to(Path(experiment["image_root"]))
    )
    frame.to_csv(frame_csv, index=False)

    with pytest.raises(ValueError, match="require absolute image_path"):
        preflight_materialized_experiment(
            bundle_root=tmp_path / "bundle",
            experiment_id=experiment["experiment_id"],
        )
