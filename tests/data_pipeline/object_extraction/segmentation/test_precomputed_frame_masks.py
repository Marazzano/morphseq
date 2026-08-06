from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.frame_masks_contract import (
    FRAME_MASKS_REQUIRED_COLUMNS,
    no_mask_frame_mask_row,
)
from data_pipeline.object_extraction.segmentation.precomputed_frame_masks import (
    prepare_precomputed_frame_masks_for_well,
    select_precomputed_frame_masks_for_well,
    write_precomputed_frame_masks_for_well,
)
from data_pipeline.shared.identifiers import build_no_mask_id


EXP = "20260802_seahub_TEST_shard001"
WELL_A = f"{EXP}_A01"
WELL_B = f"{EXP}_A02"


def _inventory(well_id: str, well_index: str) -> pd.DataFrame:
    image_id = f"{well_id}_BF_t0000"
    return pd.DataFrame(
        [
            {
                "experiment_id": EXP,
                "well_index": well_index,
                "well_id": well_id,
                "image_id": image_id,
                "channel_id": "BF",
                "time_index": 0,
                "z_index": pd.NA,
                "image_path": f"/images/{image_id}.jpg",
                "image_width_px": 100,
                "image_height_px": 80,
            }
        ]
    )


def _precomputed() -> pd.DataFrame:
    rows = [
        no_mask_frame_mask_row(_inventory(WELL_B, "A02").iloc[0]),
        no_mask_frame_mask_row(_inventory(WELL_A, "A01").iloc[0]),
    ]
    frame_masks = pd.DataFrame(rows)
    frame_masks["upstream_provenance"] = ["other-well", "authoritative"]
    return frame_masks


def _detections() -> pd.DataFrame:
    frame = _inventory(WELL_A, "A01").iloc[0]
    return pd.DataFrame(
        [
            {
                "experiment_id": EXP,
                "well_id": WELL_A,
                "image_id": frame["image_id"],
                "time_index": 0,
                "z_index": pd.NA,
                "channel_id": "BF",
                "image_path": frame["image_path"],
                "image_width_px": 100,
                "image_height_px": 80,
                "detection_id": f"{frame['image_id']}_det0000",
                "detector_backend": "groundingdino",
                "detector_model_id": "audit-only",
                "class_label": "individual embryo",
                "confidence": 0.8,
                "bbox_x_min_px": 10.0,
                "bbox_y_min_px": 11.0,
                "bbox_x_max_px": 40.0,
                "bbox_y_max_px": 41.0,
                "bbox_format": "xyxy_px_abs",
                "is_kept": True,
            }
        ]
    )


def test_selects_only_requested_well_and_emits_exact_canonical_columns() -> None:
    selected = select_precomputed_frame_masks_for_well(
        _precomputed(), _inventory(WELL_A, "A01"), well_id=WELL_A
    )

    assert list(selected.columns) == list(FRAME_MASKS_REQUIRED_COLUMNS)
    assert selected["well_id"].tolist() == [WELL_A]
    assert selected["mask_id"].tolist() == [build_no_mask_id(f"{WELL_A}_BF_t0000")]
    assert "upstream_provenance" not in selected.columns


def test_precomputed_rows_are_validated_against_per_well_inventory() -> None:
    bad = _precomputed()
    bad.loc[bad["well_id"] == WELL_A, "image_path"] = "/wrong/frame.jpg"

    with pytest.raises(ValueError, match="disagreeing with reference_frame_inventory"):
        select_precomputed_frame_masks_for_well(
            bad, _inventory(WELL_A, "A01"), well_id=WELL_A
        )


def test_detection_audit_is_required_but_cannot_change_authoritative_masks() -> None:
    source = _precomputed()
    inventory = _inventory(WELL_A, "A01")

    with pytest.raises(ValueError, match="requires a per-well frame_detections"):
        prepare_precomputed_frame_masks_for_well(
            source,
            inventory,
            well_id=WELL_A,
            frame_detections=None,
            require_detection_audit=True,
        )

    result = prepare_precomputed_frame_masks_for_well(
        source,
        inventory,
        well_id=WELL_A,
        frame_detections=_detections(),
        require_detection_audit=True,
    )
    expected = select_precomputed_frame_masks_for_well(source, inventory, well_id=WELL_A)
    pd.testing.assert_frame_equal(result.frame_masks, expected)
    assert result.detection_audit["detection_id"].tolist() == [
        f"{WELL_A}_BF_t0000_det0000"
    ]


def test_path_task_writes_per_well_masks_and_prompt_audit(tmp_path: Path) -> None:
    precomputed_csv = tmp_path / "dropin_frame_masks.csv"
    inventory_csv = tmp_path / "frame_inventory.csv"
    detections_csv = tmp_path / "frame_detections.csv"
    output_csv = tmp_path / "out" / "frame_masks.csv"
    prompt_seeds_csv = tmp_path / "out" / "prompt_seeds.csv"
    _precomputed().to_csv(precomputed_csv, index=False)
    _inventory(WELL_A, "A01").to_csv(inventory_csv, index=False)
    _detections().to_csv(detections_csv, index=False)

    write_precomputed_frame_masks_for_well(
        precomputed_frame_masks_csv=precomputed_csv,
        frame_inventory_csv=inventory_csv,
        frame_detections_csv=detections_csv,
        well_id=WELL_A,
        output_csv=output_csv,
        detection_audit_csv=prompt_seeds_csv,
        require_detection_audit=True,
    )

    written = pd.read_csv(output_csv)
    assert written["well_id"].tolist() == [WELL_A]
    assert list(written.columns) == list(FRAME_MASKS_REQUIRED_COLUMNS)
    assert pd.read_csv(prompt_seeds_csv)["detection_id"].tolist() == [
        f"{WELL_A}_BF_t0000_det0000"
    ]


def test_rule_gate_prioritizes_precomputed_over_model_server() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    orchestrator = repo_root / "src" / "data_pipeline" / "pipeline_orchestrator"
    rule_path = orchestrator / "rules" / "frame_masks.smk"
    text = rule_path.read_text(encoding="utf-8")
    task_text = (orchestrator / "tasks.py").read_text(encoding="utf-8")

    assert 'FRAME_MASKS_MODE = str(FRAME_MASKS_CONFIG.get("mode", "model"))' in text
    assert 'if FRAME_MASKS_MODE == "precomputed":' in text
    assert "rule frame_masks_per_well_precomputed:" in text
    assert "elif FRAME_MASKS_SERVED:" in text
    assert "frame_detections_validated=str(_frame_detections_validated(" in text
    assert "tasks ingest-precomputed-frame-masks" in text
    assert '--precomputed-frame-masks-csv "{input.precomputed}"' in text
    assert 'sub.add_parser("ingest-precomputed-frame-masks")' in task_text
    assert '"--precomputed-frame-masks-csv", type=Path, required=True' in task_text
