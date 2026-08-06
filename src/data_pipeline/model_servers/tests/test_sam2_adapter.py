"""CPU-only structural and parity tests for the resident SAM2 adapter."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_pipeline.shared.identifiers import build_image_id, build_well_id


class _FakePredictor:
    """Small SAM2 API stand-in that records fresh state creation per well."""

    def __init__(self) -> None:
        self.states: list[dict] = []

    def init_state(self, *, video_path: str) -> dict:
        state = {"video_path": video_path, "boxes": []}
        self.states.append(state)
        return state

    def add_new_points_or_box(
        self,
        *,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
        box: np.ndarray,
    ) -> None:
        inference_state["boxes"].append((frame_idx, obj_id, box.copy()))

    def propagate_in_video(self, inference_state: dict):
        frame_count = len(list(Path(inference_state["video_path"]).glob("*.jpg")))
        logits = np.full((1, 32, 32), -1.0, dtype=np.float32)
        logits[:, 8:16, 10:18] = 1.0
        for frame_idx in range(frame_count):
            yield frame_idx, [0], [logits]


def _write_well(tmp_path: Path, experiment_id: str, local_well: str) -> tuple[Path, Path]:
    well_id = build_well_id(experiment_id, local_well)
    inventory_rows = []
    for time_index in range(2):
        image_id = build_image_id(well_id, "BF", time_index)
        image_path = tmp_path / f"{image_id}.png"
        Image.fromarray(
            np.full((32, 32), 120 + time_index, dtype=np.uint8),
            mode="L",
        ).save(image_path)
        inventory_rows.append(
            {
                "experiment_id": experiment_id,
                "well_id": well_id,
                "image_id": image_id,
                "time_index": time_index,
                "z_index": pd.NA,
                "channel_id": "BF",
                "image_path": str(image_path),
                "image_width_px": 32,
                "image_height_px": 32,
                "image_product_type": "projection",
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    inventory_csv = tmp_path / f"{well_id}_frame_inventory.csv"
    inventory.to_csv(inventory_csv, index=False)

    seed = inventory.iloc[0]
    detections = pd.DataFrame(
        [
            {
                "detection_id": f"{seed['image_id']}_det0000",
                "image_id": seed["image_id"],
                "time_index": 0,
                "channel_id": "BF",
                "bbox_x_min_px": 8.0,
                "bbox_y_min_px": 7.0,
                "bbox_x_max_px": 19.0,
                "bbox_y_max_px": 18.0,
                "is_kept": True,
            }
        ]
    )
    detections_csv = tmp_path / f"{well_id}_frame_detections.csv"
    detections.to_csv(detections_csv, index=False)
    return inventory_csv, detections_csv


def _adapter() -> object:
    from data_pipeline.model_servers.adapters.sam2 import Sam2Adapter

    return Sam2Adapter(
        sam2_models_root="/models/sam2",
        sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_small.pt",
        sam2_model_id="sam2:test",
        device="cpu",
    )


def test_adapter_registered_under_sam2():
    from data_pipeline.model_servers.adapter_base import get_adapter_class
    from data_pipeline.model_servers.adapters.sam2 import Sam2Adapter

    assert get_adapter_class("sam2") is Sam2Adapter


def test_load_constructs_predictor_exactly_once(monkeypatch):
    import data_pipeline.model_servers.adapters.sam2 as sam2_adapter_mod

    calls = []
    predictor = _FakePredictor()

    def _fake_load(**kwargs):
        calls.append(kwargs)
        return predictor

    monkeypatch.setattr(sam2_adapter_mod, "load_sam2_video_predictor", _fake_load)
    adapter = _adapter()
    adapter.load()

    assert adapter.predictor is predictor
    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"


def test_served_output_matches_direct_frame_masks_path(tmp_path, monkeypatch):
    """The resident path must write the same two products as cmd_frame_masks."""
    import data_pipeline.models.sam2 as sam2_model_mod
    from data_pipeline.model_servers.adapters.sam2 import Sam2Adapter
    from data_pipeline.pipeline_orchestrator.tasks import cmd_frame_masks

    inventory_csv, detections_csv = _write_well(tmp_path, "20260724_test", "A01")
    direct_masks = tmp_path / "direct_frame_masks.csv"
    direct_prompts = tmp_path / "direct_prompt_seeds.csv"
    served_masks = tmp_path / "served_frame_masks.csv"
    served_prompts = tmp_path / "served_prompt_seeds.csv"

    monkeypatch.setattr(
        sam2_model_mod,
        "load_sam2_video_predictor",
        lambda **kwargs: _FakePredictor(),
    )
    cmd_frame_masks(
        Namespace(
            frame_inventory_csv=inventory_csv,
            frame_detections_csv=detections_csv,
            output_csv=direct_masks,
            prompt_seeds_csv=direct_prompts,
            sam2_models_root=Path("/models/sam2"),
            sam2_config=Path("configs/sam2.1/sam2.1_hiera_s.yaml"),
            sam2_checkpoint=Path("checkpoints/sam2.1_hiera_small.pt"),
            sam2_model_id="sam2:test",
            device="cpu",
        )
    )

    adapter = Sam2Adapter(
        sam2_models_root="/models/sam2",
        sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_small.pt",
        sam2_model_id="sam2:test",
        device="cpu",
    )
    adapter.predictor = _FakePredictor()
    adapter.handle(
        {
            "frame_inventory_csv": str(inventory_csv),
            "frame_detections_csv": str(detections_csv),
            "output_csv": str(served_masks),
            "prompt_seeds_csv": str(served_prompts),
        }
    )

    pd.testing.assert_frame_equal(
        pd.read_csv(served_masks),
        pd.read_csv(direct_masks),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(served_prompts),
        pd.read_csv(direct_prompts),
        check_dtype=False,
    )
    assert not list(tmp_path.glob(".served_*.tmp-*"))


def test_each_request_gets_fresh_tracking_state(tmp_path):
    adapter = _adapter()
    predictor = _FakePredictor()
    adapter.predictor = predictor

    output_wells = []
    for local_well in ("A01", "A02"):
        inventory_csv, detections_csv = _write_well(
            tmp_path, "20260724_test", local_well
        )
        masks_csv = tmp_path / f"{local_well}_masks.csv"
        prompts_csv = tmp_path / f"{local_well}_prompts.csv"
        adapter.handle(
            {
                "frame_inventory_csv": str(inventory_csv),
                "frame_detections_csv": str(detections_csv),
                "output_csv": str(masks_csv),
                "prompt_seeds_csv": str(prompts_csv),
            }
        )
        output_wells.append(pd.read_csv(masks_csv)["well_id"].unique().tolist())

    assert len(predictor.states) == 2
    assert predictor.states[0] is not predictor.states[1]
    assert output_wells == [
        ["20260724_test_A01"],
        ["20260724_test_A02"],
    ]
