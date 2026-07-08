from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.object_extraction.segmentation.sam2_video import (
    Sam2VideoModelConfig,
    Sam2WellInput,
    build_sam2_frame_view,
)
from data_pipeline.object_extraction.segmentation.sam2_video import run_sam2_video as sam2_runner
from data_pipeline.object_extraction.segmentation.sam2_video.model_loader import parse_sam2_video_model_config


def _model_frame_view(tmp_path: Path, *, well_id: str = "20250912_B01") -> pd.DataFrame:
    rows = []
    for t in [2, 0, 1]:
        frame = tmp_path / f"{well_id}_BF_t{t:04d}.jpg"
        frame.write_bytes(b"not-a-real-jpeg")
        rows.append(
            {
                "experiment_id": "20250912",
                "well_id": well_id,
                "image_id": f"{well_id}_BF_t{t:04d}",
                "time_index": t,
                "channel_id": "BF",
                "source_image_path": str(frame),
                "image_width_px": 100,
                "image_height_px": 80,
            }
        )
    return pd.DataFrame(rows)


def test_build_sam2_frame_view_creates_sequential_symlinks_and_mapping(tmp_path: Path) -> None:
    frames = _model_frame_view(tmp_path)

    with build_sam2_frame_view(frames, temp_root=tmp_path) as view:
        assert view.path.exists()
        names = sorted(p.name for p in view.path.iterdir())
        assert names == ["00000.jpg", "00001.jpg", "00002.jpg"]
        assert all((view.path / name).is_symlink() for name in names)

        assert view.index["sam2_frame_index"].tolist() == [0, 1, 2]
        assert view.index["time_index"].tolist() == [0, 1, 2]
        assert view.by_sam2_index[1]["image_id"] == "20250912_B01_BF_t0001"

    assert not view.path.exists()


def test_parse_sam2_video_model_config() -> None:
    cfg = parse_sam2_video_model_config(
        {
            "models_root": "/models",
            "config_path": "sam2/configs/sam2.1.yaml",
            "checkpoint_path": "checkpoints/sam2.pt",
            "device": "cpu",
            "model_id": "sam2.1_hiera_l",
        }
    )
    assert cfg.models_root == Path("/models")
    assert cfg.device == "cpu"
    assert cfg.model_id == "sam2.1_hiera_l"


def test_run_sam2_video_for_wells_loads_model_once(monkeypatch, tmp_path: Path) -> None:
    calls: list[Sam2VideoModelConfig] = []
    fake_predictor = object()

    def fake_load(config: Sam2VideoModelConfig):
        calls.append(config)
        return fake_predictor

    def fake_segment(predictor, well: Sam2WellInput) -> pd.DataFrame:
        assert predictor is fake_predictor
        return pd.DataFrame({"well_id": [well.well_id], "n_frames": [len(well.model_frame_view)]})

    monkeypatch.setattr(sam2_runner, "load_sam2_video_model", fake_load)

    cfg = Sam2VideoModelConfig(
        models_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "model.pt",
        device="cpu",
    )
    wells = [
        Sam2WellInput("20250912_B01", _model_frame_view(tmp_path, well_id="20250912_B01"), pd.DataFrame()),
        Sam2WellInput("20250912_B02", _model_frame_view(tmp_path, well_id="20250912_B02"), pd.DataFrame()),
    ]

    results = sam2_runner.run_sam2_video_for_wells(
        wells,
        model_config=cfg,
        segment_one_well=fake_segment,
    )

    assert len(calls) == 1
    assert [r.well_id for r in results] == ["20250912_B01", "20250912_B02"]
    assert results[0].frame_masks.loc[0, "n_frames"] == 3
