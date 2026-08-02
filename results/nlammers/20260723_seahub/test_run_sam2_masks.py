from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import run_sam2_masks as sam2_masks


def test_areas_only_argument_is_opt_in():
    default = sam2_masks.build_parser().parse_args([])
    areas = sam2_masks.build_parser().parse_args(["--areas-only"])
    assert default.areas_only is False
    assert areas.areas_only is True
    assert areas.prompt_box_source == "raw"


def test_area_record_contains_calibration_fields():
    row = pd.Series(
        {
            "image_id": "image-1",
            "embryo_position": 3,
            "experiment_id": "GENE1",
            "stage_hpf": 24.0,
            "image_path": "/data/image.jpg",
        }
    )
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:6, 3:8] = True
    record = sam2_masks._area_record(
        row,
        source_fov_id="fov-1",
        mask=mask,
        mask_score=0.9,
        prompt_box=(2, 1, 10, 6),
        prompt_box_source="raw",
    )
    assert record["image_id"] == "image-1"
    assert record["source_fov_id"] == "fov-1"
    assert record["embryo_position"] == 3
    assert record["mask_score"] == pytest.approx(0.9)
    assert record["mask_area_px"] == 20
    assert record["prompt_box_area_px"] == 40
    assert record["mask_to_prompt_area_ratio"] == pytest.approx(0.5)
    assert record["stage_hpf"] == 24.0


def test_checkpoint_is_atomic_and_marks_completion(tmp_path):
    output = tmp_path / "sam2_mask_areas.csv"
    rows = [
        {
            "image_id": "image-1",
            "source_fov_id": "fov-1",
            "embryo_position": 1,
            "mask_score": 0.8,
            "mask_area_px": 100,
            "mask_to_prompt_area_ratio": 0.5,
        }
    ]
    sam2_masks._write_checkpoint(
        rows,
        output,
        processed_fov_count=1,
        total_fov_count=2,
        complete=False,
    )
    partial = pd.read_csv(output)
    assert partial["processed_fov_count"].tolist() == [1]
    assert partial["total_fov_count"].tolist() == [2]
    assert partial["checkpoint_complete"].tolist() == [False]
    assert not output.with_name(f".{output.name}.tmp").exists()

    sam2_masks._write_checkpoint(
        rows,
        output,
        processed_fov_count=2,
        total_fov_count=2,
        complete=True,
    )
    complete = pd.read_csv(output)
    assert complete["checkpoint_complete"].tolist() == [True]


def test_areas_output_refuses_nonempty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sam2_masks, "HERE", tmp_path.resolve())
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.csv").write_text("stale\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh and empty"):
        sam2_masks._prepare_output(output, areas_only=True)


def test_prediction_shape_is_checked():
    masks = np.ones((2, 1, 5, 6), dtype=np.float32)
    scores = np.asarray([[0.8], [0.9]])
    normalized_masks, normalized_scores = sam2_masks._normalize_predictions(
        masks,
        scores,
        expected_count=2,
        image_shape=(5, 6),
    )
    assert normalized_masks.shape == (2, 5, 6)
    assert normalized_masks.dtype == bool
    assert normalized_scores.tolist() == pytest.approx([0.8, 0.9])

    with pytest.raises(ValueError, match="unexpected mask shape"):
        sam2_masks._normalize_predictions(
            masks[:1],
            scores,
            expected_count=2,
            image_shape=(5, 6),
        )


def test_areas_only_calls_predictor_once_per_fov_and_writes_only_csv(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sam2_masks, "HERE", tmp_path.resolve())
    manifest_rows = []
    for fov_index in range(2):
        image_id = f"image-{fov_index}"
        image_path = tmp_path / f"{image_id}.jpg"
        Image.new("RGB", (20, 10), "white").save(image_path)
        for position in (1, 2):
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "source_fov_id": f"fov-{fov_index}",
                    "image_path": str(image_path),
                    "embryo_position": position,
                    "box_x1_norm": 0.1 * position,
                    "box_y1_norm": 0.1,
                    "box_x2_norm": 0.1 * position + 0.2,
                    "box_y2_norm": 0.8,
                    "experiment_id": "GENE1",
                    "stage_hpf": 24.0,
                }
            )
    manifest_path = tmp_path / "embryo_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    class FakePredictor:
        def __init__(self):
            self.image_shape = None
            self.predict_calls = 0

        def set_image(self, image):
            self.image_shape = image.shape[:2]

        def predict(self, *, box, multimask_output):
            assert multimask_output is False
            self.predict_calls += 1
            height, width = self.image_shape
            masks = np.zeros((len(box), 1, height, width), dtype=np.float32)
            masks[:, :, 2:6, 3:8] = 1.0
            scores = np.full((len(box), 1), 0.9, dtype=np.float32)
            return masks, scores, None

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()

    predictor = FakePredictor()
    monkeypatch.setattr(
        sam2_masks, "_load_predictor", lambda: (predictor, FakeTorch)
    )
    output = tmp_path / "areas"
    sam2_masks.main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--areas-only",
        ]
    )

    assert predictor.predict_calls == 2
    assert [path.name for path in output.iterdir()] == ["sam2_mask_areas.csv"]
    result = pd.read_csv(output / "sam2_mask_areas.csv")
    assert len(result) == 4
    assert set(result["source_fov_id"]) == {"fov-0", "fov-1"}
    assert result["checkpoint_complete"].all()
    assert set(result["processed_fov_count"]) == {2}
