"""Tests for run_unet_snip runner.

Snips live on the artifact grid (= snip_frame_shape). The runner asserts each input snip is on
that grid BEFORE the per-mask try/except, so an off-grid snip kills the shard; genuine per-model
inference failures stay localized as is_valid=False rows.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.object_extraction.segmentation.backends.unet_snip.run_unet_snip import (
    run_auxiliary_mask_predictors_for_snip,
    run_unet_for_snip_inventory,
)
from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
    validate_snip_auxiliary_masks,
)

# The artifact grid the whole snip world agrees on (height, width).
ARTIFACT_SHAPE = (576, 256)


# ---------------------------------------------------------------------------
# Fake predictors (pure NumPy — no Torch). They return the artifact shape (the law),
# not the incidental input shape, mirroring FishModelSnipPredictor.
# ---------------------------------------------------------------------------

def fake_empty_mask_predictor(snip_image: np.ndarray) -> np.ndarray:
    return np.zeros(ARTIFACT_SHAPE, dtype=bool)


def fake_center_mask_predictor(snip_image: np.ndarray) -> np.ndarray:
    mask = np.zeros(ARTIFACT_SHAPE, dtype=bool)
    h, w = mask.shape
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return mask


def _fake_predictors() -> dict:
    return {mt: fake_center_mask_predictor for mt in ALLOWED_AUXILIARY_MASK_TYPES}


# ---------------------------------------------------------------------------
# snip_inventory fixture helpers — snips written at the artifact grid
# ---------------------------------------------------------------------------

def _make_snip_inventory(
    tmp_path, n: int = 2, valid: bool = True, shape: tuple[int, int] = ARTIFACT_SHAPE
) -> pd.DataFrame:
    import skimage.io as skio

    rows = []
    for i in range(n):
        snip_id = f"20250912_B0{i+1}_e01_BF_t{i:04d}"
        snip_path = tmp_path / f"{snip_id}.png"
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
        skio.imsave(str(snip_path), img, check_contrast=False)
        rows.append({
            "snip_id": snip_id,
            "physical_embryo_id": f"20250912_B0{i+1}_e01",
            "embryo_id": f"embryo_{i}",
            "experiment_id": "20250912",
            "well_id": f"20250912_B0{i+1}",
            "image_id": f"20250912_B0{i+1}_t{i:04d}",
            "time_index": i,
            "channel_id": "BF",
            "processed_snip_path": str(snip_path),
            "is_valid_snip": valid,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_auxiliary_mask_predictors_for_snip
# ---------------------------------------------------------------------------

def test_run_auxiliary_mask_predictors_keys():
    snip = np.zeros(ARTIFACT_SHAPE, dtype=np.uint8)
    result = run_auxiliary_mask_predictors_for_snip(_fake_predictors(), snip)
    assert set(result.keys()) == set(ALLOWED_AUXILIARY_MASK_TYPES)


def test_run_auxiliary_mask_predictors_return_artifact_shape():
    snip = np.zeros(ARTIFACT_SHAPE, dtype=np.uint8)
    result = run_auxiliary_mask_predictors_for_snip(_fake_predictors(), snip)
    for mt, mask in result.items():
        assert mask.shape == ARTIFACT_SHAPE, f"{mt}: expected {ARTIFACT_SHAPE}, got {mask.shape}"


# ---------------------------------------------------------------------------
# run_unet_for_snip_inventory
# ---------------------------------------------------------------------------

def test_run_unet_for_snip_inventory_contract(tmp_path):
    inv = _make_snip_inventory(tmp_path, n=2, valid=True)
    result = run_unet_for_snip_inventory(
        snip_inventory=inv,
        predictors=_fake_predictors(),
        output_dir=tmp_path / "output",
        model_id="unet_test",
        model_backend="unet_snip",
        checkpoint_paths={mt: f"weights/{mt}.pth" for mt in ALLOWED_AUXILIARY_MASK_TYPES},
        artifact_shape=ARTIFACT_SHAPE,
    )
    assert len(result) == 2 * len(ALLOWED_AUXILIARY_MASK_TYPES)
    validate_snip_auxiliary_masks(result)
    # Masks written at the artifact shape (the promise), not the snip's incidental shape.
    assert (result["mask_height_px"] == ARTIFACT_SHAPE[0]).all()
    assert (result["mask_width_px"] == ARTIFACT_SHAPE[1]).all()


def test_invalid_snips_produce_no_rows(tmp_path):
    inv = _make_snip_inventory(tmp_path, n=2, valid=False)
    result = run_unet_for_snip_inventory(
        snip_inventory=inv,
        predictors=_fake_predictors(),
        output_dir=tmp_path / "output",
        model_id="unet_test",
        model_backend="unet_snip",
        checkpoint_paths={},
        artifact_shape=ARTIFACT_SHAPE,
    )
    assert len(result) == 0


def test_predictor_failure_row_is_invalid(tmp_path):
    inv = _make_snip_inventory(tmp_path, n=1, valid=True)

    def exploding_predictor(snip_image: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated failure")

    predictors = dict(_fake_predictors())
    predictors["yolk"] = exploding_predictor

    result = run_unet_for_snip_inventory(
        snip_inventory=inv,
        predictors=predictors,
        output_dir=tmp_path / "output",
        model_id="unet_test",
        model_backend="unet_snip",
        checkpoint_paths={mt: f"weights/{mt}.pth" for mt in ALLOWED_AUXILIARY_MASK_TYPES},
        artifact_shape=ARTIFACT_SHAPE,
    )
    yolk_rows = result[result["auxiliary_mask_type"] == "yolk"]
    assert len(yolk_rows) == 1
    assert not yolk_rows.iloc[0]["is_valid_auxiliary_mask"]
    assert "simulated failure" in yolk_rows.iloc[0]["error_message"]


def test_off_grid_snip_kills_shard(tmp_path):
    """A snip not on the artifact grid is a shard-wide bug: it raises out of the runner,
    not swallowed as a per-mask is_valid=False row."""
    inv = _make_snip_inventory(tmp_path, n=1, valid=True, shape=(64, 32))
    with pytest.raises(ValueError, match="not on the snip frame"):
        run_unet_for_snip_inventory(
            snip_inventory=inv,
            predictors=_fake_predictors(),
            output_dir=tmp_path / "output",
            model_id="unet_test",
            model_backend="unet_snip",
            checkpoint_paths={mt: f"weights/{mt}.pth" for mt in ALLOWED_AUXILIARY_MASK_TYPES},
            artifact_shape=ARTIFACT_SHAPE,
        )
