"""Tests for run_unet_snip runner — Step 1 (no Torch, no FishModel)."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.segmentation.backends.unet_snip.run_unet_snip import (
    run_auxiliary_mask_predictors_for_snip,
    run_unet_for_snip_inventory,
)
from data_pipeline.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
    validate_snip_auxiliary_masks,
)


# ---------------------------------------------------------------------------
# Fake predictors (pure NumPy — no Torch)
# ---------------------------------------------------------------------------

def fake_empty_mask_predictor(snip_image: np.ndarray) -> np.ndarray:
    return np.zeros(snip_image.shape, dtype=bool)


def fake_center_mask_predictor(snip_image: np.ndarray) -> np.ndarray:
    mask = np.zeros(snip_image.shape, dtype=bool)
    h, w = mask.shape
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return mask


def _fake_predictors() -> dict:
    return {mt: fake_center_mask_predictor for mt in ALLOWED_AUXILIARY_MASK_TYPES}


# ---------------------------------------------------------------------------
# snip_inventory fixture helpers
# ---------------------------------------------------------------------------

def _make_snip_inventory(tmp_path, n: int = 2, valid: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(n):
        snip_id = f"20250912_B0{i+1}_e01_BF_t{i:04d}"
        snip_path = tmp_path / f"{snip_id}.png"
        img = np.random.randint(0, 255, (64, 32), dtype=np.uint8)
        import skimage.io as skio
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
    snip = np.zeros((64, 32), dtype=np.uint8)
    result = run_auxiliary_mask_predictors_for_snip(_fake_predictors(), snip)
    assert set(result.keys()) == set(ALLOWED_AUXILIARY_MASK_TYPES)


def test_run_auxiliary_mask_predictors_shape():
    h, w = 48, 24
    snip = np.zeros((h, w), dtype=np.uint8)
    result = run_auxiliary_mask_predictors_for_snip(_fake_predictors(), snip)
    for mt, mask in result.items():
        assert mask.shape == (h, w), f"{mt}: expected ({h},{w}), got {mask.shape}"


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
    )
    assert len(result) == 2 * len(ALLOWED_AUXILIARY_MASK_TYPES)
    validate_snip_auxiliary_masks(result)


def test_invalid_snips_produce_no_rows(tmp_path):
    inv = _make_snip_inventory(tmp_path, n=2, valid=False)
    result = run_unet_for_snip_inventory(
        snip_inventory=inv,
        predictors=_fake_predictors(),
        output_dir=tmp_path / "output",
        model_id="unet_test",
        model_backend="unet_snip",
        checkpoint_paths={},
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
    )
    yolk_rows = result[result["auxiliary_mask_type"] == "yolk"]
    assert len(yolk_rows) == 1
    assert not yolk_rows.iloc[0]["is_valid_auxiliary_mask"]
    assert "simulated failure" in yolk_rows.iloc[0]["error_message"]
