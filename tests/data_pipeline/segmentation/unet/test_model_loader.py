"""Step 2 tests for FishModelSnipPredictor adapter and checkpoint loader.

All CPU, no GPU, no real checkpoint files, no pretrained-weight downloads.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import skimage.io as skio
import torch

# Ensure src/ is on path for FishModel import
_src_dir = str(Path(__file__).resolve().parents[5] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.core.functions.core_utils_segmentation import FishModel  # type: ignore

from data_pipeline.segmentation.backends.unet_snip.model_loader import (
    FishModelSnipPredictor,
    UNetSnipModelConfig,
    load_unet_snip_predictors,
)
from data_pipeline.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
)
from data_pipeline.models.unet import load_fish_unet_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_fish_model() -> torch.nn.Module:
    return FishModel("FPN", "resnet34", in_channels=3, out_classes=1, encoder_weights=None)


def _save_plain(model: torch.nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def _save_lightning(model: torch.nn.Module, path: Path) -> None:
    torch.save({"state_dict": model.state_dict()}, path)


# ---------------------------------------------------------------------------
# load_fish_unet_model
# ---------------------------------------------------------------------------

def test_load_fish_unet_model_plain_state_dict(tmp_path):
    model = _fresh_fish_model()
    ckpt = tmp_path / "plain.pth"
    _save_plain(model, ckpt)

    loaded = load_fish_unet_model(ckpt, device="cpu")
    assert not loaded.training


def test_load_fish_unet_model_lightning_state_dict(tmp_path):
    model = _fresh_fish_model()
    ckpt = tmp_path / "lightning.ckpt"
    _save_lightning(model, ckpt)

    loaded = load_fish_unet_model(ckpt, device="cpu")
    assert not loaded.training


# ---------------------------------------------------------------------------
# FishModelSnipPredictor
# ---------------------------------------------------------------------------

def test_fish_model_snip_predictor_returns_bool_mask_with_original_shape():
    model = _fresh_fish_model().eval()
    predictor = FishModelSnipPredictor(model, im_dims=(576, 320), device="cpu")

    snip = np.random.randint(0, 255, (96, 48), dtype=np.uint8)
    mask = predictor(snip)

    assert mask.dtype == bool
    assert mask.shape == (96, 48)


# ---------------------------------------------------------------------------
# load_unet_snip_predictors
# ---------------------------------------------------------------------------

def test_load_unet_snip_predictors_returns_allowed_mask_types(tmp_path):
    model = _fresh_fish_model()
    checkpoints: dict[str, Path] = {}
    for mt in ALLOWED_AUXILIARY_MASK_TYPES:
        ckpt = tmp_path / f"{mt}.pth"
        _save_plain(model, ckpt)
        checkpoints[mt] = Path(f"{mt}.pth")

    cfg = UNetSnipModelConfig(
        models_root=tmp_path,
        checkpoints=checkpoints,
        device="cpu",
        im_dims=(576, 320),
    )
    predictors = load_unet_snip_predictors(cfg)

    assert set(predictors.keys()) == set(ALLOWED_AUXILIARY_MASK_TYPES)
    for mt, pred in predictors.items():
        assert callable(pred), f"{mt} predictor is not callable"


# ---------------------------------------------------------------------------
# Trapdoor: FishModelSnipPredictor does not mutate the Step-1 runner API
# ---------------------------------------------------------------------------

def test_fish_model_snip_predictor_does_not_mutate_runner_api(tmp_path):
    """Use FishModelSnipPredictor through run_unet_for_snip_inventory and verify the
    Step-1 contract still validates without any change to the runner."""
    import pandas as pd
    from data_pipeline.segmentation.backends.unet_snip.run_unet_snip import (
        run_unet_for_snip_inventory,
    )
    from data_pipeline.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
        validate_snip_auxiliary_masks,
    )

    # Write a tiny synthetic snip PNG
    snip_id = "20250912_B01_e01_BF_t0000"
    snip_path = tmp_path / f"{snip_id}.png"
    img = np.random.randint(0, 255, (64, 32), dtype=np.uint8)
    skio.imsave(str(snip_path), img, check_contrast=False)

    inv = pd.DataFrame([{
        "snip_id": snip_id,
        "physical_embryo_id": "20250912_B01_e01",
        "embryo_id": "embryo_0",
        "experiment_id": "20250912",
        "well_id": "20250912_B01",
        "image_id": "20250912_B01_t0000",
        "time_index": 0,
        "channel_id": "BF",
        "processed_snip_path": str(snip_path),
        "is_valid_snip": True,
    }])

    model = _fresh_fish_model().eval()
    predictors = {mt: FishModelSnipPredictor(model, im_dims=(64, 32), device="cpu")
                  for mt in ALLOWED_AUXILIARY_MASK_TYPES}

    result = run_unet_for_snip_inventory(
        snip_inventory=inv,
        predictors=predictors,
        output_dir=tmp_path / "output",
        model_id="unet_test",
        model_backend="unet_snip",
        checkpoint_paths={mt: f"weights/{mt}.pth" for mt in ALLOWED_AUXILIARY_MASK_TYPES},
    )

    assert len(result) == len(ALLOWED_AUXILIARY_MASK_TYPES)
    validate_snip_auxiliary_masks(result)
    assert result["is_valid_auxiliary_mask"].all()
