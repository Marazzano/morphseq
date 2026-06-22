"""UNet snip model config and predictor loader.

load_unet_snip_predictors() is the only public entry point for production code.
It returns dict[str, AuxiliaryMaskPredictor] — callers never see raw Torch modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import skimage.transform as st
import torch

from data_pipeline.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
)

AuxiliaryMaskPredictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class UNetSnipModelConfig:
    """Resolved config for loading UNet snip predictors.

    checkpoints maps auxiliary_mask_type → path relative to models_root.
    """
    models_root: Path
    checkpoints: dict[str, Path]
    device: str = "cuda"
    im_dims: tuple[int, int] = (576, 320)


class FishModelSnipPredictor:
    """Adapts a loaded FishModel to the AuxiliaryMaskPredictor interface.

    Input:  H×W uint8 grayscale snip (original snip dimensions)
    Output: H×W bool mask (same H×W as input)

    All Torch-specific behavior is encapsulated here:
    resize → repeat channels → tensor → device → forward → sigmoid → threshold → resize back.
    """

    def __init__(self, model: torch.nn.Module, im_dims: tuple[int, int], device: str) -> None:
        self._model = model
        self._im_dims = im_dims
        self._device = device

    def __call__(self, snip_image: np.ndarray) -> np.ndarray:
        orig_h, orig_w = snip_image.shape[:2]

        resized = st.resize(
            snip_image, self._im_dims, order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)

        # FishModel expects (B, 3, H, W)
        tensor = torch.from_numpy(
            np.stack([resized, resized, resized], axis=0)[np.newaxis]
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = logits.sigmoid()
            binary = (probs > 0.5).squeeze().cpu().numpy()

        # Resize bool mask back to original snip dimensions
        mask = st.resize(
            binary.astype(np.uint8), (orig_h, orig_w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(bool)
        return mask


def parse_unet_snip_model_config(config: Mapping[str, Any]) -> UNetSnipModelConfig:
    """Parse the unet_snip config block from a pipeline YAML."""
    if not config.get("models_root"):
        raise ValueError("unet_snip config missing required key: models_root")
    if not config.get("checkpoints"):
        raise ValueError("unet_snip config missing required key: checkpoints")

    models_root = Path(str(config["models_root"]))
    checkpoints = {k: Path(str(v)) for k, v in config["checkpoints"].items()}
    return UNetSnipModelConfig(
        models_root=models_root,
        checkpoints=checkpoints,
        device=str(config.get("device", "cuda")),
        im_dims=tuple(config.get("im_dims", (576, 320))),  # type: ignore[arg-type]
    )


def load_unet_snip_predictors(cfg: UNetSnipModelConfig) -> dict[str, AuxiliaryMaskPredictor]:
    """Load one FishModel per checkpoint and wrap each in FishModelSnipPredictor.

    Returns dict[auxiliary_mask_type, AuxiliaryMaskPredictor].
    Callers pass this directly to run_unet_for_snip_inventory.
    """
    from data_pipeline.models.unet import load_fish_unet_model

    predictors: dict[str, AuxiliaryMaskPredictor] = {}
    for mask_type in ALLOWED_AUXILIARY_MASK_TYPES:
        if mask_type not in cfg.checkpoints:
            continue
        checkpoint_path = cfg.models_root / cfg.checkpoints[mask_type]
        model = load_fish_unet_model(checkpoint_path, device=cfg.device)
        predictors[mask_type] = FishModelSnipPredictor(model, cfg.im_dims, cfg.device)
    return predictors
