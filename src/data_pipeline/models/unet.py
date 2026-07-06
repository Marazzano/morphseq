"""FishModel (FPN+ResNet34) checkpoint loader.

## What FishModel is

FishModel is a segmentation_models_pytorch FPN with ResNet34 encoder, trained as a
pytorch-lightning module. Five checkpoints exist (foreground/via/yolk/focus/bubble),
all sharing the same architecture but different weights.

## encoder_weights=None

Always pass encoder_weights=None. This prevents smp from downloading ImageNet weights
on every instantiation — we are loading from our own checkpoints, not using pretrained
encoders.

## Checkpoint formats

Two formats exist in the wild:
- Plain state dict: the file IS the state dict (torch.save(model.state_dict(), path))
- Lightning checkpoint: a dict with a "state_dict" key (saved by pl.Trainer)

load_fish_unet_model handles both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def _ensure_src_on_path() -> None:
    """Add the repo src/ dir to sys.path so FishModel is importable."""
    # Walk up from this file to find src/
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src"
        if candidate.is_dir():
            src_str = str(candidate)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return
    # Fallback: we are already inside src/data_pipeline/..., so src/ is 3 levels up
    src_dir = here.parents[3]
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def load_fish_unet_model(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Load a FishModel checkpoint from disk.

    Works with both plain state-dict files and pytorch-lightning checkpoints.
    encoder_weights=None prevents smp from downloading pretrained encoder weights.
    """
    _ensure_src_on_path()
    from src.core.functions.core_utils_segmentation import FishModel  # type: ignore

    model = FishModel("FPN", "resnet34", in_channels=3, out_classes=1, encoder_weights=None)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model
