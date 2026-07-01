"""SAM2 video model loading for the segmentation backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from data_pipeline.models.sam2 import load_sam2_video_predictor


@dataclass(frozen=True)
class Sam2VideoModelConfig:
    """Resolved file/config inputs needed to construct a SAM2 video predictor."""

    models_root: Path
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda"
    model_id: str = "sam2_video"


def parse_sam2_video_model_config(config: Mapping[str, Any]) -> Sam2VideoModelConfig:
    """Parse the `segmentation_masks.sam2_video` config block.

    The backend owns this parsing so shared segmentation code only routes by backend name.
    """
    missing = [
        key
        for key in ("models_root", "config_path", "checkpoint_path")
        if not config.get(key)
    ]
    if missing:
        raise ValueError(f"sam2_video config missing required key(s): {', '.join(missing)}")

    return Sam2VideoModelConfig(
        models_root=Path(str(config["models_root"])),
        config_path=Path(str(config["config_path"])),
        checkpoint_path=Path(str(config["checkpoint_path"])),
        device=str(config.get("device", "cuda")),
        model_id=str(config.get("model_id", "sam2_video")),
    )


def load_sam2_video_model(config: Sam2VideoModelConfig):
    """Load the SAM2 video predictor for this backend.

    This delegates the low-level import/path handling to `data_pipeline.models.sam2`.
    Keeping this facade in the backend gives the router one stable callsite.
    """
    return load_sam2_video_predictor(
        sam2_models_root=config.models_root,
        config_path=config.config_path,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
    )
