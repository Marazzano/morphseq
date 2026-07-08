"""UNet snip model config + predictor loader — and the home of shape ownership.

``load_unet_snip_predictors()`` is the only public entry point for production code.
It returns ``dict[str, AuxiliaryMaskPredictor]`` — callers never see raw Torch modules.

===================================================================================
SHAPE OWNERSHIP
===================================================================================
Four grids touch an auxiliary mask. Each has exactly one owner, one home, and a clear
answer to "does it persist to disk?". A reader should NOT have to reverse-engineer this
from the YAML + entrypoint + predictor.

snip_frame_shape    — the configured LAW.
    home:      config.yaml top-level ``snip_frame_shape: [H, W]``
    resolved:  snip_processing/snip_frame_shape.py::resolve_snip_frame_shape
    persists:  YES — snip image, embryo mask, and auxiliary masks all live here.
    consumed:  snip_processing, snip_auxiliary_masks, fraction_alive.

artifact_shape      — the derived PROMISE (what masks are written at).
    home:      NOT a YAML key. Derived: ``artifact_shape = snip_frame_shape``.
    persists:  YES — every auxiliary mask PNG is written at this shape.
    consumed:  fraction_alive.
    rule:      must equal snip_frame_shape in this pipeline generation.

model_input_shape   — the transient model DOORWAY (in).
    home:      config.yaml ``unet_snip.models.<type>.model_input_shape``
    persists:  NO. Used only to resize the snip image before inference.
    consumed:  FishModelSnipPredictor only.
    rule:      may differ from snip_frame_shape only if a checkpoint requires it.

model_output_shape  — the model EMISSION (out).
    home:      runtime-observed, never configured.
    persists:  NO. Resized back to artifact_shape before writing.
    consumed:  the predictor, internally, on the way home.

DOCTRINE:
    snip_frame_shape names the world.
    artifact_shape names the product promise.
    model_input_shape names the doorway.
    model_output_shape names what came back through it.
    Only the promise gets written.
===================================================================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    ALLOWED_AUXILIARY_MASK_TYPES,
)
from data_pipeline.object_extraction.snip_processing.snip_frame_masks import (
    assert_on_snip_frame,
    back_to_snip_frame,
    snip_image_to_model_grid,
)

AuxiliaryMaskPredictor = Callable[[np.ndarray], np.ndarray]


def _as_shape(value: Any, *, what: str) -> tuple[int, int]:
    """Return ``value`` as a validated two-int ``(height, width)`` tuple, or raise."""
    if value is None or len(value) != 2:
        raise ValueError(f"{what} must be a two-item [height, width]; got {value!r}.")
    h, w = int(value[0]), int(value[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"{what} entries must be positive; got {value!r}.")
    return h, w


@dataclass(frozen=True)
class UNetSnipPredictorConfig:
    """One fully-resolved auxiliary-mask predictor spec (one per mask type).

    Produced by :func:`parse_unet_snip_model_config`; every shape is already resolved
    so the loader/predictor never re-reads YAML or applies a default.
    """

    mask_type: str  # "via" | "yolk" | "focus" | "bubble"
    checkpoint_path: Path  # absolute, resolved against models_root
    model_input_shape: tuple[int, int]  # transient doorway (defaults to artifact_shape)
    artifact_shape: tuple[int, int]  # the law masks come home to


class FishModelSnipPredictor:
    """Adapts a loaded FishModel to the AuxiliaryMaskPredictor interface.

    Input:  H×W uint8 grayscale snip, ALREADY on artifact_shape (asserted upstream by the
            runner before the per-mask try/except, so an off-grid snip kills the shard).
    Output: H×W bool mask on artifact_shape (the promise) — never the input's incidental shape.

    All Torch-specific behavior is encapsulated here. The lifecycle, in shape terms:
        artifact_shape (in)  →  model_input_shape  →  model_output_shape (observed)  →  artifact_shape (out)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        model_input_shape: tuple[int, int],
        artifact_shape: tuple[int, int],
        device: str,
    ) -> None:
        self._model = model
        self._model_input_shape = model_input_shape
        self._artifact_shape = artifact_shape
        self._device = device

    def __call__(self, snip_image: np.ndarray) -> np.ndarray:
        # Leave the snip frame for the model's input grid (continuous resize, controlled + checked).
        resized = snip_image_to_model_grid(snip_image, self._model_input_shape).astype(np.float32)

        # FishModel expects (B, 3, H, W).
        tensor = torch.from_numpy(
            np.stack([resized, resized, resized], axis=0)[np.newaxis]
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = logits.sigmoid()
            binary = (probs > 0.5).squeeze().cpu().numpy()
        # model_output_shape is binary.shape here — observed, never configured.

        # Come home to the LAW (artifact_shape), not the input's incidental shape. Bring the
        # predicted bool mask back via the single snip-frame mask module, so the round-trip and
        # fraction_alive's alignment share one tested implementation, then assert the promise.
        artifact_mask = back_to_snip_frame(binary.astype(np.uint8), self._artifact_shape)
        assert_on_snip_frame(artifact_mask, self._artifact_shape, label="auxiliary mask")
        return artifact_mask


def parse_unet_snip_model_config(
    config: Mapping[str, Any],
    *,
    artifact_shape: tuple[int, int],
) -> list[UNetSnipPredictorConfig]:
    """Resolve the nested ``unet_snip`` config into one spec per declared model.

    Reads ``unet_snip.models.<mask_type>.{checkpoint, model_input_shape}``. Each model's
    ``model_input_shape`` defaults to ``artifact_shape`` (= snip_frame_shape, passed by the
    entrypoint) when omitted. Only declared models are returned — a mask type absent from the
    config simply has no predictor (the runner records it as invalid), which is how the retired
    ``foreground`` family stays gone.

    Raises loudly on: missing ``models_root``; the legacy flat ``unet_snip.checkpoints`` schema;
    a declared model missing ``checkpoint``; an unknown mask type; a malformed shape.
    """
    if not config.get("models_root"):
        raise ValueError("unet_snip config missing required key: models_root")

    if "checkpoints" in config:
        raise ValueError(
            "unet_snip uses nested models.<mask_type>.checkpoint and "
            "models.<mask_type>.model_input_shape. Flat unet_snip.checkpoints is no longer "
            "supported."
        )

    models = config.get("models")
    if not models:
        raise ValueError(
            "unet_snip config missing required key: models (nested "
            "models.<mask_type>.{checkpoint, model_input_shape})"
        )

    artifact_shape = _as_shape(artifact_shape, what="artifact_shape")
    models_root = Path(str(config["models_root"]))

    specs: list[UNetSnipPredictorConfig] = []
    for mask_type, model_cfg in models.items():
        if mask_type not in ALLOWED_AUXILIARY_MASK_TYPES:
            raise ValueError(
                f"unet_snip.models has unknown mask type {mask_type!r}; allowed: "
                f"{ALLOWED_AUXILIARY_MASK_TYPES}."
            )
        checkpoint = (model_cfg or {}).get("checkpoint")
        if not checkpoint:
            raise ValueError(
                f"unet_snip.models.{mask_type} missing required key: checkpoint"
            )
        raw_input_shape = (model_cfg or {}).get("model_input_shape")
        model_input_shape = (
            artifact_shape
            if raw_input_shape is None
            else _as_shape(raw_input_shape, what=f"unet_snip.models.{mask_type}.model_input_shape")
        )
        specs.append(
            UNetSnipPredictorConfig(
                mask_type=mask_type,
                checkpoint_path=models_root / str(checkpoint),
                model_input_shape=model_input_shape,
                artifact_shape=artifact_shape,
            )
        )
    return specs


def load_unet_snip_predictors(
    configs: list[UNetSnipPredictorConfig],
    *,
    device: str = "cuda",
) -> dict[str, AuxiliaryMaskPredictor]:
    """Load one FishModel per spec and wrap each in FishModelSnipPredictor.

    Returns ``dict[auxiliary_mask_type, AuxiliaryMaskPredictor]``; callers pass this directly to
    ``run_unet_for_snip_inventory``. ``device`` defaults to the production ``cuda``; tests pass
    ``device="cpu"`` (or monkeypatch ``load_fish_unet_model``).
    """
    from data_pipeline.models.unet import load_fish_unet_model

    predictors: dict[str, AuxiliaryMaskPredictor] = {}
    for cfg in configs:
        model = load_fish_unet_model(cfg.checkpoint_path, device=device)
        predictors[cfg.mask_type] = FishModelSnipPredictor(
            model,
            model_input_shape=cfg.model_input_shape,
            artifact_shape=cfg.artifact_shape,
            device=device,
        )
    return predictors
