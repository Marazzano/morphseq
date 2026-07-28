"""Tests for the unet_snip model_loader: nested config parser + law-shaped predictor.

All CPU, no GPU, no real checkpoint downloads. Uses a real (untrained) FishModel so the
to → infer → back round-trip is genuinely exercised; only checkpoint loading is mocked.

Shape ownership under test:
  artifact_shape      — the law masks come home to (= snip_frame_shape)
  model_input_shape   — per-model transient doorway (may differ from artifact_shape)
  model_output_shape  — observed at runtime, never configured
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on path for the (lazy) FishModel import used only by predictor tests.
_src_dir = str(Path(__file__).resolve().parents[5] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_pipeline.object_extraction.segmentation.backends.unet_snip import model_loader
from data_pipeline.object_extraction.segmentation.backends.unet_snip.model_loader import (
    FishModelSnipPredictor,
    UNetSnipPredictorConfig,
    load_unet_snip_predictors,
    parse_unet_snip_model_config,
)

ARTIFACT_SHAPE = (576, 256)


def _fresh_fish_model():
    # Imported lazily: FishModel pulls in heavy Torch/segmentation deps at import time, which
    # the pure-config parser tests must not pay. Only the predictor/loader tests call this.
    from src.core.functions.core_utils_segmentation import FishModel  # type: ignore

    return FishModel("FPN", "resnet34", in_channels=3, out_classes=1, encoder_weights=None).eval()


def _nested_config(**overrides):
    cfg = {
        "models_root": "models/auxiliary_masks",
        "device": "cpu",
        "models": {
            "via": {"checkpoint": "via_v1_0100", "model_input_shape": [576, 256]},
            "yolk": {"checkpoint": "yolk_v1_0050", "model_input_shape": [576, 256]},
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# parse_unet_snip_model_config
# ---------------------------------------------------------------------------

def test_parser_reads_nested_models():
    specs = parse_unet_snip_model_config(_nested_config(), artifact_shape=ARTIFACT_SHAPE)
    by_type = {s.mask_type: s for s in specs}
    assert set(by_type) == {"via", "yolk"}
    assert isinstance(by_type["via"], UNetSnipPredictorConfig)
    assert by_type["via"].checkpoint_path.as_posix() == "models/auxiliary_masks/via_v1_0100"
    assert by_type["via"].model_input_shape == ARTIFACT_SHAPE
    assert by_type["via"].artifact_shape == ARTIFACT_SHAPE


def test_parser_defaults_model_input_shape_to_artifact_shape():
    cfg = _nested_config()
    cfg["models"]["via"].pop("model_input_shape")  # omitted -> default to artifact_shape
    specs = parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)
    via = next(s for s in specs if s.mask_type == "via")
    assert via.model_input_shape == ARTIFACT_SHAPE


def test_parser_honors_distinct_model_input_shape():
    cfg = _nested_config()
    cfg["models"]["via"]["model_input_shape"] = [288, 128]
    specs = parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)
    via = next(s for s in specs if s.mask_type == "via")
    assert via.model_input_shape == (288, 128)
    assert via.artifact_shape == ARTIFACT_SHAPE  # the law is unaffected


def test_parser_rejects_missing_checkpoint():
    cfg = _nested_config()
    cfg["models"]["via"].pop("checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)


def test_parser_rejects_malformed_model_input_shape():
    cfg = _nested_config()
    cfg["models"]["via"]["model_input_shape"] = [576, 0]
    with pytest.raises(ValueError, match="model_input_shape"):
        parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)


def test_parser_rejects_legacy_flat_checkpoints():
    cfg = {"models_root": "models/auxiliary_masks", "checkpoints": {"via": "via_v1_0100"}}
    with pytest.raises(ValueError, match="Flat unet_snip.checkpoints is no longer supported"):
        parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)


def test_parser_rejects_missing_models_root():
    cfg = _nested_config()
    cfg.pop("models_root")
    with pytest.raises(ValueError, match="models_root"):
        parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)


def test_parser_rejects_unknown_mask_type():
    cfg = _nested_config()
    cfg["models"]["foreground"] = {"checkpoint": "x"}
    with pytest.raises(ValueError, match="unknown mask type"):
        parse_unet_snip_model_config(cfg, artifact_shape=ARTIFACT_SHAPE)


# ---------------------------------------------------------------------------
# FishModelSnipPredictor — comes home to the LAW (artifact_shape)
# ---------------------------------------------------------------------------

def test_predictor_returns_artifact_shape_when_input_matches():
    predictor = FishModelSnipPredictor(
        _fresh_fish_model(),
        model_input_shape=ARTIFACT_SHAPE,
        artifact_shape=ARTIFACT_SHAPE,
        device="cpu",
    )
    mask = predictor(np.random.randint(0, 255, ARTIFACT_SHAPE, dtype=np.uint8))
    assert mask.dtype == bool
    assert mask.shape == ARTIFACT_SHAPE


def test_predictor_returns_artifact_shape_when_input_differs():
    """model_input_shape != artifact_shape: the model runs on the doorway grid, but the mask
    still comes home to the artifact law."""
    predictor = FishModelSnipPredictor(
        _fresh_fish_model(),
        model_input_shape=(288, 128),  # the doorway differs from the law
        artifact_shape=ARTIFACT_SHAPE,
        device="cpu",
    )
    mask = predictor(np.random.randint(0, 255, ARTIFACT_SHAPE, dtype=np.uint8))
    assert mask.shape == ARTIFACT_SHAPE


# ---------------------------------------------------------------------------
# load_unet_snip_predictors (checkpoint loading mocked)
# ---------------------------------------------------------------------------

def test_load_unet_snip_predictors_builds_one_per_spec(monkeypatch):
    import data_pipeline.models.unet as unet_mod

    monkeypatch.setattr(
        unet_mod, "load_fish_unet_model",
        lambda path, device: _fresh_fish_model(),
        raising=False,
    )

    specs = parse_unet_snip_model_config(_nested_config(), artifact_shape=ARTIFACT_SHAPE)
    predictors = load_unet_snip_predictors(specs, device="cpu")

    assert set(predictors) == {"via", "yolk"}
    for mt, pred in predictors.items():
        assert callable(pred), f"{mt} predictor is not callable"


# ---------------------------------------------------------------------------
# No legacy attic ghosts
# ---------------------------------------------------------------------------

def test_no_legacy_576_320_default_or_im_dims():
    import inspect
    src = inspect.getsource(model_loader)
    assert "320" not in src, "legacy (576, 320) default must be gone"
    assert "im_dims" not in src, "legacy im_dims must be gone"
