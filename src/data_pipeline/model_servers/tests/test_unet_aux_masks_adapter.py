"""Tests for the snip_auxiliary_masks (UNet) adapter.

Structural/interface tests need no GPU and no real checkpoints -- they exercise
registration, construction, and config wiring only. A couple of tests are gated on
real checkpoints + real snip_inventory data being present on this filesystem (the
shared nlammers model/output trees); they are skipped, not failed, when absent, so
this file runs green in CI environments without that data mounted.

The end-to-end equivalence proof (server output vs. existing per-well path, on real
wells) lives in a standalone script, not here, per the task's deliverables --
this file covers structure only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REAL_MODELS_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/models")
REAL_OUTPUT_ROOT = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output")
REAL_WELL_ID = "20260319_A01"
REAL_SNIP_INVENTORY_CSV = (
    REAL_OUTPUT_ROOT
    / "object_extraction"
    / "20260319"
    / "snips"
    / "per_well"
    / REAL_WELL_ID
    / f"{REAL_WELL_ID}_snip_inventory.csv"
)

_HAS_REAL_DATA = REAL_MODELS_ROOT.is_dir() and REAL_SNIP_INVENTORY_CSV.is_file()

_UNET_SNIP_CONFIG = {
    "models_root": str(REAL_MODELS_ROOT),
    "device": "cpu",
    "models": {
        "via": {"checkpoint": "segmentation/via_v1_0100"},
        "yolk": {"checkpoint": "segmentation/yolk_v1_0050"},
        "focus": {"checkpoint": "segmentation/focus_v0_0100"},
        "bubble": {"checkpoint": "segmentation/bubble_v0_0100"},
    },
}


def test_adapter_registered_under_snip_auxiliary_masks():
    # Importing the adapters package registers every adapter as a side effect.
    import data_pipeline.model_servers.adapters  # noqa: F401
    from data_pipeline.model_servers.adapter_base import get_adapter_class, registered_adapter_names

    assert "snip_auxiliary_masks" in registered_adapter_names()

    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    assert get_adapter_class("snip_auxiliary_masks") is SnipAuxiliaryMasksAdapter


def test_adapter_satisfies_model_adapter_protocol():
    from data_pipeline.model_servers.adapter_base import ModelAdapter
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    adapter = SnipAuxiliaryMasksAdapter(unet_snip_config_json=json.dumps(_UNET_SNIP_CONFIG))
    assert isinstance(adapter, ModelAdapter)


def test_constructor_accepts_string_kwargs_only():
    """Constructor args arrive as strings from --adapter-arg KEY=VALUE; verify no non-string
    default breaks that contract, and that nested config travels as a JSON string."""
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    adapter = SnipAuxiliaryMasksAdapter(
        unet_snip_config_json=json.dumps(_UNET_SNIP_CONFIG),
        snip_frame_shape_json="[576, 256]",
        device="cpu",
    )
    assert adapter.snip_frame_shape == (576, 256)
    assert adapter.unet_snip_config["device"] == "cpu"
    assert adapter.predictors is None  # not loaded yet


def test_handle_before_load_raises():
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    adapter = SnipAuxiliaryMasksAdapter(unet_snip_config_json=json.dumps(_UNET_SNIP_CONFIG))
    with pytest.raises(RuntimeError, match="before load"):
        adapter.handle({"snip_inventory_csv": "x", "output_root": "y", "output_csv": "z"})


def test_device_kwarg_does_not_override_explicit_config_device():
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    cfg = dict(_UNET_SNIP_CONFIG)
    cfg["device"] = "cpu"
    adapter = SnipAuxiliaryMasksAdapter(unet_snip_config_json=json.dumps(cfg), device="cuda")
    # config already specified device=cpu; the adapter must not clobber an explicit
    # config value with the constructor default/CLI value.
    assert adapter.unet_snip_config["device"] == "cpu"


def test_config_parses_to_four_independent_specs_without_loading_models():
    """parse_unet_snip_model_config (imported, not duplicated) must resolve exactly the
    four allowed mask families with independent checkpoint paths -- this is a pure
    config-parsing check, no torch/model load involved."""
    from data_pipeline.object_extraction.segmentation.backends.unet_snip.model_loader import (
        parse_unet_snip_model_config,
    )

    specs = parse_unet_snip_model_config(_UNET_SNIP_CONFIG, artifact_shape=(576, 256))
    mask_types = {s.mask_type for s in specs}
    assert mask_types == {"via", "yolk", "focus", "bubble"}

    checkpoint_paths = {s.checkpoint_path for s in specs}
    assert len(checkpoint_paths) == 4, "all four checkpoint paths must be distinct (no shared weights)"


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="real models_root / snip_inventory not present on this filesystem")
def test_load_then_handle_produces_valid_manifest_on_real_well(tmp_path):
    """Full load() + handle() against real checkpoints (CPU) and a real, small well.

    This is a slower integration-style test (loads 4 real checkpoints on CPU), gated
    behind real-data presence so it is skipped rather than failing in environments
    without the shared nlammers mount.
    """
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter
    from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
        load_snip_auxiliary_masks,
    )

    adapter = SnipAuxiliaryMasksAdapter(unet_snip_config_json=json.dumps(_UNET_SNIP_CONFIG))
    adapter.load()
    assert adapter.predictors is not None
    assert set(adapter.predictors) == {"via", "yolk", "focus", "bubble"}

    output_csv = tmp_path / "manifest.csv"
    adapter.handle(
        {
            "snip_inventory_csv": str(REAL_SNIP_INVENTORY_CSV),
            "output_root": str(REAL_OUTPUT_ROOT),
            "output_csv": str(output_csv),
        }
    )

    df = load_snip_auxiliary_masks(output_csv)  # raises on any contract violation
    assert len(df) > 0
    assert df["is_valid_auxiliary_mask"].all(), "expected all masks valid for a clean real well"
    assert set(df["auxiliary_mask_type"]) == {"via", "yolk", "focus", "bubble"}


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="real models_root / snip_inventory not present on this filesystem")
def test_second_handle_call_is_isolated_from_first(tmp_path):
    """Two handle() calls (simulating two wells served by one resident process) must not
    interfere -- second well's output must reflect only its own input rows."""
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    adapter = SnipAuxiliaryMasksAdapter(unet_snip_config_json=json.dumps(_UNET_SNIP_CONFIG))
    adapter.load()

    out1 = tmp_path / "manifest1.csv"
    out2 = tmp_path / "manifest2.csv"
    payload = {
        "snip_inventory_csv": str(REAL_SNIP_INVENTORY_CSV),
        "output_root": str(REAL_OUTPUT_ROOT),
    }
    adapter.handle({**payload, "output_csv": str(out1)})
    adapter.handle({**payload, "output_csv": str(out2)})  # same well again, thin re-run

    import pandas as pd

    df1 = pd.read_csv(out1)
    df2 = pd.read_csv(out2)
    assert df1["snip_id"].tolist() == df2["snip_id"].tolist()
    assert df1["auxiliary_mask_type"].tolist() == df2["auxiliary_mask_type"].tolist()
