"""Tests for the GroundingDINO adapter.

Structural/interface tests (no GPU, no real weights) are the priority, matching test_harness.py's
"FakeAdapter, no torch needed" spirit as closely as this adapter's real torch import allows. This
module DOES import torch transitively (via data_pipeline.models.groundingdino / the adapter itself),
but none of these tests load real model weights or require a GPU/CUDA device -- they exercise the
adapter's construction, registration, payload handling, and failure/isolation contracts using a stub
in place of the actual GroundingDINO nn.Module.

GPU-dependent, real-weights equivalence testing (byte-identical output vs. the per-well
cmd_frame_detections path, throughput comparison) is done separately as a manual script + report,
not as part of the pytest suite, because it needs real experiment data + real weights that are not
guaranteed present in every test environment. See the task's final report for that proof.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_adapter_registered_under_grounding_dino():
    from data_pipeline.model_servers.adapter_base import get_adapter_class, registered_adapter_names
    from data_pipeline.model_servers.adapters.grounding_dino import GroundingDinoAdapter

    assert "grounding_dino" in registered_adapter_names()
    assert get_adapter_class("grounding_dino") is GroundingDinoAdapter


def test_construction_stores_string_kwargs_without_type_coercion():
    """Constructor kwargs arrive as strings from --adapter-arg KEY=VALUE; verify no premature Path()."""
    from data_pipeline.model_servers.adapters.grounding_dino import GroundingDinoAdapter

    adapter = GroundingDinoAdapter(
        gdino_repo_dir="/some/repo",
        gdino_config="/some/repo/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        gdino_weights="/some/repo/weights/groundingdino_swint_ogc.pth",
        device="cpu",
    )
    assert adapter.gdino_repo_dir == "/some/repo"
    assert adapter.device == "cpu"
    assert adapter.model is None
    assert adapter.default_detector_model_id == "SwinT_OGC"


def test_handle_before_load_raises():
    from data_pipeline.model_servers.adapters.grounding_dino import GroundingDinoAdapter

    adapter = GroundingDinoAdapter(
        gdino_repo_dir="/x", gdino_config="/x/cfg.py", gdino_weights="/x/w.pth", device="cpu"
    )
    with pytest.raises(RuntimeError, match="before load"):
        adapter.handle({"frame_inventory_csv": "/nope.csv", "output_csv": "/nope_out.csv"})


def test_handle_writes_atomically_and_matches_run_frame_detection_df(tmp_path, monkeypatch):
    """Verify handle() writes exactly run_frame_detection_df's output, atomically, using a stub model.

    We monkeypatch run_frame_detection_df itself (the shared core also used by cmd_frame_detections)
    so this test needs no real GroundingDINO weights/GPU, while still proving handle()'s plumbing:
    it reads the input CSV, calls the shared detection core with the right kwargs, and atomically
    writes exactly what that core returned -- no divergent logic of its own.
    """
    import data_pipeline.model_servers.adapters.grounding_dino as gdino_adapter_mod

    frame_inventory_csv = tmp_path / "frame_inventory.csv"
    output_csv = tmp_path / "frame_detections.csv"
    pd.DataFrame({"experiment_id": ["e1"], "well_id": ["e1_A01"]}).to_csv(frame_inventory_csv, index=False)

    expected = pd.DataFrame(
        {
            "detection_id": ["e1_A01_BF_t0000_det0"],
            "confidence": [0.87],
            "is_kept": [True],
        }
    )

    captured_kwargs = {}

    def _fake_run_frame_detection_df(reference_frame_inventory, *, backend, model, detector_model_id, config=None):
        captured_kwargs["backend"] = backend
        captured_kwargs["model"] = model
        captured_kwargs["detector_model_id"] = detector_model_id
        captured_kwargs["config"] = config
        pd.testing.assert_frame_equal(
            reference_frame_inventory.reset_index(drop=True),
            pd.read_csv(frame_inventory_csv).reset_index(drop=True),
        )
        return expected

    monkeypatch.setattr(gdino_adapter_mod, "run_frame_detection_df", _fake_run_frame_detection_df)

    adapter = gdino_adapter_mod.GroundingDinoAdapter(
        gdino_repo_dir="/x", gdino_config="/x/cfg.py", gdino_weights="/x/w.pth", device="cpu"
    )
    stub_model = object()
    adapter.model = stub_model  # bypass load(); load() itself needs real weights

    adapter.handle({"frame_inventory_csv": str(frame_inventory_csv), "output_csv": str(output_csv)})

    assert captured_kwargs["backend"] == "groundingdino"
    assert captured_kwargs["model"] is stub_model
    assert captured_kwargs["detector_model_id"] == "SwinT_OGC"
    assert captured_kwargs["config"].device == "cpu"

    written = pd.read_csv(output_csv)
    pd.testing.assert_frame_equal(written, expected)

    leftover = list(tmp_path.glob(".frame_detections.csv.tmp-*"))
    assert leftover == [], f"leftover temp file(s) after atomic write: {leftover}"


def test_handle_respects_payload_detector_model_id_override(tmp_path, monkeypatch):
    import data_pipeline.model_servers.adapters.grounding_dino as gdino_adapter_mod

    frame_inventory_csv = tmp_path / "frame_inventory.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame({"col": [1]}).to_csv(frame_inventory_csv, index=False)

    captured = {}

    def _fake(reference_frame_inventory, *, backend, model, detector_model_id, config=None):
        captured["detector_model_id"] = detector_model_id
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(gdino_adapter_mod, "run_frame_detection_df", _fake)

    adapter = gdino_adapter_mod.GroundingDinoAdapter(
        gdino_repo_dir="/x", gdino_config="/x/cfg.py", gdino_weights="/x/w.pth",
        detector_model_id="default_id",
    )
    adapter.model = object()

    adapter.handle(
        {
            "frame_inventory_csv": str(frame_inventory_csv),
            "output_csv": str(output_csv),
            "detector_model_id": "override_id",
        }
    )
    assert captured["detector_model_id"] == "override_id"


def test_per_request_isolation_no_mutable_state_carried_between_calls(tmp_path, monkeypatch):
    """Two consecutive handle() calls with different inputs must not leak state between them.

    GDINO is stateless (verified in the adapter's module docstring by reading the full call chain:
    run_frame_detection_df / detect_frame / detect_embryos / groundingdino.util.inference.predict --
    none of them write attributes onto `model`, unlike SAM2's inference_state). This test asserts
    that behavior end-to-end through the adapter: the same `model` object is passed on both calls
    (never rebuilt or reset), each call's output reflects only its own input, and the adapter object
    itself carries no per-call counters/caches that would make call 2's result depend on call 1.
    """
    import data_pipeline.model_servers.adapters.grounding_dino as gdino_adapter_mod

    seen_models = []

    def _fake(reference_frame_inventory, *, backend, model, detector_model_id, config=None):
        seen_models.append(model)
        # Echo back something derived ONLY from this call's own input.
        marker = reference_frame_inventory["marker"].iloc[0]
        return pd.DataFrame({"marker_echo": [marker]})

    monkeypatch.setattr(gdino_adapter_mod, "run_frame_detection_df", _fake)

    adapter = gdino_adapter_mod.GroundingDinoAdapter(
        gdino_repo_dir="/x", gdino_config="/x/cfg.py", gdino_weights="/x/w.pth"
    )
    stub_model = object()
    adapter.model = stub_model

    for i in range(3):
        frame_inventory_csv = tmp_path / f"fi_{i}.csv"
        output_csv = tmp_path / f"out_{i}.csv"
        pd.DataFrame({"marker": [f"well-{i}"]}).to_csv(frame_inventory_csv, index=False)
        adapter.handle({"frame_inventory_csv": str(frame_inventory_csv), "output_csv": str(output_csv)})
        result = pd.read_csv(output_csv)
        assert result["marker_echo"].iloc[0] == f"well-{i}"

    # The identical model object was reused (resident-server amortization), never rebuilt per call.
    assert seen_models == [stub_model, stub_model, stub_model]


def test_handle_missing_payload_keys_raises_keyerror():
    from data_pipeline.model_servers.adapters.grounding_dino import GroundingDinoAdapter

    adapter = GroundingDinoAdapter(
        gdino_repo_dir="/x", gdino_config="/x/cfg.py", gdino_weights="/x/w.pth"
    )
    adapter.model = object()
    with pytest.raises(KeyError):
        adapter.handle({})
