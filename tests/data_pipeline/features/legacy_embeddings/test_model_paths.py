"""Contract tests for the legacy VAE model-dir resolver.

Pins the resolved path, the tolerated nested ``final_model/`` layout, and the
fail-loud message (important words, not exact prose) — not implementation
internals. Runs under the main pipeline env (no torch, no real weights).
"""

from __future__ import annotations

import pytest

from data_pipeline.features.legacy_embeddings.model_paths import (
    legacy_models_dir,
    resolve_legacy_model_dir,
)

MODEL_NAME = "20241107_ds_sweep01_optimum"


def _make_model_dir(models_root, *, with_config: bool, nested: bool = False):
    """Create <models_root>/legacy/<name>/ and optionally a config (flat or nested)."""
    model_dir = legacy_models_dir(models_root) / MODEL_NAME
    config_dir = model_dir / "final_model" if nested else model_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    if with_config:
        (config_dir / "model_config.json").write_text("{}")
    return model_dir, config_dir


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Resolution — the worked examples
# ─────────────────────────────────────────────────────────────────────────────────────────────

def test_resolves_flat_model_dir(tmp_path):
    model_dir, _ = _make_model_dir(tmp_path, with_config=True)

    resolved = resolve_legacy_model_dir(tmp_path, MODEL_NAME)

    assert resolved == model_dir
    assert resolved == tmp_path / "legacy" / MODEL_NAME


def test_resolves_into_nested_final_model_when_config_lives_there(tmp_path):
    _, nested_dir = _make_model_dir(tmp_path, with_config=True, nested=True)

    resolved = resolve_legacy_model_dir(tmp_path, MODEL_NAME)

    assert resolved == nested_dir
    assert resolved.name == "final_model"


def test_returns_top_dir_when_no_config_anywhere(tmp_path):
    # Directory exists but carries no config and no final_model/ — a warning, not an error.
    model_dir, _ = _make_model_dir(tmp_path, with_config=False)

    resolved = resolve_legacy_model_dir(tmp_path, MODEL_NAME)

    assert resolved == model_dir


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Fail-loud — the error path names the missing dir AND the fix
# ─────────────────────────────────────────────────────────────────────────────────────────────

def test_missing_model_dir_fails_loud_with_path_and_fix(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_legacy_model_dir(tmp_path, MODEL_NAME)

    msg = str(exc.value)
    # Names the exact missing path...
    assert str(tmp_path / "legacy" / MODEL_NAME) in msg
    # ...and tells the reader what to do.
    assert "model_config.json" in msg
    assert "model_name" in msg
