"""Tests for resolve_under_input_root (stored-absolute path, re-anchored on move).

Run with: PYTHONPATH=src pytest src/data_pipeline/shared/tests/test_path_roots.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline.shared.path_roots import resolve_under_input_root

_LABEL = "test"


def _make(root: Path, *parts: str) -> Path:
    p = root.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


def test_default_returns_stored_path_as_is(tmp_path):
    # full_root_fallback=False: the stored absolute path is returned verbatim (existing behavior).
    stored = "/some/absolute/raw_image_data/Keyence/exp/XY01/img.tif"
    assert resolve_under_input_root(stored, input_root=tmp_path, scope_label=_LABEL) == Path(stored)


def test_fallback_uses_stored_path_when_it_exists(tmp_path):
    f = _make(tmp_path, "raw_image_data", "Keyence", "exp", "XY01", "img.tif")
    # Even with fallback on, an existing stored path is used unchanged.
    got = resolve_under_input_root(
        f, input_root=tmp_path / "other_root", scope_label=_LABEL, full_root_fallback=True
    )
    assert got == f


def test_fallback_reanchors_stale_path_onto_input_root(tmp_path):
    # Real file lives under a NEW input_root; the stored path points at an OLD (nonexistent) root.
    new_root = tmp_path / "new"
    real = _make(new_root, "raw_image_data", "Keyence", "exp", "XY01", "img.tif")
    stale = "/OLD/pipeline/input/raw_image_data/Keyence/exp/XY01/img.tif"

    got = resolve_under_input_root(
        stale, input_root=new_root, scope_label=_LABEL, full_root_fallback=True
    )
    assert got == real.resolve()


def test_fallback_raises_when_no_raw_image_data_segment(tmp_path):
    stale = "/OLD/somewhere/no_pivot/img.tif"  # missing, and no raw_image_data/ to pivot on
    with pytest.raises(ValueError, match="raw_image_data"):
        resolve_under_input_root(
            stale, input_root=tmp_path, scope_label=_LABEL, full_root_fallback=True
        )


def test_fallback_raises_when_no_input_root(tmp_path):
    stale = "/OLD/pipeline/input/raw_image_data/Keyence/exp/XY01/img.tif"  # missing
    with pytest.raises(ValueError, match="no input_root"):
        resolve_under_input_root(
            stale, input_root=None, scope_label=_LABEL, full_root_fallback=True
        )
