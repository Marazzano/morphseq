"""Keyence acquisition-inventory source-path resolution (stored absolute, re-anchored on move).

Run with:
    PYTHONPATH=src pytest \
      src/data_pipeline/acquisition/metadata_ingest/scope/keyence/tests/test_keyence_source_paths.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import (
    assert_keyence_acquisition_sources_readable,
)

_LABEL = "test"


def test_readability_uses_stored_absolute_path_when_present(monkeypatch, tmp_path):
    tiff = tmp_path / "raw_image_data" / "Keyence" / "exp" / "XY01" / "img.tif"
    tiff.parent.mkdir(parents=True)
    tiff.write_bytes(b"")

    opened: list[str] = []
    import skimage.io as skio

    monkeypatch.setattr(skio, "imread", lambda p: opened.append(str(p)) or None)

    df = pd.DataFrame({"source_tiff_path": [str(tiff)]})  # full absolute path that exists
    assert_keyence_acquisition_sources_readable(df, scope_label=_LABEL, input_root=None)
    assert opened == [str(tiff)]


def test_readability_reanchors_stale_absolute_under_input_root(monkeypatch, tmp_path):
    new_root = tmp_path / "new"
    real = new_root / "raw_image_data" / "Keyence" / "exp" / "XY01" / "img.tif"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"")

    opened: list[str] = []
    import skimage.io as skio

    monkeypatch.setattr(skio, "imread", lambda p: opened.append(str(p)) or None)

    stale = "/OLD/pipeline/input/raw_image_data/Keyence/exp/XY01/img.tif"
    df = pd.DataFrame({"source_tiff_path": [stale]})
    assert_keyence_acquisition_sources_readable(df, scope_label=_LABEL, input_root=new_root)
    assert opened == [str(real.resolve())]


def test_readability_stale_path_no_pivot_raises(tmp_path):
    df = pd.DataFrame({"source_tiff_path": ["/OLD/no_pivot/img.tif"]})  # missing, no raw_image_data/
    with pytest.raises(ValueError, match="raw_image_data"):
        assert_keyence_acquisition_sources_readable(df, scope_label=_LABEL, input_root=tmp_path)
