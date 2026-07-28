"""Step 6 — scaffold a starter drop-in frame_inventory from an image directory."""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from data_pipeline.acquisition.metadata_ingest.frame_inventory.scaffold_dropin_inventory import (
    discover_image_files,
    scaffold_dropin_inventory,
    scaffold_row_for_image,
)


def _png(path, w=16, h=24):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h)).save(path)
    return path


def test_scaffold_reads_dims_and_infers_atoms(tmp_path):
    img = _png(tmp_path / "imgs" / "my_experiment_B01_BF_t0000.png", w=20, h=30)
    row = scaffold_row_for_image(img, image_root=tmp_path / "imgs")
    assert row["channel_id"] == "BF"
    assert row["time_index"] == 0
    assert row["image_width_px"] == 20
    assert row["image_height_px"] == 30
    # The user must still fill these — the scaffold leaves them blank.
    assert row["experiment_id"] == ""
    assert row["well_index"] == ""
    assert row["image_micrometers_per_pixel"] == ""
    # well_id / image_id are never authored.
    assert "well_id" not in row
    assert "image_id" not in row


def test_scaffold_relative_path_when_under_root(tmp_path):
    img = _png(tmp_path / "imgs" / "a_B01_BF_t0001.png")
    row = scaffold_row_for_image(img, image_root=tmp_path / "imgs")
    assert row["image_path"] == "a_B01_BF_t0001.png"


def test_scaffold_unparseable_name_leaves_blanks(tmp_path):
    img = _png(tmp_path / "imgs" / "random.png")
    row = scaffold_row_for_image(img, image_root=tmp_path / "imgs")
    assert row["channel_id"] == ""
    assert row["time_index"] == ""


def test_scaffold_writes_csv_for_all_images(tmp_path):
    _png(tmp_path / "imgs" / "exp_A01_BF_t0000.png")
    _png(tmp_path / "imgs" / "exp_A01_BF_t0001.png")
    out = tmp_path / "dropin_frame_inventory.csv"
    df = scaffold_dropin_inventory(tmp_path / "imgs", out)
    assert out.exists()
    written = pd.read_csv(out)
    assert len(written) == 2
    assert list(df["time_index"]) == [0, 1]


def test_scaffold_empty_dir_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        discover_image_files(tmp_path / "empty")
