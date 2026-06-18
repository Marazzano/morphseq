"""Tests for the shared acquisition check primitives and the YX1 acquisition inventory.

Run with: PYTHONPATH=src pytest src/data_pipeline/metadata_ingest/scope/tests/test_acquisition_inventory.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.metadata_ingest.scope.shared.acquisition_checks import (
    assert_channel_mapping_consistent,
    assert_columns_present,
    assert_positive_column,
    assert_unique_on_key,
)
from data_pipeline.metadata_ingest.scope.yx1 import acquisition_inventory as acq_mod
from data_pipeline.metadata_ingest.scope.yx1.acquisition_inventory import (
    YX1_ACQUISITION_CELL_KEY,
    YX1_ACQUISITION_INVENTORY_COLUMNS,
    assert_acquisition_sources_readable,
    assert_channel_id_in_vocabulary,
    build_yx1_acquisition_inventory,
    build_yx1_acquisition_inventory_rows,
    validate_yx1_acquisition_inventory,
)

_LABEL = "test"


# ── shared primitives ────────────────────────────────────────────────────────────────────────


def test_assert_columns_present_passes_and_fails():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert_columns_present(df, ["a", "b"], scope_label=_LABEL)
    with pytest.raises(ValueError, match="Missing required columns"):
        assert_columns_present(df, ["a", "c"], scope_label=_LABEL)


def test_assert_positive_column_rejects_zero_nan_negative_missing():
    assert_positive_column(pd.DataFrame({"x": [1.0, 2.0]}), "x", scope_label=_LABEL)

    for bad in ([0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0]):
        with pytest.raises(ValueError, match="must be present and > 0"):
            assert_positive_column(pd.DataFrame({"x": bad}), "x", scope_label=_LABEL)

    with pytest.raises(ValueError, match="is missing"):
        assert_positive_column(pd.DataFrame({"y": [1.0]}), "x", scope_label=_LABEL)


def test_assert_unique_on_key_detects_collision():
    clean = pd.DataFrame({"p": [0, 0, 1], "t": [0, 1, 0]})
    assert_unique_on_key(clean, ["p", "t"], scope_label=_LABEL)

    collided = pd.DataFrame({"p": [0, 0], "t": [0, 0]})
    with pytest.raises(ValueError, match="not unique on the cell key"):
        assert_unique_on_key(collided, ["p", "t"], scope_label=_LABEL)


def test_assert_channel_mapping_consistent():
    good = pd.DataFrame(
        {"channel_index": [0, 0, 1], "raw_channel_name": ["EYES - Dia", "EYES - Dia", "EYES - GFP"],
         "channel": ["BF", "BF", "GFP"]}
    )
    assert_channel_mapping_consistent(good, scope_label=_LABEL)

    # one index maps to two raw names → inconsistent
    bad = pd.DataFrame(
        {"channel_index": [0, 0], "raw_channel_name": ["EYES - Dia", "EYES - GFP"],
         "channel": ["BF", "GFP"]}
    )
    with pytest.raises(ValueError, match="inconsistent channel mapping"):
        assert_channel_mapping_consistent(bad, scope_label=_LABEL)


# ── YX1 inventory builder ────────────────────────────────────────────────────────────────────


def _make_inventory(*, n_t=2, n_z=3, channels=None):
    channels = channels or [(0, "BF", "EYES - Dia"), (1, "GFP", "EYES - GFP")]
    return build_yx1_acquisition_inventory(
        experiment_id="20250912",
        n_t=n_t,
        n_z=n_z,
        timestamps=[100.0 * t for t in range(n_t)],
        channels=channels,
        stage_xy={0: (10.0, 20.0), 1: (30.0, 40.0)},
        micrometers_per_pixel=3.25,
        image_width_px=2048,
        image_height_px=2048,
        objective_magnification="4x",
        source_nd2_path="/data/20250912/exp.nd2",
    )


def test_inventory_grain_is_position_z_channel_time():
    n_t, n_z = 2, 3
    df = _make_inventory(n_t=n_t, n_z=n_z)
    n_positions, n_channels = 2, 2
    assert len(df) == n_positions * n_t * n_z * n_channels


def test_inventory_has_tensor_axes_and_channel_mapping():
    df = _make_inventory()
    for col in ("position_index", "z_index", "channel_index", "time_index",
                "raw_channel_name", "channel_id", "source_nd2_path", "n_z"):
        assert col in df.columns
    # Z is exploded, not collapsed.
    assert sorted(df["z_index"].unique().tolist()) == [0, 1, 2]
    # full schema is exactly the declared column set.
    assert list(df.columns) == list(YX1_ACQUISITION_INVENTORY_COLUMNS)
    # no per-plane path — only the ND2.
    assert "source_image_path" not in df.columns
    assert (df["source_nd2_path"] == "/data/20250912/exp.nd2").all()


def test_inventory_is_unique_on_the_tensor_cell():
    df = _make_inventory()
    assert not df.duplicated(subset=list(YX1_ACQUISITION_CELL_KEY)).any()


def test_inventory_validates_clean():
    df = _make_inventory()
    validate_yx1_acquisition_inventory(df)  # no raise


def test_validate_rejects_bad_calibration():
    df = _make_inventory()
    df.loc[0, "micrometers_per_pixel"] = 0.0
    with pytest.raises(ValueError, match="micrometers_per_pixel"):
        validate_yx1_acquisition_inventory(df)


def test_validate_rejects_unknown_channel_id():
    # An unrecognized raw channel falls through the normalizer as its own channel_id; the vocabulary
    # gate must catch it HERE (at the minting point) rather than let it flow downstream.
    df = _make_inventory()
    df["channel_id"] = "Cy5"  # not in VALID_CHANNEL_NAMES
    with pytest.raises(ValueError, match="not in the allowed channel vocabulary"):
        validate_yx1_acquisition_inventory(df)


def test_assert_channel_id_in_vocabulary_passes_on_known_tokens():
    df = _make_inventory()  # channels default to BF/GFP — both valid
    assert_channel_id_in_vocabulary(df, scope_label=_LABEL)  # no raise


# ── source readability (check_sources mode) ──────────────────────────────────────────────────


class _FakeND2:
    """Stand-in for nd2.ND2File: opening succeeds, close() is a no-op."""

    def __init__(self, path):
        self.path = path

    def close(self):
        pass


def test_check_sources_false_is_default_and_skips_file_io(monkeypatch):
    # _make_inventory points source_nd2_path at a nonexistent file; build-mode must NOT touch it.
    def _boom(path):  # would fire if the validator opened the file
        raise AssertionError("ND2File must not be opened when check_sources=False")

    monkeypatch.setattr(acq_mod.nd2, "ND2File", _boom)
    df = _make_inventory()
    validate_yx1_acquisition_inventory(df)  # default check_sources=False → no raise, no open


def test_check_sources_true_passes_when_nd2_opens(monkeypatch, tmp_path):
    nd2_file = tmp_path / "exp.nd2"
    nd2_file.write_bytes(b"")  # exists; open is faked below
    monkeypatch.setattr(acq_mod.nd2, "ND2File", _FakeND2)
    df = _make_inventory()
    df["source_nd2_path"] = str(nd2_file)
    validate_yx1_acquisition_inventory(df, check_sources=True)  # no raise


def test_check_sources_true_fails_when_nd2_missing():
    df = _make_inventory()  # source_nd2_path = "/data/20250912/exp.nd2" (does not exist)
    with pytest.raises(ValueError, match=r"does not exist.*/data/20250912/exp\.nd2|/data/20250912/exp\.nd2.*does not exist"):
        validate_yx1_acquisition_inventory(df, check_sources=True)


def test_check_sources_true_fails_when_nd2_unopenable(monkeypatch, tmp_path):
    nd2_file = tmp_path / "broken.nd2"
    nd2_file.write_bytes(b"not a real nd2")

    def _raise(path):
        raise RuntimeError("bad magic bytes")

    monkeypatch.setattr(acq_mod.nd2, "ND2File", _raise)
    df = _make_inventory()
    df["source_nd2_path"] = str(nd2_file)
    with pytest.raises(ValueError, match="failed to open"):
        validate_yx1_acquisition_inventory(df, check_sources=True)


def test_assert_sources_readable_opens_each_unique_path_once(monkeypatch, tmp_path):
    nd2_file = tmp_path / "shared.nd2"
    nd2_file.write_bytes(b"")
    calls: list[str] = []

    def _record(path):
        calls.append(str(path))
        return _FakeND2(path)

    monkeypatch.setattr(acq_mod.nd2, "ND2File", _record)
    # Two positions sharing ONE ND2 → many rows, one unique path → one open.
    df = _make_inventory()
    df["source_nd2_path"] = str(nd2_file)
    assert len(df) > 1
    assert_acquisition_sources_readable(df, scope_label="test")
    assert calls == [str(nd2_file)]


def test_rows_builder_is_pure_and_orders_positions():
    rows = build_yx1_acquisition_inventory_rows(
        experiment_id="E",
        n_t=1,
        n_z=1,
        timestamps=[0.0],
        channels=[(0, "BF", "EYES - Dia")],
        stage_xy={1: (1.0, 1.0), 0: (0.0, 0.0)},
        micrometers_per_pixel=1.0,
        image_width_px=10,
        image_height_px=10,
        objective_magnification="4x",
        source_nd2_path="x.nd2",
    )
    assert [r["position_index"] for r in rows] == [0, 1]
    assert rows[0]["raw_position_label"] == "0"
