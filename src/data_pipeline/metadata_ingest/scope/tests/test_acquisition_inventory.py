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
from data_pipeline.metadata_ingest.scope.yx1.acquisition_inventory import (
    YX1_ACQUISITION_CELL_KEY,
    YX1_ACQUISITION_INVENTORY_COLUMNS,
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
                "raw_channel_name", "channel", "source_nd2_path", "n_z"):
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
