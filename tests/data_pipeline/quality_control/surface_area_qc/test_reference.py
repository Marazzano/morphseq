"""surface_area_qc reference tests — packaged reference loads, validates, interpolates."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.surface_area_qc.reference import (
    interpolate_reference_band,
    load_packaged_surface_area_reference,
    packaged_reference_path,
)
from data_pipeline.quality_control.surface_area_qc.reference_contract import (
    SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS,
    validate_surface_area_reference,
)


def test_packaged_v1_exists_and_validates():
    assert packaged_reference_path("v1").exists()
    ref = load_packaged_surface_area_reference("v1")
    assert list(ref.columns[:5]) == SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS
    assert len(ref) > 0


def test_missing_version_fails_loud():
    with pytest.raises(FileNotFoundError, match="reference_version"):
        load_packaged_surface_area_reference("does_not_exist")


def test_interpolation_monotone_band():
    ref = pd.DataFrame(
        {
            "stage_hpf": [10.0, 20.0, 30.0],
            "p5": [100.0, 200.0, 300.0],
            "p50": [150.0, 250.0, 350.0],
            "p95": [200.0, 400.0, 600.0],
            "n": [50, 60, 70],
        }
    )
    validate_surface_area_reference(ref)
    p5, p95 = interpolate_reference_band(15.0, ref)
    assert p5 == pytest.approx(150.0)  # midway 100..200
    assert p95 == pytest.approx(300.0)  # midway 200..400


def test_reference_out_of_order_percentiles_fails():
    bad = pd.DataFrame(
        {"stage_hpf": [10.0], "p5": [300.0], "p50": [200.0], "p95": [100.0], "n": [5]}
    )
    with pytest.raises(ValueError, match="p5 <= p50 <= p95"):
        validate_surface_area_reference(bad)


def test_reference_unsorted_stage_fails():
    bad = pd.DataFrame(
        {"stage_hpf": [20.0, 10.0], "p5": [1, 1], "p50": [2, 2], "p95": [3, 3], "n": [1, 1]}
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_surface_area_reference(bad)
