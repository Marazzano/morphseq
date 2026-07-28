"""Tests for the reusable string validator ``validate_physical_embryo_id``."""

from __future__ import annotations

import pytest

from data_pipeline.shared.identifiers import validate_physical_embryo_id


def test_valid_physical_embryo_id_returns_unchanged():
    assert validate_physical_embryo_id("20250912_B01_e01") == "20250912_B01_e01"


def test_valid_with_matching_well_id():
    assert (
        validate_physical_embryo_id("20250912_B01_e02", well_id="20250912_B01")
        == "20250912_B01_e02"
    )


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_physical_embryo_id("")


def test_zero_index_rejected():
    # _e00 is not a valid one-based identity (parser enforces >= 1).
    with pytest.raises(ValueError, match="one-based|< 1|e00"):
        validate_physical_embryo_id("20250912_B01_e00")


def test_malformed_shape_rejected():
    with pytest.raises(ValueError, match="parse_physical_embryo_id|cannot parse"):
        validate_physical_embryo_id("not-an-embryo-id")


def test_leaked_local_well_id_rejected():
    # The embedded well must be a GLOBAL well_id, not a bare local label.
    with pytest.raises(ValueError, match="local|global"):
        validate_physical_embryo_id("B01_e01")


def test_well_id_mismatch_rejected():
    with pytest.raises(ValueError, match="encodes well_id|must match"):
        validate_physical_embryo_id("20250912_B01_e01", well_id="20250912_C04")
