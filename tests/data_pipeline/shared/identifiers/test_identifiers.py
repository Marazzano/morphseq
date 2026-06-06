"""Scope-1 contract tests for shared/identifiers/.

Locks in (a) the re-export shim — every public name still importable from the old
flat path — and (b) the CURRENT (local-well_id) constructor grammar, unchanged by
the package split. The Scope-2 scaffolds must raise until the global-well_id flip.

Run: PYTHONPATH=src pytest src/data_pipeline/shared/identifiers/tests/
"""

import pytest

from data_pipeline.shared.identifiers import (
    build_well_id,
    build_image_id,
    build_embryo_id,
    build_snip_id,
    sanitize_experiment_id,
    normalize_embryo_local_track_id,
    split_well_id,
    validate_well_id,
)


def test_reexport_from_legacy_flat_path():
    # The 10 call sites import these names from data_pipeline.shared.identifiers;
    # the package __init__ must keep them all resolvable.
    import data_pipeline.shared.identifiers as ids

    for name in (
        "build_well_id",
        "build_image_id",
        "build_embryo_id",
        "build_snip_id",
        "sanitize_experiment_id",
        "normalize_embryo_local_track_id",
        "split_well_id",
        "validate_well_id",
    ):
        assert hasattr(ids, name), name


def test_reexport_from_shared_package():
    from data_pipeline.shared import build_well_id as bwi

    assert bwi("A01") == "A01"


# --- CURRENT (local-well_id) constructor grammar -------------------------------


def test_build_well_id_is_local_label():
    assert build_well_id("A01") == "A01"
    assert build_well_id("  A01 ") == "A01"


def test_build_image_id_sanitizes_experiment():
    assert build_image_id("2024 0418", "A01", "BF", 3) == "2024_0418_A01_BF_t0003"
    assert build_image_id("20240418", "A01", "BF", 3) == "20240418_A01_BF_t0003"


def test_build_embryo_and_snip_id():
    eid = build_embryo_id("20240418", "A01", 7)
    assert eid == "20240418_A01_e07"
    assert build_snip_id(eid, 3) == "20240418_A01_e07_t0003"


def test_sanitize_experiment_id():
    assert sanitize_experiment_id("  my exp!! ") == "my_exp"


def test_normalize_embryo_local_track_id():
    assert normalize_embryo_local_track_id("embryo_5") == 5
    assert normalize_embryo_local_track_id(2.0) == 2
    assert normalize_embryo_local_track_id(3) == 3
    with pytest.raises(ValueError):
        normalize_embryo_local_track_id(True)
    with pytest.raises(ValueError):
        normalize_embryo_local_track_id("no-digits")


# --- Scope-2 scaffolds must not be usable yet ----------------------------------


def test_split_well_id_is_scope2_stub():
    with pytest.raises(NotImplementedError):
        split_well_id("A01")


def test_validate_well_id_is_scope2_stub():
    with pytest.raises(NotImplementedError):
        validate_well_id("A01")
