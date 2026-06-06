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

    assert bwi("20240418", "A01") == "20240418_A01"


# --- GLOBAL well_id constructor grammar ----------------------------------------


def test_build_well_id_is_global_and_sanitizes_experiment():
    assert build_well_id("20240418", "A01") == "20240418_A01"
    assert build_well_id("  20240418 ", "  A01 ") == "20240418_A01"
    # experiment_id is sanitized ONCE here, so well_id is born clean
    assert build_well_id("2024 0418", "A01") == "2024_0418_A01"


def test_build_image_id_is_well_id_first():
    well_id = build_well_id("20240418", "A01")
    assert build_image_id(well_id, "BF", 3) == "20240418_A01_BF_t0003"


def test_build_embryo_and_snip_id_well_id_first():
    well_id = build_well_id("20240418", "A01")
    eid = build_embryo_id(well_id, 7)
    assert eid == "20240418_A01_e07"
    assert build_snip_id(eid, 3) == "20240418_A01_e07_t0003"


def test_full_grammar_composes():
    # experiment_id + well_index -> well_id -> image/embryo/snip, all global
    well_id = build_well_id("20240418", "A01")
    assert build_image_id(well_id, "BF", 3) == "20240418_A01_BF_t0003"
    assert build_snip_id(build_embryo_id(well_id, 7), 3) == "20240418_A01_e07_t0003"


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


# --- split_well_id: inverse of build_well_id -----------------------------------


def test_split_well_id_roundtrips_build_well_id():
    well_id = build_well_id("20240418", "A01")
    assert split_well_id(well_id) == ("20240418", "A01")


def test_split_well_id_rejects_bare_local_label():
    with pytest.raises(ValueError):
        split_well_id("A01")


# --- validate_well_id: guard rail against leaked local labels -------------------


def test_validate_well_id_accepts_global():
    assert validate_well_id("20240418_A01") == "20240418_A01"


def test_validate_well_id_rejects_bare_local_label():
    with pytest.raises(ValueError):
        validate_well_id("A01")
    with pytest.raises(ValueError):
        validate_well_id("B12")


def test_validate_well_id_rejects_empty():
    with pytest.raises(ValueError):
        validate_well_id("")
