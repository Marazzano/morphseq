"""Tests for shared/identifiers/ — constructors, parsers, and validators.

Philosophy: tests use public constructors/parsers only. No duplicating regexes in tests.
No reaching into real data files. Identifier strings are opaque; tests exercise the
public API, not the grammar directly.

Run: PYTHONPATH=src pytest tests/data_pipeline/shared/identifiers/
"""

import pytest

from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_mask_id,
    build_no_mask_id,
    build_physical_embryo_id,
    build_snip_id,
    build_track_id,
    build_well_id,
    normalize_embryo_local_track_id,
    parse_embryo_id,
    parse_embryo_local_track_id,
    parse_image_id,
    parse_mask_id,
    parse_physical_embryo_id,
    parse_snip_id,
    parse_track_id,
    sanitize_experiment_id,
    split_well_id,
    track_index_to_embryo_index,
    validate_well_id,
)

# ── Small scoped constants (not real data files) ──────────────────────────────

EXP = "20250912"
WELL = f"{EXP}_B01"
CHANNEL = "BF"
T = 7

PHYSICAL = f"{WELL}_e01"
IMAGE = f"{WELL}_{CHANNEL}_t{T:04d}"
EMBRYO = f"{PHYSICAL}_{CHANNEL}"
SNIP = f"{EMBRYO}_t{T:04d}"


# ── re-export shim ────────────────────────────────────────────────────────────


def test_reexport_from_legacy_flat_path():
    import data_pipeline.shared.identifiers as ids

    for name in (
        "build_well_id",
        "build_image_id",
        "build_mask_id",
        "build_no_mask_id",
        "build_track_id",
        "build_physical_embryo_id",
        "build_embryo_id",
        "build_snip_id",
        "sanitize_experiment_id",
        "parse_embryo_local_track_id",
        "normalize_embryo_local_track_id",
        "parse_image_id",
        "parse_mask_id",
        "parse_track_id",
        "parse_physical_embryo_id",
        "parse_embryo_id",
        "parse_snip_id",
        "split_well_id",
        "track_index_to_embryo_index",
        "validate_well_id",
    ):
        assert hasattr(ids, name), name


def test_reexport_from_shared_package():
    from data_pipeline.shared import build_well_id as bwi

    assert bwi("20240418", "A01") == "20240418_A01"


# ── build_well_id ─────────────────────────────────────────────────────────────


def test_build_well_id_is_global_and_sanitizes_experiment():
    assert build_well_id("20240418", "A01") == "20240418_A01"
    assert build_well_id("  20240418 ", "  A01 ") == "20240418_A01"
    assert build_well_id("2024 0418", "A01") == "2024_0418_A01"


# ── build_image_id ────────────────────────────────────────────────────────────


def test_build_image_id_is_well_id_first():
    well_id = build_well_id("20240418", "A01")
    assert build_image_id(well_id, "BF", 3) == "20240418_A01_BF_t0003"


# ── sanitize_experiment_id ────────────────────────────────────────────────────


def test_sanitize_experiment_id():
    assert sanitize_experiment_id("  my exp!! ") == "my_exp"


# ── parse_embryo_local_track_id (renamed from normalize_embryo_local_track_id) ─


def test_parse_embryo_local_track_id():
    assert parse_embryo_local_track_id("embryo_5") == 5
    assert parse_embryo_local_track_id(2.0) == 2
    assert parse_embryo_local_track_id(3) == 3
    with pytest.raises(ValueError):
        parse_embryo_local_track_id(True)
    with pytest.raises(ValueError):
        parse_embryo_local_track_id("no-digits")


def test_normalize_embryo_local_track_id_is_alias():
    # Old name still works — deprecated alias.
    assert normalize_embryo_local_track_id("embryo_0") == 0
    assert normalize_embryo_local_track_id(3) == 3


# ── track_index_to_embryo_index ───────────────────────────────────────────────


def test_track_index_to_embryo_index_zero_based_to_one_based():
    assert track_index_to_embryo_index(0) == 1
    assert track_index_to_embryo_index(1) == 2
    assert track_index_to_embryo_index(11) == 12


def test_track_index_to_embryo_index_rejects_negative():
    with pytest.raises(ValueError):
        track_index_to_embryo_index(-1)


# ── build_physical_embryo_id ──────────────────────────────────────────────────


def test_build_physical_embryo_id():
    assert build_physical_embryo_id(WELL, 1) == PHYSICAL
    assert build_physical_embryo_id(WELL, 12) == f"{WELL}_e12"  # double-digit


def test_build_physical_embryo_id_rejects_zero_and_negative():
    with pytest.raises(ValueError):
        build_physical_embryo_id(WELL, 0)
    with pytest.raises(ValueError):
        build_physical_embryo_id(WELL, -1)


# ── parse_physical_embryo_id ──────────────────────────────────────────────────


def test_parse_physical_embryo_id():
    assert parse_physical_embryo_id(PHYSICAL) == (WELL, 1)
    assert parse_physical_embryo_id(f"{WELL}_e12") == (WELL, 12)


def test_parse_physical_embryo_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_physical_embryo_id(WELL)  # no _e suffix


def test_parse_physical_embryo_id_rejects_zero_index():
    with pytest.raises(ValueError):
        parse_physical_embryo_id(f"{WELL}_e00")  # zero is not one-based


# ── parse_image_id ────────────────────────────────────────────────────────────


def test_parse_image_id():
    assert parse_image_id(IMAGE) == (WELL, CHANNEL, T)


def test_parse_image_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_image_id("not_an_image_id")


def test_parse_image_id_rejects_non_canonical_channel():
    with pytest.raises(ValueError):
        parse_image_id(f"{WELL}_Cy5_t0007")  # Cy5 not in VALID_CHANNEL_NAMES


# ── mask_id and track_id ──────────────────────────────────────────────────────


def test_mask_id_round_trip():
    assert parse_mask_id(build_mask_id(IMAGE, 1)) == (IMAGE, 1, False)


def test_no_mask_id_round_trip():
    assert parse_mask_id(build_no_mask_id(IMAGE)) == (IMAGE, None, True)


def test_parse_mask_id_allows_underscores_in_image_id():
    assert parse_mask_id("20250912_B01_BF_t0007_m0001") == (
        "20250912_B01_BF_t0007",
        1,
        False,
    )
    assert parse_mask_id("20250912_B01_BF_t0007_mask_none") == (
        "20250912_B01_BF_t0007",
        None,
        True,
    )


def test_track_id_round_trip_and_zero_based_compatibility():
    assert build_track_id("WELL", 0) == "WELL_track0000"
    assert parse_track_id("WELL_track0000") == ("WELL", 0)
    assert build_track_id("20250912_B01", 0) == "20250912_B01_track0000"
    assert parse_track_id("20250912_B01_track0000") == ("20250912_B01", 0)


def test_build_mask_id_rejects_negative_index():
    with pytest.raises(ValueError, match="local_mask_index must be zero or greater"):
        build_mask_id(IMAGE, -1)


def test_build_track_id_rejects_negative_index():
    with pytest.raises(ValueError, match="track_index must be zero or greater"):
        build_track_id(WELL, -1)


@pytest.mark.parametrize(
    "mask_id",
    ["", "m0001", "image_mask", "image_m1", "image_mabc", "image_m"],
)
def test_parse_mask_id_rejects_malformed(mask_id):
    with pytest.raises(ValueError):
        parse_mask_id(mask_id)


@pytest.mark.parametrize(
    "track_id",
    ["", "track0000", "WELL_track1", "WELL_track", "WELL_trackabc", "WELL-trace0000"],
)
def test_parse_track_id_rejects_malformed(track_id):
    with pytest.raises(ValueError):
        parse_track_id(track_id)


# ── build_embryo_id ───────────────────────────────────────────────────────────


def test_build_embryo_id():
    assert build_embryo_id(PHYSICAL, IMAGE) == EMBRYO


def test_build_embryo_id_roundtrips_with_parse():
    physical_embryo_id, channel_id = parse_embryo_id(EMBRYO)
    assert physical_embryo_id == PHYSICAL
    assert channel_id == CHANNEL


def test_build_embryo_id_rejects_well_mismatch():
    other_image = f"{EXP}_C01_{CHANNEL}_t{T:04d}"
    with pytest.raises(ValueError, match="well_id mismatch"):
        build_embryo_id(PHYSICAL, other_image)


# ── parse_embryo_id ───────────────────────────────────────────────────────────


def test_parse_embryo_id():
    assert parse_embryo_id(EMBRYO) == (PHYSICAL, CHANNEL)


def test_parse_embryo_id_rejects_non_canonical_channel():
    with pytest.raises(ValueError):
        parse_embryo_id(f"{PHYSICAL}_Cy5")


# ── build_snip_id ─────────────────────────────────────────────────────────────


def test_build_snip_id():
    assert build_snip_id(EMBRYO, IMAGE) == SNIP


def test_build_snip_id_rejects_channel_mismatch():
    gfp_image = f"{WELL}_GFP_t{T:04d}"
    with pytest.raises(ValueError, match="channel_id mismatch"):
        build_snip_id(EMBRYO, gfp_image)  # EMBRYO is BF; image is GFP


def test_build_snip_id_rejects_well_mismatch():
    other_image = f"{EXP}_C01_{CHANNEL}_t{T:04d}"
    with pytest.raises(ValueError, match="well_id mismatch"):
        build_snip_id(EMBRYO, other_image)


# ── parse_snip_id ─────────────────────────────────────────────────────────────


def test_parse_snip_id():
    assert parse_snip_id(SNIP) == (EMBRYO, T)


def test_parse_snip_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_snip_id("no_time_suffix")


# ── full grammar composition ───────────────────────────────────────────────────


def test_full_grammar_composes():
    well_id = build_well_id(EXP, "B01")
    image_id = build_image_id(well_id, CHANNEL, T)
    physical = build_physical_embryo_id(well_id, 1)
    embryo = build_embryo_id(physical, image_id)
    snip = build_snip_id(embryo, image_id)

    assert well_id == WELL
    assert image_id == IMAGE
    assert physical == PHYSICAL
    assert embryo == EMBRYO
    assert snip == SNIP


def test_track_index_chain_composes():
    # Typical snip_processing chain: tracker gives "embryo_0" -> physical_embryo_id
    raw_track_index = parse_embryo_local_track_id("embryo_0")
    local_embryo_index = track_index_to_embryo_index(raw_track_index)
    physical = build_physical_embryo_id(WELL, local_embryo_index)
    assert physical == f"{WELL}_e01"


# ── split_well_id ─────────────────────────────────────────────────────────────


def test_split_well_id_roundtrips_build_well_id():
    well_id = build_well_id("20240418", "A01")
    assert split_well_id(well_id) == ("20240418", "A01")


def test_split_well_id_rejects_bare_local_label():
    with pytest.raises(ValueError):
        split_well_id("A01")


# ── validate_well_id ──────────────────────────────────────────────────────────


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
