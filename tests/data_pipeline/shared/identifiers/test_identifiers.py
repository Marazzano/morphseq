import pytest

from data_pipeline.shared.identifiers import (
    build_mask_id,
    build_no_mask_id,
    build_track_id,
    parse_mask_id,
    parse_track_id,
)


def test_mask_id_round_trip() -> None:
    image_id = "20250912_B01_BF_t0007"

    assert parse_mask_id(build_mask_id(image_id, 1)) == (image_id, 1, False)


def test_no_mask_id_round_trip() -> None:
    image_id = "20250912_B01_BF_t0007"

    assert parse_mask_id(build_no_mask_id(image_id)) == (image_id, None, True)


def test_zero_based_track_id_round_trip() -> None:
    assert build_track_id("WELL", 0) == "WELL_track0000"
    assert parse_track_id("WELL_track0000") == ("WELL", 0)


def test_track_id_allows_underscores_in_well_id() -> None:
    assert build_track_id("20250912_B01", 0) == "20250912_B01_track0000"
    assert parse_track_id("20250912_B01_track0000") == ("20250912_B01", 0)


def test_parse_mask_id_allows_underscores_in_image_id() -> None:
    assert parse_mask_id("20250912_B01_BF_t0007_m0001") == (
        "20250912_B01_BF_t0007",
        1,
        False,
    )


def test_parse_no_mask_id_allows_underscores_in_image_id() -> None:
    assert parse_mask_id("20250912_B01_BF_t0007_mask_none") == (
        "20250912_B01_BF_t0007",
        None,
        True,
    )


def test_build_track_id_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="track_index must be zero or greater"):
        build_track_id("WELL", -1)


def test_build_mask_id_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="local_mask_index must be zero or greater"):
        build_mask_id("20250912_B01_BF_t0007", -1)


@pytest.mark.parametrize(
    "mask_id",
    [
        "",
        "m0001",
        "image_mask",
        "image_m1",
        "image_mabc",
        "image_m",
    ],
)
def test_parse_mask_id_rejects_malformed_ids(mask_id: str) -> None:
    with pytest.raises(ValueError):
        parse_mask_id(mask_id)


@pytest.mark.parametrize(
    "track_id",
    [
        "",
        "track0000",
        "WELL_track1",
        "WELL_track",
        "WELL_trackabc",
        "WELL-trace0000",
    ],
)
def test_parse_track_id_rejects_malformed_ids(track_id: str) -> None:
    with pytest.raises(ValueError):
        parse_track_id(track_id)


def test_parse_track_id_allows_zero_index() -> None:
    assert parse_track_id("WELL_track0000") == ("WELL", 0)
