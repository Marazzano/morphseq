import pytest

from data_pipeline.snip_processing.snip_frame_shape import (
    DEFAULT_SNIP_FRAME_SHAPE,
    resolve_snip_frame_shape,
)


def test_reads_top_level_single_source_of_truth() -> None:
    assert resolve_snip_frame_shape({"snip_frame_shape": [576, 256]}) == (576, 256)


def test_falls_back_to_legacy_output_shape() -> None:
    cfg = {"snip_processing": {"output_shape": [600, 300]}}
    assert resolve_snip_frame_shape(cfg) == (600, 300)


def test_top_level_wins_over_legacy() -> None:
    cfg = {"snip_frame_shape": [576, 256], "snip_processing": {"output_shape": [600, 300]}}
    assert resolve_snip_frame_shape(cfg) == (576, 256)


def test_default_when_absent() -> None:
    assert resolve_snip_frame_shape({}) == DEFAULT_SNIP_FRAME_SHAPE
    assert resolve_snip_frame_shape(None) == DEFAULT_SNIP_FRAME_SHAPE


def test_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        resolve_snip_frame_shape({"snip_frame_shape": [576, 256, 1]})


def test_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        resolve_snip_frame_shape({"snip_frame_shape": [0, 256]})
