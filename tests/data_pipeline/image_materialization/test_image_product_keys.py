"""Tests for image materialization product keys."""

import pytest

from data_pipeline.image_materialization.image_product_keys import (
    build_image_product_key,
    parse_image_product_key,
)


def test_projection_product_key_includes_method():
    assert (
        build_image_product_key(
            channel_id="BF",
            image_product_type="projection",
            projection_method="focus_stack",
        )
        == "BF__projection__focus_stack"
    )


def test_z_stack_product_key_omits_method():
    assert (
        build_image_product_key(channel_id="BF", image_product_type="z_stack")
        == "BF__z_stack"
    )


def test_parse_projection_product_key():
    assert parse_image_product_key("BF__projection__focus_stack") == (
        "BF",
        "projection",
        "focus_stack",
    )


def test_parse_z_stack_product_key():
    assert parse_image_product_key("BF__z_stack") == ("BF", "z_stack", None)


def test_projection_requires_method():
    with pytest.raises(ValueError, match="projection_method"):
        build_image_product_key(channel_id="BF", image_product_type="projection")


def test_z_stack_rejects_method():
    with pytest.raises(ValueError, match="must not set"):
        build_image_product_key(
            channel_id="BF",
            image_product_type="z_stack",
            projection_method="focus_stack",
        )


def test_parse_rejects_noncanonical_key():
    with pytest.raises(ValueError, match="not canonical|cannot parse"):
        parse_image_product_key("BF__z_stack__focus_stack")
