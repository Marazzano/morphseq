"""Tests for materialization_plan — vocabulary + plan shapes + config loading."""

import pytest

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ImageMaterializationPlan,
    ImageProductRequest,
    SUPPORTED_CHANNELS,
    UnsupportedMaterializationRequest,
    load_image_materialization_plan,
)
from data_pipeline.shared.channel_vocabulary import VALID_CHANNEL_NAMES


class TestVocabulary:
    def test_channels_imported_not_redefined(self):
        # The channel vocabulary IS shared/channel_vocabulary — one source of truth.
        assert SUPPORTED_CHANNELS == frozenset(VALID_CHANNEL_NAMES)


class TestRequestDefaults:
    def test_xy_composition_defaults_to_auto(self):
        req = ImageProductRequest(
            channel_id="BF", image_product_type="projection", projection_method="focus_stack"
        )
        assert req.xy_composition == "auto"


class TestLoadPlan:
    def test_none_config_yields_default_bf_product(self):
        plan = load_image_materialization_plan(None)
        assert isinstance(plan, ImageMaterializationPlan)
        assert len(plan.products) == 1
        p = plan.products[0]
        assert (p.channel_id, p.image_product_type, p.projection_method, p.xy_composition) == (
            "BF", "projection", "focus_stack", "auto",
        )

    def test_explicit_empty_products_rejected(self):
        # Absent key → default; explicit empty list → error (could mean "no products").
        with pytest.raises(UnsupportedMaterializationRequest, match="empty"):
            load_image_materialization_plan({"image_materialization": {"products": []}})

    def test_products_not_a_list_rejected(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="must be a list"):
            load_image_materialization_plan({"image_materialization": {"products": "BF"}})

    def test_product_not_a_dict_rejected(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="must be a dict"):
            load_image_materialization_plan({"image_materialization": {"products": ["BF"]}})

    def test_explicit_products_loaded(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "BF", "image_product_type": "projection",
             "projection_method": "focus_stack", "xy_composition": "mosaic"},
        ]}}
        plan = load_image_materialization_plan(cfg)
        assert plan.products[0].xy_composition == "mosaic"

    def test_unknown_channel_rejected(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "ZZZ", "image_product_type": "projection",
             "projection_method": "focus_stack"},
        ]}}
        with pytest.raises(UnsupportedMaterializationRequest, match="channel_id"):
            load_image_materialization_plan(cfg)

    def test_unknown_projection_method_rejected(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "BF", "image_product_type": "projection",
             "projection_method": "wavelet"},
        ]}}
        with pytest.raises(UnsupportedMaterializationRequest, match="projection_method"):
            load_image_materialization_plan(cfg)

    def test_unknown_xy_composition_rejected(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "BF", "image_product_type": "projection",
             "projection_method": "focus_stack", "xy_composition": "blend"},
        ]}}
        with pytest.raises(UnsupportedMaterializationRequest, match="xy_composition"):
            load_image_materialization_plan(cfg)


class TestProductShapeGrammar:
    def test_projection_requires_method(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "BF", "image_product_type": "projection"},  # no method
        ]}}
        with pytest.raises(UnsupportedMaterializationRequest, match="requires"):
            load_image_materialization_plan(cfg)

    def test_z_stack_forbids_method(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "BF", "image_product_type": "z_stack",
             "projection_method": "focus_stack"},  # method on a z_stack
        ]}}
        with pytest.raises(UnsupportedMaterializationRequest, match="must not set"):
            load_image_materialization_plan(cfg)

    def test_z_stack_without_method_ok(self):
        cfg = {"image_materialization": {"products": [
            {"channel_id": "BF", "image_product_type": "z_stack"},
        ]}}
        plan = load_image_materialization_plan(cfg)
        assert plan.products[0].projection_method is None
