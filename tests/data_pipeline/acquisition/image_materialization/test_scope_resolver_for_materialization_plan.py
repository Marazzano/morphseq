"""Tests for the scope resolver — request → resolved, per-scope quirks."""

import pytest

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ImageMaterializationPlan,
    ImageProductRequest,
    UnsupportedMaterializationRequest,
    UnsupportedScopeError,
)
from data_pipeline.acquisition.image_materialization.scope.scope_resolver_for_materialization_plan import (
    resolve_materialization_plan,
    _resolve_keyence_xy_composition,
)


def _plan(**overrides):
    base = dict(
        channel_id="BF",
        image_product_type="projection",
        projection_method="focus_stack",
        xy_composition="auto",
    )
    base.update(overrides)
    return ImageMaterializationPlan(products=(ImageProductRequest(**base),))


class TestScopeRouting:
    def test_unknown_scope_raises(self):
        with pytest.raises(UnsupportedScopeError, match="No live materialization resolver"):
            resolve_materialization_plan(scope_name="zeiss", requested_plan=_plan())

    def test_keyence_routes_and_resolves_to_mosaic(self):
        out = resolve_materialization_plan(scope_name="keyence", requested_plan=_plan(xy_composition="auto"))
        assert out.products[0].xy_composition == "mosaic"

    def test_keyence_identity_raises(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="mosaic"):
            resolve_materialization_plan(scope_name="keyence", requested_plan=_plan(xy_composition="identity"))

    def test_keyence_z_stack_resolves_to_mosaic(self):
        out = resolve_materialization_plan(
            scope_name="keyence",
            requested_plan=_plan(image_product_type="z_stack", projection_method=None),
        )
        product = out.products[0]
        assert product.image_product_type == "z_stack"
        assert product.projection_method is None
        assert product.xy_composition == "mosaic"


class TestYX1XYComposition:
    def test_auto_resolves_to_identity(self):
        out = resolve_materialization_plan(scope_name="yx1", requested_plan=_plan(xy_composition="auto"))
        assert out.products[0].xy_composition == "identity"

    def test_identity_stays_identity(self):
        out = resolve_materialization_plan(scope_name="yx1", requested_plan=_plan(xy_composition="identity"))
        assert out.products[0].xy_composition == "identity"

    def test_mosaic_resolves_to_identity_with_warning(self, caplog):
        with caplog.at_level("WARNING"):
            out = resolve_materialization_plan(
                scope_name="yx1", requested_plan=_plan(xy_composition="mosaic")
            )
        assert out.products[0].xy_composition == "identity"
        assert any("mosaic" in r.message and "identity" in r.message for r in caplog.records)


class TestYX1RequiredAxes:
    def test_non_bf_channel_rejected(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="channel_id='BF'"):
            resolve_materialization_plan(scope_name="yx1", requested_plan=_plan(channel_id="GFP"))

    def test_z_stack_resolves_for_bf_with_no_projection_method(self):
        out = resolve_materialization_plan(
            scope_name="yx1",
            requested_plan=_plan(image_product_type="z_stack", projection_method=None),
        )
        product = out.products[0]
        assert product.image_product_type == "z_stack"
        assert product.projection_method is None
        assert product.xy_composition == "identity"

    def test_non_focus_stack_rejected(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="projection_method='focus_stack'"):
            resolve_materialization_plan(
                scope_name="yx1", requested_plan=_plan(projection_method="max")
            )

    def test_non_bf_z_stack_rejected(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="channel_id='BF'"):
            resolve_materialization_plan(
                scope_name="yx1",
                requested_plan=_plan(channel_id="GFP", image_product_type="z_stack", projection_method=None),
            )


class TestKeyenceReservedSketch:
    # Keyence is NOT routed by the public resolver in Step 6 (see TestScopeRouting). These cover
    # the reserved private helper so the intended rule is captured for the next migration.
    def test_auto_resolves_to_mosaic(self):
        assert _resolve_keyence_xy_composition("auto") == "mosaic"

    def test_mosaic_stays_mosaic(self):
        assert _resolve_keyence_xy_composition("mosaic") == "mosaic"

    def test_identity_rejected(self):
        with pytest.raises(UnsupportedMaterializationRequest, match="mosaic"):
            _resolve_keyence_xy_composition("identity")
