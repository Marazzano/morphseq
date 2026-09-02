"""Tests for the scope resolver — request → resolved, per-scope quirks."""

import pytest

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    IMPLEMENTED_PROJECTION_METHODS,
    ImageMaterializationPlan,
    ImageProductRequest,
    SUPPORTED_CHANNELS,
    SUPPORTED_PROJECTION_METHODS,
    UnsupportedMaterializationRequest,
    UnsupportedScopeError,
)
from data_pipeline.acquisition.image_materialization.scope.scope_resolver_for_materialization_plan import (
    resolve_materialization_plan,
    _resolve_keyence_xy_composition,
)
from data_pipeline.shared.channel_vocabulary import BRIGHTFIELD_CHANNELS

# Derived from the contracts, never re-typed here. Both the vocabulary (SUPPORTED_*) and the
# capability set (IMPLEMENTED_PROJECTION_METHODS) are owned by materialization_plan; the resolver
# imports the latter rather than restating it, and the YX1 executor asserts its primitive registry
# covers it exactly. Adding a channel or implementing a method extends coverage automatically.
_FLUORESCENCE_CHANNELS = sorted(SUPPORTED_CHANNELS - set(BRIGHTFIELD_CHANNELS))
_UNIMPLEMENTED_METHODS = sorted(SUPPORTED_PROJECTION_METHODS - IMPLEMENTED_PROJECTION_METHODS)


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
    """The resolver gates on CAPABILITY (is there a primitive?) and product SHAPE only.

    The former BF-only / focus_stack-only gates were removed on 2026-07-29. Channel availability is a
    per-experiment fact this scope-generic resolver cannot see (it is checked at the executor seam
    that holds the acquisition inventory), and channel x method taste is the product plan's call. The
    three tests asserting the old rule were deleted rather than adapted — the rule itself is gone.
    """

    def test_z_stack_resolves_for_bf_with_no_projection_method(self):
        out = resolve_materialization_plan(
            scope_name="yx1",
            requested_plan=_plan(image_product_type="z_stack", projection_method=None),
        )
        product = out.products[0]
        assert product.image_product_type == "z_stack"
        assert product.projection_method is None
        assert product.xy_composition == "identity"

    @pytest.mark.parametrize("channel_id", sorted(SUPPORTED_CHANNELS))
    @pytest.mark.parametrize("projection_method", sorted(IMPLEMENTED_PROJECTION_METHODS))
    def test_every_channel_resolves_with_every_implemented_method(
        self, channel_id, projection_method
    ):
        """No channel x method policy: the full cross product of the two contracts must resolve."""
        out = resolve_materialization_plan(
            scope_name="yx1",
            requested_plan=_plan(channel_id=channel_id, projection_method=projection_method),
        )
        product = out.products[0]
        assert product.channel_id == channel_id
        assert product.projection_method == projection_method
        assert product.xy_composition == "identity"

    @pytest.mark.parametrize("channel_id", _FLUORESCENCE_CHANNELS)
    def test_fluorescence_z_stack_resolves(self, channel_id):
        out = resolve_materialization_plan(
            scope_name="yx1",
            requested_plan=_plan(
                channel_id=channel_id, image_product_type="z_stack", projection_method=None
            ),
        )
        assert out.products[0].channel_id == channel_id
        assert out.products[0].image_product_type == "z_stack"

    @pytest.mark.skipif(
        not _UNIMPLEMENTED_METHODS,
        reason="every method in the vocabulary now has an implementation",
    )
    @pytest.mark.parametrize("scope_name", ["yx1", "keyence"])
    def test_unimplemented_method_rejected_on_every_scope(self, scope_name):
        """A method in the vocabulary with no primitive must be refused at the resolver, per scope."""
        for method in _UNIMPLEMENTED_METHODS:
            with pytest.raises(UnsupportedMaterializationRequest, match="no implementation"):
                resolve_materialization_plan(
                    scope_name=scope_name, requested_plan=_plan(projection_method=method)
                )

    def test_write_index_map_passes_through_unchanged(self):
        out = resolve_materialization_plan(
            scope_name="yx1", requested_plan=_plan(write_index_map=True)
        )
        assert out.products[0].write_index_map is True


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
