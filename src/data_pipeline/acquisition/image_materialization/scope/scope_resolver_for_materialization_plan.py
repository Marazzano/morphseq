"""Scope resolver — the VERB that turns a request into a scope's commitment.

This is the customs checkpoint between *what was requested* and *what a microscope will do*.
It is the ONE place that knows scope quirks (e.g. "YX1 acquires one tile per well/channel/time,
so mosaic XY composition is equivalent to identity"). Everything above it (the plan vocabulary)
is scope-agnostic; everything below it (the backend executors) is request-agnostic.

Three kinds of axis:
  - STRUCTURAL axes (image_product_type, projection_method): does the code have a primitive for
    this product at all? ``projection`` vs ``z_stack`` shape rules, and methods that exist in the
    vocabulary but have no implementation (``mean``). A mismatch is a hard
    ``UnsupportedMaterializationRequest``.
  - BYPASSABLE axes (xy_composition): the request is a wish the scope normalizes. YX1 resolves
    every request to ``identity`` (one tile); an explicit ``mosaic`` resolves to ``identity`` too,
    but logs a warning so a wrong-microscope config is noticed rather than silently honored.
  - NOT RESOLVED HERE (channel_id): which channels an experiment actually contains is a
    per-experiment fact in the acquisition inventory, which this scope-generic resolver never sees.
    The absent-channel check belongs at the executor/resolve seam that holds the inventory.

There is deliberately NO channel x method policy. ``RFP x focus_stack`` resolves fine: the product
plan is where a method is chosen, so a resolver that second-guesses it on an unvalidated hunch about
optics would contradict the plan's own authority. Capability is checked; taste is not.

The reasoning lives here as code + log sentences, deliberately NOT in an AXIS_POLICY dict —
the moment a rule needs a sentence ("one tile, so mosaic == identity"), it has outgrown a dict.

Import rules: imports the plan nouns from ``materialization_plan``. It MUST NOT import backends,
orchestration, tasks, or Snakemake rules — it returns a resolved plan; routing to a backend is
``materialize_well.py``'s job.
"""

from __future__ import annotations

import logging

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ImageMaterializationPlan,
    ImageProductRequest,
    ResolvedImageProduct,
    ResolvedMaterializationPlan,
    UnsupportedMaterializationRequest,
    UnsupportedScopeError,
)

log = logging.getLogger(__name__)


# Projection methods with a real primitive behind them. ``SUPPORTED_PROJECTION_METHODS`` in
# materialization_plan.py is the GRAMMAR (what the vocabulary can express); this is CAPABILITY (what
# the code can actually execute). "mean" is grammatical but unimplemented, so it is rejected here
# rather than failing deeper with a confusing message.
_IMPLEMENTED_PROJECTION_METHODS: frozenset[str] = frozenset({"focus_stack", "max"})


def _assert_projection_method_implemented(
    request: ImageProductRequest, *, scope_label: str
) -> None:
    """Reject a projection method that has no primitive behind it.

    This gates on what the CODE can do, not on what suits a channel. There is deliberately NO
    channel x method restriction: an earlier design made focus_stack brightfield-only on the theory
    that LoG focus scoring is noise-driven on sparse fluorescence, but that is an unvalidated
    inference, and the product plan is the declared place to choose a method. A resolver that
    overrides the plan on a guess is incoherent with the plan being the choice. If a fluorescence
    focus_stack looks wrong, that is evidence — and only then does a rule get written, with a reason.
    """
    if request.image_product_type != "projection":
        return
    if request.projection_method not in _IMPLEMENTED_PROJECTION_METHODS:
        raise UnsupportedMaterializationRequest(
            f"[{scope_label}] projection_method={request.projection_method!r} is in the plan "
            f"vocabulary but has no implementation. Implemented: "
            f"{sorted(_IMPLEMENTED_PROJECTION_METHODS)}. To add one, write the primitive and wire it "
            "in the scope executor, then extend _IMPLEMENTED_PROJECTION_METHODS."
        )


# ---------------------------------------------------------------------------
# Scope router — an explicit if-chain is correct here (it IS the scope dispatch).
# ---------------------------------------------------------------------------


def resolve_materialization_plan(
    *,
    scope_name: str,
    requested_plan: ImageMaterializationPlan,
) -> ResolvedMaterializationPlan:
    """Normalize a requested plan for one microscope → a concrete, executable plan.

    Routes to the per-scope resolver, which maps each ``ImageProductRequest`` to a
    ``ResolvedImageProduct`` (required axes strict, xy_composition normalized).

    Both YX1 and Keyence are live. Keyence routes to ``_resolve_keyence`` (auto→mosaic, identity
    raises). The backend executor (``scope/keyence/materialize_well_keyence.py``) is wired in
    ``run_materialize_well.py``.
    """
    if scope_name == "yx1":
        return _resolve_yx1(requested_plan)
    if scope_name == "keyence":
        return _resolve_keyence(requested_plan)
    raise UnsupportedScopeError(
        f"No live materialization resolver for scope {scope_name!r}. "
        "Supported scopes: 'yx1', 'keyence'."
    )


# ---------------------------------------------------------------------------
# YX1 — single-tile scope. xy_composition always resolves to identity.
# ---------------------------------------------------------------------------


def _resolve_yx1(plan: ImageMaterializationPlan) -> ResolvedMaterializationPlan:
    return ResolvedMaterializationPlan(
        products=tuple(_resolve_yx1_product(p) for p in plan.products)
    )


def _resolve_yx1_product(request: ImageProductRequest) -> ResolvedImageProduct:
    if request.image_product_type not in ("projection", "z_stack"):
        raise UnsupportedMaterializationRequest(
            f"YX1 supports image_product_type='projection' or 'z_stack' only; "
            f"got {request.image_product_type!r}."
        )
    if request.image_product_type == "z_stack" and request.projection_method is not None:
        raise UnsupportedMaterializationRequest(
            "YX1 z_stack products must not set projection_method; z_stack preserves Z planes."
        )
    # Channel is NOT gated here: which channels exist is a per-EXPERIMENT fact recorded in the
    # acquisition inventory, and this resolver has no inventory. The absent-channel check therefore
    # lives at the executor/resolve seam where the inventory is in hand.
    _assert_projection_method_implemented(request, scope_label="YX1")

    xy = _resolve_yx1_xy_composition(request.xy_composition)
    return ResolvedImageProduct(
        channel_id=request.channel_id,
        image_product_type=request.image_product_type,
        projection_method=request.projection_method,
        xy_composition=xy,
        write_index_map=request.write_index_map,
    )


def _resolve_yx1_xy_composition(requested: str) -> str:
    """YX1 has one acquired tile per well/channel/time → XY composition is always identity."""
    if requested in ("auto", "identity"):
        return "identity"
    if requested == "mosaic":
        log.warning(
            "xy_composition='mosaic' requested for yx1, but YX1 acquires one tile per "
            "well/channel/time; mosaic composition is equivalent to identity. "
            "Resolved to 'identity'."
        )
        return "identity"
    raise UnsupportedMaterializationRequest(
        f"Unknown xy_composition {requested!r} for yx1."
    )


# ---------------------------------------------------------------------------
# Keyence — multi-tile scope; auto resolves to mosaic. Backend: materialize_well_keyence.py.
# ---------------------------------------------------------------------------


def _resolve_keyence(plan: ImageMaterializationPlan) -> ResolvedMaterializationPlan:
    return ResolvedMaterializationPlan(
        products=tuple(_resolve_keyence_product(p) for p in plan.products)
    )


def _resolve_keyence_product(request: ImageProductRequest) -> ResolvedImageProduct:
    # Moved UP from materialize_well_keyence.py so both scopes reject an unimplemented method at the
    # same layer, with the same message, before any disk work.
    _assert_projection_method_implemented(request, scope_label="Keyence")
    xy = _resolve_keyence_xy_composition(request.xy_composition)
    return ResolvedImageProduct(
        channel_id=request.channel_id,
        image_product_type=request.image_product_type,
        projection_method=request.projection_method,
        xy_composition=xy,
        write_index_map=request.write_index_map,
    )


def _resolve_keyence_xy_composition(requested: str) -> str:
    """Keyence acquisitions may contain multiple XY tiles → default composition is mosaic."""
    if requested in ("auto", "mosaic"):
        return "mosaic"
    if requested == "identity":
        raise UnsupportedMaterializationRequest(
            "Keyence materialization requires xy_composition='mosaic' (multi-tile); "
            "'identity' is not supported until a single-tile Keyence mode exists."
        )
    raise UnsupportedMaterializationRequest(
        f"Unknown xy_composition {requested!r} for keyence."
    )
