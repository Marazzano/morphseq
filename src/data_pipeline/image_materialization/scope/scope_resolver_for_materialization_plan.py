"""Scope resolver — the VERB that turns a request into a scope's commitment.

This is the customs checkpoint between *what was requested* and *what a microscope will do*.
It is the ONE place that knows scope quirks (e.g. "YX1 acquires one tile per well/channel/time,
so mosaic XY composition is equivalent to identity"). Everything above it (the plan vocabulary)
is scope-agnostic; everything below it (the backend executors) is request-agnostic.

Two kinds of axis:
  - REQUIRED axes (channel_id, image_product_type, projection_method): a scope either produces
    the product or it does not. A mismatch is a hard ``UnsupportedMaterializationRequest`` — you
    cannot fabricate an image the scope never captured.
  - BYPASSABLE axes (xy_composition): the request is a wish the scope normalizes. YX1 resolves
    every request to ``identity`` (one tile); an explicit ``mosaic`` resolves to ``identity`` too,
    but logs a warning so a wrong-microscope config is noticed rather than silently honored.

The reasoning lives here as code + log sentences, deliberately NOT in an AXIS_POLICY dict —
the moment a rule needs a sentence ("one tile, so mosaic == identity"), it has outgrown a dict.

Import rules: imports the plan nouns from ``materialization_plan``. It MUST NOT import backends,
orchestration, tasks, or Snakemake rules — it returns a resolved plan; routing to a backend is
``materialize_well.py``'s job.
"""

from __future__ import annotations

import logging

from data_pipeline.image_materialization.materialization_plan import (
    ImageMaterializationPlan,
    ImageProductRequest,
    ResolvedImageProduct,
    ResolvedMaterializationPlan,
    UnsupportedMaterializationRequest,
    UnsupportedScopeError,
)

log = logging.getLogger(__name__)


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

    Step 6 is YX1-only live. Keyence resolution logic exists below (``_resolve_keyence``) as a
    reserved sketch, but is deliberately NOT routed here — the resolver must not claim Keyence is
    known while the backend cannot execute it. Keyence gets a route + backend together later.
    """
    if scope_name == "yx1":
        return _resolve_yx1(requested_plan)
    raise UnsupportedScopeError(
        f"No live materialization resolver for scope {scope_name!r}. "
        "Step 6 supports only 'yx1'. Keyence is planned but not wired."
    )


# ---------------------------------------------------------------------------
# YX1 — single-tile scope. xy_composition always resolves to identity.
# ---------------------------------------------------------------------------


def _resolve_yx1(plan: ImageMaterializationPlan) -> ResolvedMaterializationPlan:
    return ResolvedMaterializationPlan(
        products=tuple(_resolve_yx1_product(p) for p in plan.products)
    )


def _resolve_yx1_product(request: ImageProductRequest) -> ResolvedImageProduct:
    # Required axes — Step 6 supports exactly the one accepted YX1 product.
    if request.channel_id != "BF":
        raise UnsupportedMaterializationRequest(
            f"YX1 Step 6 supports channel_id='BF' only; got {request.channel_id!r}."
        )
    if request.image_product_type != "projection":
        raise UnsupportedMaterializationRequest(
            f"YX1 Step 6 supports image_product_type='projection' only; "
            f"got {request.image_product_type!r}."
        )
    if request.projection_method != "focus_stack":
        raise UnsupportedMaterializationRequest(
            f"YX1 Step 6 supports projection_method='focus_stack' only; "
            f"got {request.projection_method!r}."
        )

    xy = _resolve_yx1_xy_composition(request.xy_composition)
    return ResolvedImageProduct(
        channel_id=request.channel_id,
        image_product_type=request.image_product_type,
        projection_method=request.projection_method,
        xy_composition=xy,
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
# Keyence — RESERVED SKETCH, intentionally NOT routed by resolve_materialization_plan.
# Multi-tile scope; auto resolves to mosaic. This captures the intended rule so the next
# migration has a starting point, but it stays unreachable until a Keyence backend exists
# (no cardboard doorway opening into a wall). Wire route + backend together when Keyence is real.
# ---------------------------------------------------------------------------


def _resolve_keyence(plan: ImageMaterializationPlan) -> ResolvedMaterializationPlan:
    return ResolvedMaterializationPlan(
        products=tuple(_resolve_keyence_product(p) for p in plan.products)
    )


def _resolve_keyence_product(request: ImageProductRequest) -> ResolvedImageProduct:
    xy = _resolve_keyence_xy_composition(request.xy_composition)
    return ResolvedImageProduct(
        channel_id=request.channel_id,
        image_product_type=request.image_product_type,
        projection_method=request.projection_method,
        xy_composition=xy,
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
