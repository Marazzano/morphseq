"""Materialization-plan vocabulary and dataclass shapes (no scope behavior).

Defines request plans and resolved plans, but does not resolve scope behavior. Request
vocabulary differs from resolved vocabulary:

  - ``ImageMaterializationPlan``: config/user intent; may contain ``xy_composition='auto'``.
  - ``ResolvedMaterializationPlan``: scope commitment; only executable values.

Scope-specific narrowing lives in ``scope_resolver_for_materialization_plan.py``. Backends
accept only a ``ResolvedMaterializationPlan``.

This module enforces GLOBAL product grammar (channel/type/method vocab + product-shape rules)
but no scope-specific rules. Channels are imported from ``shared/channel_vocabulary`` (the
one canonical list — never redefined here). It MUST NOT import scope backends, orchestration,
tasks, or Snakemake rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_pipeline.shared.channel_vocabulary import VALID_CHANNEL_NAMES

# ---------------------------------------------------------------------------
# Global vocabulary — the universe of what the CODEBASE can express.
# These are nouns (static vocab), not behavior. A scope unlocks a SUBSET of these;
# that subset (and the reasoning behind it) lives in the resolver, not here.
# ---------------------------------------------------------------------------

# Channels are imported, never redefined — shared/channel_vocabulary.py is the one source.
SUPPORTED_CHANNELS: frozenset[str] = frozenset(VALID_CHANNEL_NAMES)

# Image product SHAPE (encoded in the path tree); method lives in frame_inventory, not the path.
SUPPORTED_IMAGE_PRODUCT_TYPES: frozenset[str] = frozenset({"projection", "z_stack"})

# How a Z-stack collapses to one 2D frame.
SUPPORTED_PROJECTION_METHODS: frozenset[str] = frozenset({"focus_stack", "max", "mean"})

# REQUEST vocabulary for XY composition (what a config/user may ask for).
#   auto     — let the scope decide (the recommended default)
#   identity — one tile becomes one frame (no XY composition)
#   mosaic   — many tiles composed into one frame
SUPPORTED_XY_COMPOSITION_REQUESTS: frozenset[str] = frozenset({"auto", "identity", "mosaic"})

# RESOLVED vocabulary — what a backend actually executes. ``auto`` is never resolved.
RESOLVED_XY_COMPOSITIONS: frozenset[str] = frozenset({"identity", "mosaic"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnsupportedScopeError(ValueError):
    """Raised when a microscope has no registered materialization resolver."""


class UnsupportedMaterializationRequest(ValueError):
    """Raised when an image-materialization request is invalid or unsupported."""


# ---------------------------------------------------------------------------
# Plan shapes — the two halves of the request≠resolved seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageProductRequest:
    """ONE image product a config/user *asked for* — a wish, not a commitment.

    Fields carry REQUEST vocabulary: ``xy_composition`` may be the soft token ``"auto"``,
    meaning "let the scope decide." A request is scope-agnostic; the same plan can be handed
    to any microscope's resolver, which narrows it to that scope's reality.
    """

    channel_id: str
    image_product_type: str
    projection_method: str | None
    xy_composition: str = "auto"


@dataclass(frozen=True)
class ResolvedImageProduct:
    """ONE image product a scope *committed to* — concrete, executable.

    A resolved product never carries soft tokens: ``xy_composition`` is ``"identity"`` or
    ``"mosaic"`` (never ``"auto"``). This is the only product shape a backend executor accepts;
    by construction it cannot be ambiguous about what to do.
    """

    channel_id: str
    image_product_type: str
    projection_method: str | None
    xy_composition: str


@dataclass(frozen=True)
class ImageMaterializationPlan:
    """A whole REQUEST: the set of image products to materialize for one well.

    A plan is a *tuple of products*, not a single product — one well job may produce several
    products (channels × methods × shapes). The tuple is the unit the resolver maps over.
    """

    products: tuple[ImageProductRequest, ...]


@dataclass(frozen=True)
class ResolvedMaterializationPlan:
    """A whole COMMITMENT: what one scope will actually produce for one well.

    Output of the resolver, input to the backend. Each product is a ``ResolvedImageProduct``;
    the backend executes exactly this tuple and nothing it has to interpret.
    """

    products: tuple[ResolvedImageProduct, ...]


# ---------------------------------------------------------------------------
# Loading a requested plan from config
# ---------------------------------------------------------------------------


def load_image_materialization_plan(config: dict | None) -> ImageMaterializationPlan:
    """Build an ``ImageMaterializationPlan`` from a config dict.

    Reads ``config["image_materialization"]["products"]`` — a list of product dicts. Each dict
    must carry ``channel_id`` and ``image_product_type``; ``xy_composition`` defaults to
    ``"auto"``. ``projection_method`` is required for ``projection`` and forbidden for
    ``z_stack`` (global product grammar, enforced here — not scope behavior). Every value is
    validated against the SUPPORTED_* vocabularies. Scope rules are NOT applied here.

    If ``image_materialization.products`` is absent, returns the Step-6 default (one BF /
    projection / focus_stack product). An explicitly provided but empty list is an error —
    silently materializing the default would surprise a caller who meant "no products."
    """
    products_cfg = None
    if config:
        products_cfg = (config.get("image_materialization") or {}).get("products")

    if products_cfg is None:
        return _default_step6_plan()
    if not isinstance(products_cfg, list):
        raise UnsupportedMaterializationRequest(
            "config['image_materialization']['products'] must be a list of product dicts; "
            f"got {type(products_cfg).__name__}."
        )
    if not products_cfg:
        raise UnsupportedMaterializationRequest(
            "config['image_materialization']['products'] was provided but empty. "
            "Remove the key to use the Step-6 default, or list at least one product."
        )

    requests: list[ImageProductRequest] = []
    for i, p in enumerate(products_cfg):
        if not isinstance(p, dict):
            raise UnsupportedMaterializationRequest(
                f"product[{i}] must be a dict, got {type(p).__name__}."
            )
        channel_id = p["channel_id"]
        image_product_type = p["image_product_type"]
        projection_method = p.get("projection_method")
        xy_composition = p.get("xy_composition", "auto")

        _assert_in("channel_id", channel_id, SUPPORTED_CHANNELS, i)
        _assert_in("image_product_type", image_product_type, SUPPORTED_IMAGE_PRODUCT_TYPES, i)
        if projection_method is not None:
            _assert_in("projection_method", projection_method, SUPPORTED_PROJECTION_METHODS, i)
        _assert_in("xy_composition", xy_composition, SUPPORTED_XY_COMPOSITION_REQUESTS, i)
        _validate_product_shape(
            image_product_type=image_product_type,
            projection_method=projection_method,
            product_index=i,
        )

        requests.append(
            ImageProductRequest(
                channel_id=channel_id,
                image_product_type=image_product_type,
                projection_method=projection_method,
                xy_composition=xy_composition,
            )
        )
    return ImageMaterializationPlan(products=tuple(requests))


def _default_step6_plan() -> ImageMaterializationPlan:
    """The Step-6 default product set: the one accepted YX1 product."""
    return ImageMaterializationPlan(
        products=(
            ImageProductRequest(
                channel_id="BF",
                image_product_type="projection",
                projection_method="focus_stack",
                xy_composition="auto",
            ),
        )
    )


def _assert_in(field: str, value: str, allowed: frozenset[str], product_index: int) -> None:
    """Fail loud if ``value`` is outside the global vocabulary for ``field``."""
    if value not in allowed:
        raise UnsupportedMaterializationRequest(
            f"product[{product_index}] {field}={value!r} is not a supported {field}. "
            f"Supported: {sorted(allowed)}."
        )


def _validate_product_shape(
    *,
    image_product_type: str,
    projection_method: str | None,
    product_index: int,
) -> None:
    """Enforce global product-shape grammar (not scope behavior).

    ``projection`` requires a ``projection_method``; ``z_stack`` forbids one (it preserves z
    planes). This is product ontology — true for every microscope — so it lives here.
    """
    if image_product_type == "projection" and projection_method is None:
        raise UnsupportedMaterializationRequest(
            f"product[{product_index}] image_product_type='projection' requires "
            "projection_method to be set."
        )
    if image_product_type == "z_stack" and projection_method is not None:
        raise UnsupportedMaterializationRequest(
            f"product[{product_index}] image_product_type='z_stack' must not set "
            f"projection_method={projection_method!r}; z_stack preserves z planes."
        )
