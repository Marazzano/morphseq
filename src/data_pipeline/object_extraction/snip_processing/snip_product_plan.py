"""What snip products a run is asking for.

SNIP PRODUCTS NEST UNDER THE IMAGE PRODUCT THEY CROP:

    image_materialization:
      products:
        - channel_id: BF
          image_product_type: projection
          projection_method: focus_stack
          snips:
            - snip_recipe: clahe_blend
        - channel_id: RFP
          image_product_type: projection
          projection_method: max
          snips:
            - snip_recipe: no_change

THE NESTING IS THE VALIDATION. A flat ``snip_processing.products`` list would have to carry
``channel_id`` and ``source_image_product_key`` on every entry, and then cross-validate that the two
agree — a check that exists only because the flat shape lets them disagree in the first place.
Nesting makes the mismatch UNREPRESENTABLE: a snip's source IS its parent, so there is nothing to
reconcile. It also turns "the source product must be materialized" from a runtime failure (a job
starts, reads frame_inventory, finds nothing) into a config-time structural fact.

``products`` stays a LIST rather than becoming a mapping keyed by product key.
``load_image_materialization_plan`` asserts ``isinstance(products_cfg, list)`` with its own error
message, and the absent/empty/non-empty rule is written against a list. A keyed mapping would read
marginally better and would be a breaking change to a shipped schema for a cosmetic gain.

THE THREE-WAY RULE APPLIES AT EACH LEVEL INDEPENDENTLY:

    snips: absent    -> this image product renders no snips
    snips: []        -> ERROR (as with products: [], an explicit empty list is a mistake, not a
                        request for the default)
    snips: [ ... ]   -> exactly these

Note "absent -> none" here, NOT "absent -> default". At the image-product level absent means "use
the default product set" because there is no parent to inherit from. Here the parent is explicit, so
absence means the caller did not ask for snips of it -- defaulting would silently render products
nobody requested.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_pipeline.acquisition.image_materialization.image_product_keys import (
    build_image_product_key,
)
from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
    build_snip_product_key,
)
from data_pipeline.object_extraction.snip_processing.snip_recipes import (
    SUPPORTED_SNIP_RECIPES,
)


class UnsupportedSnipRequest(ValueError):
    """A snip product was requested that this pipeline cannot render."""


@dataclass(frozen=True)
class SnipProductRequest:
    """One requested snip product: which source pixels, and what happens to them.

    ``channel_id`` is deliberately ABSENT -- it is derivable from ``source_image_product_key``, and
    storing it here would create a second representation of one fact that can disagree with the
    first.
    """

    source_image_product_key: str
    snip_recipe: str

    @property
    def snip_product_key(self) -> str:
        return build_snip_product_key(
            source_image_product_key=self.source_image_product_key,
            snip_recipe=self.snip_recipe,
        )


@dataclass(frozen=True)
class SnipProductPlan:
    """The whole REQUEST: every snip product to render for one well.

    A tuple of products, not one product — a per-well job may render several, and the tuple is the
    unit the fanout maps over.
    """

    products: tuple[SnipProductRequest, ...]

    @property
    def snip_product_keys(self) -> tuple[str, ...]:
        return tuple(p.snip_product_key for p in self.products)


def _default_plan() -> SnipProductPlan:
    """What runs when no config mentions snips at all: today's BF product, unchanged.

    Byte-identical to the pre-migration behavior, so an existing config keeps producing exactly what
    it produced before this seam existed.
    """
    source, recipe = DEFAULT_BF_SNIP_PRODUCT_KEY.rsplit("__", 1)
    return SnipProductPlan(
        products=(SnipProductRequest(source_image_product_key=source, snip_recipe=recipe),)
    )


def load_snip_product_plan(config: dict | None) -> SnipProductPlan:
    """Build the snip plan by walking ``image_materialization.products[*].snips``.

    Reads the SAME config block the image plan reads, because a snip product is a child of an image
    product rather than an independent request. When no product declares ``snips``, returns the
    default BF plan -- which is what a pre-migration config means.
    """
    products_cfg = None
    if config:
        products_cfg = (config.get("image_materialization") or {}).get("products")

    if products_cfg is None:
        return _default_plan()
    if not isinstance(products_cfg, list):
        raise UnsupportedSnipRequest(
            "config['image_materialization']['products'] must be a list of product dicts; "
            f"got {type(products_cfg).__name__}."
        )

    requests: list[SnipProductRequest] = []
    for i, product in enumerate(products_cfg):
        if not isinstance(product, dict):
            raise UnsupportedSnipRequest(
                f"product[{i}] must be a dict, got {type(product).__name__}."
            )
        if "snips" not in product:
            continue

        snips_cfg = product["snips"]
        if not isinstance(snips_cfg, list):
            raise UnsupportedSnipRequest(
                f"product[{i}] 'snips' must be a list of snip dicts; got "
                f"{type(snips_cfg).__name__}."
            )
        if not snips_cfg:
            raise UnsupportedSnipRequest(
                f"product[{i}] 'snips' was provided but empty. Remove the key to render no snips "
                "of this product, or list at least one recipe."
            )

        # The source key is BUILT from the parent, never read from the snip entry. That is the whole
        # point of nesting: there is no second copy to disagree with.
        try:
            source_key = build_image_product_key(
                channel_id=product["channel_id"],
                image_product_type=product["image_product_type"],
                projection_method=product.get("projection_method"),
            )
        except (KeyError, ValueError) as exc:
            raise UnsupportedSnipRequest(
                f"product[{i}] declares 'snips' but is not a valid image product ({exc}). A snip "
                "cannot be rendered from a source that will not be materialized."
            ) from exc

        for j, snip in enumerate(snips_cfg):
            if not isinstance(snip, dict):
                raise UnsupportedSnipRequest(
                    f"product[{i}].snips[{j}] must be a dict, got {type(snip).__name__}."
                )
            if "channel_id" in snip or "source_image_product_key" in snip:
                raise UnsupportedSnipRequest(
                    f"product[{i}].snips[{j}] may not set 'channel_id' or "
                    "'source_image_product_key' — the source is the PARENT product, and a second "
                    "copy could disagree with it. Remove the key."
                )
            recipe = snip.get("snip_recipe")
            if recipe not in SUPPORTED_SNIP_RECIPES:
                raise UnsupportedSnipRequest(
                    f"product[{i}].snips[{j}] snip_recipe={recipe!r} is not supported. "
                    f"Supported: {sorted(SUPPORTED_SNIP_RECIPES)}."
                )
            requests.append(
                SnipProductRequest(source_image_product_key=source_key, snip_recipe=recipe)
            )

    if not requests:
        return _default_plan()

    keys = [r.snip_product_key for r in requests]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise UnsupportedSnipRequest(
            f"duplicate snip product(s) requested: {duplicates}. Each key names one output path, so "
            "two requests for it would have two jobs writing the same files."
        )
    return SnipProductPlan(products=tuple(requests))
