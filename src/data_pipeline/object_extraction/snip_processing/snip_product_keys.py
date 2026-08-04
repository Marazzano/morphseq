"""Snip product-key helpers.

A snip product key names one rendered snip product: WHICH source pixels, and WHAT was done to them
photometrically.

    snip_product_key = {source_image_product_key}__{snip_recipe}

    BF__projection__focus_stack__clahe_blend
    RFP__projection__max__no_change
    BF__z_stack__no_change

THE SOURCE PRODUCT IS IN THE KEY, NOT JUST THE CHANNEL. An earlier design used
``{channel_id}__{snip_recipe}``. That COLLIDES: ``build_image_product_key`` emits both
``BF__projection__focus_stack`` and ``BF__z_stack``, which carry the same ``channel_id`` but
different pixels AND different calibration — ``BF__z_stack`` ships ``downsample_factor: 4`` while the
BF projection is native, so per-product divergence is already live in the shipped config. Under the
channel grammar both would collapse to ``BF__no_change``: one key, two materially different products.

``channel_id`` is therefore never written separately on a snip product — it is derivable from the
source key, and storing it twice is how two representations of one fact drift apart.

Pipeline vocabulary, not a biological identifier, so it lives beside the stage that mints it rather
than in ``shared/identifiers``.
"""

from __future__ import annotations

from data_pipeline.acquisition.image_materialization.image_product_keys import (
    parse_image_product_key,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_PRODUCT_KEY_COLUMN,
)

#: The BF product every legacy consumer means when it says "the snip". The compatibility resolver
#: filters to this rather than letting each caller assume "first row per snip_id" — row order is a
#: roulette wheel once a second product exists.
DEFAULT_BF_SNIP_PRODUCT_KEY = "BF__projection__focus_stack__clahe_blend"


def build_snip_product_key(*, source_image_product_key: str, snip_recipe: str) -> str:
    """Compose the canonical snip product key.

    The source key is validated by parsing it: a malformed source must fail HERE, where the message
    can name it, rather than downstream where the failure looks like a missing file.
    """
    source = str(source_image_product_key)
    recipe = str(snip_recipe)

    if not source or not recipe:
        raise ValueError(
            f"build_snip_product_key: both parts are required; got "
            f"source_image_product_key={source_image_product_key!r}, snip_recipe={snip_recipe!r}."
        )
    if "__" in recipe:
        raise ValueError(
            f"build_snip_product_key: snip_recipe {recipe!r} may not contain '__' — that is the "
            "field separator, and a recipe containing it would make the key ambiguous to parse."
        )
    # Raises if the source is not itself a canonical image product key.
    parse_image_product_key(source)

    return f"{source}__{recipe}"


def parse_snip_product_key(product_key: str) -> tuple[str, str]:
    """Split into ``(source_image_product_key, snip_recipe)``.

    THE SPLIT IS FROM THE RIGHT, and it has to be: the source key itself contains ``__`` (two or
    three fields of its own), so splitting from the left would tear the source apart. The recipe is
    the LAST field, which is why ``build_snip_product_key`` forbids ``__`` inside a recipe.

    Round-trips through the builder and requires string equality, so a non-canonical key — right
    shape, wrong spelling — is rejected rather than silently accepted and later mismatched against
    a directory name.
    """
    text = str(product_key)
    source, separator, recipe = text.rpartition("__")
    if not separator or not source or not recipe:
        raise ValueError(
            f"parse_snip_product_key: cannot parse {product_key!r}. Expected "
            "{source_image_product_key}__{snip_recipe}, e.g. 'RFP__projection__max__no_change'."
        )

    try:
        expected = build_snip_product_key(source_image_product_key=source, snip_recipe=recipe)
    except ValueError as exc:
        raise ValueError(
            f"parse_snip_product_key: {product_key!r} is not canonical. {exc}"
        ) from exc
    if text != expected:
        raise ValueError(
            f"parse_snip_product_key: {product_key!r} is not canonical (expected {expected!r})."
        )
    return source, recipe


def channel_id_for_snip_product_key(product_key: str) -> str:
    """The channel a snip product's pixels came from, DERIVED rather than stored.

    Exists so no caller is tempted to keep a parallel ``channel_id`` column beside the key: two
    representations of one fact can disagree, and this makes the disagreement unrepresentable.
    """
    source, _recipe = parse_snip_product_key(product_key)
    channel_id, _type, _method = parse_image_product_key(source)
    return channel_id


def snip_product_key_for_row(row: object) -> str | None:
    """Read a snip product key off an inventory ROW, normalizing CSV round-trip nulls.

    A missing value survives a CSV round-trip as ``NaN`` / ``""`` / ``"nan"`` / ``"<NA>"`` depending
    on the writer and the column's inferred dtype. All of those mean ABSENT; returning them as
    strings would produce a product key that looks real and matches nothing.
    """
    try:
        value = row["snip_product_key"]  # type: ignore[index]
    except (KeyError, TypeError, IndexError):
        return None

    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "<na>", "none"):
        return None
    return text


def select_default_snip_product(
    inventory,
    *,
    default_snip_product_key: str = DEFAULT_BF_SNIP_PRODUCT_KEY,
):
    """Filter a multi-product inventory down to the one product legacy consumers mean.

    THE COMPATIBILITY SEAM, AND IT IS CENTRALIZED ON PURPOSE. Once a well holds several products,
    a consumer that still thinks in terms of "the snip for this snip_id" is asking an ambiguous
    question. The dangerous version of this helper is the one nobody writes: each of ~40 downstream
    readers independently doing ``df.drop_duplicates("snip_id")`` or taking ``.iloc[0]``, which
    silently means "whichever product happened to be written first". Row order is a roulette wheel
    the moment a second product exists, and the resulting bug — a QC metric computed on RFP pixels
    because the merge ordering changed — looks like a data problem, not a code problem.

    Pre-migration shards have no product column at all; those pass through unchanged, so this is
    safe to call from a consumer that may see either shape.

    TODO(deprecate-legacy-snip-symlinks): delete once consumers select their product explicitly.
    """
    if SNIP_PRODUCT_KEY_COLUMN not in getattr(inventory, "columns", ()):
        return inventory

    present = set(inventory[SNIP_PRODUCT_KEY_COLUMN].dropna().astype(str))
    if present and default_snip_product_key not in present:
        raise ValueError(
            f"select_default_snip_product: {default_snip_product_key!r} is not present in this "
            f"inventory; found {sorted(present)}. Returning some other product would silently hand "
            "a legacy caller pixels it did not ask for."
        )
    return inventory[
        inventory[SNIP_PRODUCT_KEY_COLUMN].astype(str) == default_snip_product_key
    ].copy()
