"""Image materialization product-key helpers.

An image product key names one materialized image product within the image-materialization stage.
It is pipeline vocabulary, not a global biological/sample identifier, so it lives here rather than
in ``shared/identifiers``.
"""

from __future__ import annotations


def parse_image_product_key(product_key: str) -> tuple[str, str, str | None]:
    """Parse ``{channel_id}__{image_product_type}[__{projection_method}]``."""
    parts = str(product_key).split("__")
    if len(parts) == 2:
        channel_id, image_product_type = parts
        projection_method = None
    elif len(parts) == 3:
        channel_id, image_product_type, projection_method = parts
    else:
        raise ValueError(
            f"parse_image_product_key: cannot parse product_key {product_key!r}. "
            "Expected {channel_id}__{image_product_type}[__{projection_method}]."
        )
    try:
        expected = build_image_product_key(
            channel_id=channel_id,
            image_product_type=image_product_type,
            projection_method=projection_method,
        )
    except ValueError as exc:
        raise ValueError(
            f"parse_image_product_key: product_key {product_key!r} is not canonical. {exc}"
        ) from exc
    if str(product_key) != expected:
        raise ValueError(
            f"parse_image_product_key: product_key {product_key!r} is not canonical "
            f"(expected {expected!r})."
        )
    return channel_id, image_product_type, projection_method


def image_product_key_for_frame_row(
    *,
    channel_id: str,
    image_product_type: str,
    projection_method: object,
) -> str:
    """Return the product_key for one frame_inventory ROW.

    A frame_inventory row carries the product as three required columns (``channel_id``,
    ``image_product_type``, ``projection_method``); this composes them into the canonical
    ``product_key``. The one row-level normalization is that a CSV round-trip turns a null
    ``projection_method`` (z_stack rows) into ``NaN``/``"" `` — map that back to ``None`` so the
    grammar sees a true absent method. ``image_product_type`` is required and trusted as-is (no
    back-compat synthesis: every row in this target carries it by contract).

    Row-level sibling of ``image_product_key_for_resolved_product`` — one grammar, two callers.
    """
    method = projection_method
    if isinstance(method, float) and method != method:  # NaN from a CSV round-trip
        method = None
    if method is not None:
        method_text = str(method).strip()
        method = None if method_text == "" or method_text.lower() in ("nan", "<na>") else method_text

    return build_image_product_key(
        channel_id=str(channel_id),
        image_product_type=str(image_product_type),
        projection_method=method,
    )


def build_image_product_key(
    *,
    channel_id: str,
    image_product_type: str,
    projection_method: str | None = None,
) -> str:
    """Return the canonical image product key.

    Grammar: ``{channel_id}__{image_product_type}[__{projection_method}]``.
    Projection products require a method; z-stack products preserve planes and must not set one.
    """
    if image_product_type == "projection":
        if projection_method is None:
            raise ValueError(
                "build_image_product_key: projection products require projection_method."
            )
        return f"{channel_id}__{image_product_type}__{projection_method}"

    if image_product_type == "z_stack":
        if projection_method is not None:
            raise ValueError(
                "build_image_product_key: z_stack products must not set projection_method."
            )
        return f"{channel_id}__{image_product_type}"

    raise ValueError(
        f"build_image_product_key: unsupported image_product_type {image_product_type!r}; "
        "expected 'projection' or 'z_stack'."
    )
