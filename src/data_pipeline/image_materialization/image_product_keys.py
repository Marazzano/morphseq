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
