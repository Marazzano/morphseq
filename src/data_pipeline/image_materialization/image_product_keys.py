"""Image materialization product-key helpers.

An image product key names one materialized image product within the image-materialization stage.
It is pipeline vocabulary, not a global biological/sample identifier, so it lives here rather than
in ``shared/identifiers``.
"""

from __future__ import annotations


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
