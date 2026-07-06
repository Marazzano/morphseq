"""The snip coordinate space — one (height, width) governing the whole snip world.

Everything that lives at "snip resolution" must agree on a single frame shape: the snip_processing
crop output, the cropped embryo mask saved beside it, and the per-snip UNet auxiliary masks. If any
two disagree, fraction_alive's mask AND is comparing different grids. This resolver is the ONE place
that reads ``snip_frame_shape`` from config, so changing that single number reshapes every consumer
without re-touching per-step wiring.
"""

from __future__ import annotations

from collections.abc import Mapping

# Default snip frame shape (height, width), in pixels. Tall embryo axis first (NumPy/torch H, W).
DEFAULT_SNIP_FRAME_SHAPE: tuple[int, int] = (576, 256)

SNIP_FRAME_SHAPE_KEY = "snip_frame_shape"


def resolve_snip_frame_shape(config: Mapping | None) -> tuple[int, int]:
    """Return the ``(height, width)`` snip frame shape from config (single source of truth).

    Reads the top-level ``snip_frame_shape: [H, W]`` key. Falls back to the legacy per-step
    ``snip_processing.output_shape`` only if ``snip_frame_shape`` is absent, then to the default —
    so existing configs keep working while new configs set one value that governs every consumer.
    """
    config = config or {}
    shape = config.get(SNIP_FRAME_SHAPE_KEY)
    if shape is None:
        shape = (config.get("snip_processing") or {}).get("output_shape")
    if shape is None:
        return DEFAULT_SNIP_FRAME_SHAPE
    if len(shape) != 2:
        raise ValueError(
            f"{SNIP_FRAME_SHAPE_KEY} must be a two-item [height, width]; got {shape!r}."
        )
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{SNIP_FRAME_SHAPE_KEY} entries must be positive; got {shape!r}.")
    return height, width
