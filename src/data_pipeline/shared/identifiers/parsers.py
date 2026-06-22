"""Parser functions for canonical shared identifiers."""

from .constructors import (
    MASK_INDEX_PREFIX,
    MASK_INDEX_WIDTH,
    NO_MASK_SUFFIX,
    TRACK_INDEX_WIDTH,
    TRACK_SUFFIX_SEPARATOR,
)


def parse_mask_id(mask_id: str) -> tuple[str, int | None, bool]:
    """Parse a mask identifier into ``(image_id, local_mask_index, is_no_mask)``."""
    _require_non_empty_text(mask_id, field_name="mask_id")

    no_mask_suffix = f"_{NO_MASK_SUFFIX}"
    if mask_id.endswith(no_mask_suffix):
        image_id = mask_id[: -len(no_mask_suffix)]
        _require_non_empty_text(image_id, field_name="image_id")
        return image_id, None, True

    marker = f"_{MASK_INDEX_PREFIX}"
    image_id, separator, mask_index_text = mask_id.rpartition(marker)
    if not separator or not image_id:
        raise ValueError(
            "mask_id must be constructor-minted as '<image_id>_m####' or "
            "'<image_id>_mask_none'."
        )
    if not mask_index_text.isdigit() or len(mask_index_text) < MASK_INDEX_WIDTH:
        raise ValueError("mask_id local mask index must use constructor-minted '_m####' form.")

    return image_id, int(mask_index_text), False


def parse_track_id(track_id: str) -> tuple[str, int]:
    """Parse a track identifier into ``(well_id, track_index)``."""
    _require_non_empty_text(track_id, field_name="track_id")

    well_id, separator, track_index_text = track_id.rpartition(TRACK_SUFFIX_SEPARATOR)
    if not separator or not well_id:
        raise ValueError("track_id must be constructor-minted as '<well_id>_track####'.")
    if not track_index_text.isdigit() or len(track_index_text) < TRACK_INDEX_WIDTH:
        raise ValueError("track_id index must use constructor-minted '_track####' form.")

    return well_id, int(track_index_text)


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
