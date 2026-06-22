"""Constructor functions for canonical shared identifiers."""

MASK_INDEX_PREFIX = "m"
NO_MASK_SUFFIX = "mask_none"
TRACK_SUFFIX_SEPARATOR = "_track"
MASK_INDEX_WIDTH = 4
TRACK_INDEX_WIDTH = 4


def build_mask_id(image_id: str, local_mask_index: int) -> str:
    """Build the canonical mask identifier for one image-local mask."""
    _require_non_empty_text(image_id, field_name="image_id")
    _require_non_negative_int(local_mask_index, field_name="local_mask_index")
    return f"{image_id}_{MASK_INDEX_PREFIX}{local_mask_index:0{MASK_INDEX_WIDTH}d}"


def build_no_mask_id(image_id: str) -> str:
    """Build the explicit placeholder mask identifier for an image with no masks."""
    _require_non_empty_text(image_id, field_name="image_id")
    return f"{image_id}_{NO_MASK_SUFFIX}"


def build_track_id(well_id: str, track_index: int) -> str:
    """Build the canonical zero-based track identifier for one well-local track."""
    _require_non_empty_text(well_id, field_name="well_id")
    _require_non_negative_int(track_index, field_name="track_index")
    return f"{well_id}{TRACK_SUFFIX_SEPARATOR}{track_index:0{TRACK_INDEX_WIDTH}d}"


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_non_negative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
