"""Canonical identifier parsers.

Parsing is the inverse of minting: it DECOMPOSES an existing identifier back into
its parts. This is the only sanctioned place for that — it replaces ad-hoc
``.split("_")`` scattered through the call sites.

Identifier strings are opaque outside ``shared/identifiers``. Code outside this
package must use constructors and parsers — never string splitting, regex matching,
or f-string minting. Tiny fence, giant moat.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

import re

from data_pipeline.schemas.channel_normalization import VALID_CHANNEL_NAMES


_LOCAL_ID_RE = re.compile(r"(\d+)$")

# image_id grammar (projection): {well_id}_{channel_id}_t{time_index:04d+}
# channel_id has no underscores (canonical constraint); suffix is _t followed by digits.
_IMAGE_ID_RE = re.compile(r"^(.+)_([A-Za-z0-9]+)_t(\d{4,})$")

# image_id grammar (z_stack): {well_id}_{channel_id}_z{z_index:04d+}_t{time_index:04d+}
# A z-stack image_id names one materialized Z plane; the _z token disambiguates it from projection.
_Z_STACK_IMAGE_ID_RE = re.compile(r"^(.+)_([A-Za-z0-9]+)_z(\d{4,})_t(\d{4,})$")

# physical_embryo_id grammar: {well_id}_e{local_embryo_index:02d+}
# The _e token followed by digits is unambiguous because well_index never ends in _e\d+.
_PHYSICAL_EMBRYO_ID_RE = re.compile(r"^(.+)_e(\d+)$")

# embryo_id grammar: {physical_embryo_id}_{channel_id}
# physical_embryo_id ends in _e\d+; channel_id is the final _-delimited token with no underscores.
_EMBRYO_ID_RE = re.compile(r"^(.+_e\d+)_([A-Za-z0-9]+)$")

# snip_id grammar: {embryo_id}_t{time_index:04d+}
_SNIP_ID_RE = re.compile(r"^(.+)_t(\d{4,})$")

# mask_id grammar: {image_id}_m{local_mask_index:04d+} or {image_id}_mask_none
_MASK_ID_RE = re.compile(r"^(.+)_m(\d{4,})$")
_NO_MASK_SUFFIX = "_mask_none"

# track_id grammar: {well_id}_track{track_index:04d+}
_TRACK_ID_RE = re.compile(r"^(.+)_track(\d{4,})$")

_WELL_INDEX_RE = re.compile(r"^[A-Za-z]\d{1,3}$")


def parse_embryo_local_track_id(value: object) -> int:
    """Parse tracker-native embryo IDs like ``"embryo_0"`` to a zero-based integer track index.

    Returns a zero-based integer (the raw backend index). Use
    ``track_index_to_embryo_index`` to convert to one-based before minting
    ``physical_embryo_id``.

    snip_processing chain:
        raw_track_index = parse_embryo_local_track_id(track_id)
        local_embryo_index = track_index_to_embryo_index(raw_track_index)
        physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
    """
    if isinstance(value, bool):
        raise ValueError("embryo local track id cannot be boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    match = _LOCAL_ID_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse embryo local track id from {value!r}")
    return int(match.group(1))


def normalize_embryo_local_track_id(value: object) -> int:
    """Deprecated alias for ``parse_embryo_local_track_id``. Use that name instead."""
    return parse_embryo_local_track_id(value)


def track_index_to_embryo_index(raw_track_index: int) -> int:
    """Convert a zero-based backend/object track index to a one-based local embryo index.

    0 → 1, 1 → 2, etc. Rejects negative values (fail loud).
    This is the ONE place the zero-to-one conversion lives — not inline arithmetic.

    snip_processing chain:
        raw_track_index = normalize_embryo_local_track_id(track_id)
        local_embryo_index = track_index_to_embryo_index(raw_track_index)
        physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
    """
    idx = int(raw_track_index)
    if idx < 0:
        raise ValueError(
            f"track_index_to_embryo_index: raw_track_index must be >= 0, got {raw_track_index!r}. "
            "Negative track indices are not valid."
        )
    return idx + 1


def parse_image_id(image_id: str) -> tuple[str, str, int]:
    """Decompose a PROJECTION image_id into (well_id, channel_id, time_index).

    Validates that channel_id is in ``VALID_CHANNEL_NAMES``. Fails loud on malformed input.
    Example: ``"20250912_B01_BF_t0007"`` → ``("20250912_B01", "BF", 7)``

    A z-stack image_id (``..._z{z:04d}_t{t:04d}``) is REJECTED loud rather than silently dropping
    its ``z_index`` — use ``parse_image_id_with_z_index`` for ids that may carry a plane index.
    """
    text = str(image_id).strip()
    if _Z_STACK_IMAGE_ID_RE.match(text):
        raise ValueError(
            f"parse_image_id: {image_id!r} is a z-stack image_id (carries a _z plane index). "
            "parse_image_id returns a 3-tuple and would silently drop the z_index — call "
            "parse_image_id_with_z_index instead."
        )
    match = _IMAGE_ID_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_image_id: cannot parse {image_id!r}. "
            "Expected format: {well_id}_{channel_id}_t{time_index:04d} "
            "(channel_id has no underscores; time suffix is _t followed by ≥4 digits)."
        )
    well_id = match.group(1)
    channel_id = match.group(2)
    time_index = int(match.group(3))
    if channel_id not in VALID_CHANNEL_NAMES:
        raise ValueError(
            f"parse_image_id: image_id {image_id!r} contains channel_id {channel_id!r} "
            f"which is not in the canonical vocabulary {sorted(VALID_CHANNEL_NAMES)}. "
            "A canonical image_id must embed a canonical channel_id."
        )
    return well_id, channel_id, time_index


def parse_image_id_with_z_index(image_id: str) -> tuple[str, str, int, int | None]:
    """Decompose any image_id into (well_id, channel_id, time_index, z_index).

    The z-aware inverse of ``build_image_id``: it accepts BOTH grammars and never drops the plane
    index. ``z_index`` is ``None`` for a projection id and the integer plane for a z-stack id.
    Validates that channel_id is in ``VALID_CHANNEL_NAMES``. Fails loud on malformed input.

    Examples:
        ``"20250912_B01_BF_t0007"``        → ``("20250912_B01", "BF", 7, None)``
        ``"20250912_B01_BF_z0003_t0007"``  → ``("20250912_B01", "BF", 7, 3)``
    """
    text = str(image_id).strip()
    z_match = _Z_STACK_IMAGE_ID_RE.match(text)
    if z_match:
        well_id = z_match.group(1)
        channel_id = z_match.group(2)
        z_index = int(z_match.group(3))
        time_index = int(z_match.group(4))
        if channel_id not in VALID_CHANNEL_NAMES:
            raise ValueError(
                f"parse_image_id_with_z_index: image_id {image_id!r} contains channel_id "
                f"{channel_id!r} which is not in the canonical vocabulary "
                f"{sorted(VALID_CHANNEL_NAMES)}. A canonical image_id must embed a canonical channel_id."
            )
        return well_id, channel_id, time_index, z_index
    well_id, channel_id, time_index = parse_image_id(text)
    return well_id, channel_id, time_index, None


def parse_mask_id(mask_id: str) -> tuple[str, int | None, bool]:
    """Decompose mask_id into (image_id, local_mask_index, is_no_mask).

    Actual masks use ``{image_id}_m{local_mask_index:04d+}``. Explicit no-mask
    placeholders use ``{image_id}_mask_none`` and return ``local_mask_index is None``.
    """
    text = str(mask_id).strip()
    if not text:
        raise ValueError("parse_mask_id: mask_id must be a non-empty string.")

    if text.endswith(_NO_MASK_SUFFIX):
        image_id = text[: -len(_NO_MASK_SUFFIX)]
        if not image_id:
            raise ValueError(
                "parse_mask_id: no-mask placeholder must include a non-empty image_id "
                "before '_mask_none'."
            )
        return image_id, None, True

    match = _MASK_ID_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_mask_id: cannot parse {mask_id!r}. Expected constructor-minted "
            "{image_id}_m{local_mask_index:04d} or {image_id}_mask_none."
        )
    return match.group(1), int(match.group(2)), False


def parse_track_id(track_id: str) -> tuple[str, int]:
    """Decompose zero-based track_id into (well_id, track_index)."""
    text = str(track_id).strip()
    if not text:
        raise ValueError("parse_track_id: track_id must be a non-empty string.")

    match = _TRACK_ID_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_track_id: cannot parse {track_id!r}. Expected constructor-minted "
            "{well_id}_track{track_index:04d}."
        )
    return match.group(1), int(match.group(2))


def parse_physical_embryo_id(physical_embryo_id: str) -> tuple[str, int]:
    """Decompose physical_embryo_id into (well_id, local_embryo_index).

    Fails loud on malformed input or local_embryo_index < 1 (one-based; _e00 is rejected).
    Example: ``"20250912_B01_e01"`` → ``("20250912_B01", 1)``
    """
    text = str(physical_embryo_id).strip()
    match = _PHYSICAL_EMBRYO_ID_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_physical_embryo_id: cannot parse {physical_embryo_id!r}. "
            "Expected format: {well_id}_e{local_embryo_index:02d+} (e.g. 20250912_B01_e01)."
        )
    well_id = match.group(1)
    local_embryo_index = int(match.group(2))
    if local_embryo_index < 1:
        raise ValueError(
            f"parse_physical_embryo_id: local_embryo_index in {physical_embryo_id!r} is "
            f"{local_embryo_index}, which is < 1. physical_embryo_id uses a one-based index "
            "(embryo 1 is the first embryo, not embryo 0). _e00 is not a valid identity."
        )
    return well_id, local_embryo_index


def parse_embryo_id(embryo_id: str) -> tuple[str, str]:
    """Decompose embryo_id into (physical_embryo_id, channel_id).

    Validates that channel_id is in ``VALID_CHANNEL_NAMES``. Fails loud on malformed input.
    Example: ``"20250912_B01_e01_BF"`` → ``("20250912_B01_e01", "BF")``
    """
    text = str(embryo_id).strip()
    match = _EMBRYO_ID_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_embryo_id: cannot parse {embryo_id!r}. "
            "Expected format: {physical_embryo_id}_{channel_id} "
            "(physical_embryo_id ends in _e\\d+; channel_id has no underscores)."
        )
    physical_embryo_id = match.group(1)
    channel_id = match.group(2)
    if channel_id not in VALID_CHANNEL_NAMES:
        raise ValueError(
            f"parse_embryo_id: embryo_id {embryo_id!r} contains channel_id {channel_id!r} "
            f"which is not in the canonical vocabulary {sorted(VALID_CHANNEL_NAMES)}. "
            "A canonical embryo_id must embed a canonical channel_id."
        )
    return physical_embryo_id, channel_id


def parse_snip_id(snip_id: str) -> tuple[str, int]:
    """Decompose snip_id into (embryo_id, time_index).

    Fails loud on malformed input.
    Example: ``"20250912_B01_e01_BF_t0007"`` → ``("20250912_B01_e01_BF", 7)``
    """
    text = str(snip_id).strip()
    match = _SNIP_ID_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_snip_id: cannot parse {snip_id!r}. "
            "Expected format: {embryo_id}_t{time_index:04d} "
            "(time suffix is _t followed by ≥4 digits)."
        )
    embryo_id = match.group(1)
    time_index = int(match.group(2))
    return embryo_id, time_index


def split_well_id(well_id: str) -> tuple[str, str]:
    """Decompose a GLOBAL ``well_id`` into ``(experiment_id, well_index)``.

    Inverse of ``build_well_id``: ``"20240418_A01" -> ("20240418", "A01")``. The
    well_index is the final underscore-delimited token (a local plate label like
    ``A01``); everything before it is the experiment id. This replaces the ad-hoc
    ``.rsplit("_", 1)`` / ``re.match(r"^(.+)_([A-H]\\d{2})$", …)`` scattered through
    the per-well boundaries and the legacy ``video_id`` parsing.
    """
    text = str(well_id).strip()
    if "_" not in text:
        raise ValueError(
            f"well_id {well_id!r} is not global ({{experiment_id}}_{{well_index}}); "
            "it looks like a bare local well_index."
        )
    experiment_id, well_index = text.rsplit("_", 1)
    if not experiment_id or not well_index:
        raise ValueError(f"Could not split well_id {well_id!r} into (experiment_id, well_index)")
    return experiment_id, well_index
