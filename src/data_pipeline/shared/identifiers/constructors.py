"""Canonical identifier constructors.

These functions MINT identifiers. They are dumb: they perform no biology
inference and no name mapping (e.g. they do not map ``Brightfield -> BF``).
Normalization of local tokens (``channel_id`` from ``raw_channel_name``) happens
upstream in metadata ingest; constructors only assemble the canonical string.

Identifier strings are opaque outside ``shared/identifiers``. Code outside this
package must use constructors and parsers — never string splitting, regex matching,
or f-string minting. Tiny fence, giant moat.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.

Canonical model (the sign on the door):
    experiment_id       = 20240418
    well_index          = A01
    well_id             = 20240418_A01                    global well id
    channel_id          = BF                              no underscores in channel_id (canonical constraint)
    image_id            = {well_id}_{channel_id}_t{time_index:04d}
                          or {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}
    physical_embryo_id  = {well_id}_e{local_embryo_index:02d}  (one-based; ≥ 1)
    embryo_id           = {physical_embryo_id}_{channel_id}
    snip_id             = {embryo_id}_t{time_index:04d}

Tiny doctrine:
    Physical embryo ID names the animal.
    Embryo ID names the animal in a channel.
    Snip ID names the animal-channel at a time.

BREAKING CHANGE (snip world update):
    build_embryo_id(well_id, local_track_id)  →  build_embryo_id(physical_embryo_id, image_id)
    build_snip_id(embryo_id, time_index)      →  build_snip_id(embryo_id, image_id)
    NEW: build_physical_embryo_id(well_id, local_embryo_index)

    Production callers using the old signatures fail loudly at import time and must migrate.
    See docs/refactors/streamline-snakemake/target/specs/detect-seg-track/targets/snip_world.md.

Compositional grammar — every id is ``parent_id + local_token``. ``build_well_id``
is the ONE join point where ``experiment_id`` and ``well_index`` meet; every id
downstream is ``well_id``-first and inherits its global, sanitized prefix.
"""

from __future__ import annotations

import re

from .parsers import parse_embryo_id, parse_image_id, parse_physical_embryo_id


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SEP_RE = re.compile(r"[_-]{2,}")
_MASK_INDEX_PREFIX = "m"
_NO_MASK_SUFFIX = "mask_none"
_TRACK_SUFFIX_SEPARATOR = "_track"
_MASK_INDEX_WIDTH = 4
_TRACK_INDEX_WIDTH = 4


def sanitize_experiment_id(value: str) -> str:
    """Return a deterministic, path-safe experiment identifier."""
    cleaned = str(value).strip()
    cleaned = cleaned.replace(" ", "_")
    cleaned = _SANITIZE_RE.sub("_", cleaned)
    cleaned = _SEP_RE.sub("_", cleaned)
    return cleaned.strip("_-")


def build_well_id(experiment_id: str, well_index: str) -> str:
    """Return the canonical GLOBAL well id, e.g. ``20240418_A01``.

    This is the one place ``experiment_id`` and the local ``well_index`` are
    joined; ``experiment_id`` is sanitized here so every downstream id inherits a
    clean global prefix.
    """
    return f"{sanitize_experiment_id(experiment_id)}_{str(well_index).strip()}"


def build_image_id(
    well_id: str,
    channel_id: str,
    time_index: int,
    *,
    z_index: int | None = None,
) -> str:
    """Return the canonical image id for one channel at one timepoint.

    ``z_index=None`` preserves the historical projection grammar byte-for-byte.
    A real ``z_index`` identifies one materialized z-stack plane.
    """
    if z_index is None:
        return f"{str(well_id)}_{str(channel_id)}_t{int(time_index):04d}"
    if isinstance(z_index, bool):
        raise ValueError("z_index must be an integer or None.")
    try:
        z = int(z_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("z_index must be an integer or None.") from exc
    if z < 0:
        raise ValueError("z_index must be zero or greater.")
    return f"{str(well_id)}_{str(channel_id)}_z{z:04d}_t{int(time_index):04d}"


def build_mask_id(base_id: str, local_mask_index: int) -> str:
    """Return the canonical mask id for one local mask under a parent id.

    ``base_id`` is the parent identity the mask belongs to. A frame-object mask is
    scoped to an ``image_id`` (``{image_id}_m{NNNN}``); a snip-scoped mask (e.g. an
    auxiliary/VIA mask paired with one snip) is scoped to a ``snip_id``
    (``{snip_id}_m{NNNN}``). One constructor, both grammars — the mask id always reads
    ``{parent}_m{index}`` and ``parse_mask_id`` round-trips either parent verbatim.

    Examples:
        ``("20250912_B01_BF_t0007", 1)``           -> ``"20250912_B01_BF_t0007_m0001"``
        ``("20250912_B01_e01_BF_t0007", 1)``       -> ``"20250912_B01_e01_BF_t0007_m0001"``
    """
    _require_non_empty_text(base_id, field_name="base_id")
    _require_non_negative_int(local_mask_index, field_name="local_mask_index")
    return f"{base_id}_{_MASK_INDEX_PREFIX}{local_mask_index:0{_MASK_INDEX_WIDTH}d}"


def build_no_mask_id(image_id: str) -> str:
    """Return the explicit placeholder mask id for an image with no masks."""
    _require_non_empty_text(image_id, field_name="image_id")
    return f"{image_id}_{_NO_MASK_SUFFIX}"


def build_track_id(well_id: str, track_index: int) -> str:
    """Return the canonical zero-based track id for one well-local track."""
    _require_non_empty_text(well_id, field_name="well_id")
    _require_non_negative_int(track_index, field_name="track_index")
    return f"{well_id}{_TRACK_SUFFIX_SEPARATOR}{track_index:0{_TRACK_INDEX_WIDTH}d}"


def build_physical_embryo_id(well_id: str, local_embryo_index: int) -> str:
    """Return the stable animal identity within a well (no image/time/channel context).

    ``local_embryo_index`` must be >= 1 (one-based — biologist-facing).
    Use ``track_index_to_embryo_index(raw_track_index)`` to convert zero-based backend
    object IDs before calling this function.

    Example: ``("20250912_B01", 1)`` → ``"20250912_B01_e01"``
    """
    idx = int(local_embryo_index)
    if idx < 1:
        raise ValueError(
            f"build_physical_embryo_id: local_embryo_index must be >= 1 (one-based), got {idx!r}. "
            "Use track_index_to_embryo_index(raw_track_index) to convert zero-based backend IDs. "
            "Embryo 1 is the first embryo — zero is not a valid identity."
        )
    return f"{str(well_id)}_e{idx:02d}"


def build_embryo_id(physical_embryo_id: str, image_id: str) -> str:
    """Return the image-channel-contextual embryo identity.

    Parses ``physical_embryo_id`` and ``image_id`` to extract and cross-check their
    embedded ``well_id``s, then extracts ``channel_id`` from ``image_id``.
    Fails loud on well_id mismatch.

    Example: ``("20250912_B01_e01", "20250912_B01_BF_t0007")`` → ``"20250912_B01_e01_BF"``
    """
    embryo_well_id, _local_idx = parse_physical_embryo_id(physical_embryo_id)
    image_well_id, channel_id, _time_index = parse_image_id(image_id)
    if embryo_well_id != image_well_id:
        raise ValueError(
            f"build_embryo_id: well_id mismatch — physical_embryo_id encodes well_id "
            f"{embryo_well_id!r} but image_id encodes well_id {image_well_id!r}. "
            "Cannot combine an embryo from one well with an image from a different well."
        )
    return f"{str(physical_embryo_id)}_{channel_id}"


def build_snip_id(embryo_id: str, image_id: str) -> str:
    """Return the crop artifact identity for one animal-channel at one time.

    Parses ``embryo_id`` and ``image_id``, then asserts well_id AND channel_id agreement
    (fails loud on mismatch). Extracts ``time_index`` from ``image_id``.

    Example: ``("20250912_B01_e01_BF", "20250912_B01_BF_t0007")`` → ``"20250912_B01_e01_BF_t0007"``
    """
    physical_embryo_id, embryo_channel_id = parse_embryo_id(embryo_id)
    embryo_well_id, _local_idx = parse_physical_embryo_id(physical_embryo_id)
    image_well_id, image_channel_id, time_index = parse_image_id(image_id)

    if embryo_well_id != image_well_id:
        raise ValueError(
            f"build_snip_id: well_id mismatch — embryo_id encodes well_id "
            f"{embryo_well_id!r} but image_id encodes well_id {image_well_id!r}."
        )
    if embryo_channel_id != image_channel_id:
        raise ValueError(
            f"build_snip_id: channel_id mismatch — embryo_id encodes channel_id "
            f"{embryo_channel_id!r} but image_id encodes channel_id {image_channel_id!r}. "
            "A snip must derive from the same channel as its embryo identity."
        )
    return f"{str(embryo_id)}_t{time_index:04d}"


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_non_negative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be zero or greater.")
