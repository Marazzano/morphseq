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

from data_pipeline.shared.channel_vocabulary import VALID_CHANNEL_NAMES


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

# ── Experiment collection grammar (see docs/EXPERIMENT_GROUP_PLATE_MODEL.md) ──
#
# A COLLECTION is a raw directory whose name ends in ``_coll``. Inside it, each
# child folder/file is named positionally:
#
#     {date}_{plate_token}_{event_label}   e.g. 20260607_plate01_t45hpf
#
#     date        = 8 digits            → acquisition fact, NOT identity (dropped)
#     plate_token = plate01             → the PLATE identity; this IS the experiment
#     event_label = t45hpf (optional)   → declared_hpf axis (dropped from identity)
#
# The MERGE model: date and event_label are dropped from identity; the plate token
# IS the id. Multiple t-events of one plate share ``experiment_id = {coll}_{plate}``.
_COLLECTION_SUFFIX = "_coll"

# child name: {date}_{plate_token}[_{event_label}], date is the 8-digit split anchor.
# Group 1 = plate_token, group 3 = event_label (may be absent → None).
#
# NOTE: this positional reading assumes the plate token is the FIRST underscore-delimited
# token after the date. Real acquisitions also carry a free-form descriptive middle, e.g.
#   20260624_pbx_flouresence_bf_pilot_plate01_t33hpf
# where positionally group 1 would be "pbx" (a project label, not the plate) and group 3
# "flouresence_bf_pilot_plate01_t33hpf" (which no declared-hpf pattern matches). So the
# token-anchored patterns below take PRECEDENCE, and this positional form is the fallback
# for names that carry no recognizable plate/age token.
_COLLECTION_CHILD_RE = re.compile(r"^\d{8}_([^_]+)(_(.+))?$")

# ANCHORED tokens — matched by SHAPE anywhere in the name, not by position. These are what
# make a descriptive middle harmless. Both are anchored to token boundaries (start/underscore
# … underscore/end) so they cannot match inside a longer word.
#   plate token: plate01, plate1, PLATE02 → the PLATE identity
_PLATE_TOKEN_RE = re.compile(r"(?:^|_)(plate\d+)(?:_|$)", flags=re.IGNORECASE)
#   declared-age token: t45hpf → the age axis. Trailing/anywhere; the LAST one wins.
_AGE_TOKEN_RE = re.compile(r"(?:^|_)(t\d+hpf)(?:_|$)", flags=re.IGNORECASE)

# The 8-digit date prefix every collection child must carry (the split anchor).
_CHILD_DATE_PREFIX_RE = re.compile(r"^\d{8}_")

# event_label declared-age token: t<NN>hpf → NN.
_DECLARED_HPF_RE = re.compile(r"^t(\d+)hpf$")


def is_collection(name: str) -> bool:
    """Return True iff ``name`` marks an experiment COLLECTION (``_coll`` suffix).

    The marker is a pure suffix test — collection names are experiment-id-shaped
    strings, so no filesystem probe is needed to classify an input entry. A folder
    WITHOUT the marker is a legacy single experiment (unchanged behavior).
    """
    return str(name).strip().endswith(_COLLECTION_SUFFIX)


_COLLECTION_ID_MARKER = _COLLECTION_SUFFIX + "_"  # "_coll_" — the marker INSIDE a plate id


def is_collection_plate_id(experiment_id: str) -> bool:
    """Return True iff ``experiment_id`` is a merged collection plate id (``{collection}_coll_{plate}``).

    Distinct from ``is_collection`` (which tests a collection *name*, ending in ``_coll``): a plate
    id has the ``_coll_`` marker INSIDE it (``cilia_snapshots_coll_plate01``). Used at the DAG seam
    to decide whether an experiment is a single scope read or a collection UNION.
    """
    return _COLLECTION_ID_MARKER in str(experiment_id).strip()


def parse_collection_name_from_plate_id(experiment_id: str) -> str:
    """``{collection}_coll_{plate}`` -> ``{collection}_coll`` (the raw dir name to glob).

    Inverse-facing helper for acquisition ingest: the collection dir is the id up to and
    including the ``_coll`` marker. Fails loud if the id is not a collection plate id.
    """
    text = str(experiment_id).strip()
    if _COLLECTION_ID_MARKER not in text:
        raise ValueError(
            f"parse_collection_name_from_plate_id: {experiment_id!r} is not a collection plate id "
            f"(no '{_COLLECTION_ID_MARKER}' marker)."
        )
    return text[: text.index(_COLLECTION_ID_MARKER)] + _COLLECTION_SUFFIX


def parse_plate_token(child_name: str) -> str:
    """Extract the ``plate_token`` from a collection child name.

    The token is found by SHAPE (``plate<N>``) anywhere in the name, so a free-form
    descriptive middle is harmless::

        "20260607_plate01_t45hpf"                          -> "plate01"
        "20260607_plate01"                                 -> "plate01"
        "20260624_pbx_flouresence_bf_pilot_plate01_t33hpf"  -> "plate01"   (not "pbx")

    Falls back to the positional reading (``{date}_{plate_token}[_{event_label}]``) for
    names that carry no ``plate<N>``-shaped token, preserving the original behavior for
    plate tokens that are named differently. Fails loud on a name that does not match the
    collection child grammar (e.g. a missing/non-8-digit date).

    Returned lowercased when matched by shape so the token is a stable identity component
    regardless of source casing (``PLATE01`` and ``plate01`` are the same plate).
    """
    text = str(child_name).strip()
    if not _CHILD_DATE_PREFIX_RE.match(text):
        raise ValueError(
            f"parse_plate_token: cannot parse {child_name!r}. Expected a collection child "
            "named {date}_{plate_token}[_{event_label}] with date = 8 digits "
            "(e.g. 20260607_plate01_t45hpf)."
        )

    # Anchored token wins: it is the plate identity wherever it sits in the name.
    anchored = _PLATE_TOKEN_RE.search(text)
    if anchored:
        return anchored.group(1).lower()

    # Fallback: the original positional reading (first token after the date).
    match = _COLLECTION_CHILD_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_plate_token: cannot parse {child_name!r}. Expected a collection child "
            "named {date}_{plate_token}[_{event_label}] with date = 8 digits "
            "(e.g. 20260607_plate01_t45hpf)."
        )
    return match.group(1)


def parse_event_label(child_name: str) -> str | None:
    """Extract the optional ``event_label`` from a collection child name.

    The declared-age token is found by SHAPE (``t<NN>hpf``) so a free-form descriptive
    middle is harmless::

        "20260607_plate01_t45hpf"                           -> "t45hpf"
        "20260607_plate01"                                  -> None        (no event)
        "20260624_pbx_flouresence_bf_pilot_plate01_t33hpf"   -> "t33hpf"

    Falls back to the positional reading (everything after ``{date}_{plate_token}_``) for
    names whose event label is not age-shaped (e.g. ``sci``), preserving the original
    behavior. Fails loud on a name that does not match the collection child grammar.

    Returned lowercased when matched by shape, so ``T45HPF`` and ``t45hpf`` agree.
    """
    text = str(child_name).strip()
    if not _CHILD_DATE_PREFIX_RE.match(text):
        raise ValueError(
            f"parse_event_label: cannot parse {child_name!r}. Expected a collection child "
            "named {date}_{plate_token}[_{event_label}] with date = 8 digits "
            "(e.g. 20260607_plate01_t45hpf)."
        )

    # Anchored age token wins wherever it sits. The LAST match is the event: a descriptive
    # middle could itself contain an age-shaped word, and the event label is conventionally
    # the trailing token.
    anchored = list(_AGE_TOKEN_RE.finditer(text))
    if anchored:
        return anchored[-1].group(1).lower()

    # Fallback: the original positional reading (may be a non-age label like "sci", or None).
    match = _COLLECTION_CHILD_RE.match(text)
    if not match:
        raise ValueError(
            f"parse_event_label: cannot parse {child_name!r}. Expected a collection child "
            "named {date}_{plate_token}[_{event_label}] with date = 8 digits "
            "(e.g. 20260607_plate01_t45hpf)."
        )
    return match.group(3)


def parse_declared_hpf(event_label_or_name: str | None) -> int | None:
    """Return the declared (planned) age in hpf, or ``None`` if none is declared.

    Accepts either a bare event_label (``"t45hpf"``) or a full collection child name
    (``"20260607_plate01_t45hpf"``) — a full name is reduced to its event_label first.
    ``t<NN>hpf -> NN``; no ``t...hpf`` token (e.g. ``sci``, absent event) -> ``None``.
    """
    if event_label_or_name is None:
        return None
    text = str(event_label_or_name).strip()
    if not text:
        return None
    # Reduce a full child name to its event_label if it looks like one.
    if _COLLECTION_CHILD_RE.match(text):
        text = parse_event_label(text) or ""
    match = _DECLARED_HPF_RE.match(text)
    if not match:
        return None
    return int(match.group(1))


def compose_collection_experiment_id(collection_name: str, child_name: str) -> str:
    """Compose ``experiment_id = {collection}_{plate_token}`` for a collection child.

    The plate token IS the id: date and event_label are DROPPED (the MERGE model — all
    t-events of one plate share this id). The composed id is run through
    ``sanitize_experiment_id`` so it inherits the canonical grammar. This is the ONE
    place a collection id is minted — the plural resolver consumes it, never re-mints.

    Example: ``("cilia_snapshots_coll", "20260607_plate01_t45hpf")``
             -> ``"cilia_snapshots_coll_plate01"``
    """
    from data_pipeline.shared.identifiers.constructors import sanitize_experiment_id

    plate_token = parse_plate_token(child_name)
    return sanitize_experiment_id(f"{str(collection_name).strip()}_{plate_token}")


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


def parse_well_row_col(well_id: str) -> tuple[int, int]:
    """Decompose a GLOBAL ``well_id`` into zero-based plate ``(row, col)`` grid indices.

    ``"20250912_B11" -> (1, 10)``: row A–H maps to 0–7, column 1–12 maps to 0–11. Composes the
    other well parsers so grammar lives in exactly one place — ``split_well_id`` strips the
    experiment prefix and ``validate_well_index`` canonicalizes + range-checks the local label
    (fails loud on anything outside the 8×12 plate). Callers that need physical plate geometry
    (e.g. a 96-well heatmap) use this instead of re-parsing the label with ``ord()``.
    """
    from data_pipeline.shared.identifiers.validators import validate_well_index

    _, well_index = split_well_id(well_id)
    canonical = validate_well_index(well_index)  # 'B11' -> validated 'B11'
    row = ord(canonical[0]) - ord("A")
    col = int(canonical[1:]) - 1
    return row, col
