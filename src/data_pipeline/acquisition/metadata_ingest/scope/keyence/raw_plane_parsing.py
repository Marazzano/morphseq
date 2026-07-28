"""Keyence raw-plane filename grammar — the ONE place the BZ-X TIFF naming is parsed.

Keyence raw data on disk is **per-Z-plane, per-channel, per-tile TIFFs** named
``...XY##_NNNNN_Z###_CH#.tif`` (e.g. ``embryo__XY16_00003_Z001_CH1.tif``). The address of a plane —
``(well, tile, time, z, channel)`` — lives entirely in the path + filename; unlike YX1 there is no
single tensor file whose axes encode it.

This module owns that grammar so the acquisition inventory (the system of record) and the legacy
``stitched_index/materialize_stitched_images.py`` materializer share **one parser** during the
strangler overlap. The functions were lifted verbatim from the legacy materializer; the only
addition is ``_parse_keyence_time_z_channel`` which ALSO returns ``channel_index`` (the legacy
``_parse_keyence_time_and_z`` matched ``_CH\\d+`` but discarded the number and the materializer
hardcoded channel 0 — fine for its single-channel use, wrong for a faithful inventory).

Import direction: this module imports stdlib only. It MUST NOT import stages, Snakemake/tasks,
stitch, or scope-inventory logic — those import IT.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_WELL_MARKER_RE = re.compile(r"_([A-H](?:0[1-9]|1[0-2]))$", flags=re.IGNORECASE)


def _well_from_w_index(raw: int) -> str:
    row = (raw - 1) // 12
    col = (raw - 1) % 12 + 1
    return f"{chr(65 + row)}{col:02d}"


def _normalize_well_marker(value: str) -> str:
    marker = str(value).strip().upper()
    if not re.fullmatch(r"[A-H](?:0[1-9]|1[0-2])", marker):
        raise ValueError(f"Invalid Keyence well marker {value!r}; expected A01-H12.")
    return marker


def _keyence_xy_position_dir(path: Path) -> Path | None:
    for idx, part in enumerate(path.parts):
        if re.fullmatch(r"XY\d+", part, flags=re.IGNORECASE):
            return Path(*path.parts[: idx + 1])
    return None


def _parse_keyence_xy_position_index(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"XY(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    # Legacy W###/P#####/T#### layout (2023 experiments have no XY## dir): the well directory
    # W### is the acquisition position — one capture location per well; the P##### tile within it
    # is parsed separately by _extract_keyence_well_and_tile. Mirror the W-handling that the
    # well/tile parser and _discover_keyence_wells already do, so the position index resolves too.
    for part in path.parts:
        match = re.fullmatch(r"W0?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


@lru_cache(maxsize=None)
def _read_keyence_well_marker(position_dir: Path) -> str:
    """Read the explicit Keyence well marker under an ``XY##`` directory.

    Newer BZ-X plate-map exports observed in production write a zero-byte marker
    file named like ``_A01`` inside each ``XY##`` position directory. That marker
    is the authoritative well label; ``XY##`` is only the acquisition position.

    Memoized: the answer is per-directory but this is consulted once per plane
    (thousands of times over NFS). ``position_dir`` is a hashable ``Path`` and the
    marker file is immutable within a run, so caching collapses the repeated
    ``iterdir()`` scans to one per ``XY##`` directory.
    """
    markers = []
    for child in Path(position_dir).iterdir():
        match = _WELL_MARKER_RE.fullmatch(child.name)
        if match:
            markers.append(_normalize_well_marker(match.group(1)))

    if len(markers) == 1:
        return markers[0]
    if not markers:
        raise ValueError(
            f"No Keyence well marker found in {position_dir}. Expected exactly one marker file "
            "named like '_A01'. Refusing to infer plate wells from XY capture order."
        )
    raise ValueError(
        f"Multiple Keyence well markers found in {position_dir}: {sorted(markers)}. "
        "Expected exactly one marker file named like '_A01'."
    )


def _extract_keyence_well_and_tile(path: Path) -> tuple[str | None, int]:
    tile_id: int | None = None

    for part in path.parts:
        p_match = re.fullmatch(r"P(\d+)", part, flags=re.IGNORECASE)
        if p_match:
            tile_id = int(p_match.group(1))
            break

    for part in path.parts:
        xy_match = re.fullmatch(r"XY(\d+)([A-Za-z]?)", part, flags=re.IGNORECASE)
        if xy_match:
            xy_raw = int(xy_match.group(1))
            suffix = xy_match.group(2)
            if suffix:
                well_index = f"{suffix.upper()}{xy_raw:02d}"
                if tile_id is None:
                    tile_id = max(ord(suffix.lower()) - 96, 1)
            else:
                position_dir = _keyence_xy_position_dir(path)
                if position_dir is None:
                    return None, tile_id or 1
                well_index = _read_keyence_well_marker(position_dir)
                if tile_id is None:
                    # Legacy XY layout without P*/T* encodes sub-position in filename,
                    # e.g. embryo__XY16_00003_Z001_CH1.tif -> tile 3 at time 0.
                    legacy_token = part[-4:]
                    if legacy_token in path.name:
                        suffix_str = path.name.split(legacy_token, 1)[1]
                        tile_match = re.match(r"_(\d+)", suffix_str)
                        if tile_match:
                            tile_id = int(tile_match.group(1))
            return well_index, tile_id or 1

    for part in path.parts:
        w_match = re.fullmatch(r"W0?(\d+)", part, flags=re.IGNORECASE)
        if w_match:
            return _well_from_w_index(int(w_match.group(1))), tile_id or 1

    name_match = re.search(r"([A-H](?:0[1-9]|1[0-2]))", path.name)
    if name_match:
        return name_match.group(1), tile_id or 1

    return None, tile_id or 1


def _parse_keyence_time_z_channel(path: Path) -> tuple[int, int, int] | None:
    """Parse ``(time_index, z_index, channel_index)`` from a Keyence plane filename.

    The faithful parser the acquisition inventory uses. ``channel_index`` is the integer in the
    ``_CH#`` token (1-based as written on disk; recorded as-is — it is part of the raw cell key, so
    keeping the on-disk value avoids inventing a re-basing the rest of the system would have to undo).
    Returns ``None`` when the filename is not a Keyence ``_Z###_CH#`` plane (caller skips it).
    """
    zc_match = re.search(r"_Z(\d+)_CH(\d+)", path.name, flags=re.IGNORECASE)
    if not zc_match:
        return None

    time_index = 0
    # Prefer directory timepoint (legacy Keyence layout: .../T0034/...).
    for part in path.parts:
        t_match = re.fullmatch(r"T(\d+)", part, flags=re.IGNORECASE)
        if t_match:
            time_index = max(int(t_match.group(1)) - 1, 0)
            break

    # Fallback for layouts that encode explicit T in filename.
    if time_index == 0:
        t_name_match = re.search(r"_T(\d+)_Z\d+_CH\d+", path.name, flags=re.IGNORECASE)
        if t_name_match:
            time_index = max(int(t_name_match.group(1)) - 1, 0)

    z_index = int(zc_match.group(1))
    channel_index = int(zc_match.group(2))
    return time_index, z_index, channel_index


def _parse_keyence_time_and_z(path: Path) -> tuple[int, int] | None:
    """Legacy 2-tuple ``(time_index, z_index)`` parser — kept byte-compatible for the materializer.

    Thin wrapper over ``_parse_keyence_time_z_channel`` that drops the channel index, preserving the
    exact return shape the legacy ``materialize_stitched_images`` + ``test_keyence_parsing_semantics``
    depend on. New code should call ``_parse_keyence_time_z_channel`` instead.
    """
    parsed = _parse_keyence_time_z_channel(path)
    if parsed is None:
        return None
    time_index, z_index, _channel_index = parsed
    return time_index, z_index


def _infer_keyence_stack_lookup(raw_images_dir: Path) -> dict[tuple[str, int], dict[int, list[Path]]]:
    lookup: dict[tuple[str, int], dict[int, list[tuple[int, Path]]]] = {}
    for path in raw_images_dir.rglob("*CH*.tif"):
        well_index, tile_id = _extract_keyence_well_and_tile(path)
        if well_index is None:
            continue

        parsed = _parse_keyence_time_and_z(path)
        if parsed is None:
            continue
        time_index, z_index = parsed
        key = (well_index, time_index)
        lookup.setdefault(key, {}).setdefault(tile_id, []).append((z_index, path))

    out: dict[tuple[str, int], dict[int, list[Path]]] = {}
    for key, tile_dict in lookup.items():
        out[key] = {}
        for tile_id, z_pairs in tile_dict.items():
            out[key][tile_id] = [path for _, path in sorted(z_pairs, key=lambda pair: pair[0])]
    return out
