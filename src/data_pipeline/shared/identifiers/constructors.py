"""Canonical identifier constructors.

These functions MINT identifiers. They are dumb: they perform no biology
inference and no name mapping (e.g. they do not map ``Brightfield -> BF``).
Normalization of local tokens (``channel_id`` from ``raw_channel_name``) happens
upstream in metadata ingest; constructors only assemble the canonical string.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.

Canonical model (the sign on the door):
    experiment_id = 20240418            global experiment id
    well_index    = A01                 LOCAL well label, unique within an experiment
    well_id       = 20240418_A01        GLOBAL well id = {experiment_id}_{well_index}
    channel_id    = BF                  local channel token
    image_id      = {well_id}_{channel_id}_t{time_int:04d}
    embryo_id     = {well_id}_e{local_embryo_index:02d}
    snip_id       = {embryo_id}_t{time_int:04d}

Compositional grammar — every id is ``parent_id + local_token``. ``build_well_id``
is the ONE join point where ``experiment_id`` and ``well_index`` meet; every id
downstream is ``well_id``-first and inherits its global, sanitized prefix. See
target/well_id_throughline_refactor_plan.md and
target/front_end_naming_and_flow.md (Decision 7: well_id is the canonical key
everywhere after the fan; well_index survives only as a scope-table column).
"""

from __future__ import annotations

import re


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SEP_RE = re.compile(r"[_-]{2,}")


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


def build_image_id(well_id: str, channel_id: str, time_int: int) -> str:
    """Return the canonical image id for one channel at one timepoint."""
    return f"{str(well_id)}_{str(channel_id)}_t{int(time_int):04d}"


def build_embryo_id(well_id: str, local_track_id: int) -> str:
    """Return the canonical embryo id for one tracked embryo within one well."""
    return f"{str(well_id)}_e{int(local_track_id):02d}"


def build_snip_id(embryo_id: str, time_int: int) -> str:
    """Return the canonical snip id for one embryo at one timepoint."""
    return f"{str(embryo_id)}_t{int(time_int):04d}"
