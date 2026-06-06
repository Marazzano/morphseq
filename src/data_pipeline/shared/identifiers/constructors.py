"""Canonical identifier constructors.

These functions MINT identifiers. They are dumb: they perform no biology
inference and no name mapping (e.g. they do not map ``Brightfield -> BF``).
Normalization of local tokens (``channel_id`` from ``raw_channel_name``) happens
upstream in metadata ingest; constructors only assemble the canonical string.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.

NOTE (Scope 1, 2026-06-05): signatures here are the CURRENT (local-``well_id``)
forms, unchanged by the package split. ``well_id`` is the plate-LOCAL label
(``A01``) today. The TARGET refactor (Scope 2) flips ``build_well_id`` to
``(experiment_id, well)`` so ``well_id`` becomes global ``{experiment_id}_{well}``
and ``build_image_id``/``build_embryo_id`` become ``well_id``-first; see
target/well_id_throughline_refactor_plan.md.
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


def build_well_id(well_index: str) -> str:
    """Return the canonical plate-local well ID, e.g. A01."""
    return str(well_index).strip()


def build_image_id(experiment_id: str, well_id: str, channel_id: str, time_int: int) -> str:
    """Return the canonical image ID for one channel at one timepoint."""
    return f"{sanitize_experiment_id(experiment_id)}_{build_well_id(well_id)}_{str(channel_id)}_t{int(time_int):04d}"


def build_embryo_id(experiment_id: str, well_id: str, local_track_id: int) -> str:
    """Return the canonical embryo ID for one tracked embryo within one well."""
    return f"{sanitize_experiment_id(experiment_id)}_{build_well_id(well_id)}_e{int(local_track_id):02d}"


def build_snip_id(embryo_id: str, time_int: int) -> str:
    """Return the canonical snip ID for one embryo at one timepoint."""
    return f"{str(embryo_id)}_t{int(time_int):04d}"
