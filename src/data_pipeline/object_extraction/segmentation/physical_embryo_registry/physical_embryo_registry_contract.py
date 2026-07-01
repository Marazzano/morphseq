"""`physical_embryo_registry` contract constants (schema only — no logic).

Grain: ONE row per ``physical_embryo_id`` (one row per animal). The registry is the
identity-origination boundary — the only place ``track_id → physical_embryo_id``
resolution happens (see ``physical_embryo_registry_world.md``).

MVP carries identity + minimal provenance only. The bright line for any future
column: it must describe WHERE the animal came from (provenance — allowed) and never
WHETHER to trust the animal (judgment — that is the QC table's job). A count column,
if ever added, must be ``n_detected_masks`` — never ``n_valid_masks``.
"""

from __future__ import annotations

import pandas as pd


PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS: tuple[str, ...] = (
    # Identity (the promise this artifact makes)
    "physical_embryo_id",   # primary key — the animal; build_physical_embryo_id(well_id, local_embryo_index)
    "experiment_id",
    "well_id",
    "local_embryo_index",   # one-based integer >= 1 (well-scoped)
    # Provenance (where this animal came from — provenance, NOT judgment)
    "track_id",             # the tracking identity this entity was resolved from
    "track_id_source",      # how track identity was established (e.g. "frame_masks")
)

PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY: tuple[str, ...] = ("physical_embryo_id",)


def empty_physical_embryo_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)
