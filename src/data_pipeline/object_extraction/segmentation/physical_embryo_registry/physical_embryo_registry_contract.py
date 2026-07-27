"""`physical_embryo_registry` contract constants (schema only — no logic).

Grain: ONE row per ``physical_embryo_id`` (one row per animal). The registry is the
identity-origination boundary — the only place ``track_id → physical_embryo_id``
resolution happens (see ``physical_embryo_registry_world.md``).

MVP carries identity + minimal provenance + the merge-policy payload. The bright line
for any provenance/identity column: it must describe WHERE the animal came from
(provenance — allowed) and never WHETHER to trust the animal (judgment — that is the QC
table's job). A count column, if ever added for detection, must be ``n_detected_masks`` —
never ``n_valid_masks``.

The two PAYLOAD columns (``merge_policy``, ``n_sources``) earn their place by auditing a
non-obvious identity decision IN-PLACE: when a well merges several snapshot acquisitions,
whether its ``physical_embryo_id``s were minted NORMAL / BRIDGED / FRACTURED (see
``EmbryoMergePolicy`` and EXPERIMENT_GROUP_PLATE_MODEL.md "physical_embryo_id merge
policy"). ``n_sources`` is echoed from ``frame_inventory`` so the decision is legible
without re-joining upstream.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────────────
# Controlled vocabulary — the merge policy, defined ONCE
# ─────────────────────────────────────────────────────────────────────────────────────
class EmbryoMergePolicy(str, Enum):
    """How a well's ``physical_embryo_id``s were minted from its tracks.

    Driven by ``n_sources`` (raw acquisitions merged into the well, from frame_inventory)
    and ``n_tracks`` (distinct ``track_id`` in the well):

    * NORMAL   (``n_sources == 1``): one physical_embryo_id per track — today's behavior.
    * BRIDGE   (``n_sources > 1`` AND ``n_tracks == 1``): ONE physical_embryo_id spanning
               the well's timepoints — the only correspondence possible, like a timelapse.
    * FRACTURE (``n_sources > 1`` AND ``n_tracks > 1``): which-maps-to-which is ambiguous;
               do NOT guess — disjoint ``_e`` blocks per source, every animal a distinct id.

    The enum *value* (``"normal"`` / ``"bridged"`` / ``"fractured"``) is what lands in the
    ``merge_policy`` column.
    """

    NORMAL = "normal"
    BRIDGE = "bridged"
    FRACTURE = "fractured"


# The three legal ``merge_policy`` cell values (validator membership check, defined once).
MERGE_POLICY_VALUES: frozenset[str] = frozenset(p.value for p in EmbryoMergePolicy)


# ─────────────────────────────────────────────────────────────────────────────────────
# Contract — column families
# ─────────────────────────────────────────────────────────────────────────────────────
# Identity (the promise this artifact makes) — minted here, imported downstream.
PHYSICAL_EMBRYO_REGISTRY_SPINE_COLUMNS: tuple[str, ...] = (
    "physical_embryo_id",   # primary key — the animal; build_physical_embryo_id(well_id, local_embryo_index)
    "experiment_id",
    "well_id",
    "local_embryo_index",   # one-based integer >= 1 (well-scoped)
)

# Provenance (where this animal came from — provenance, NOT judgment).
PHYSICAL_EMBRYO_REGISTRY_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "track_id",             # the tracking identity this entity was resolved from
    "track_id_source",      # how track identity was established (e.g. "frame_masks")
)

# Payload (owned by this product — audits the merge decision made at the mint site).
PHYSICAL_EMBRYO_REGISTRY_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "merge_policy",         # EmbryoMergePolicy value: "normal" / "bridged" / "fractured"
    "n_sources",            # int >= 1, raw acquisitions merged into the well (echoed from frame_inventory)
)

PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS: tuple[str, ...] = (
    PHYSICAL_EMBRYO_REGISTRY_SPINE_COLUMNS
    + PHYSICAL_EMBRYO_REGISTRY_PROVENANCE_COLUMNS
    + PHYSICAL_EMBRYO_REGISTRY_PAYLOAD_COLUMNS
)

PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY: tuple[str, ...] = ("physical_embryo_id",)


def empty_physical_embryo_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)
