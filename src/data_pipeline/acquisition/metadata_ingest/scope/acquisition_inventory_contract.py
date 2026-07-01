"""Scope-neutral acquisition-inventory contract — the SHARED core schema.

The acquisition inventory is each scope's record of what the microscope ACQUIRED. Its schema is two
tiers (see ``specs/acquisition_inventory_schema_policy.md``):

  - **Tier 1 — SHARED, HARD-CHECKED.** Columns whose meaning is identical across scopes, declared HERE
    once. Every scope's acquisition inventory MUST carry these; the scope validator hard-checks them.
  - **Tier 2 — SCOPE-SPECIFIC, SOFT.** Whatever a scope happens to know (e.g. YX1 tensor axes, Keyence
    exposure/gain). Declared in the per-scope contract, ALLOWED but not required, not forced downstream.

A scope's full column tuple is ``REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS + (scope-specific…)``.
This module owns ONLY the shared core — it imports no scope code and names no microscope.

Nuances baked into the core choice:
  - ``elapsed_time_s`` is the CONVERGED downstream time unit and is the ONLY time column in core — we
    do not carry two ways of looking at time. The RAW per-frame timestamp it is derived from is
    scope-specific (YX1 ``acquisition_time_s``; Keyence ``experiment_time_s`` / interval) and lives in
    the per-scope Tier-2 set (kept for audit/re-derivation), NOT here. See the time-atom policy.
  - ``channel_id`` is the normalized token (the converged name used by ``frame_inventory`` and all
    downstream); ``raw_channel_name`` is its provenance. Both are core.
  - ``well_index`` is NOT core: at ingest there is no well yet (only a raw position). The well label is
    attached later by ``apply_position_to_well_mapping``. The core is the schema AS ACQUIRED.
"""

from __future__ import annotations

# The shared Tier-1 block — identical meaning across scopes; hard-checked by every scope validator.
REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS: tuple[str, ...] = (
    "experiment_id",            # global experiment id
    "position_index",           # acquisition position (well attached later by mapping)
    "channel_id",               # normalized channel token (BF/GFP/RFP) — converged downstream name
    "raw_channel_name",         # raw scope channel string (provenance for the normalization)
    "time_index",               # T axis, 0-based contiguous
    "elapsed_time_s",           # the ONE converged time unit — seconds since the position's 1st frame
    "micrometers_per_pixel",    # calibration µm/px — validated > 0
    "image_width_px",
    "image_height_px",
    "microscope_id",            # which scope produced this row
)
