# Retiring `schemas/` — the last central holder

**Status:** active migration, mdcolon 2026-07-01. Companion to
[`half_retired_central_holders.md`](half_retired_central_holders.md) item **#5**. That doc explains
*why* (`feature_world.md` doctrine: each product owns its `contract.py`; no shared schema bucket).
This doc is the *how* — the per-member plan and the tier order, so the retirement is one deliberate
list and not a scavenger hunt.

**Goal state:** `schemas/` is **empty and deleted**. Every `REQUIRED_COLUMNS_*` / vocabulary
constant lives in the product folder that mints the artifact it describes, in that product's
`contract.py` (or `<product>_contract.py`, matching the folder's existing convention).

---

## The one move, applied uniformly

For each member the mechanical shape is identical (proven on `plate_metadata`,
`stage_predictions`, `auxiliary_masks`):

1. Confirm/locate the **co-located home** — the product folder that owns the artifact.
2. If a home contract already carries the constant → **repoint importers** at it.
   If not → **move the constant verbatim** into the product's `contract.py`, then repoint.
3. Repoint every live importer (grep `schemas.<member>`), aliasing the name if the co-located
   constant is spelled differently (e.g. `REQUIRED_PLATE_METADATA_COLUMNS as REQUIRED_COLUMNS_PLATE_METADATA`).
4. `git rm schemas/<member>.py`.
5. Import-check the touched modules. Repeat per member; commit per tier.

`schemas/__init__.py` is inert (no re-exports), so members retire one file at a time with no
package-level coupling.

---

## Tier 0 — DONE

| Member | Home | Note |
|---|---|---|
| `stage_predictions` | `feature_extraction/stage_predictions/contract.py` | deleted (0 importers) — commit `49cdc2dd` |
| `auxiliary_masks` | `segmentation/backends/unet_snip/snip_auxiliary_masks_contract.py` | deleted (0 importers) — commit `49cdc2dd` |
| `plate_metadata` | `metadata_ingest/plate/plate_metadata_contract.py` | shim deleted; 2 importers repointed |

## Tier 1 — clean move + repoint (mechanical, low risk) — DONE

No semantic reconciliation; the home folder exists and the constant has one authoritative spelling.
All three migrated + verified live-in-DAG + tests green (42 passed).

| Member | Importers | Home |
|---|---|---|
| `scope_metadata` | 2 (scope/keyence + scope/yx1 extractors) | **new** `metadata_ingest/scope/scope_metadata_contract.py` (NOT merged into `acquisition_inventory_contract.py` — distinct artifact: raw per-series scope metadata vs. tiered acquisition inventory) |
| `stitched_image_index` | 1 (metadata_ingest/stitched_index/validate) | **new** `metadata_ingest/stitched_index/contract.py` (also dropped a duplicated `time_int` entry in the legacy list) |
| `snip_processing` | 2 (snip_processing/io, feature_extraction/io) | **new** `snip_processing/contract.py` |

> **`frame_contract` is NOT here — it is deprecated-cluster debt, not a clean move.** Its build rule
> is commented out in the Snakefile (`# DEPRECATED: build_frame_contract is superseded by the
> per-well frame_inventory rules`). The live product is `frame_inventory` (`frame_inventory_contract.py`,
> ~14 rules). `REQUIRED_COLUMNS_FRAME_CONTRACT` has **no live consumer**: `metadata_ingest/frame_contract/{build,validate}`
> is off-DAG, and the only other importer is the old `feature_extraction/io/loaders.py::load_frame_contract`,
> reachable only through the zero-importer `feature_extraction/__init__.py` facade + `core/` shims.
> `schemas/frame_contract.py` stays put; it retires **with its deprecated product** (see Tier 3, the
> `core/` + old-`consolidate_features.py` deprecation arc), not as a mechanical move. **Do not give a
> dead product a fresh co-located contract.**

## Tier 2 — `segmentation` (create home, migrate 8)

| Member | Importers | Home |
|---|---|---|
| `segmentation` | 8 (segmentation_and_tracking/normalizers ×5, csv_formatter, validate_seg_and_tracking, feature_extraction/io) | `segmentation_and_tracking/` has no `contract.py` — create `segmentation_and_tracking/contract.py`, move `REQUIRED_COLUMNS_SEGMENTATION_TRACKING`, repoint 8 |

Larger blast radius but still a pure move — the vocabulary has one spelling and one authoritative
home. Do it as its own commit so the 8-site repoint is reviewable in isolation.

## Tier 3 — parked cluster: retire as one unit (do NOT piecemeal-move)

`features`, `quality_control`, `analysis_ready` are **entangled**, not independent moves:

- **`analysis_ready` (4) + `quality_control` (3)** — every importer lives inside the parked
  `analysis_ready/` subsystem (`validators.py`, `io/{loaders,writers}.py`, `core/assemble.py`,
  `assemble_features_qc_embeddings.py`). Confirmed **not wired into the DAG** (Snakefile has only
  `ANALYSIS_READY_DIR` var + an "out of scope" comment; no rule). Retire the whole `analysis_ready/`
  tree together, deleting `schemas/{analysis_ready,quality_control}.py` with it.
- **`features` (4)** — `schemas/features.py` carries a flat legacy `REQUIRED_COLUMNS_FEATURES`.
  A modern co-located contract (`feature_extraction/consolidated_features/contract.py`) already
  exists but uses a **different** spine-based vocabulary (`CONSOLIDATED_FEATURES_TABLE_COLUMNS`),
  so this is a *reconciliation*, not a move. Two of the 4 importers are the parked QC/analysis_ready
  loaders; the live ones are `feature_extraction/io/writers.py` + `consolidate_features.py`.
  **Decision (mdcolon):** feature outputs should flow *through* analysis_ready rather than force the
  flat list onto the spine contract. So `features` retirement is **coupled to the analysis_ready
  redesign** — resolve it when that subsystem is rebuilt/retired, not before.

  > Also part of this cluster: `feature_extraction/core/` remainder (`mask_geometry`,
  > `pose_kinematics`, `fraction_alive`, `stage_inference`, `consolidate_features`) + the
  > `feature_extraction/__init__.py` re-export facade (0 external importers) — see
  > `half_retired_central_holders.md` #3. Migrate those into their product folders in the same arc.

- **`frame_contract` (schema + product) — deprecated, retire whole.** `schemas/frame_contract.py`,
  `metadata_ingest/frame_contract/{build,validate}_frame_contract.py`, and the old
  `feature_extraction/io/loaders.py::load_frame_contract` all belong to the superseded whole-experiment
  frame_contract path (replaced by per-well `frame_inventory`). None is on the live DAG. Delete the
  cluster together; the old `consolidate_features.py` path that calls `load_frame_contract` retires
  with the `core/`/facade work above.

## Keep (already doctrine-correct — do not move)

| Member | Why it stays |
|---|---|
| `channel_normalization` | This *is* the co-located home — it owns the canonical channel **vocabulary** (a cross-scope language, not a per-product artifact schema). Doctrine (`acquisition_inventory_schema_policy.md`): *vocabularies define language, validators guard contracts.* Per-scope dialect maps already live with their scope. Optional cosmetic relocation only; no importer change. |

---

## Do-not-touch (verified live, easy to mistake for debt)

- **`image_materialization/frame_inventory_contract.py`** — NOT out of spec. 14 live importers across
  detection, image_materialization, metadata_ingest. It is a central *but correctly-placed* contract
  for the materialized-frame inventory. Leave it.

## After the last member

Delete `schemas/__init__.py` and the empty `schemas/` dir; grep the tree for any surviving
`data_pipeline.schemas` import (should be zero); update `half_retired_central_holders.md` #5 to PAID.
