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

## Tier 2 — `segmentation` (create home, migrate 8) — DONE

| Member | Importers | Home |
|---|---|---|
| `segmentation` | 8 (segmentation_and_tracking/normalizers ×5, csv_formatter, validate_seg_and_tracking, feature_extraction/io) | **moved** `schemas/segmentation.py` → `segmentation_and_tracking/contract.py` (holds all 6 sub-contracts: segmentation_tracking, frame_detections, seed_selection, track_instances, mask_rle, V2). 8 importers repointed. |

Pure move — one spelling, one home. Verified behavior-neutral: import-checked all 9 touched
modules; the Phase-3 normalizer/ingestor suite shows the **same 5 failures before and after** the
change (pre-existing `video_id`/`SeedSelection` drift, unrelated to schemas — flag separately).

Gotcha logged: `csv_formatter.py` used a relative `...schemas.segmentation`; the naive
`..segmentation_and_tracking.contract` rewrite resolved to `segmentation.segmentation_and_tracking`
(wrong parent). Switched to an absolute `data_pipeline.segmentation_and_tracking.contract` import.
When repointing **relative** imports across sibling packages, prefer the absolute path.

## Tier 3 — DONE (retired as one off-DAG island, mdcolon 2026-07-01)

The four remaining members (`features`, `quality_control`, `analysis_ready`, `frame_contract`) were
NOT four independent problems — they were **one connected off-DAG island**. The live DAG runs
exclusively through the per-product feature folders (`mask_geometry`, `fraction_alive`,
`pose_kinematics`, `curvature_metrics`, `stage_predictions`) and the QC products (`snip_qc`,
`death_detection`, …). Nothing on the DAG imported any of the four. Retired together in commits
`fc5d5f83`…`cfef9bc8`; through-line DAG dry-run + affected test subsets (374 tests) green.

**Key correction to the earlier framing:** `features` was believed to be a *live reconciliation
blocker* coupled to an "analysis_ready redesign." It was not. All 4 importers of
`REQUIRED_COLUMNS_FEATURES` were parked (`feature_extraction/io/writers.py` stamped it into an
unused sidecar; `consolidate_features.py` + the two QC/analysis_ready loaders were all off-DAG). The
live consolidation path (`consolidated_features/`) never imported it — and that path was itself
removed (see below).

- **`consolidated_features` removed entirely (redundant op).** First-principles finding: it minted
  nothing — a pure 1:1 join of feature shards that already share the spine — its "contract" only
  re-declared `mask_geometry`'s columns, and it had **zero live readers** (the QC products read the
  per-product shards directly; no rule consumed the merged table; not a through-line target). Deleted
  folder + 3 rules + path-registry entry + tasks.py commands.
- **`analysis_ready` + `quality_control`** — the whole off-DAG `analysis_ready/` chain
  (validators, io, core/assemble, entrypoints) + `quality_control/io/` + `quality_control/validators.py`
  deleted. `analysis_ready/` left as a **stub** `__init__.py` that imports spine + snip_qc payload
  from their mint sites (re-declaring nothing); its intended role is an optional downstream product
  (`snip_qc` stays the through-line terminal). Legacy `schemas/{analysis_ready,quality_control}.py`
  deleted with the chain.
- **`feature_extraction/core/` + facade** — `mask_geometry`, `pose_kinematics`, `fraction_alive`,
  `stage_inference`, `consolidate_features` under `core/` + the 0-importer `__init__.py` re-export
  facade + the dead `feature_extraction/io/` dir: all deleted (every family has a live product folder
  + per-well DAG rule). See `half_retired_central_holders.md` #3.
- **`frame_contract` (schema + product)** — `schemas/frame_contract.py`,
  `metadata_ingest/frame_contract/`, `load_frame_contract`, and the commented-out Snakefile build
  rule: all deleted (superseded by per-well `frame_inventory`).

## Keep (already doctrine-correct — do not move)

| Member | Why it stays |
|---|---|
| `channel_normalization` | This *is* the co-located home — it owns the canonical channel **vocabulary** (a cross-scope language, not a per-product artifact schema). Doctrine (`acquisition_inventory_schema_policy.md`): *vocabularies define language, validators guard contracts.* Per-scope dialect maps already live with their scope. Optional cosmetic relocation only; no importer change. |

---

## Do-not-touch (verified live, easy to mistake for debt)

- **`image_materialization/frame_inventory_contract.py`** — NOT out of spec. 14 live importers across
  detection, image_materialization, metadata_ingest. It is a central *but correctly-placed* contract
  for the materialized-frame inventory. Leave it.

## After the last member — DONE

The four island members are deleted; the tree has **zero** surviving
`data_pipeline.schemas.{features,quality_control,analysis_ready,frame_contract}` imports.
`schemas/` is NOT empty and NOT deleted — `channel_normalization` remains as the doctrine-correct
keeper (a cross-scope vocabulary, not a per-product artifact schema), so `schemas/__init__.py` + the
dir stay. Its docstring now says so and warns against re-adding per-product column contracts. A
future cosmetic pass may relocate `channel_normalization` and then drop the dir; that is out of scope.
`half_retired_central_holders.md` #5 → PAID.
