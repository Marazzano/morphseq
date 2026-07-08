# Tech Debt: Half-Retired Central Holders (co-location refactor left the old home behind)

**Status:** known pattern + tracked inventory, mdcolon 2026-07-01. Some items paid down this
session; the rest are named here so the retirement is one deliberate list, not a scavenger hunt.

---

## The root-cause pattern (the one smell behind all of these)

The refactor moved from **central holders** to **co-location**: a contract/config/schema now lives
next to where it is minted or used, not in a shared `schemas/` / `config.py` bucket. `feature_world.md`
states the doctrine directly — each product owns its `contract.py` / `config.py`; spine columns are
imported from their minting site, never re-declared; and *"retirement is part of the definition of
done for each product — it is not a separate cleanup pass."*

Every "snake" found this session is the **same fracture, one displacement**: the co-located home got
built, but the old central holder was **not deleted in the same change**. That always yields one of
two failures:

1. **Two sources of truth** — the co-located home and the surviving central one drift (e.g. a
   threshold edited in the visible central file that nothing actually reads).
2. **A live import hazard** — worse, when the old holder is a flat `module.py` sitting beside a new
   same-named `module/` **package**. Python silently imports the *package*; any name still living only
   in the shadowed `.py` becomes an **ImportError** the moment someone imports it by that path.

> **The rule (hunt for this at review time):** when a concept moves from a shared holder to a
> co-located home, **delete the shared holder in the same change.** A co-located contract plus a
> surviving central one is two sources of truth; if the central one shares a name with a new package,
> it is also a live import hazard.

---

## Inventory

### PAID DOWN this session

| Old central holder | Co-located home | Fix |
|---|---|---|
| `quality_control/config.py` (`QC_DEFAULTS`/`DEFAULT_QC_CONFIG`) | each product's `config.py` (`DEATH_DETECTION_DEFAULTS`, `MOTION_BLUR_QC_DEFAULTS`, …) | **deleted**; consumers repointed to `death_detection/config.py`; fixed a latent `QC_DEFAULTS['dead_lead_time_hours']` KeyError (key never existed) → `['lead_time_hr']` (commit `eef64557`) |
| `quality_control/death_detection.py` (flat module) | `death_detection/` package | **deleted** — was already unreachable (package shadowed it) (commit `eef64557`) |
| `segmentation/backends.py` (flat module) | `backends/` package | **moved** into `backends/selection.py` + re-exported; this was a **live ImportError** at `segmentation_and_tracking.py:13` (commit `5bb3462b`) |
| `feature_extraction/core/curvature_metrics.py` + `curvature_skeletonization.py` | `curvature_metrics/skeletonization.py` (product-local) | **migrated**; dropped uncalled `extract_curvature_metrics_batch` + dead re-exports (commit `5bb3462b`) |
| empty husk dirs: `quality_control/core/`, `quality_control/entrypoints/`, `quality_control/segmentation_qc/`, `quality_control/auxiliary_mask_qc/` | n/a (source migrated elsewhere) | **removed** (source-empty, unimported) |
| `schemas/stage_predictions.py` (`REQUIRED_COLUMNS_STAGE_PREDICTIONS`, `UNIQUE_KEY_STAGE_PREDICTIONS`) | `feature_extraction/stage_predictions/contract.py` (`STAGE_PREDICTION_TABLE_COLUMNS`) | **deleted** — 0 importers; legacy vocab (`time_int`, `pipeline_version`) superseded by co-located contract |
| `schemas/auxiliary_masks.py` (`REQUIRED_COLUMNS_AUXILIARY_MASKS`) | `segmentation/backends/unet_snip/snip_auxiliary_masks_contract.py` (`SNIP_AUXILIARY_MASKS_REQUIRED_COLUMNS`) | **deleted** — 0 importers; legacy full-frame vocab superseded by per-snip contract |

### PAID DOWN 2026-07-01 (the off-DAG island — #3 remainder, #4, #5)

Items #3 (remainder), #4, and #5 below were **one connected off-DAG island**, retired together
(commits `fc5d5f83`…`cfef9bc8`). The live DAG never imported any of it. Summary of what changed:

- `schemas/{features,quality_control,analysis_ready,frame_contract}.py` — **deleted**. `schemas/`
  now holds only `channel_normalization` (the doctrine-correct keeper).
- `analysis_ready/` legacy subsystem — **deleted**, replaced by a stub `__init__.py` that imports
  spine + snip_qc payload from their mint sites (re-declares nothing). Optional future product;
  `snip_qc` stays the through-line terminal.
- `feature_extraction/core/` + the 0-importer `__init__.py` facade + the dead `feature_extraction/io/`
  dir + `consolidate_features.py` — **deleted** (every feature family already has a live product
  folder + per-well DAG rule).
- `metadata_ingest/frame_contract/` + `load_frame_contract` + the commented Snakefile build rule —
  **deleted** (superseded by per-well `frame_inventory`).
- `quality_control/io/` + `quality_control/validators.py` — **deleted** (fed only the retired chain;
  zero importers).
- **Also removed: the `consolidated_features` product** (folder + 3 rules + path-registry entry +
  tasks.py commands) — a redundant no-op join with zero live readers. The `features` schema was
  never a live reconciliation blocker; all its importers were parked. See
  `schemas_retirement_plan.md` Tier 3 for the full reasoning.

The full detail of each below is retained for history; treat #3-remainder / #4 / #5 as PAID.

---

**#5 — `schemas/` is itself a surviving central holder (the biggest one).** — **PAID (see above).**
- The two deletions above were the *dead* members. The rest of `schemas/` is a shared bucket of
  `REQUIRED_COLUMNS_*` contracts — a direct violation of the `feature_world.md` doctrine that each
  product owns its own `contract.py`. `schemas/__init__.py` is inert (no re-exports), so members
  retire one file at a time.
- **Live but mis-homed** (co-locate into the product's `contract.py` the way curvature moved, *then*
  delete the schema member): `schemas.segmentation` (8 importers), `schemas.channel_normalization` (8),
  `schemas.features` (4), `schemas.frame_contract` (3), `schemas.plate_metadata` (3),
  `schemas.snip_processing` (2), `schemas.scope_metadata` (2), `schemas.stitched_image_index` (1).
- **Parked** (retire with the analysis_ready unit — see #4): `schemas.quality_control` (3),
  `schemas.analysis_ready` (5). Every importer of these two lives inside the parked chain.
- **To pay down:** treat `schemas/` as the last central holder. Migrate the live members product by
  product; delete the parked members with the analysis_ready subsystem; the goal is an empty `schemas/`.

**#4 — `schemas/quality_control.py` (legacy `qc_flags` vocabulary).** — **PAID (see above).**
- Co-located replacement exists: `quality_control/snip_qc/contract.py::SNIP_QC_EXCLUSION_FLAGS`. The
  live `snip_qc` does **not** read the schema module.
- The old schema (`SNIP_EXCLUSION_FLAGS`, `REQUIRED_COLUMNS_QC`, `QC_OUTPUT_COLUMNS`) carries an
  **incompatible** vocabulary (`viability_flag`, `dead_flag`, `motion_flag`,
  `death_inflection_time_int`) — it is NOT a drop-in for the new contract.
- It still feeds a **parked** chain: `quality_control/validators.py`, `quality_control/io/{loaders,writers}.py`,
  `schemas/analysis_ready.py` → `analysis_ready/*`. **None of this is wired into the live DAG**
  (only a stray `ANALYSIS_READY_DIR` var + an "out of scope" comment in the Snakefile).
- **To pay down:** retire the analysis_ready / qc_flags legacy subsystem as one unit, then delete
  `schemas/quality_control.py`. An inline retirement note is on the file.

**#3 remainder — the rest of `feature_extraction/core/`.** — **PAID (see above): `core/` +
`consolidate_features.py` + the `__init__.py` facade + the dead `io/` dir all deleted; each feature
family already had a live product folder.**
- `mask_geometry.py`, `pose_kinematics.py`, `fraction_alive.py`, `stage_inference.py`,
  `consolidate_features.py` still live under `core/`, re-exported through
  `feature_extraction/__init__.py` (a legacy public-API facade with **zero live importers**).
- Only curvature was migrated this session (it was the one with a live consumer:
  `curvature_metrics/compute.py`). Each remaining family should move to its product folder the same
  way, then `core/` and the `__init__` facade get deleted together. Verify each `extract_*_batch`
  disk-path function is uncalled before dropping it (curvature's was).

---

## Where to look when paying the outstanding debt

- The last central holder: `schemas/` (drive it toward empty — see #5). Legacy QC schema
  `schemas/quality_control.py` (+ its `analysis_ready` chain) is the parked slice.
- Live QC contract to converge on: `quality_control/snip_qc/contract.py`.
- Legacy feature layer: `feature_extraction/core/` + `feature_extraction/__init__.py` re-export facade.
- The doctrine being enforced: `feature_world.md` (Legacy Domain Retirement; Config/Stage-table
  patterns; "one concept, built in exactly one place").

## How to catch the next one before it lands

- Grep for a flat `X.py` sitting next to a same-named `X/` package — that is a shadow (silent
  ImportError waiting to happen). Import `X` and check `X.__file__` resolves to the package, not the
  file, and that every name callers pull from `X` is re-exported by the package.
- **Higher yield:** grep for a `schemas/<product>.py` (or any shared-holder module) when a co-located
  `<product>/…/contract.py` already exists — that pairing *is* the two-sources-of-truth smell, and it
  is not import-forced so nothing crashes to flag it. Count importers of the schema member: zero →
  delete now; nonzero → migrate the live consumers onto the co-located contract, then delete. The two
  `schemas/{stage_predictions,auxiliary_masks}.py` deletions were exactly this (0 importers, contract
  already present).
- Grep for the same constant/threshold/flag list defined in more than one module. The co-located
  home is authoritative; the other is drift.
- When adding a product folder, confirm the corresponding central-holder entry is deleted in the same
  change (the `feature_world.md` "retirement is part of done" rule).
