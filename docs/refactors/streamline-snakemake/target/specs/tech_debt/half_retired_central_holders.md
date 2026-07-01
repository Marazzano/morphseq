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

### STILL OUTSTANDING (not import-forced — retire deliberately, not mid-sweep)

**#4 — `schemas/quality_control.py` (legacy `qc_flags` vocabulary).**
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

**#3 remainder — the rest of `feature_extraction/core/`.**
- `mask_geometry.py`, `pose_kinematics.py`, `fraction_alive.py`, `stage_inference.py`,
  `consolidate_features.py` still live under `core/`, re-exported through
  `feature_extraction/__init__.py` (a legacy public-API facade with **zero live importers**).
- Only curvature was migrated this session (it was the one with a live consumer:
  `curvature_metrics/compute.py`). Each remaining family should move to its product folder the same
  way, then `core/` and the `__init__` facade get deleted together. Verify each `extract_*_batch`
  disk-path function is uncalled before dropping it (curvature's was).

---

## Where to look when paying the outstanding debt

- Legacy QC schema: `schemas/quality_control.py` (+ its `analysis_ready` chain).
- Live QC contract to converge on: `quality_control/snip_qc/contract.py`.
- Legacy feature layer: `feature_extraction/core/` + `feature_extraction/__init__.py` re-export facade.
- The doctrine being enforced: `feature_world.md` (Legacy Domain Retirement; Config/Stage-table
  patterns; "one concept, built in exactly one place").

## How to catch the next one before it lands

- Grep for a flat `X.py` sitting next to a same-named `X/` package — that is a shadow (silent
  ImportError waiting to happen). Import `X` and check `X.__file__` resolves to the package, not the
  file, and that every name callers pull from `X` is re-exported by the package.
- Grep for the same constant/threshold/flag list defined in more than one module. The co-located
  home is authoritative; the other is drift.
- When adding a product folder, confirm the corresponding central-holder entry is deleted in the same
  change (the `feature_world.md` "retirement is part of done" rule).
