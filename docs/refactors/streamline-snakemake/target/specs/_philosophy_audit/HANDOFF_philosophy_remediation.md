# HANDOFF — Pipeline Philosophy Remediation (in progress)

**Date:** 2026-06-24. **Read first:** [`README.md`](README.md) (cross-zone synthesis) and the rubric
[`../pipeline_file_philosophy.md`](../pipeline_file_philosophy.md). Zone reports under [`zones/`](zones/).

This doc records what's been **fixed**, what's **left**, and the key decisions/discoveries so the next
session can pick up without re-deriving context.

---

## What this is

Five Sonnet agents audited `src/data_pipeline/` against `pipeline_file_philosophy.md`:
**16 BLOCKER / 35 MAJOR / 27 MINOR**. The one-line verdict: *the new recipe-conformant per-product
code is strong; nearly every BLOCKER lives in an un-retired legacy layer, several of which now actively
crash.* Remediation was scoped (with the user) to **P0 (runtime-breaking) + P1 (hard-constraint
violations in live code)**; legacy modules kept as drift benchmarks are **marked + guarded, not deleted**
(build04 still consumes them — see Decision 1).

---

## DONE this session (committed)

| Fix | File | What changed |
|---|---|---|
| P0 — QC crash | `quality_control/death_detection.py` | `from src.data_pipeline` → `data_pipeline`; flat `QC_DEFAULTS['…']` → nested `QC_DEFAULTS['death_detection']['persistence_threshold'\|'decline_rate_threshold'\|'lead_time_hr']`; added RETIRED header. |
| P0 — retire marker | `quality_control/surface_area_outlier_detection.py` | RETIRED header (forked `validate_sa_reference` noted; kept live for build04 + drift gate). |
| P0 — raise-on-call tripwire | `feature_extraction/fraction_alive/compute.py` | dropped `resolve_data_root_relative_path` (now raises); resolve mask path directly against `output_root`. |
| P0 — import-time tripwire | `feature_extraction/mask_geometry_metrics.py` | removed `require_existing_path` import (warned at import; would crash the dead legacy batch fn); replaced its body with direct fail-loud path check. |
| P0 — cross-boundary import | `image_building/scope/keyence/stitched_ff_builder.py` + `image_building/utils/frame_tiler.py` | `from src.build.export_utils import trim_to_shape` → promoted local `_trim_to_shape` to public `trim_to_shape`, import from `frame_tiler`. |
| P1 — inline id split (Hard Constraint 1) | `segmentation/backends/unet_snip/run_unet_snip.py` | `snip_id.split("_")[1]` (yielded LOCAL well slug, wrong grain) → pass GLOBAL `well_id` from the snip row into `_mask_output_path`. |
| P1 — inline id mint (Hard Constraint 1) | `segmentation/video_generation/video_generator.py` | `generate_image_id` literal `f"{video_id}_t{…}"` → delegates to `build_image_id` (drops the deprecated `_t`-literal mint). |

**Deleted (untracked dead dirs):** `feature_extraction/entrypoints/` and `feature_extraction/pipelines/`
(zero importers; `pipelines/` was the raw-path-string entrypoints = FE B3/B4).

**Verification:** all changed modules import clean (no tripwire warnings); targeted suites
`feature_extraction/{fraction_alive,mask_geometry,pose_kinematics}` + `segmentation/` = **159 passed**,
0 failures. (The new `generate_image_id` round-trips: `('20250912_B01_BF', 7)` → `20250912_B01_BF_t0007`.)

---

## KEY DECISIONS

1. **Don't delete the legacy QC drift-benchmark modules yet.** `src/build/build04_perform_embryo_qc.py`
   still *calls* `compute_dead_flag2_persistence` (death_detection.py) and `compute_sa_outlier_flag`
   (surface_area_outlier_detection.py) — and build04 is the producer of the legacy ground-truth the
   feature-extraction legacy-drift gate compares against. The spec's own retirement clause says keep
   them live until build04 is migrated **and** the drift gate signs off. So: **mark + fix-the-crash,
   don't delete.** (User chose "delete" initially; reversed to "mark + guard" once the live build04
   consumer was found.)

2. **Two audit findings were FALSE POSITIVES — verified, not fixed:**
   - **FE M5 ("no `.validated` sentinel in any of the 5 feature entrypoints")** — WRONG. The sentinel is
     written by the paired `cmd_validate_<product>` verb in `tasks.py` (e.g. `cmd_validate_mask_geometry`
     at `tasks.py:346-360` does `validate_*(...)` then `output_flag.write_text("ok\n")`). The
     compute/validate split into two verbs is the **correct** pattern (same as frame_masks/registry).
     The agent judged the entrypoint in isolation and missed the validate verb. **No fix needed.**
   - This is why each finding was re-verified against source before acting. Treat the audit as leads,
     not gospel.

---

## LEFT TO DO (next session)

### P0/P1 — the `path_contracts` tripwire is wider than the audit's single B1
`data_pipeline.shared.path_contracts` is a deliberate raise-on-call tripwire (`resolve_data_root_relative_path`
and `require_existing_path` both `raise RuntimeError`). The audit flagged ONE site (fraction_alive). It
actually still infects **5 more feature files**, all live-or-legacy crash paths:

- `feature_extraction/pose_kinematics_metrics.py:19,117,145` — **LIVE** (imported by `pose_kinematics/compute.py`).
- `feature_extraction/io/loaders.py:12,13,32,44-66` — legacy loader layer; `resolve_data_root_relative_path`
  + `require_existing_path` both called. Also carries the `time_int`/`frame_index` compat shim (FE M7).
- `feature_extraction/fraction_alive/_legacy_compute.py:19,68,78,86` — called by `fraction_alive/compute.py`
  via `compute_fraction_alive` (so this one IS on the live path — confirm fixtures don't mask it).
- `feature_extraction/core/curvature_metrics.py:10,92` — legacy `core/` (see P2 below).

**Recommended fix:** same pattern already applied — drop the import, resolve paths directly against an
explicit root, fail loud with a message that names the fix. Consider a single small local helper
(`_require_existing(path, *, field, row_id)`) in `feature_extraction/io/` rather than 6 inline copies.
Then **delete `shared/path_contracts.py`** once no importer remains.

### P1 — remaining hard-constraint / id items not yet touched
- `segmentation/video_generation/results_adapter.py:27-54` — `_derive_experiment_id_from_video` /
  `_derive_video_id_from_image_stem` re-implement id grammar with local regex (handle legacy `_f`/`_ch00`
  stems the canonical parsers reject). Decide: migrate to `parse_image_id` and drop legacy-format support,
  or quarantine as an explicit legacy-video adapter. NOT a blind swap — canonical `parse_image_id` raises
  on non-`VALID_CHANNEL_NAMES`.
- `segmentation/grounded_sam2/` subtree — segmentation zone's principal offender: stale `time_int` column,
  shadow `validate_csv_schema` diverging from `validate_frame_mask_block`, foreign schema imports.
- `metadata_ingest/` well-index formula `chr(65+row)+f"{col:02d}"` duplicated ×4 → one constructor in
  `shared/identifiers/`.

### P2 — structural retirement & de-duplication (explicitly out of this session's scope)
- Delete `feature_extraction/core/` — BUT `core/curvature_metrics.py` is a **real impl still imported** by
  `curvature_metrics/compute.py:14`; move it to a product-private helper first, don't blind-delete. Same
  care for root `stage_inference.py` / `pose_kinematics_metrics.py` / `mask_geometry_metrics.py` (still
  imported as pure-function delegates).
- QC orphan cluster: `quality_control/{io/, validators.py, config.py}` forked validators + pycache-ghost
  dirs (`segmentation_qc/`, `auxiliary_mask_qc/`, `core/`, `entrypoints/` — source deleted, `__pycache__`
  retained, look importable).
- `data_pipeline/features/` ghost package (spec forbids it; risks shadowing `feature_extraction`).
- Finish `image_building/` → `image_materialization/` migration (duplicate package; `materialize_well_yx1.py`
  imports private `_determine_bf_channel`/`_get_stack` from the package it should replace). Then delete the
  `image_building/{yx1,keyence}/` star-import shims + their module-scope `logging.basicConfig()`.
- Land the locked 2026-06-16 stitch-split in `metadata_ingest/stitched_index/materialize_stitched_images.py`
  (`if YX1 / elif Keyence` fat materializer).
- `metadata_ingest/scope/yx1/generate_xy_reference.py` — exploration script with hardcoded machine paths
  leaked into `src/`; move to `results/` or delete.

### P3 — validator/dispatcher hygiene
- Collapse forked validators → one-per-contract + `check_sources=` flag (QC `validators.py`,
  `surface_area_outlier_detection.validate_sa_reference`, grounded_sam2 shadow; metadata_ingest validators
  that do unconditional disk I/O need the flag, B3).
- Thin out `tasks.py:cmd_validate_snip_inventory` (embeds a column manifest inline — only non-thin validate
  verb); delete duplicated private sentinel helpers (`_sentinel_path`, `qc_sentinel_path`).

---

## Pre-existing test debt (separate from this work)
The full suite has **4 collection errors** from a stale import: `ensure_frame_time_alias` is imported by
`tests/test_sam2_staleness.py` / `tests/test_time_helpers.py` but no longer exists in
`src/data_pipeline/metadata_ingest/time_helpers.py`. Not caused by this work; whoever owns the time_helpers
refactor should restore/rename or skip those tests.

---

## NOT mine in the working tree
These were already modified before this session (initial `git status`) and were **not** touched here — do
not attribute them to this remediation: `consolidated_features/entrypoint.py`,
`fraction_alive/{_legacy_compute,entrypoint}.py`, `feature_extraction/io/loaders.py`,
`segmentation/masks/__init__.py`, `snip_processing/entrypoints/run_snip_processing.py`, plus several spec docs.
