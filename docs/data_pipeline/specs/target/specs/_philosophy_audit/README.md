# Pipeline Philosophy Audit — cross-zone synthesis

**Date:** 2026-06-24. **Rubric:** [`pipeline_file_philosophy.md`](../pipeline_file_philosophy.md)
(the two hard constraints + the 14-point conformance checklist).

Five parallel reviewers audited `src/data_pipeline/` against the philosophy doc, one per zone. Each
zone report (under [`zones/`](zones/)) cites `file:line` + snippet + violated rule + severity + fix.
This index is the cross-zone read: the shared themes, a prioritized blocker list, and where the code
is genuinely strong.

## Scoreboard

| Zone | BLOCKER | MAJOR | MINOR | Report |
|---|---|---|---|---|
| feature_extraction | 5 | 10 | 7 | [feature_extraction.md](zones/feature_extraction.md) |
| segmentation | 4 | 6 | 5 | [segmentation.md](zones/segmentation.md) |
| metadata_ingest | 3 | 8 | 7 | [metadata_ingest.md](zones/metadata_ingest.md) |
| quality_control | 2 | 5 | 4 | [quality_control.md](zones/quality_control.md) |
| orchestration + image | 2 | 6 | 4 | [orchestration_and_image.md](zones/orchestration_and_image.md) |
| **Total** | **16** | **35** | **27** | |

## The one-sentence verdict

**The new, recipe-conformant code is strong almost everywhere; nearly every BLOCKER lives in a
pre-doctrine legacy layer that was never deleted when its per-product successor went live — and
several of those dead layers now actively break (runtime crashes, dead imports, raise-on-call
tripwires) rather than sitting quietly.** This is good news: the refactor's *target* shape is sound;
the work is **finishing the retirement**, not redesigning.

---

## Cross-cutting themes (these recur across ≥3 zones — fix the theme, not just the instance)

### THEME 1 — Un-retired legacy layers are the dominant smell (and some now crash)
Every zone has a pre-recipe layer that should have been deleted when the per-product subfolders
landed, and the audit found several are not merely dead but **runtime-broken**:

- `feature_extraction/`: legacy `core/`, `entrypoints/`, `pipelines/` + root-level `.py` files still
  present. `fraction_alive/compute.py` calls `resolve_data_root_relative_path`, now a **raise-on-call
  tripwire** → every `compute_fraction_alive_features` call crashes (B1). `entrypoints/compute_fraction_alive.py`
  imports a **deleted** function → `ImportError` on load (B2). (feature_extraction B1–B5)
- `quality_control/`: legacy `death_detection.py` does flat-key lookups into a nested `QC_DEFAULTS`
  → **`KeyError` at runtime** (BLOCKER-1), and imports `from src.data_pipeline...` → breaks under the
  project's `PYTHONPATH=src` convention (BLOCKER-2). Plus pycache-only ghost dirs (`segmentation_qc/`,
  `auxiliary_mask_qc/`, `core/`, `entrypoints/`) that *look* importable. (quality_control B-1, B-2, M-5)
- `segmentation/`: the `grounded_sam2/` subtree is the principal non-conformance area — stale schema
  vocabulary, shadow validators, raw paths. (segmentation, "grounded_sam2 is the problem zone")
- `metadata_ingest/`: `scope/yx1/generate_xy_reference.py` is an exploration script (hardcoded machine
  paths, module-level globals, retired `series_number_map`) leaked into importable `src/`. (B2)
- `orchestration+image`: `image_building/` is a whole pre-doctrine **duplicate package** of
  `image_materialization/` (see Theme 4).

> **Action:** treat retirement as an explicit, tracked step. For each retired module either DELETE it
> or mark it (`# RETIRED — canonical impl now in <pkg>/; kept only for <reason> until <date>`). A
> legacy module kept "live for the legacy-drift gate" (QC's death/SA modules — see project memory)
> must still be marked and must not crash on call. Delete pycache-only ghost directories.

### THEME 2 — Inline id mint/split (Hard Constraint #1) — still present where legacy lingers
The crudest hard-constraint violation, concentrated in older/adapter code:

- `segmentation/backends/unet_snip/run_unet_snip.py:58` — `snip_id.split("_")[1]` to recover `well_id`
  (BLOCKER); use `parse_*` or pass `well_id`.
- `segmentation/video_generation/results_adapter.py` + `video_generator.py:206` — re-implement
  `image_id` grammar with local regex / f-strings, using the **deprecated `_t` suffix**; `build_image_id`
  / `parse_image_id` already exist (BLOCKER ×2).
- `feature_extraction/fraction_alive/_legacy_compute.py:67` — inline `snip_id` split to recover
  `image_id` (M8).
- `metadata_ingest/` — the well-index formula `chr(65+row)+f"{col:02d}"` is duplicated inline in ≥4
  places; it belongs once in `shared/identifiers/` (M1).

### THEME 3 — Raw path strings bypassing the registry (Hard Constraint #1) + missing/duplicated sentinels
- Raw path tokens (`"per_well"`, `"computed_features"`, `"snip_auxiliary_masks"`, sentinel filenames)
  hardcoded instead of `artifact_path()`/`validated_path()`: `feature_extraction/pipelines/*` (B3, B4),
  `segmentation/.../run_unet_snip.py:55-58` (BLOCKER), `image_building/*` (B-1).
- **Sentinel handling is inconsistent across the whole pipeline:**
  - *Missing:* none of the 5 new `feature_extraction` product entrypoints write the `.validated`
    sentinel the WellRunner contract requires → downstream rules depending on it never fire
    (feature_extraction M5; a **systemic gap**, not one file).
  - *Duplicated:* private `_sentinel_path` / `qc_sentinel_path` re-implement `validated_path()` in
    `feature_extraction/io/writers.py` (M6) and `quality_control/io/paths.py` (M-2) — violates
    "sidecars derived via the one helper, never re-derived."

> **Action:** one cross-zone pass — every product entrypoint writes its sentinel via
> `validated_path(output_csv)`, and every private sentinel/path helper is deleted in favor of the
> orchestration helper.

### THEME 4 — Duplicate / ghost packages (the "one concept, two homes → drift" smell at package scale)
- `image_building/` vs `image_materialization/` — **confirmed duplicate package**: both write the same
  pixel product to different paths; `image_materialization/` is the doctrine-compliant successor but
  still imports private helpers (`_determine_bf_channel`, `_get_stack`) from the old one, so the
  migration is half-done (orchestration+image B-1, M-4).
- `image_building/scope/keyence/stitched_ff_builder.py` imports `from src.build.export_utils` — a
  root-relative import outside the `data_pipeline` boundary (orchestration+image B-2).
- `data_pipeline/features/` — ghost package (only `__init__.py` + stale pyc) that the spec **explicitly
  forbids**; risks shadowing `feature_extraction` imports (feature_extraction M10).

### THEME 5 — Forked validators / disk-checks not behind a mode flag (Checklist: one validator + `check_sources=`)
- `quality_control/validators.py` — a **forked second validator set** using old column names
  (`dead_flag`, `death_inflection_time_int`) that contradict the authoritative per-product contracts;
  its only caller is itself orphaned (M-3). `surface_area_outlier_detection.py:validate_sa_reference`
  is a second, diverged validator for the surface-area reference contract (M-1).
- `segmentation/grounded_sam2/csv_formatter.py` — shadow `validate_csv_schema` diverging from the
  authoritative `validate_frame_mask_block`.
- `metadata_ingest/` — `validate_stitched_image_index` and `validate_frame_contract` bake
  `path.exists()` disk I/O in unconditionally (no `check_sources=` flag), unsafe at Snakemake planning
  time (B3).

### THEME 6 — Fat dispatchers & microscope/stale vocabulary leaking past the scope boundary
- `tasks.py:cmd_validate_snip_inventory` embeds a column manifest + uniqueness check inline — the one
  non-thin `cmd_validate_*` (orchestration+image M-2); `tasks.py` also leaks pre-doctrine
  stitched-index/scope-aware args via a `materialize-stitched` verb (M-3).
- `metadata_ingest/stitched_index/materialize_stitched_images.py` — fat pre-split materializer with
  `if microscope=="YX1" / elif "Keyence"` branching (B1); the **locked 2026-06-16 stitch-split** has
  not landed here.
- Stale ND2 `series_number`/`well_series_mapping` vocabulary downstream of the scope boundary
  (orchestration+image M-1; metadata_ingest).

---

## Prioritized fix list

### P0 — runtime-breaking (code that crashes or fails to import when called)
1. `quality_control/death_detection.py` — `KeyError` on `QC_DEFAULTS` flat-key lookups + `from src.data_pipeline` import. (QC B-1, B-2)
2. `feature_extraction/fraction_alive/compute.py` — calls `resolve_data_root_relative_path`, a raise-on-call tripwire. (FE B1)
3. `feature_extraction/entrypoints/compute_fraction_alive.py` — imports a deleted function → `ImportError`. (FE B2)
4. `image_building/scope/keyence/stitched_ff_builder.py` — `from src.build.export_utils` import that breaks outside the repo root. (O+I B-2)

### P1 — hard-constraint violations in live code paths
5. Inline id split/mint: `run_unet_snip.py:58`, `results_adapter.py`, `video_generator.py:206`, `_legacy_compute.py:67`, the 4× well-index formula. (Theme 2)
6. Raw-path-string entrypoints bypassing the registry: `feature_extraction/pipelines/*`, `run_unet_snip.py` path tokens, `image_building/*`. (Theme 3)
7. Missing `.validated` sentinel writes in all 5 new feature product entrypoints. (FE M5)

### P2 — structural retirement & de-duplication
8. Delete/retire legacy layers: `feature_extraction/{core,entrypoints,pipelines}` + root `.py`; QC orphan cluster (`io/`, `validators.py`, `config.py`) + pycache ghosts; `data_pipeline/features/` ghost; `metadata_ingest/scope/yx1/generate_xy_reference.py`. (Themes 1, 4)
9. Finish the `image_building/` → `image_materialization/` migration; remove the cross-boundary private-helper imports. (O+I B-1, M-4)
10. Land the locked stitch-split in `materialize_stitched_images.py`. (MI B1)

### P3 — validator/dispatcher hygiene
11. Collapse forked validators to one-per-contract + `check_sources=` mode flag; remove shadow validators. (Theme 5)
12. Thin out `tasks.py` dispatchers; delete duplicated private sentinel/path helpers. (Themes 3, 6)

---

## Where the code is genuinely strong (keep these as the exemplars)

- **`pipeline_orchestrator/orchestration/` (`paths.py`, `well_runner.py`)** — the doc's claim that
  these *embody* the doctrine is justified: pure path construction, the planning-vs-runtime collector
  split, controlled-vocabulary constants, fail-loud messages.
- **`image_materialization/` core modules** — the doctrine-compliant materialization successor.
- **`segmentation/physical_embryo_registry/`** and the new **`backends/`** packages — clean identity
  spine; the registry is the sole minting site and downstream joins to it.
- **`feature_extraction/` per-product packages** (`mask_geometry/`, `curvature_metrics/`,
  `pose_kinematics/`, `stage_predictions/`, `consolidated_features/`, `legacy_embeddings/`) — good to
  excellent; contracts correctly import spine columns from the minting site.
- **`quality_control/` per-product packages** (`death_detection/`, `surface_area_qc/`,
  `mask_quality_qc/`, `snip_qc/`) — follow the recipe almost perfectly.
- **`metadata_ingest/` acquisition-inventory modules (both scopes)** — exemplary docstrings, section
  banners, clean import direction, `check_sources`-style validators.

> The pattern is consistent enough to be a rule of thumb for the rest of the refactor: **trust the
> per-product subfolder packages; distrust anything still sitting in a `core/`, `entrypoints/`,
> `pipelines/`, or root-level legacy slot.**
