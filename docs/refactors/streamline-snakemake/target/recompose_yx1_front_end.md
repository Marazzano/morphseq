# Recompose — YX1 Front End (🟢 TARGET, microscope-scoped)

**Status:** the YX1 recomposition plan, from a read-only research pass 2026-06-07.
**Owner of this doc:** the **YX1 model only.** Keyence is a *separate* doc/model
(`recompose_keyence_front_end.md`) — **do not read or import Keyence logic.** The two microscopes
must not fuse. They meet only at the convergence line (the shared join), which neither doc owns.
**Terminology guard:** `raw_position_label` means the raw ND2 position token. Do **not** call it `well_index` until it has been resolved to the local plate well label by mapping.
**North star (read first, in order):**
1. `pipeline_file_philosophy.md` — the conventions every change must satisfy (the two hard
   constraints, nouns-for-steps, fail-loud, the conformance checklist). **Judge every file against it.**
2. `front_end_naming_and_flow.md` — the target flow + naming (the "ONE raw read" + convergence-line
   decisions this doc implements).
3. `stitched_handoff_contract.md` — only for Phase 2 (the stitched seam + `well_id`-keyed tree).

---

## 🧭 THE THREE LAYERS (hold them distinct — do not collapse)
- **L1 — legacy/source reality** (real-data lessons): `src/build/build01B_compile_yx1_images_torch.py` (853 lines).
- **L2 — current refactored code** (partially built, may not read as narrative):
  `metadata_ingest/scope/yx1/{extract_scope_metadata,map_series_to_wells,validate_xy_reference_grid,generate_xy_reference}.py`
  + the LIVE Snakefile front-end rules + `tasks.py` verbs.
- **L3 — cohesive narrative target** (what L2 should become to honor L1's lessons AND the philosophy).

This doc maps L1→L2→L3 for YX1 and prescribes the recomposition.

---

## 🚦 THE TWO PHASES — a HARD line at the GPU boundary

YX1's native-microscope entry mode is **two phases with different resource profiles**. Keep them separated; do not fuse the
flow into one job. The shared pipeline begins at the canonical stitched handoff tree; the external-handoff entry mode starts there.

```
  PHASE 1 — METADATA (CPU, implement NOW)                    │  PHASE 2 — STITCH (GPU, specified, deferred)
  ingest_scope_metadata → map_series_to_wells →             │  stitch_well[well_id]
  join_series_mapping_to_scope_metadata → discover_wells    │  (LoG focus-projection + frame_tiler stitch)
            (no GPU; the 2-well smoke run needs this)        │  (needs a GPU; depends on Phase 1)
                            └──────────── CONVERGENCE LINE (microscope gone) ────────────┘
```

> **Implement Phase 1 now.** It is lower-risk (no GPU scheduling), it is the "ONE raw read" fix, and
> it is exactly what the 2-well smoke run needs. Phase 2 is fully specified below but built after
> Phase 1 runs clean.
>
> **"ONE raw read" here means one raw metadata read in Phase 1.** It does not prohibit later raw
> image reads in Phase 2.

---

# ════════ PHASE 1 — METADATA (CPU) — the implement-now recompose ════════

## Phase 1 split: required recomposition vs safety parity

### 1A — Required recomposition
- no premature IDs
- add `x_um`/`y_um`
- CSV→CSV `map_series_to_wells`
- config-sourced reference path

### 1B — Safety parity (stay out of 1A unless the immediate smoke run needs it)
- BF env override
- timestamp jump detection
- KMeans QC
- cleanup / dead-code removal

## What L2 has today (inventory + narrative judgment)

| File | What it does | Narrative verdict (vs philosophy) |
|---|---|---|
| `yx1/extract_scope_metadata.py` (271) | opens ND2, reads calibration/dims/channels, imputes+monotonizes timestamps, normalizes channel, writes `scope_metadata__yx1.csv` | **Partial.** Honest name/docstring, BUT mints `well_id`/`image_id` at ingest (lines ~212-213) before a real well exists — violates "well_id born at the join." Does NOT emit stage XY. Duplicate `'time_int'` dict key. |
| `yx1/map_series_to_wells.py` (510) | **re-opens the ND2** for stage XY, KD-tree matches to a reference grid, writes `series_well_mapping.csv` + `.provenance.json` | **Fails the central rule.** Target says map is **CSV→CSV**; this re-reads raw (the 2nd raw metadata read in Phase 1). Hardcoded absolute `DEFAULT_REF_XY_PATH` (line ~21). Dead helpers `_parse_series_number_map`/`_build_implicit_mapping`. |
| `yx1/validate_xy_reference_grid.py` (200) | geometric sanity-check on the reference grid; fail-loud with previews | **Good — the exemplar.** Keyword-only, fail-loud-with-the-words. Match this altitude. |
| `yx1/generate_xy_reference.py` (232) | offline one-shot tool: builds the `well,x_um,y_um` reference CSV from a verified experiment | **Out-of-band, fine.** NOT a DAG node. Hardcodes ref experiment + `morphseq_playground` paths; duplicates `extract_nd2_stage_positions` (drift risk). |

## The real-data lessons from L1 (`build01B`) — must survive
1. **Distance tolerance = half-grid-spacing** (`max_distance_um=4500` ≈ 9000/2). A match beyond half a
   well pitch is rejected. The single most important tolerance — faithfully in L2, keep it.
2. **Series is 1-based; ND2 position is 0-based** (`series = P + 1`). Off-by-one silently shifts every
   well. Preserve.
3. **Stage-XY frame addressing:** position `w` at T=0 is frame `w*(Z*C)`; read
   `channels[0].position.stagePositionUm`. Preserve exactly.
4. **Column-major Excel series order** (8×12 grid read down columns). Load-bearing if any Excel-grid
   code is touched.
5. **KMeans match-QC cross-check** (L1 `_qc_well_assignments`): clusters stage Y→rows, X→cols and
   asserts they match assigned labels — catches a transposed/flipped reference. **L2 DROPPED this**
   (kept only reference-grid-shape validation + distance rejection). Gap — see risks.
6. **Timestamp imputation + `cummax()` monotonic** — ported to L2. But L1's mid-acquisition **jump
   detection** (`_fix_nd2_timestamp`) is **NOT ported** — latent correctness gap on jumpy ND2s.
7. **BF channel detection is fuzzy** (`BF`/`EYES - Dia`/`Empty`/single-channel fallback/fail-loud).
   L2 covers most but not the `YX1_BF_CHANNEL_INDEX` env override.
8. **Well label is an opaque string** in image building (dict key + dir component, never parsed) —
   confirms switching the image tree to `well_id` is a pure substitution.

## Current → target mapping (Phase 1)

| Target step | Existing code | Reuse | Reorient | Rewrite | Missing |
|---|---|---|---|---|---|
| `ingest_scope_metadata` (the ONE raw metadata read in Phase 1) | `extract_scope_metadata.py`; live rule; `tasks.py::cmd_extract_scope` | timestamp impute, channel normalize, ND2 open, schema-validate+write | **stop minting `well_id`/`image_id`** (emit only `raw_position_label`, not a resolved well label) | **ADD `x_um`/`y_um` stage-XY columns per series** so map can be CSV→CSV | 1B: BF env override; timestamp jump-detection (only pull into 1A if the smoke run proves it is required) |
| `map_series_to_wells` (CSV→CSV) | `map_series_to_wells.py`; live rule; `cmd_map_series` | KD-tree XY match, grid-validator call, distance tol, provenance + gap/dup warnings | **read stage XY from `scope_metadata__yx1.csv` columns** instead of re-opening the ND2; drop `nd2_path` from signature + rule input + verb | replace hardcoded `DEFAULT_REF_XY_PATH` with a config-sourced path; delete dead fallbacks | 1B: KMeans match-QC (only pull into 1A if the smoke run proves it is required) |
| `join_series_mapping_to_scope_metadata` ← **CONVERGENCE; NOT this doc's to design** | `scope/shared/apply_series_mapping.py` (shared) | the whole join; `well_id` minted here (correct point) | — | normalize `.validated` suffix (open audit) | — |
| `discover_wells` ← shared, already TARGET-shaped | `checkpoint discover_wells` | reads `well_id` col → `discovered_wells.txt` | — | — | — |

**Where YX1 converges out:** at the **input** to `join_series_mapping_to_scope_metadata`. The only
YX1-specific front-end modules are `extract_scope_metadata.py` and `map_series_to_wells.py` (both
under `scope/yx1/`). The join and everything after are shared — out of scope here.

## Phase-1 recomposition plan (what to change)
1. **`extract_scope_metadata.py` — reorient to a pure raw-reader.**
   - Emit acquisition facts **+ stage XY (`x_um`,`y_um`) per series**.
   - Emit only `raw_position_label` (the raw ND2 position label). **Remove `well_id`/`image_id` minting** —
     those are born at the join. Fix the duplicate `'time_int'` key.
   - Phase 1B: port the BF-channel env override + timestamp jump-detection (lessons 6-7) only if the smoke run needs them.
   - Schema: add `x_um`/`y_um` to `REQUIRED_COLUMNS_SCOPE_METADATA` (⚠️ shared schema — coordinate; see
     "Cross-cutting" below).
2. **`map_series_to_wells.py` — rewrite the read path to CSV→CSV.**
   - Read `x_um`/`y_um` from `scope_metadata__yx1.csv`; **delete** `extract_nd2_stage_positions` +
     `nd2_path`. Remove the ND2 glob from `tasks.py::cmd_map_series`.
   - Source the reference grid path from config (`scope_metadata.yx1.ref_xy_csv` or similar), not a
     module constant.
   - Phase 1B: port a lightweight `_qc_well_assignments` as a post-match assertion (lesson 5) only if the smoke run needs it.
   - Delete dead `_parse_series_number_map`/`_build_implicit_mapping`.
3. **`validate_xy_reference_grid.py` — keep as-is** (the exemplar).
4. **`generate_xy_reference.py` — relocate** to a `tools/`-style location (not a DAG node);
   de-duplicate `extract_nd2_stage_positions` (import one shared `nd2_stage_positions` helper).

## Philosophy violations to fix (YX1, Phase 1)
- NO-LEAKAGE: `extract_scope_metadata.py` mints `well_id`/`image_id` at the wrong moment → move to join.
- Second raw metadata read in Phase 1: `map_series_to_wells.py` re-opens the ND2 → CSV→CSV.
- Hardcoded path string: `DEFAULT_REF_XY_PATH` → config.
- Minor / safety parity: duplicate `'time_int'` key; dead helpers; `cmd_map_series` self-globs the ND2 (delete once CSV→CSV).
- Keep BF env override, timestamp jump detection, and KMeans QC in 1B unless the immediate smoke run forces one of them into 1A.

---

# ════════ PHASE 2 — STITCH (GPU) — specified, build AFTER Phase 1 ════════

> **GPU-gated, per-well, depends on Phase 1.** Do not start until the metadata phase runs clean for
> the 2 wells. Different resource profile (needs `device=cuda`).

**Reuse the shared engine — do NOT rewrite stitching.** `image_building/utils/frame_tiler.py` (419
lines: `stitch_frame_tiles`, `FrameTilingConfig`, `FallbackParams`, QC, legacy-canvas fallback) is
the LIVE shared stitch engine. The YX1 `stitch_well` **calls it**; it does not reimplement tiling.
This is the native-microscope entry mode; the shared pipeline does not care about YX1 vs Keyence
once the canonical stitched handoff tree exists.

**Target shape:**
- `stitch_well[well_id]` — per-well fanout (not the current experiment-grain loop). Reads this well's
  raw frames + its mapping rows; writes `built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/`
  keyed on **`well_id`** (compose via `shared/identifiers`, never an f-string in a path helper);
  sentinel `.well_{well_id}.done`.
- Off-registry image path comes from a `stitched_handoff/paths.py` helper, not an inline string.
  That helper is the practical seam between native microscope mode and the shared pipeline.
- YX1 focus/stitch specifics stay in the YX1 backend; the cross-microscope tiling stays in `frame_tiler`.

**Orphaned stitch code (deferred until after Phase 1 — do not clean up yet):** `image_building/yx1/
stitched_ff_builder.py` (4-line shim → delete) and `image_building/scope/yx1/stitched_ff_builder.py`
(279 lines, orphaned old builder → **diff against the inlined logic in `materialize_stitched_images.py`
first**, confirm no unique capability, then delete). `frame_tiler.py` is KEPT.

---

## 🔗 CROSS-CUTTING (shared with the Keyence doc — coordinate, don't fuse)
These touch shared surfaces; resolve them jointly so the two microscopes stay consistent. The
shared side begins at the canonical stitched handoff tree; the native side ends there:
- **Scope-metadata schema** (`REQUIRED_COLUMNS_SCOPE_METADATA`) — both ingests change it (YX1 adds
  `x_um`/`y_um`; Keyence adds position). Change once, together.
- **`.validated` sentinel suffix** (leading vs trailing dot) at the join — shared open audit.
- **The orphan-shim deletion** — both microscopes have a `scope/{scope}/stitched_ff_builder.py`;
  verify-then-delete each against the inlined logic, but only after Phase 1 is green. `frame_tiler.py` stays for both.
- **The convergence line itself** (`join_...`, `discover_wells`) — neither microscope doc designs it. The seam that matters for the shared pipeline is the canonical stitched handoff tree.

## ✅ DONE-FOR-PHASE-1 (YX1)
- `ingest_scope_metadata` emits `raw_position_label` + stage XY, no premature `well_id`.
- `map_series_to_wells` is CSV→CSV (no ND2 re-read), ref path from config.
- `map_series_to_wells` has no `nd2_path` and does not reopen the ND2.
- `scope_metadata_mapped.csv` (global `well_id`) + `discovered_wells.txt` produced for 2 YX1 wells of
  `20250912`, via snakemake, conforming to the philosophy doc. Front-end tests green.
