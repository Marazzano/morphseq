# Current State & Next Steps — the STATUS doc

**Status:** the "where are we RIGHT NOW" anchor. The current snapshot below (2026-06-17) is the live
truth; the dated sections further down are earlier verified state, kept for history. Design lives in
`specs/`; the active front-half plan is `front_half_reorg_roadmap.md`. See `README.md` for the map.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-17 (session: Steps 4+5 — comparison gate + two-well fan PASSED)

**What shipped:**
- `acquisition_inventory__yx1.csv` generated for `20250912` (161,025 rows).
- `position_well_mapping.csv` regenerated for `20250912` in new contract format (`experiment_id, position_index, well_index, well_id, mapping_method`).
- `tasks.py::cmd_materialize_yx1_well_candidate` fixed: joins `position_well_mapping.csv` (`--position-well-mapping-csv`) to add `well_index/well_id` before filtering; uses `validate_position_well_mapping` guard. Parser and join tests updated.
- B01 smoke: **113 frames written**, all paths resolve, correct schema. mdcolon visual sign-off: **accepted**.
- **Step 4 comparison gate PASSED:**
  - 113/113 frames compared. 0 byte-identical (expected — legacy=JPEG, candidate=PNG; format difference, not algorithm difference).
  - Numeric diff: max_abs_diff 10–12/255 (mean ~10.3), mean_abs_diff ~1.5, p99 5.0 — consistent with JPEG compression artifacts only.
  - QC evidence saved to `data_pipeline_output/stitch_candidate_qc/20250912/20250912_B01/`: `comparison_summary.csv` (113 rows, `human_review_status=accepted`), `frame_diff_metrics.csv`, 113 side-by-side JPEGs, `side_by_side.mp4`.
  - **human_review_status=accepted** (mdcolon visual sign-off on candidate frames; diff is JPEG codec noise only).
- 56/56 targeted tests pass.
- **Step 5 two-well fan smoke PASSED:** C01 (`20250912_C01`, position_index=2) ran cleanly — 113 frames, 0 missing paths, done flag exists. Only B01 has legacy stitched output so numeric diff on C01 skipped; B01 is the accepted comparison baseline. The fan is real: B01→position_index=1, C01→position_index=2 (different positions, not hardcoded).
- `side_by_side.mp4` (219 MB) saved alongside comparison evidence.
**What's broken/half-done:** nothing. Steps 1–5 complete.
**Next concrete action:** Step 6 — PROMOTE candidate to live spine. **Naming/layout updated 2026-06-17:**
the producer is named for the STAGE (`materialize_well`), not one op (`stitch_well` was too narrow —
stitch is one composition step, peer to projection). Files are FLAT under `image_materialization/`
(filenames carry the moment; only `scope/` is a subfolder) — the `stitched/` nesting from Steps 2–5
collapses. See `front_half_reorg_roadmap.md` → Target Package Layout for the exact filenames and the
compose/adapter/axes/Keyence-compat plumbing. **Do the move/rename as its own mechanical commit
(verified green) BEFORE wiring new logic.**

(1) Restructure to the flat target shape: `materialization_plan.py` (intent only — product set, no
microscope geometry), `materialized_image_paths.py` (where one product file lands; was
`stitched/layout.py`), `materialize_well.py` (thin dispatcher; was `stitch_well.py`),
`frame_inventory_contract.py` (observed manifest; was `stitched/contracts/`), YX1 fork at
`scope/yx1/materialize_well_yx1.py` (was `scope/yx1/materialize_yx1_stitched_images.py`).
`materialized_image_paths.py` takes `built_image_data_dir` explicitly and imports no orchestration
paths. (2) Add/wire the live `materialize_well` orchestration path/sentinel in `paths.py`; `candidate/`
vs live layout stays owned by `materialized_image_paths.py` via `candidate=True/False`, not the
registry. (3) Add `materialize_well` Snakemake rule fanned over `discovered_wells.txt`, call the
accepted materializer with `candidate=False` (replaces experiment-grain `materialize_stitched_images`).
(4) Keep grain **per well**: the sentinel means "the configured product set for this well finished."
Step 6's set is deliberately minimal — `BF` projection via `focus_stack`; channel/method/z-slice
expansion happens inside the well job later, recorded as `frame_inventory` rows, not Snakemake
wildcards. (5) Wire `validate_frame_inventory_for_well` to read the materializer-emitted shard (not the
`frame_contract.csv` adapter). (6) Add the per-well `{well_id}_frame_inventory.csv.validated` target to
`rule all` / front-end target. Verify: `20250912` runs end-to-end
`ingest→…→materialize_well[well_id]→validate_frame_inventory_for_well` and produces per-well frame
inventories keyed on `time_index` with resolving paths.

> **Compose/axes/Keyence are NOT Step 6.** Step 6 promotes the EXISTING accepted YX1 materializer under
> the new names. The acquired-image-tiles seam, `compose_xy_mosaic`, `materialize_image_product`
> branch table, max/z_stack, and the clean-vs-legacy `FrameTilingConfig` are later work — they're
> specified in the roadmap so the names are reserved, but Step 6 ships the rename + live wiring only.
**Open decisions:** none blocking Step 6.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-17 (session: Step 3 — stitch_well_candidate / layout.py)

**What shipped:**
- `image_materialization/stitched/layout.py` — locked image layout (`materialized_images/{candidate/}{well_id}/projection/{channel_id}/{image_id}.png`). One generic constructor `materialized_image_path` + `projection_frame_path` wrapper. Product-type-aware (`projection` / `z_stack`), enforces `z_index` consistency, validates `well_id` and extension. Pure path math. 15/15 tests pass at `tests/data_pipeline/image_materialization/stitched/test_layout.py`.
- `image_materialization/stitched/scope/__init__.py` + `scope/yx1/__init__.py` — package stubs.
- `image_materialization/stitched/scope/yx1/materialize_yx1_stitched_images.py` — two image-math primitives (`materialize_ff_projection` / `materialize_max_projection`) + per-well orchestrator `materialize_yx1_well`. Emits flat frame-inventory schema with `z_index` (pd.NA for projections), `image_product_type`, `projection_method`. Entry guard checks well_id/well_index/experiment_id consistency + unambiguous position_index. 11/11 tests pass at `tests/data_pipeline/image_materialization/stitched/scope/yx1/test_materialize_yx1.py`.
- `pipeline_orchestrator/tasks.py` — added `cmd_materialize_yx1_well_candidate` (thin dispatcher; explicit `--built-image-data-dir` and `--position-well-mapping-csv` args).
- **Key design decisions locked:** `layout.py` takes `built_image_data_dir` (not `DATA_ROOT`); no registry row for Step 3 (standalone B01 only, no `well_runner`); `materialize_max_projection` stub defined to lock naming convention; `nd2` imported at module level (not locally) so tests can patch it.
**What's broken/half-done:** nothing — targeted tests pass. B01 smoke run on real ND2 data is Step 4's gate (not yet run — requires GPU).
**Next concrete action:** Step 3 B01 smoke — run `cmd_materialize_yx1_well_candidate` on `20250912_B01` with real ND2 data. CLI: `PYTHONPATH=src /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python -m data_pipeline.pipeline_orchestrator.tasks materialize-yx1-well-candidate --experiment 20250912 --well-id 20250912_B01 --well-index B01 --acquisition-inventory-csv <path> --position-well-mapping-csv <position_well_mapping.csv> --nd2-path <path> --built-image-data-dir <BUILT_IMAGE_DATA_DIR> --frame-inventory-csv <out.csv> --done-flag <out.done>`. Confirm images land under `materialized_images/candidate/20250912_B01/projection/BF/` and frame_inventory CSV has correct row count. Then Step 4: comparison gate vs legacy stitched output.
**Open decisions:** none blocking the B01 smoke.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-17 (session: Step 2 — frame_inventory_contract.py)

**What shipped:** `image_materialization/stitched/contracts/frame_inventory_contract.py` — the shared microscope-agnostic handoff seam. Contains: `REQUIRED_FRAME_INVENTORY_COLUMNS` (8 atoms), `DERIVED_FRAME_INVENTORY_COLUMNS` (well_id, image_id), `derive_well_id`/`derive_image_id` helpers, `assert_derived_ids_consistent` guard (recomputes from atoms, fails loud), and three frozen dataclasses (`StitchedHandoffSpec`, `FrameInventorySpec`, `WellHandoff`). `__init__.py` chain created for `image_materialization/`, `stitched/`, `contracts/`. 16/16 tests pass at `tests/data_pipeline/image_materialization/stitched/contracts/test_frame_inventory_contract.py` (no flags, no `__init__.py` in test tree — Option A convention confirmed). Decision: acquisition_inventory and frame_inventory are fully separate; no shared base; `time_index` name overlap is intentional vocabulary alignment only.
**What's broken/half-done:** nothing — fully verified.
**Next concrete action:** Step 3 — build `stitch_well_candidate` beside the legacy stitcher. Create `src/data_pipeline/image_materialization/stitched/stitch_well_candidate.py` (per-well, reads the acquisition inventory + frame_inventory contract, writes stitched images + `{well_id}_frame_inventory.csv` for ONE well). Run on B01 of `20250912`. No Snakemake wiring yet — the candidate runs standalone.
**Open decisions:** none blocking Step 3.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-17 (session: Step 1 — extract well_discovery/)

**What shipped:** `well_discovery/` package extracted from `tasks.py` (no behavior change). Three new files: `__init__.py`, `discovered_wells_contract.py` (read/write/validate), `discover_wells_from_scope_metadata.py` (the logic, now with `validate_discovered_wells` guard). `tasks.py::cmd_discover_wells` is now a pure 3-line delegator (no pandas, no business logic). 12/12 smoke tests pass at `tests/data_pipeline/metadata_ingest/test_well_discovery.py`.
**What's broken/half-done:** nothing — fully verified.
**Next concrete action:** Step 2 decision gate interview with mdcolon (see Open decisions below), then build `image_materialization/stitched/contracts/frame_inventory_contract.py`.
**Open decisions:** Step 2 — acquisition/frame contract shape interview (how `scope/yx1/acquisition_inventory.py` schema/key relates to `frame_inventory_contract.py` atoms). Must interview mdcolon before building Step 2.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-17 (session: quickstart + handoff protocol setup)

**What shipped:** `AGENT_QUICKSTART.md` created (cold-start + end-of-session protocol, pointer map, conceptual anchors). No code changes this session — doc/process work only.
**What's broken/half-done:** nothing in flight.
**Next concrete action:** Step 1 — extract `well_discovery/` from `tasks.py`. Create `src/data_pipeline/metadata_ingest/well_discovery/__init__.py` + `discovered_wells_contract.py` + `discover_wells_from_scope_metadata.py`; edit `tasks.py` to delegate (no pandas). Verify: `snakemake -n` parses; `20250912` produces identical `discovered_wells.txt`.
**Open decisions:** Step 2 decision gate — acquisition/frame contract shape (how the shipped `scope/yx1/acquisition_inventory.py` schema/key relates to `frame_inventory_contract.py`). Interview mdcolon before building Step 2.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-17

**Where we are (front half, YX1):**
- ✅ **Shipped:** `shared/identifiers/` (built + wired) · YX1 metadata spine (`ingest_scope_metadata`
  → `map_positions_to_wells` → join → `discover_wells`, smoke-verified on `20250912`) · YX1
  `acquisition_inventory__yx1.csv` (record-only, 161k-row smoke).
- ❌ **Not built:** the per-well stitch (`stitch_well`) · the live per-well `frame_inventory` spine
  (the rules exist but the branch is DEAD — segmentation still reads legacy `frame_contract.csv`) ·
  `image_materialization/` package · `well_discovery/` package (discovery still inline in `tasks.py`).

**The plan (LOCKED, building next — in stages):** `front_half_reorg_roadmap.md` — a **7-step STRANGLER
migration** to a validated per-well `frame_inventory` shard (the end of the microscope-aware pipeline):
> 1 extract `well_discovery` · 2 domain contracts (as consumed) · 3 `stitch_well_candidate` beside
> legacy (per-well, inventory-fed, isolated paths; run B01) · 4 comparison gate on B01 (byte → numeric
> diff → side-by-side video; **mdcolon's visual sign-off**) · 5 two-well fan smoke · 🏁 6 promote to
> live spine (finish line) · 7 strangle legacy.

**Next concrete action:** **Step 1 — extract `well_discovery/` from `tasks.py`** (low-risk, no behavior
change, legacy stays green). The well-discovery shape is now intentionally boring: contract +
`discover_wells_from_scope_metadata.py`, no dispatcher until a second source exists. The remaining 🎤
decision gate is the acquisition/frame contract shape — interview before building that.

> ⚠️ The dated sections below predate the 2026-06-17 reorg + strangler plan. Trust the snapshot above
> and the roadmap for anything front-half; the older material is Scope-1/3 + per-well-pattern context.

---

## 📂 (historical, 2026-06-04) Doc coverage — superseded by `README.md`

**Companion to:** `per_well_throughline_findings.md` (north star), `well_id_throughline_refactor_plan.md`
(the formal Scopes), `front_end_naming_and_frame_inventory_flow.md` (front-end), `frame_inventory_handoff_contract.md`
(the drop-in seam).

---

## 📂 Doc coverage — what's in `target/` now

All refactor docs live in `target/` (the two throughline docs were `git mv`'d here 2026-06-04):

| Doc | Covers | Status |
|---|---|---|
| `per_well_throughline_findings.md` | north star: grain model, registry, well-runner, DAG mechanics | design complete |
| `well_id_throughline_refactor_plan.md` | the formal Scopes 1–5 | design complete |
| `front_end_naming_and_frame_inventory_flow.md` | ingest lineages, fan, post-fan tail | design complete |
| `frame_inventory_handoff_contract.md` | the stitched drop-in seam + frame_inventory contract | design complete |
| `model_input_handoff_contract.md` | the model/embedding seam (legacy build_06): inference-first encode of snips → latents; symlink view of `processed_snip_path`; reuses `gen_embeddings` + the `mseq_pipeline_py3.9` sub-env | design (inference) complete; training deferred |
| `current_state_and_next_steps.md` | **this doc** — verified on-disk state | living |
| `OVERALL_PLAN.md` | top-level plan/index: goal, the ordered spine (family/fanout/execution/status per stage), build order, per-stage status table, open items, **audit findings** (2026-06-05) | living |

**Not yet a dedicated target doc** (specified *inside* the findings doc, not broken out):
- the **per-well stage pattern** for features + QC (Scope 5) — see "The Per-Well Stage Pattern" below
- the **registry / well-runner code** (`lib/paths.py`, `lib/well_runner.py`) — designed, not built

---

## 🆔 Scope 1 — Identifiers: EMPTY (create, not split)

**On disk:** `src/data_pipeline/identifiers/__init__.py` is **0 bytes**; nothing else in the package.

**IDs are minted inline as f-strings today** (verified 2026-06-04):
- `metadata_ingest/scope/yx1_scope_metadata.py:208` → `well_id = f"{experiment_id}_{well_index}"`
- `metadata_ingest/scope/keyence_scope_metadata.py:261` → `well_id`; `:265` → `image_id` (uses `_f{time_int:04d}`)
- `metadata_ingest/mapping/series_well_mapper_keyence.py:177` → `well_id`

**What goes in it** (per the plan + the immutable-key decision in `frame_inventory_handoff_contract.md`):
```
identifiers/
    __init__.py     re-exports the public names
    constructors.py build_well_id(experiment_id, well_index)  → sanitize ONCE here
                    build_image_id(well_id, channel_id, time_index)
                    build_embryo_id, build_snip_id            — compose from ATOMS
    parsers.py      split_well_id(well_id) -> (experiment_id, well_index); parse_image_id; _LOCAL_ID_RE
    validators.py   validate_well_id (fail loud on bare A01); recompute_and_check(atoms, supplied_id)
    README.md       the "sign on the door": atoms are truth; well_id/image_id are DERIVED
```

- This is the layer the handoff contract leans on: the validator's "recompute well_id/image_id from
  atoms and fail loud on disagreement" = `validators.recompute_and_check()`.
- **Reconcile on build:** the handoff doc standardized `time_index` + `_t{time_index:04d}`, but the
  Keyence inline code uses `_f{time_int:04d}`. `build_image_id` must use the canonical `_t####` form
  (part of the `frame_index`/`time_int` → `time_index` collapse).
- **Zero-risk / additive** — no existing importers of these names to preserve (the package is empty),
  so Scope 1 is purely "write the constructors + optionally repoint the inline f-strings."

---

## ⚙️ Scope 3 — Config + Environment: HALF DONE

| Piece | State |
|---|---|
| `config.yaml` (science knobs) | ✅ **exists** — `src/data_pipeline/pipeline_orchestrator/config.yaml` |
| `env.yaml` (machine/paths) | ❌ **does not exist** |
| `env.example.yaml` (template) | ❌ **does not exist** |
| path routing decoupled from code location | ❌ **still welded** |

**The welding, verified (Snakefile):**
- `:17` `PROJECT_ROOT = WORKFLOW_DIR.parent.parent.parent` — data location derived from code location.
- `:34-36` `DATA_ROOT = ... PROJECT_ROOT / "data_pipeline_output"` — outputs pile up inside the repo.
- `:23` `PYTHON = "/net/.../mdcolon/.../bin/python"` — **interpreter hardcoded to a home dir**; no
  other machine/user can run without editing the Snakefile.

**Also:** `config.yaml` mixes machine concerns into the science config — `device: "cuda"` (×2),
absolute model checkpoint paths (`weights_path`, `checkpoint_path`). Scope 3 moves those to `env.yaml`.

**What Scope 3 builds** (per findings doc / refactor plan): `env.yaml` (gitignored) +
`env.example.yaml` (committed) beside the Snakefile, with `input_root` / `output_root` / `models_root`
as first-class absolute paths and `python` / `device` moved out of `config.yaml`. `project_root` keeps
being derived (works with zero setup). This is a **separate, later track** from identifiers.

---

## 🔁 The Per-Well Stage Pattern (features + QC) — ONE pattern, not two

**Verified 2026-06-04: features and QC rules are the SAME template.** `compute_mask_geometry` and
`compute_pose_kinematics` (Snakefile:618, :635) are **byte-identical except three tokens**: the stage
name, the module path, and the output filename. All 13 stages follow it:
- Features: `compute_mask_geometry`, `compute_pose_kinematics`, `compute_fraction_alive`,
  `compute_stage_predictions`, `consolidate_features`.
- QC: `compute_segmentation_qc`, `compute_viability_qc`, `compute_death_detection`,
  `compute_surface_area_qc`, `compute_auxiliary_mask_qc`, `compute_focus_qc`, `compute_motion_qc`,
  `consolidate_qc`.

**The three things that vary per stage** (everything else is the template):
```
1. stage name      compute_mask_geometry        vs  compute_pose_kinematics
2. module path     ...entrypoints.compute_mask_geometry  vs  ...compute_pose_kinematics
3. output file     mask_geometry/mask_geometry_metrics.csv  vs  pose_kinematics/pose_kinematics_metrics.csv
```
Features vs QC differ only in **`family`** (`computed_features/` vs `quality_control/`) and inputs.

**This is exactly the registry's job** (findings doc): one `STAGES` row per stage drives the path,
the rule body is a copy-paste template. "Add a feature / a QC metric = **one registry row + one
compute fn + one rule from the template**." Features and QC are the *same* recipe with a different
`family` field.

**The one structural difference to handle (Scope 5 wiring):** QC stages produce **flags**, and a few
currently read the **merged/consolidated** contract (e.g. `surface_area_qc` reads
`consolidated_features`) — the accidental merge-wall the findings doc flagged. The math is per-snip /
per-embryo (no cohort stat — findings doc audit), so the conversion is **rewiring those rules to read
per-well shards**, not rewriting algorithms.

> **Recommendation (mdcolon asked: "should I flesh out the features/QC pattern? they're the same"):**
> **YES — as ONE documented pattern.** The code proves they're copy-paste-with-find-replace, so the
> valuable artifact is a single **"add a per-well stage" recipe**: (1) the template rule, (2) the
> registry row, (3) the three tokens that vary, (4) the per-well conversion checklist (incl. the few
> QC merge-wall reads to rewire). This turns 13 near-identical rules into "one pattern + a table" —
> the whole point of the registry. Write it as a sibling doc (`per_well_stage_pattern.md`) once
> `lib/paths.py` exists (the pattern *imports* the registry), OR sketch it now and wire it when the
> registry lands.

---

## 🛑 OPEN DECISION (mdcolon 2026-06-16) — mandate the stitched layout + split materialization per-scope

Raised while building the YX1 acquisition inventory. Two coupled changes, **own focused pass**
(the handoff-contract rewrite + a stitch refactor); recorded here so it isn't lost.

**The smell.** `materialize_stitched_images.py` (661 lines) is **one microscope-mixed file** —
`if microscope == "YX1" / elif "Keyence"` branches plus a shared `drop_duplicates(..., keep="first")`
that *pretends both scopes stitch the same way.* They do not: YX1 = ND2 tensor-slice + LoG focus;
Keyence = TIFF-tile mosaic + re-acquisition collision resolution. DRY-over-a-false-sameness is
exactly what gets dangerous and annoying to debug.

**Recommendation (independent analysis).** Separate the two questions the handoff contract conflates:

- **A — on-disk layout.** For the **NATIVE** producer (the path we control), *mandate* the canonical
  `stitched_ff_images/{well_id}/{channel}/{well_id}_{channel}_t{time:04d}.{ext}` tree: stitch writes
  it, the validator derives each path **from the frame key**, `source_image_path` becomes derived,
  and a layout/filename mismatch is a **hard FAIL** (today it is a *warning* —
  `frame_inventory_handoff_contract.md:139-142`, Decisions 11–12). The free-form
  `source_image_path`-points-anywhere flexibility earns its keep **only** at the **external drop-in**
  ingress (a user who can't reorganize) — keep warning-not-fail *there only*. Net: the native path
  loses the arbitrary-path-resolution machinery and gets simpler; the drop-in escape hatch survives
  where it's actually needed.
- **B — split materialization into per-scope routes (the real win).** `stitch_well` dispatches to a
  **YX1 backend** and a **Keyence backend** that genuinely differ; they **share only** the honestly
  shared surface: the `identifiers/` path/id constructors, the `frame_inventory` **schema +
  validator**, and the focus primitive (`image_building/shared/log_focus.py`). The mandate is **same
  OUTPUT contract (tree layout + schema), independently produced** — NOT same stitching code.

> **The clean seam:** scope-divergent producers → ONE enforced handoff (canonical tree + frame
> inventory schema). The acquisition inventory is the scope-specific *input* to each route; the
> frame inventory is the shared *output*. This is the microscope boundary, made structural.

**Scope of the rewrite when it happens:** flip `frame_inventory_handoff_contract.md` (recommended →
required for native; mismatch warning → fail; `source_image_path` derived natively, tolerated at
drop-in) and reconcile Decisions 11–12; split `materialize_stitched_images.py` into
`stitched_index/scope/{yx1,keyence}/` backends + a thin dispatcher (Task #9). Self-document each
backend as scope-specific up to the stitched handoff boundary. Keep it lean — don't pre-build shared
modules the second scope doesn't force.

## 🧭 Recommended order

```
Scope 1  identifiers/       ← DO NEXT. Empty, additive, zero-risk; unblocks everything keying on well_id;
                              the layer the handoff validator depends on.
Scope 3  env.yaml           ← separate later track (config exists; env + path-decoupling do not).
lib/paths.py + STAGES       ← the registry; precedes the well-runner.
Scope 5  per-well features/QC ← the ONE pattern above; mostly rewiring (merge-wall reads), not new math.
```

**Next concrete action:** build Scope 1 (`identifiers/`) — `constructors.py` / `parsers.py` /
`validators.py` / `README.md` — using the canonical `_t{time_index:04d}` form, against the four inline
f-string call-sites above.
