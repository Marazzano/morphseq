# Current State & Next Steps — the STATUS doc

**Status:** the "where are we RIGHT NOW" anchor. The current snapshot below (2026-06-18) is the live
truth; the dated sections further down are earlier verified state, kept for history. Design lives in
`specs/`; the active front-half plan is `front_half_reorg_roadmap.md`. See `README.md` for the map.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 14:52 (session: acquisition schema policy + time atom wired through — Beat 2 foundation, Parts 1–2)

**The plan:** `~/.claude/plans/committed-working-tree-flickering-rocket.md` — acquisition-inventory
schema policy (two-tier) + time atom + legacy frame_contract cutover. Parts 1–2 DONE; Part 3 (legacy
delete, clean-room cut) and Part 4 (back-half repoint) pending.

**What shipped:**
- **Part 1 (commit `eb2ba16f`):** `specs/acquisition_inventory_schema_policy.md` — the two-tier schema
  doctrine (Tier 1 shared/hard-checked vs Tier 2 scope-specific/soft; validator hard-checks the core,
  allows+ignores scope extras) + the time-atom policy (acquisition OWNS+DERIVES `elapsed_time_s`,
  frame_inventory CARRIES, downstream reads one name) + the scope→inventory adapter seam as DESIGN
  intent (config-driven `raw_channel_name→channel_id` routed through the scope backend, mirrors the
  detection backend pattern — built separately). Linked from README.
- **Part 2 (commit `2cef1094`):** time atom wired through YX1 end-to-end.
  - NEW `metadata_ingest/scope/acquisition_inventory_contract.py` owns
    `REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS` (Tier-1 shared). YX1 schema = CORE + YX1 Tier-2
    extras. Validator hard-checks core then full YX1 schema.
  - YX1 acquisition derives `elapsed_time_s` (per-position rebased) reusing
    `time_helpers.add_elapsed_time_columns(experiment_time_col="acquisition_time_s")`; finite/non-neg
    assertion added.
  - `frame_inventory_contract.py` requires `elapsed_time_s` + `acquisition_time_s` (carried block; NOT
    in the unique key). `materialize_well_yx1` SELECTS them through (no derivation in the pixel stage).
  - **RENAMED `channel` → `channel_id`** in the YX1 acquisition inventory (the converged downstream
    name) — schema tuple, builder, channel-mapping validator, + test fixtures.

**Verification:** 164 passed (`tests/data_pipeline/`). front_half smoke **9/9 (100%)** on real ND2 for
both wells (B01,C01) after force-rerunning `ingest_scope_metadata` (to regenerate the acquisition
inventory under the new schema) + `materialize_well`. Both shards carry `channel_id` + per-well-rebased
`elapsed_time_s` (t=0 → 0.0) + raw `acquisition_time_s`.

**What's broken/half-done:** nothing. The on-disk `acquisition_inventory__yx1.csv` for 20250912 was
regenerated under the new schema (channel_id, elapsed_time_s present).

**Next concrete action:** **Part 3 — delete the legacy frame_contract producer** (clean-room cut:
`materialize_stitched_images` + `validate_stitched_image_index` + `build_frame_contract` +
`validate_frame_contract` + `schemas/frame_contract.py` + the inline Snakefile rules + the
`cmd_materialize_stitched` task). `rule all` goes INTENTIONALLY RED at the segmentation/features seam;
`front_half` stays green. Then **Part 4** repoints the back half onto the per-well shard stage-by-stage
(turns `rule all` green again). `time_helpers.py` STAYS (the shared derivation).

**Deferred (specified, not built):** full Tier-2 optical extraction expansion (nd2 `Microscope` struct
for YX1; BZ-X exposure/gain/binning for Keyence) + the scope→inventory channel adapter refactor.

**Open decisions:** none blocking Part 3.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 13:35 (session: front-end de-legacy sweep — delete two verified-dead clusters)

**What shipped (commit `17d471b3`):** the front end no longer carries duplicate/orphaned legacy
modules and rule files. Two provably-dead clusters removed (8 files, −890 lines, zero blast radius):
- **Cluster A — dead Python (no importers):** `metadata_ingest/microscope_data_ingest/` (a never-wired
  SECOND `build_frame_contract`) + `metadata_ingest/stitched_index/debug_keyence_stitch.py` (standalone
  debug script).
- **Cluster B — orphaned rule files:** `rules/frame_contracts.smk`, `scope_ingest.smk`,
  `segmentation_and_tracking.smk`, `snip_processing.smk`, `stage_predictions.smk`. The Snakefile
  `include:` list is ONLY `frame_inventory.smk` + `quality_control.smk` (Snakefile:151–152), and every
  rule in the five deleted files has a LIVE inline Snakefile equivalent (e.g.
  `segment_and_track_well`→`segment_and_track_per_well`, `snip_processing_well`→
  `run_snip_processing_per_well`, `extract_scope_metadata_yx1`→inline `ingest_scope_metadata`). They
  were duplicate definitions, not parked work. `rules/` now holds exactly the two included files.

**Verification (the core safety check):** captured `snakemake -n` job-stats BEFORE deleting, then again
AFTER — **IDENTICAL**: `rule all` = 17 jobs, `front_half` = 5 jobs. Proves nothing live was removed.
Plus: **164 passed** (`tests/data_pipeline/`, unchanged), `tasks.py` imports clean (it still imports the
LIVE `materialize_stitched_images`, untouched), `grep microscope_data_ingest|debug_keyence_stitch` over
`src/`+`tests/` → no hits.

**Process note:** first commit attempt used `git add -A` and swept in unrelated detect-seg-track doc
edits + the two known strays; soft-reset and re-committed ONLY the 8 deletions. The detect-seg-track doc
mods + `may_need_to_be_domunented!.md` + `tests/improvements/` are left dangling in the working tree,
untouched (per standing instruction).

**What's broken/half-done:** nothing. The live `frame_contract.csv` producer chain
(`materialize_stitched_images` + `validate_stitched_image_index` + `build_frame_contract` +
`validate_frame_contract` + `schemas/frame_contract.py` + the inline Snakefile rules at 299/331) is the
deferred "key events" set — LEFT FULLY INTACT because `rule all` / the back half still consumes it.

**Next concrete action:** **Beat 2** — repoint the downstream already-agnostic stages (segmentation →
snips → features → QC + aux masks) to read the per-well `frame_inventory` shard instead of
`frame_contract.csv`, via `well_runner.py`. ONLY after Beat 2 lands does deleting the `frame_contract`
producer chain become safe (= the rest of the roadmap's full Step 7). Recommended shape: incremental
strangler, one stage at a time (segmentation first), keeping `rule all` green throughout. The MEDIUM
`file_organization_audit.md` banner cleanups remain opportunistic.

**Open decisions:** Beat 2 scoping (incremental-per-stage vs full sweep vs a compat-shim bridge) — to
be decided when Beat 2 planning opens.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 13:12 (session: Step 7 — strangle the dead build_frame_inventory_for_well legacy adapter)

**What shipped (commit `e114fbb1`):** the off-DAG legacy `build_frame_inventory_for_well` adapter is
gone. It was the OLD way of splitting `frame_contract.csv` into per-well shards; Step 6's
`materialize_well` now emits the shard directly, and NO Snakemake rule ever invoked the adapter (it was
reachable only by manually naming its CLI command). Removed:
- `build_frame_inventory_for_well` + `_validate_unique_keys_legacy` + the now-orphaned
  `schemas/frame_contract` import in `metadata_ingest/frame_inventory/frame_inventory.py` (the live
  `validate_frame_inventory` / `merge_frame_inventory_shards` + their helpers are UNTOUCHED).
- the `build-for-well` CLI subcommand (`frame_inventory.py`) and the `build-frame-inventory-for-well`
  task (`tasks.py`: `cmd_build_frame_inventory_for_well` + its subparser).
- the package export (`frame_inventory/__init__.py`) and the 2 legacy unit tests + their `_row` /
  `_write_frame_contract` helpers (`test_frame_inventory.py`).
- `paths.py` docstrings repointed from the dead verb to the live `materialize_well` producer.

**Scope decision (mdcolon-approved):** dead-adapter ONLY. The roadmap's *full* Step 7 (delete
`materialize_stitched_images` / `build_frame_contract` / `frame_contract.csv`) is STILL BLOCKED:
`frame_contract.csv` is live for `rule all` / Beat 2 — 8+ rules consume it (`segment_and_track_*`,
`snip_processing_well`, `generate_auxiliary_masks_well`, `compute_mask_geometry/curvature/pose/
fraction_alive/stage_predictions`, `consolidate_features`). Those must repoint onto the per-well shard
(= **Beat 2**) before the producer chain can be deleted. `schemas/frame_contract.py`,
`build_frame_contract.py`, `validate_frame_contract.py`, `materialize_stitched_images.py` all LEFT
INTACT.

**Verification:** **164 passed** (`tests/data_pipeline/`; = prior 181-baseline minus the 2 removed
legacy tests, plus the count is the live-only suite). Import smoke clean (adapter symbol gone,
`validate`/`merge` present, `tasks.py` imports). `grep` for `build_frame_inventory_for_well` /
`build-frame-inventory-for-well` / `_validate_unique_keys_legacy` over `src/`+`tests/` → only the
strangle docstring + the untracked `tests/improvements/` scratch (not in the suite). `snakemake
front_half -n` parses through the spine; `snakemake -n` (default `rule all` / Beat 2) still builds its
17-job DAG unbroken.

**What's broken/half-done:** nothing. Committed. Stray untracked files
(`specs/front_end/may_need_to_be_domunented!.md`, `tests/improvements/`) left in place by request.

**Next concrete action:** **Beat 2** — repoint the downstream already-agnostic stages
(segmentation → snips → features → QC, + auxiliary masks) to read the per-well `frame_inventory` shard
instead of `frame_contract.csv`, via `well_runner.py`. ONLY after that lands does the full Step 7
deletion (`materialize_stitched_images` + `frame_contract` chain + `schemas/frame_contract.py`) become
safe. The MEDIUM `file_organization_audit.md` banner cleanups remain opportunistic (apply when touching
those files).

**Open decisions:** none blocking Beat 2.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 12:08 (session: next-batch #1 — ND2-exists/opens hardening at the acquisition-inventory CONSUME boundary)

**What shipped (item #1 of the next batch — the small bridge before Step 7):**
- **One authoritative validator, two modes.** `metadata_ingest/scope/yx1/acquisition_inventory.py`
  grew `assert_acquisition_sources_readable(df, *, scope_label)` (each UNIQUE `source_nd2_path` must
  exist AND open via `nd2.ND2File(path).close()` — open-then-close, no tensor read; O(#ND2s) not
  O(#rows)) and a keyword-only `check_sources: bool = False` on `validate_yx1_acquisition_inventory`.
  `nd2` imported at module level (matches the extract/materialize modules; lets tests patch it).
  Build time keeps `check_sources=False` (the ND2 was just opened to build the inventory — re-opening
  is tautological); the CONSUME boundary passes `True`.
- **Consume call lives in the YX1 backend's entry guard** (`image_materialization/scope/yx1/
  materialize_well_yx1.py`), right after the `source_nd2_path.nunique()==1` tripwire and BEFORE the
  ND2 is opened: `validate_yx1_acquisition_inventory(well_acquisition_inventory_df, check_sources=True)`.
  Placement rationale (mdcolon-approved): `run_materialize_well.py` is deliberately scope-NEUTRAL, so a
  YX1-specific validator can't live there; the YX1 backend is the *immediate domain consumer* and
  already owns the `source_nd2_path` guards. `tasks.py` stays a thin CLI adapter (untouched). The
  readability LOGIC is owned by the acquisition contract; the backend only CALLS it. New import is
  YX1→YX1 (image_materialization/scope/yx1 → metadata_ingest/scope/yx1) — import-coherent, does not
  touch the identity/orchestration kingdoms. The `nunique()==1` guard stays as a local tripwire.
- **frame_inventory validator UNTOUCHED** (stays microscope-agnostic — immutable anchor).
- **Doctrine recorded** in `specs/pipeline_file_philosophy.md`: new structural-conventions section
  "**Validators live where the contract lives; call them where the risk appears**" (one contract → one
  authoritative validator owned where the product lives; lifecycle differences are a MODE FLAG, not a
  forked second validator; source/disk checks fire at the consume boundary) + a matching CONFORMANCE
  CHECKLIST line.
- Tests: **161 passed** (`tests/data_pipeline/`) + the in-source acquisition suite. Added 5
  source-readability cases to `metadata_ingest/scope/tests/test_acquisition_inventory.py`
  (default-skips-IO, opens-ok, missing-file, unopenable, opens-each-unique-path-once) and a backend
  guard `test_missing_source_nd2_fails_loud_before_tensor_read` to
  `tests/.../test_materialize_well_yx1.py`. The materializer test fixture `_make_inventory` was made
  **schema-complete** (carries every `YX1_ACQUISITION_INVENTORY_COLUMN`) because the consume-side
  validator re-runs the full contract; the patched-ND2 tests now point `source_nd2_path` at a real
  empty file under `tmp_path` so `.exists()` passes before the open is faked.

**Verification:** `snakemake front_half --configfile config_smoke_front_half_20250912.yaml
--rerun-triggers mtime --cores 1 --forcerun materialize_well` → **5 of 5 steps (100%) done**, both
wells re-materialized through the new consume-side check, real ND2 exists/opens → passes silently,
`AUTO_MODE_CHOSEN -> CPU` printed, `SMOKE_FRAME_CAP_ACTIVE` honored. (This snakemake build parses a
bare positional as a configfile, so the target goes BEFORE `--configfile`: `snakemake front_half
--configfile ...`, not `snakemake --configfile ... front_half`.)

**What's broken/half-done:** nothing. NOT yet committed (code + tests + 2 docs staged in working
tree). Smoke knobs in `config_smoke_front_half_20250912.yaml` are the overlay; main `config.yaml`
inert as before.

**ALSO this session — organization/concern-mixing audit (mdcolon-requested):**
- Applied the **Contract → Validation → Builder** section-banner reorg to the exemplar
  `acquisition_inventory.py` (the concrete example mdcolon gave) — 15 in-source tests still green.
- Codified the convention in `pipeline_file_philosophy.md`: new "**One file reads top-to-bottom in
  flow order, with section banners marking each concern**" section (+ the "banners separate, a SPLIT
  fixes a real mix" caveat) + a CONFORMANCE CHECKLIST line.
- Ran a 3-zone read-only audit (image_materialization / metadata_ingest / orchestration) and recorded
  it in `specs/file_organization_audit.md` (severity table; HIGH=split, MEDIUM=banner+small extract,
  LOW=clean).
- Added internal flow banners to `materialize_yx1_well` (Entry guard → ND2 setup → Materialize loop →
  Inventory assembly) using the file's existing `# ---` style — no behavior change.
- **HIGH audit item #2 FIXED (thin-dispatcher restore):** the position→well join + per-well row
  selection moved out of `tasks.py::cmd_materialize_well` into a NEW pure domain function
  `image_materialization/select_well_acquisition_rows.py` (DataFrames + well_id in → row-slice out;
  imports identity, NOT orchestration — verified `image_materialization/` imports `well_runner` 0
  times). Named for the INPUT it slices, NOT `…_shard` (that's well_runner's OUTPUT vocabulary). See
  the audit doc's "Placement note" for the full `well_runner` boundary rationale. `cmd_materialize_well`
  is now read+validate (file boundary) → adapter → `run_materialize_well` → write. +5 unit tests.
  **181 passed**; smoke re-run **5/5 steps** through the extracted adapter (both wells).
- The remaining HIGH audit item — `materialize_stitched_images.py` per-scope split — stays for **Step 7**
  (the locked split decision + legacy strangler), NOT a drive-by edit.

**Next concrete action:** Step 7 — the legacy strangler (`materialize_stitched_images` /
`build_frame_contract` / `build_frame_inventory_for_well` + its legacy tests + `schemas/frame_contract.py`),
scoped to dead-on-front_half pieces only (Beat 2 still reads `frame_contract.csv` — don't delete its
production until Beat 2 repoints downstream). Fold the MEDIUM banner cleanups from
`file_organization_audit.md` in as you touch those files.

**Open decisions:** none blocking Step 7. The MEDIUM audit items are opportunistic (apply when a file
is already being edited), not a standalone pass.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 11:18 (session: Step 6 no-GPU SMOKE PASSED + device auto-resolution + frame_inventory validator repoint)

**What shipped (the smoke walk-across — Beat 1 proven end-to-end on real data):**
- **`snakemake front_half` for `20250912` (wells B01,C01) → `4 of 4 steps (100%) done`.** The live
  branch ran `…→ materialize_well[B01,C01] → validate_frame_inventory_for_well → front_half` on real
  ND2 data, no GPU. Both per-well shards + `.validated` sentinels landed under
  `built_image_data/20250912/per_well/20250912_{B01,C01}/`.
- **Device auto-resolution wired (the gap the user named).** The new `materialize_well` branch was the
  one path BYPASSING the project-wide resolver. Fixed in
  `image_materialization/run_materialize_well.py`: now imports
  `utils/cuda_diagnostics.resolve_device`, resolves the preference at the sequencer seam, passes a
  concrete device to the backend, and ANNOUNCES it — `print("AUTO_MODE_CHOSEN: requested='auto' -> CPU")`
  (also `log.info`). Default flipped `cuda`→`auto`. `config.yaml image_building.device: "auto"` (durable).
  On this no-GPU box `auto` → `CPU` (confirmed `torch.cuda.is_available()==False`); the run printed
  `AUTO_MODE_CHOSEN: requested='auto' -> CPU` for both wells.
- **frame_inventory VALIDATOR repointed to the live contract (real bug the smoke caught).** Commit 2 had
  repointed the validator's INPUT PATH but not its SCHEMA CHECK — `validate_frame_inventory` still
  enforced the LEGACY `schemas/frame_contract` columns (`well_id`, `image_id`, `stitched_image_path`,
  `time_int`, `micrometers_per_pixel`), so the correct new shard was rejected. Now
  `metadata_ingest/frame_inventory/frame_inventory.py` validates the LIVE microscope-agnostic contract
  from `image_materialization/frame_inventory_contract.py`:
  `REQUIRED_FRAME_INVENTORY_COLUMNS` + identity-anchored unique key + `assert_derived_ids_consistent`.
- **Identity-anchored key (per mdcolon: keep the identifiers in the loop).** Added
  `UNIQUE_FRAME_INVENTORY_KEY_COLUMNS` + `frame_inventory_image_ids()` to the contract. The validator's
  uniqueness check does NOT trust the raw atom tuple — it recomposes the derived `image_id` via
  `build_well_id`→`build_image_id` (routing through `shared/identifiers/`) and validates the intermediate
  `well_id` with `validate_well_id` (a leaked bare-local label fails loud HERE). Uniqueness is on the
  derived image_id, so the key can never drift from the constructors.
- **Legacy `build_frame_inventory_for_well` kept self-consistent (Step-7 strangler debt).** It still reads
  the legacy `frame_contract.csv`, so it validates the legacy schema LOCALLY (`_validate_unique_keys_legacy`)
  — NOT resurrected onto the live helper. Marked do-not-extend in the module docstring.
- Tests: **160 passed** (`tests/data_pipeline/`; +1 new leaked-local-well_id guard test). The 4
  live validate/merge tests moved to the new-contract fixture; the 2 legacy build tests stay on the
  legacy fixture.

**Verification (all checklist items PASS, both wells):** csv+`.validated` exist; exactly 3 `time_index`
(0,1,2) per well (smoke cap honored); `z_index` all-NA on projection rows; every row
BF/projection/focus_stack; `source_image_path` unique, all under `materialized_images/.../projection/BF/`
(NO `candidate/`), every PNG exists on disk; input-side `source_nd2_path nunique==1` per well;
`AUTO_MODE_CHOSEN -> CPU` printed.

**What's broken/half-done:** nothing. Beat 1 is proven end-to-end. NOT yet committed (code + config +
tests + this doc are staged in the working tree). **`config.yaml` smoke knobs NOT yet reverted** —
`target_wells: {20250912: [B01,C01]}` and `image_materialization.smoke_max_time_indices: 3` are still in
config (drop them for a full run; `device: "auto"` is the keeper). The `--rerun-triggers mtime` flag was
needed (provenance triggers wanted to re-run the expensive ND2 ingest; mtime mode skips it). First run
with `--cores 2` OOM-killed C01 (two parallel ND2 loads on CPU) — use `--cores 1` for the smoke.

**Next concrete action:** (1) commit this (suggested split: commit A = device auto-resolution +
AUTO_MODE print + config `device: auto`; commit B = frame_inventory validator repoint + identity-anchored
key + tests; keep smoke knobs out of the committed config or revert them). (2) THEN the separate
ND2-exists/opens hardening commit in the acquisition-inventory validator (still deferred — not done here).
(3) THEN Step 7: strangle legacy `materialize_stitched_images` / `frame_contract` /
`build_frame_inventory_for_well`.

**Open decisions:** whether to persist the smoke knobs (`target_wells`, `smoke_max_time_indices`) in the
committed config or strip them before commit (recommend strip; `device: auto` stays).

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 (session: Step 6 commit 2 — promote materialize_well to the live spine = BEAT 1 FINISH LINE)

**What shipped (commit 2 — orchestration wiring; the branch is now LIVE up to the validated shard):**
- `orchestration/paths.py` — added the live `materialize_well` step (`stage=built_image_data`,
  `fanout=PER_WELL_THEN_MERGE`): per-well `inventory` shard + `done` sentinel. `candidate/` vs live
  stays owned by `materialized_image_paths.py`, not the registry. Paths resolve to
  `built_image_data/{exp}/per_well/{well_id}/{well_id}_frame_inventory.csv` (+ `.validated`, `.done`).
- `rules/frame_inventory.smk` — new `rule materialize_well` fanned over `discovered_wells.txt`
  (via `_frame_inventory_run_wells`), calls `tasks materialize-well … --candidate false`. Repointed
  `rule validate_frame_inventory_for_well` to consume the MATERIALIZER-emitted shard (dropped the
  `frame_contract.csv` adapter; `build_frame_inventory_for_well` removed).
- `Snakefile` — added `rule front_half` (the Beat-1 target) whose input is the validated per-well
  shards for every discovered well. `rule all` (default → features) is UNTOUCHED. Dry-run
  `snakemake -n front_half` parses and the checkpoint-gated DAG resolves
  (ingest→map→join→discover→materialize_well→validate_frame_inventory_for_well).
- **ND2 path now travels INSIDE the acquisition inventory** (`source_nd2_path`), not a CLI/sequencer
  arg — conceptually correct (the inventory is the record of what was acquired). Dropped `nd2_path`
  from the backend, `run_materialize_well`, and the CLI. `well_index` is derived from `well_id` via
  `split_well_id` (no `.split` in the rule). `cmd_materialize_well` is a pure CLI adapter calling
  `run_materialize_well`. Smoke cap (`smoke_max_time_indices`, config `image_materialization.smoke_max_time_indices`;
  ≤0 = no cap) wired through the rule for the no-GPU 2-well × first-3-frames run.
- Tests: **159 passed** (full `tests/data_pipeline/` tree). Snakemake dry-run clean.

**What's broken/half-done:** nothing for Beat 1. The branch runs to the validated per-well
frame_inventory shard. NOT yet committed at the moment of writing this block (about to commit).

**Next concrete action:** run the real no-GPU smoke — `snakemake front_half` for `20250912` with
`image_materialization.smoke_max_time_indices: 3` and target_wells = 2 wells (B01,C01), confirm
per-well shards + `.validated` land under `built_image_data/20250912/per_well/`. Then Step 7
(strangle the legacy `materialize_stitched_images` / `frame_contract` chain) is the cleanup, and
Beat 2 (repoint segmentation/features/QC onto the per-well shard) is the far side of the handoff.

**Open decisions:** Source-image validation (does the ND2 exist/open) conceptually belongs in the
acquisition-inventory validator, not the materializer — trivial for YX1 (one ND2, guarded by the
`source_nd2_path.nunique()==1` check). Add an explicit file-exists/opens assertion there if desired
(small separate edit; not blocking). Stray untracked files left in place by request.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 (session: Step 6 commit 1 — materialization-plan capability gate)

**What shipped (commit 1 of Step 6 — the gate; NO orchestration wiring yet):**
- `image_materialization/materialization_plan.py` — the NOUNS: global vocab (`SUPPORTED_CHANNELS`
  IMPORTED from `schemas/channel_normalization.VALID_CHANNEL_NAMES`, never redefined;
  `SUPPORTED_IMAGE_PRODUCT_TYPES`, `SUPPORTED_PROJECTION_METHODS`, `SUPPORTED_XY_COMPOSITION_REQUESTS`,
  `RESOLVED_XY_COMPOSITIONS`), 4 frozen dataclasses (`ImageProductRequest`/`ResolvedImageProduct`
  + plan tuples), `load_image_materialization_plan(config)`, and global product-shape grammar
  (`projection` requires a method; `z_stack` forbids one). No scope behavior. Exceptions are
  scope-neutral.
- `image_materialization/scope/scope_resolver_for_materialization_plan.py` — the semantic
  TRANSLATOR: `resolve_materialization_plan(scope_name, requested_plan) → ResolvedMaterializationPlan`.
  Does NOT touch a backend. YX1 live (required axes strict; `xy_composition` auto/identity→identity,
  mosaic→identity+warning). Keyence is a RESERVED private sketch — deliberately NOT routed by the
  public resolver (no cardboard doorway).
- `image_materialization/run_materialize_well.py` — the SEQUENCER (renamed from materialize_well.py):
  load plan → resolve → call backend with the raw inputs it was HANDED (it does not gather/resolve
  file args — tasks.py/Snakemake does). Step-6 backend selection is YX1-only; `nd2_path` is a
  YX1-shaped Step-6 input (scope-neutral input bundles are the next migration).
- `scope/yx1/materialize_well_yx1.py` — EXECUTOR: now takes `resolved_plan`, asserts every product
  resolves to `xy_composition=='identity'` (guards a future resolver bug), and honors a TEMPORARY
  `smoke_max_time_indices` cap (no-GPU 2-well × first-3-frames smoke).
- Tests: 68 passed. New `test_materialization_plan.py` + `test_scope_resolver_for_materialization_plan.py`;
  updated `test_materialize_well_yx1.py` (resolved-plan signature, identity-assert, smoke cap). Tests
  never mint ids (use `build_well_id`). Real-import smoke of the 4-module graph clean.

**Key locked design (request ≠ resolved vocabulary):** config requests products → resolver normalizes
per scope → backend executes only resolved. `stitch` is renamed `xy_composition` (single-tile YX1
doesn't "skip stitching" — its XY composition RESOLVES to identity). Separation of concerns:
plan=nouns, resolver=translator (no backend), run_materialize_well=sequencer (backend selection),
backend=executor. See memory `project_materialization_plan_resolver.md`.

**What's broken/half-done:** `tasks.py` still has the old `cmd_materialize_yx1_well_candidate` calling
the backend directly — NOT yet repointed to `run_materialize_well`. That is the first move of commit 2.
Nothing committed for commit 2 yet.

**Next concrete action (Step 6 commit 2 — promote to live spine):**
(1) Repoint `tasks.py`: `cmd_materialize_yx1_well_candidate` → a dispatcher that resolves CLI args
(nd2_path, inventory shard, config) and calls `run_materialize_well(...)`. tasks.py stays a pure CLI
adapter. (2) `orchestration/paths.py`: add the live `materialize_well` step row
(`fanout=PER_WELL_THEN_MERGE`, per-well shard + done sentinel); `candidate/` vs live stays owned by
`materialized_image_paths.py`, not the registry. (3) `rules/frame_inventory.smk`: add a
`materialize_well` rule fanned over `discovered_wells.txt` calling the materializer `candidate=False`;
repoint `validate_frame_inventory_for_well` to read the materializer-emitted shard (drop the
`frame_contract.csv` adapter). (4) `Snakefile`: add per-well `{well_id}_frame_inventory.csv.validated`
to the front-end target so the branch goes LIVE. Verify: `20250912` runs
`ingest→…→materialize_well[well_id]→validate_frame_inventory_for_well` end-to-end (per-well inventories
key on `time_index`, derived ids recompute, paths resolve). = Beat 1 DONE.

**Open decisions:** none blocking commit 2. (Stray untracked files
`specs/front_end/may_need_to_be_domunented!.md` + `tests/improvements/` are left in place by request —
not work products, clean up later.)

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
