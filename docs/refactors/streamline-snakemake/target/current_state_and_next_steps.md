# Current State & Next Steps — the STATUS doc

**Status:** the "where are we RIGHT NOW" anchor. The current snapshot at the top is the live truth;
the dated sections further down are earlier verified state, kept for history. Design lives in
`specs/`; the active front-half plan is `front_half_reorg_roadmap.md`. See `README.md` for the map.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-25 (ingest acquisition path grouping)

**What shipped (commit `cb9343ee`):** gave the five experiment-grain acquisition-ingest steps a
`product_dir` so the front-of-pipeline tables stop laying flat under `acquisition/{exp}/`.

| Step (key UNCHANGED) | folder now |
|---|---|
| `ingest_plate_metadata` | `ingest_metadata/` |
| `ingest_scope_metadata` (BOTH artifacts) | `ingest_metadata/` |
| `apply_position_to_well_mapping` | `ingest_metadata/` |
| `map_positions_to_wells` | `well_identities/` |
| `discover_wells` | `well_identities/` |

Resulting acquisition tree:
```
acquisition/{exp}/
  ingest_metadata/   plate_metadata.csv, scope_metadata__{scope}.csv,
                     acquisition_inventory__{scope}.csv, scope_metadata_mapped.csv (+.validated)
  well_identities/   position_well_mapping.csv (+.provenance.json), discovered_wells.txt
  materialized_images/ ...
  frame_inventory/   (commit 6e879632)
```

**Key design move:** putting BOTH `ingest_scope_metadata` artifacts (`scope_metadata__` +
`acquisition_inventory__`) under one `ingest_metadata/` folder dissolved the need for artifact-level
`product_dir` — the one real machinery question. `ingest_metadata/` (not `acquisition_metadata/`)
because the parent regime is already `acquisition/`; the qualifier says "tables emitted by ingest."
`well_identities/` (not `wells/`) names exactly what that layer owns: position→well identity
resolution. The mapped scope table stays in `ingest_metadata/` (noun is "scope metadata," mapping is
an adjective; `well_identities/` owns identity resolution, not its consumers).

**Scope discipline:** only `PIPELINE_STEPS` `product_dir` values + path tests changed. Step keys,
function/verb/rule names, CLI args unchanged. All Snakefile path constants resolve through
`artifact_path(STEP, ...)`, so the move is automatic — verified nothing hardcodes these paths.

**Verified:** 186 tests pass (`tests/data_pipeline/{pipeline_orchestrator,metadata_ingest}`,
importlib); resolved-path probe confirms the target tree.

**Next concrete action:** the deferred **code-vocabulary rename** (folder↔function drift from both
grouping commits) — `discover_product_shards_for_well` → `list_available_products_for_well`,
`frame_inventory_product*` → `product_inventory*`, etc. Best folded into the Commit-5 behavior work
or done as a dedicated vocabulary-only pass. Also still uncommitted: the z-stack audit gap fixes
(detection projection filter, `parse_image_id_with_z_index`) — see snapshot below.

**Open decisions:** when/how to do the code-vocabulary rename.

---

## ⭐ SNAPSHOT — 2026-06-25 (frame_inventory path grouping)

**What shipped (commit `6e879632`):** grouped the product-shard frame_inventory lineage under
`frame_inventory/` so the on-disk folder names tell the build story. Pure path-contract change.

| Step key (UNCHANGED) | folder before | folder now |
|---|---|---|
| `resolved_product_plans` | `resolved_product_plans/` | `frame_inventory/resolved_product_plans/` |
| `frame_inventory_products` | `frame_inventory_products/` | `frame_inventory/product_inventories/` |
| `discovered_product_shards` | `discovered_product_shards/` | `frame_inventory/available_products/` |

Plus the availability manifest renamed at the contract level:
`{well_id}_available_products.csv` (was `…_discovered_product_shards.csv`); manifest columns
`product_inventory_{csv,validated}` (were `frame_inventory_product_{csv,validated}`). Canonical
merged inventory stays at `frame_inventory/{experiment_id}_frame_inventory.csv` (shared
`PATH_MODE_MERGED` convention — deliberately NOT moved to a `merged/` subdir).

**Scope discipline:** only `PIPELINE_STEPS` `product_dir` values + the manifest filename/columns +
path tests changed. Step keys, function/verb names (`discover_product_shards_for_well`,
`frame_inventory_product*` CLI args), and rule names are intentionally unchanged — the
**code-vocabulary rename is deferred** to the behavior commit (prefer `list_available_products_for_well`,
`product_inventories`, `product_inventory_csv` there). **Ingest-stage cleanup is deferred** (future:
group flat acquisition CSVs into `metadata/`, `acquisition_inventory/`, `well_identities/` — NOT
one-folder-per-CSV; needs per-artifact `product_dir` since `ingest_scope_metadata` emits two families).

**Verified:** 193 tests pass (`tests/data_pipeline/{pipeline_orchestrator,image_materialization}`,
importlib). Resolved-path probe confirms the target tree and that executor-write / discovery-scan /
validate-input dirs all move together through the registry (no hardcoded path can drift).

**Next concrete action:** decide Commit 2 (ingest-stage folder grouping) separately. Do NOT start it
implicitly. The audit's gap #1/#2 fixes (detection projection filter, `parse_image_id_with_z_index`)
remain uncommitted in the working tree as a distinct change — see the snapshot below.

**Open decisions:** Commit 2 ingest layout (per-artifact `product_dir` mechanism); when to do the
deferred code-vocabulary rename.

---

## ⭐ SNAPSHOT — 2026-06-25 (z-stack audit gap closures, uncommitted)

**What shipped:** closed the two remaining z-stack wire-through gaps the audit surfaced — both were
spec deliverables that were never landed in the Commit-0–5 run.

1. **Detection consumer safety (spec §10 / plan A4b).** `detection/run_frame_detection.py` now
   selects frames via a named `_projection_bf_rows(frame_inventory)` helper:
   `channel_id == "BF" AND image_product_type == "projection"`, with a column-absent back-compat
   branch (pre-z_stack inventories keep all BF rows). Previously the router filtered on `channel_id`
   alone, so any assembled inventory carrying a `BF__z_stack` shard would have pushed every Z plane
   into SAM/detection/tracking. Tests: mixed inventory → projection-only; column-absent → all BF;
   router skips planes.
2. **Z-aware parser (spec §1 identity layer).** `shared/identifiers/parsers.py` gains
   `parse_image_id_with_z_index` (the z-aware inverse of `build_image_id`; `z_index=None` for
   projection, integer plane for z_stack), exported from the package `__init__`. `parse_image_id`
   now **rejects** a z-stack id loudly and names the sibling, instead of mis-parsing the `_z` token
   as a channel — closing the "silent z drop" hole. All existing `parse_image_id` callers
   (`build_embryo_id`, `build_snip_id`, snip_processing, snip_identity_contract) operate on
   projection frames, so the new guard adds protection without regression.

**Verified:** `PYTHONPATH=src "$PYTHON" -m pytest` (importlib) over
`tests/data_pipeline/{detection,image_materialization,shared}/`,
`tests/data_pipeline/metadata_ingest/test_frame_inventory.py`,
`tests/data_pipeline/segmentation/physical_embryo_registry/`,
`tests/data_pipeline/snip_processing/` → all green (228 + 46 + 68 across runs). No real-data run
needed: both edits are pure inventory/identity logic with mocked-backend coverage.

**What's broken/half-done:** nothing new. Prior note still holds — the real additive smoke
overwrote the B01 canonical smoke inventory; rerun the desired product overlay before using B01
canonical outputs downstream.

**Next concrete action:** unchanged — choose the next scale gate (recommended two-well additive
smoke `20250912_B01`/`20250912_C01`, `smoke_max_time_indices: 1` through `front_half`), then decide
on a committed smoke overlay vs. ad hoc operator proofs.

**Open decisions:** none.

---

## ⭐ EARLIER SNAPSHOT — 2026-06-24 22:53

**What shipped:** Commit 5 is complete and committed (`5246407b`). Native product materialization now
feeds canonical `frame_inventory` through:
validated `frame_inventory_products` shard → `discover_product_shards_for_well` →
`assemble_well_frame_inventory` → existing strict `validate_frame_inventory_for_well`. Product
discovery scans the on-disk product-shard directory and includes only CSVs with `.csv.validated`
sidecars; config product keys are DAG triggers, not the discovery universe. The old native
`materialize_well` Snakemake writer is no longer the canonical path producer. Also committed
`e44e5a4a`, which writes the merged Snakemake runtime config to
`data_pipeline_output/_snakemake_runtime/merged_config.yaml` and passes that to task entrypoints, so
`--configfile` overlays and `IMAGE_PRODUCT_KEYS` cannot diverge.

**Verified:** targeted Python suite passed:
`PYTHONPATH=src "$PYTHON" -m pytest tests/data_pipeline/image_materialization/test_image_product_keys.py
tests/data_pipeline/image_materialization/test_product_shard_assembly.py
tests/data_pipeline/pipeline_orchestrator/test_tasks_parser.py
tests/data_pipeline/pipeline_orchestrator/test_paths.py
tests/data_pipeline/metadata_ingest/test_frame_inventory.py
tests/data_pipeline/metadata_ingest/test_frame_inventory_strict_gate.py` → 87 passed. Forced B01
dry-run for canonical validation plans the intended chain:
`write_resolved_product_plan_for_well` → `materialize_image_product_for_well` →
`validate_frame_inventory_product_for_well` → `discover_product_shards_for_well` →
`assemble_well_frame_inventory` → `validate_frame_inventory_for_well`. Real CUDA z-stack smoke on
20250912_B01 with one timepoint completed 6/6: A100 visible, `AUTO_MODE_CHOSEN: requested='cuda' ->
CUDA`, product shard `20250912_B01_BF__z_stack_frame_inventory.csv` has 15 rows for `time_index=0`
and `z_index=0..14`, 15 PNGs were written under
`materialized_images/20250912_B01/z_stack/BF/`, discovery listed `BF__z_stack`, canonical
`20250912_B01_frame_inventory.csv` assembled to the same 15 z-stack rows, and both product/canonical
`.validated` sentinels exist. Additive proof also completed 6/6 on real 20250912_B01 with one
timepoint: projection materialization produced 1 product-shard row, discovery listed both
`BF__projection__focus_stack` and `BF__z_stack`, canonical assembly produced 16 rows
(`projection: 1`, `z_stack: 15`), projection `z_index` is NA, z-stack `z_index` spans `0..14`, and
strict canonical validation passed.

**What's broken/half-done:** no known code blocker in the product-shard/assembly path. The real
additive smoke intentionally overwrote the canonical B01 smoke inventory with a one-timepoint
projection+z_stack assembled manifest; rerun the desired product overlay before using B01 canonical
outputs for downstream projection-only work.

**Next concrete action:** choose the next scale gate. Recommended: run a two-well additive smoke
(`20250912_B01`, `20250912_C01`, `smoke_max_time_indices: 1`) through `front_half` with forced
discovery/assembly, then decide whether to add a committed smoke overlay for product/additive tests
or keep these as ad hoc operator proofs.

**Open decisions:** none.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-24 22:33

**What shipped:** product-grain materialization fanout is committed (`b4d99350`). Added
`resolved_product_plans.py`, which writes/loads one schema-versioned resolved product plan JSON per
`(well_id, product_key)` and validates that `product_key` matches the resolved product fields. Added
task verbs `write-resolved-product-plan-for-well` and `materialize-image-product-for-well`. The
Snakefile now derives `IMAGE_PRODUCT_KEYS` through the real config→plan→scope-resolver path, not a
hardcoded list, and exposes a `front_half_products` target that stops at validated product
frame-inventory shards.

**Verified:** product-shard dry-run for
`.../frame_inventory_products/per_well/20250912_B01/20250912_B01_BF__projection__focus_stack_frame_inventory.csv.validated`
plans the intended chain:
`write_resolved_product_plan_for_well` → `materialize_image_product_for_well` →
`validate_frame_inventory_product_for_well`. `front_half_products -n` parses and reaches the
discovery checkpoint. Pytest slice
`tests/data_pipeline/image_materialization/test_image_product_keys.py
tests/data_pipeline/image_materialization/test_resolved_product_plans.py
tests/data_pipeline/image_materialization/test_materialized_image_paths.py
tests/data_pipeline/image_materialization/test_scope_resolver_for_materialization_plan.py
tests/data_pipeline/image_materialization/scope/yx1/test_materialize_well_yx1.py
tests/data_pipeline/pipeline_orchestrator/test_tasks_parser.py
tests/data_pipeline/pipeline_orchestrator/test_paths.py` → 107 passed, 1 existing projection dtype
warning.

**What's broken/half-done:** canonical per-well `frame_inventory` is still not assembled from product
shards. The old `materialize_well`/`front_half` canonical path remains in place as the existing
pre-assembly path; the new product-grain path is exposed separately as `front_half_products` and
direct product-shard targets.

**Next concrete action:** Commit 5 — add `discovered_product_shards` discovery and
`assemble_well_frame_inventory`. Discovery should list active validated product shard CSVs for one
well; assembly should concatenate those product shards into the old canonical
`frame_inventory/per_well/{well_id}/{well_id}_frame_inventory.csv`, then the existing strict
`validate_frame_inventory_for_well` gate can validate the assembled canonical shard. Verify
projection-only assembly reproduces the old canonical per-well frame inventory.

**Open decisions:** none for product fanout. Keep materialization writing product shards only;
canonical writes start in the assembly rule.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-24 22:21

**What shipped:** product-shard path contract is now committed (`c8bd4f2d`). Added
`image_materialization/image_product_keys.py` with `build_image_product_key`
(`BF__projection__focus_stack`, `BF__z_stack`), plus `PIPELINE_STEPS` rows for
`resolved_product_plans`, `frame_inventory_products`, and `discovered_product_shards`. Product keys
are filename-level `format_vars`, not a new path mode. Product frame-inventory shard sentinels use
the existing `validated_path(...)` trailing sidecar convention (`.csv.validated`).

**Verified:** `PYTHONPATH=src "$PYTHON" -m pytest tests/data_pipeline/image_materialization/test_image_product_keys.py
tests/data_pipeline/pipeline_orchestrator/test_paths.py` → 52 passed. While touching the path tests,
the stale `auxiliary_masks` assertions were aligned to the already-registered `snip_auxiliary_masks`
step.

**What's broken/half-done:** no DAG behavior is wired yet by design. Product shard paths now exist,
but no rule emits resolved product plan JSONs, no product materialization rule writes
`frame_inventory_products`, and canonical per-well `frame_inventory` is not assembled from product
shards yet.

**Next concrete action:** Commit 4 — add the product-resolution/fanout layer. Implement a resolver
entrypoint that writes one `resolved_product_plans/per_well/{well_id}/{product_key}_resolved_product_plan.json`
per `(well_id, product_key)`, then update `run_materialize_well.py` / `tasks.py` / `materialize_well_native.smk`
so materialization consumes exactly one resolved product plan and writes only
`frame_inventory_products/per_well/{well_id}/{well_id}_{product_key}_frame_inventory.csv`. Verify
projection-only still works through product-shard materialization.

**Open decisions:** none for path layout. Assembly semantics are decided: the old canonical
`frame_inventory/per_well/{well_id}/{well_id}_frame_inventory.csv` is written only by the later
assembly step, not by product materialization.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-24 21:50

**What shipped:** z-stack materialization is wired through the contract and YX1 single-product
executor in two commits:
- `0f5a613d` — `build_image_id(..., z_index=None)` preserves projection IDs byte-for-byte and emits
  z-aware IDs for planes; `frame_inventory` now requires-present nullable `z_index`, derives
  z-aware `image_id`s, and checks uniqueness through derived `image_id` so projection `NA` rows still
  collide while distinct z planes pass.
- `960a7667` — YX1 resolver accepts BF `z_stack`; `z_stack_frame_path` is live; the YX1 materializer
  has a product-grain helper that writes inventory-declared Z planes without projection and emits
  z-aware frame-inventory rows. Projection behavior remains covered.

**Verified:** `PYTHONPATH=src "$PYTHON" -m pytest tests/data_pipeline/shared/identifiers/test_identifiers.py
tests/data_pipeline/image_materialization/ tests/data_pipeline/metadata_ingest/test_frame_inventory.py
tests/data_pipeline/metadata_ingest/test_frame_inventory_strict_gate.py` → 166 passed, 1 existing
projection dtype warning.

**What's broken/half-done:** Snakemake/orchestration product fanout is not wired yet. The live
`materialize_well` rule is still well-grain and writes one per-well `frame_inventory` shard; the
compatibility wrapper now fails loud if a resolved plan contains more than one product. A config with
only BF `z_stack` can reach the YX1 executor path, but a config requesting both projection and
z_stack still needs product-key fanout before it can run as separate product jobs.

**Next concrete action:** wire product fanout at the orchestration seam: add a product key/selection
path through `src/data_pipeline/pipeline_orchestrator/rules/materialize_well_native.smk`,
`src/data_pipeline/pipeline_orchestrator/tasks.py::cmd_materialize_well`, and
`src/data_pipeline/image_materialization/run_materialize_well.py` so each job passes exactly one
resolved `ResolvedImageProduct` into `materialize_yx1_product_for_well`; then verify a `snakemake -n`
for config products `{projection, z_stack}` fans out cleanly and the targeted pytest slice still
passes.

**Open decisions:** decide the on-disk contract for product-specific frame-inventory shards in
`PIPELINE_STEPS`/rules before wiring fanout, because today there is only one per-well
`frame_inventory` artifact path.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-24 (session: QC half of feature_world.md — all 4 MVP QC products)

**What shipped (the QC half of `feature_world.md`, built product-by-product, one commit each):**
The `quality_control/` domain now matches `feature_extraction/` in shape + wiring. Four MVP QC
products in the target per-product layout, each: contract (spine-first, imported from the minting
site) + compute + config + entrypoint, a `PIPELINE_STEPS` row, thin `tasks.py` verbs, a per-well
build→validate→merge `.smk` included in the Snakefile, and tests in the parallel tree.
- `surface_area_qc` (ffd47280) — stage-binned `sa_outlier_flag`; packaged
  `references/surface_area_reference_v1.csv` (no paths.py row); canonical k=1.4/0.7 (legacy 1.2/0.9
  dropped); self-documenting band statement. 22 tests.
- `mask_quality_qc` (89e3b3fa) — `edge`/`discontinuous_mask`/`overlapping_mask` flags, re-pointed
  off raw SAM2 `mask_rle` onto canonical `frame_masks` via `decode_binary_mask_rle`; overlap per
  `image_id` within well between DISTINCT `physical_embryo_id`s; iou=0.10. No composite. 14 tests.
- `death_detection` (fd729289) — FULL re-architecture: group `physical_embryo_id`, sort
  `time_index`, hours-based lead-time via `frame_inventory.elapsed_time_s`, TWO output grains
  (`death_detection_qc` per snip + `death_event` per animal, no `embryo_id`). Files: compute /
  persistence / death_event / grain_reconciliation / contract / config / entrypoint. 17 tests incl.
  the two-grain integration test + non-uniform-interval lead-time test. Spec synced (file renames).
- `snip_qc` (91a05ad6) — final `use_snip` + `qc_fail_reasons` verdict (parquet), ORed from the three
  flag families via the registry-resolved `inputs.py`; pure `build.py`. 13 tests.

**Verified:** `pytest tests/data_pipeline/quality_control/` → 66 passed; full feature+QC → 150 passed.
`snakemake -n` for the **merged `snip_qc` target** plans the ENTIRE QC DAG end-to-end
(mask_geometry / stage_predictions / fraction_alive / frame_masks / snip_processing →
surface_area_qc + mask_quality_qc + death_detection → snip_qc), 43 jobs, all registry-resolved, no
raw-path errors. Also fixed a pre-existing `rule all` NameError (`_paths_mod` → `PATH_MODE_MERGED`).

**Legacy retired (conservative — grep-confirmed no live importers before each delete; never `-A`):**
`core/` and `entrypoints/` are empty; `consolidation/`, `segmentation_qc/`, `morphology_qc/` deleted;
the orphaned raw-path `rules/quality_control.smk` deleted. No `QUALITY_CONTROL_DIR` rule paths remain.
**HELD (live external importers — do NOT delete this pass):**
- top-level `quality_control/death_detection.py` + `surface_area_outlier_detection.py` —
  `src/build/build04_perform_embryo_qc.py` still imports `compute_dead_flag2_persistence` /
  `compute_sa_outlier_flag`;
- `schemas/quality_control.py` + `quality_control/validators.py` + `quality_control/io/` —
  `analysis_ready/io/loaders.py` still imports `REQUIRED_COLUMNS_QC`. This cluster retires with the
  (deferred) `analysis_ready` wiring effort, not here. `quality_control/reporting/` is a pre-existing
  empty stub, left as-is.

**Next concrete action:** none required for the QC products (complete + verified). When
`analysis_ready` is wired as a real pipeline step, it should consume `snip_qc`'s `verdict` parquet
(per-snip `use_snip`/`qc_fail_reasons`) and `death_event`, then the held
`schemas/quality_control.py` + `validators.py` + `quality_control/io/` cluster can be retired and
`build04`'s two top-level legacy imports migrated.

**Open decisions:** none.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: legacy embeddings — wire `latent_embeddings` into orchestration)

**What shipped (the producer side — encode → validate → merge fully wired):** the pure encode code was
already built + tested; this pass runs it in the pipeline.
- `feature_extraction/legacy_embeddings/entrypoint.py` (NEW) — the Python-3.9 batch body. Glue only:
  resolve_legacy_model_dir → load_legacy_vae_encoder (loaded ONCE per invocation) → collect_snip_inputs →
  encode_snips → stamp `embedding_model_name` → validate_latent_embeddings → write parquet. Takes paired
  `--snip-inventory-csv`/`--output-parquet` lists (the rule passes one pair per well).
- `feature_extraction/legacy_embeddings/contract.py` — resolved its own TODO: `embedding_model_name` is now
  a REQUIRED, non-null column (latents without model provenance are unlabeled vials); validator enforces it.
- `orchestration/paths.py` — NEW `latent_embeddings` `PIPELINE_STEPS` row (FIRST `stage="features"` step;
  product_dir=`latent_embeddings`, fanout=PER_WELL_THEN_MERGE, execution=RUN_BATCH; latents parquet artifacts).
  Resolves to `features/<exp>/latent_embeddings/per_well/<well>/<well>_latents.parquet` (+ merged + sentinels).
- `pipeline_orchestrator/tasks.py` — `validate-latent-embeddings` + `merge-latent-embeddings` verbs (run under
  the normal 3.10 RUN env — they only read/validate parquet).
- `Snakefile` — NEW `MODEL_RUN` prefix: prefers `env.yaml.runtime.model_python_executable` (direct 3.9 path),
  falls back to `conda run -n {model_python_env}`; `None` if neither set. `include`s `latent_embeddings.smk`.
- `rules/latent_embeddings.smk` (NEW) — `encode_latent_embeddings_for_well` (runs under MODEL_RUN/3.9) +
  per-well validate + merge + merged validate. Encode depends on the VALIDATED snip_inventory shard.

**Verified:**
- `pytest tests/data_pipeline/feature_extraction/legacy_embeddings/` — 59 passed (incl. NEW
  `test_entrypoint.py`: mocked encoder, 2 wells, asserts model loads ONCE, one validated provenance-stamped
  parquet per well, row order preserved, count-mismatch fails loud).
- `snakemake -n` merged-latents target plans 95 encode + 95 validate + 1 merge, no errors; all 4 rules
  register; both task verbs registered.

**Deliberately deferred (NOT done — these are real, separate efforts):**
1. **Real-weights end-to-end smoke.** The legacy VAE weights are still NOT staged on disk
   (`models_root/legacy/20241107_ds_sweep01_optimum`). Everything is verified with a mocked encoder; the
   real-load gate (`load_model_smoke.py` exit 0; a real encode) waits for weights to be staged.
2. **`analysis_ready` join.** Spec "Done When" wants `analysis_ready` to gain `latents.parquet`, join on
   `snip_id`, set `embedding_calculated=True`. BUT the existing `analysis_ready/` module is an OLDER
   standalone surface (uses `z0/z1` columns + `time_int`/`well_index` vocabulary) and is NOT wired into the
   pipeline at all (no PIPELINE_STEPS row, no .smk). Wiring it — and reconciling `z_mu_*` vs `z0` and the
   stale vocabulary — is its own stage-by-stage effort, not part of "wire the embeddings producer."

**RUN_BATCH note:** the registry row says execution=RUN_BATCH (load once across wells), but the rule is
declared per-well (one `{well_id}` job) — identical to frame_masks (also RUN_BATCH). A true
single-process-all-wells batch needs Snakemake's `--batch` mechanism, which no rule here uses yet; the
`execution` field documents intent, the optimization is deferred. Model currently loads per well.

**Next concrete action:** stage the legacy VAE weights, run `load_model_smoke.py` (must exit 0), then run
`encode_latent_embeddings_for_well` on one real well and confirm a non-trivial `<well>_latents.parquet`.
Then (separate effort) wire `analysis_ready` as a real pipeline step and add the latents join.

**Open decisions:** the analysis_ready column/vocabulary reconciliation (`z_mu_*` vs `z0`) — defer to the
analysis_ready wiring effort.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: frame_masks validate rule — fix cross-agent merge break + finish TODO)

**Why:** A parallel agent's frame_masks conformance commit (`a4f284d9`) switched `merge_frame_masks` to
`collect_well_shard_paths` (which only returns shards that carry a `.validated` sentinel) but never added a
frame_masks validate rule — so NO sentinel was ever written, the merge would `collect` → `[]` → fail on empty
input. frame_masks was also the only product in the detect→seg→snip chain WITHOUT per-well validation
(frame_inventory/registry/snip_inventory all validate). This pass fixes both and resolves the
`# TODO(frame_masks-validate)` left in the registry rule.

**What shipped:**
- `pipeline_orchestrator/tasks.py` — NEW `cmd_validate_frame_masks` + `validate-frame-masks` subparser. The
  full contract `validate_frame_masks(frame_masks, frame_inventory)` cross-checks masks against frame
  identity, so the verb takes BOTH `--input-csv` and `--frame-inventory-csv` (+ `--output-flag`).
- `rules/frame_masks.smk` — NEW `rule validate_frame_masks_for_well` (writes the per-well `.validated`
  sentinel from the frame_masks shard + its frame_inventory); `_frame_masks_validated` +
  `_frame_masks_validated_for_run` helpers; `merge_frame_masks` now also depends on the per-well
  `.validated` sentinels (so `collect_well_shard_paths` finds validated shards — the empty-merge bug is fixed).
- `rules/physical_embryo_registry.smk` — resolved the TODO: `build_physical_embryo_registry_for_well` now
  depends on the frame_masks per-well `.validated` sentinel (not the raw CSV).
- `rules/snip_processing.smk` — `snip_processing_per_well` now also depends on the frame_masks `.validated`
  sentinel (consistency: every frame_masks consumer waits on a validated shard).

**Verified:**
- `validate-frame-masks` verb on real 20250912_B01 (frame_masks + its frame_inventory) → passes, writes
  sentinel.
- `snakemake -n` snip target chains materialize → validate_frame_inventory → frame_detections → frame_masks
  → **validate_frame_masks** → build/validate registry → snip, no errors. Merged frame_masks target plans
  95× `validate_frame_masks_for_well` feeding `merge_frame_masks` (was previously a guaranteed empty-merge).
- `pytest` snip + registry suites → 29 passed.

**Now consistent:** every per-well product in the chain (frame_inventory, frame_masks,
physical_embryo_registry, snip_inventory) builds → validates (per-well `.validated`) → merges, and every
consumer depends on the VALIDATED upstream shard.

**Open decisions:** none.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: physical_embryo_registry — Stage 4 snip_processing cutover)

**What shipped (Stage 4 of 4 — the LAST stage; snip_processing now JOINS identity instead of minting it):**
The `physical_embryo_registry` is now the sole identity-origination boundary. `snip_processing` consumes it.
- `snip_processing/entrypoints/run_snip_processing.py` — DELETED the per-mask mint chain
  (`parse_embryo_local_track_id` / `track_index_to_embryo_index` / `build_physical_embryo_id`) and its
  imports. Added a `physical_embryo_registry_csv` param + `_physical_embryo_id_by_track` lookup; the crop
  loop now JOINS `physical_embryo_id` on `(well_id, track_id)` and **fails loud** (`ValueError`, message
  names the fix) if a valid mask's track has no registry row. KEPT `build_embryo_id` / `build_snip_id`
  (crop-product naming legitimately stays here).
- `pipeline_orchestrator/tasks.py` — `cmd_snip_processing` threads the new arg; `--physical-embryo-registry-csv`
  added to the `snip-processing` subparser (required).
- `pipeline_orchestrator/rules/snip_processing.smk` — `snip_processing_per_well` now depends on the
  **per-well** `physical_embryo_registry` shard + its `.validated` sentinel (LOCKED: per-well, NOT merged —
  the crop loop is per-well, so depending on the merged table would serialize all wells). Passes the CSV to
  the task verb; docstring updated (join, not mint). (A linter independently upgraded `merge_snip_inventory`
  to use well_runner's `collect_well_shard_paths`/`concat_well_shards_to_file` — kept.)
- Tests: `tests/data_pipeline/snip_processing/test_run_snip_processing.py` — updated the e2e to build +
  pass a registry (via the real Stage-2 builder) and added two cases: join reproduces the registry's
  physical_embryo_id, and fail-loud-on-missing-registry-match. Also patched the untracked smoke caller.

**Verified:**
- `pytest tests/data_pipeline/snip_processing/ tests/data_pipeline/segmentation/physical_embryo_registry/`
  → **29 passed** (3 snip + 26 registry).
- **Identical-to-pre-cutover gate (the required Stage-4 proof):** on real 20250912_B01 data, the
  registry-join `physical_embryo_id` for every valid mask is IDENTICAL to the old per-mask mint chain
  (3 valid masks → 1 animal `20250912_B01_e01`).
- `snakemake -n` for the B01 snip target plans `build_physical_embryo_registry_for_well` →
  `validate_physical_embryo_registry_for_well` → `snip_processing_per_well` cleanly — the new per-well
  registry dependency wires into the DAG with no errors.

**physical_embryo_registry is now COMPLETE (all 4 stages shipped + verified).** The mint chain lives in
exactly one place (the registry builder); snip_processing, and any future identity-carrying consumer, joins.

**Next concrete action:** none for this product. Deferred-per-spec items remain out of scope (channel-free
`physical_embryo_occurrence` table; auditability columns `n_detected_masks`/`first|last_time_index`; a
discrete `tracks` gap/swap QC stage) — add only when a consumer needs them. Separately tracked: the
frame_masks conformance pass (its own agent) that will add the per-well frame_masks `.validated` sentinel,
at which point the `# TODO(frame_masks-validate)` in `rules/physical_embryo_registry.smk` should be resolved.

**Open decisions:** none.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: physical_embryo_registry — Stage 3 orchestration wiring)

**What shipped (Stage 3 of 4 — wire the proven product into orchestration; meaning of no existing stage
changed):**
- `pipeline_orchestrator/orchestration/paths.py` — added the `physical_embryo_registry` `PIPELINE_STEPS`
  row (stage=`object_extraction`, fanout=`PER_WELL_THEN_MERGE`, execution=`EXECUTION_PER_WELL` — cheap CPU,
  deliberately NOT `RUN_BATCH` like frame_masks). Per-well + merged artifact templates. Placed between the
  `frame_masks` (its input) and `snip_inventory` rows.
- `pipeline_orchestrator/tasks.py` — three thin dispatcher verbs + subparsers:
  `build-physical-embryo-registry` (frame_masks CSV → Stage-2 builder → CSV),
  `validate-physical-embryo-registry` (CSV → Stage-2 validator → `.validated` sentinel),
  `merge-physical-embryo-registry` (per-well CSVs → Stage-2 `merge_physical_embryo_registry` which
  re-validates GLOBAL uniqueness → merged CSV). Zero domain logic in tasks.py.
- `pipeline_orchestrator/rules/physical_embryo_registry.smk` (NEW) — build+validate(per-well)+merge+
  validate(merged), mirroring the conformant `frame_inventory.smk` template. Merge input set resolved by
  the planning-time, disk-blind `run_well_shard_paths` (well_runner); merge waits on per-well `.validated`
  sentinels. `include:`d in the Snakefile between frame_masks and snip_processing.
- Upstream dep is the per-well frame_masks **CSV only** (frame_masks has no per-well `.validated` yet); a
  `# TODO(frame_masks-validate)` marks where to add the sentinel input once a SEPARATE agent's frame_masks
  conformance pass lands (that agent is fixing frame_masks' inline-shell merge + adding its validate rules).

**Verified:**
- Task verbs on real 20250912 data: B01 build → 1 row `20250912_B01_e01` (one-based `local_embryo_index=1`);
  validate writes sentinel; C01 build; merge → 2 rows, passes global-uniqueness validation.
- **Cross-check vs old mint (the Stage-4 gate):** for B01 the registry `physical_embryo_id` per
  `(well_id, track_id)` is IDENTICAL to the current snip_processing mint chain over its valid masks — proves
  Stage 4's join will reproduce old IDs exactly.
- Stage-2 suite still green: `pytest tests/data_pipeline/segmentation/physical_embryo_registry/` → 26 passed.
- **snakemake IS on the env** (blocker did not apply): `snakemake -n --list` registers all four new rules;
  a dry-run of the merged target for 20250912 plans 95 build + 95 validate + 1 merge with no errors and
  correct PER_WELL_THEN_MERGE paths.

**Next concrete action (Stage 4 — MANUALLY GATED, do NOT auto-run):** cut `snip_processing` over to JOIN the
per-well registry instead of minting. Delete the mint chain in `run_snip_processing.py:114–116`, add a
`physical_embryo_registry_csv` param + left-join on `(well_id, track_id)` (fail loud on missing match), thread
the arg through the snip task verb + `rules/snip_processing.smk`. LOCKED: per-well snip depends on the
**per-well** registry shard + its `.validated` (NOT the merged table; merged is for experiment-level QC).
Start only after a human reviews the Stage-3 outputs above. Plan: `.claude/plans/handoff-physical-embryo-registry-zazzy-waffle.md` §"Stage 4".

**Open decisions:** none. (frame_masks conformance is a separate agent's task, tracked via the TODO marker.)

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: physical_embryo_registry — Stage 2 registry product)

**What shipped (Stage 2 of 4 — contract + table validator + builder):** the `physical_embryo_registry`
data product itself, mirroring the `frame_masks_contract.py`/`validate_frame_masks.py` split. Files (all
under `src/data_pipeline/segmentation/physical_embryo_registry/`):
- `physical_embryo_registry_contract.py` — `PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS` (identity +
  minimal provenance: physical_embryo_id, experiment_id, well_id, local_embryo_index, track_id,
  track_id_source), `PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY`, `empty_physical_embryo_registry()`.
- `validate_physical_embryo_registry.py` — table validator: grain uniqueness (one row per animal,
  ENFORCES global uniqueness on the merged table), one-based index, round-trip via
  `validate_physical_embryo_id`, no dup (well_id, local_embryo_index)/(well_id, track_id), one track →
  one animal.
- `build_physical_embryo_registry.py` — `build_physical_embryo_registry(frame_masks)`: mints ONE row per
  distinct `(well_id, track_id)` from the DETECTED set (drops NA-track no-mask rows, keeps tracks
  regardless of is_valid_mask), running the mint chain once per animal; `merge_physical_embryo_registry`
  concats well shards + re-validates global uniqueness. Both validate before returning.
- Tests: `tests/data_pipeline/segmentation/physical_embryo_registry/test_physical_embryo_registry.py`.

**Verified:** `pytest tests/data_pipeline/segmentation/physical_embryo_registry/` — **26 passed** (14 new
Stage-2 + 12 Stage-1 spine tests in the same dir).

**Next concrete action (Stage 3 — orchestration wiring):** (1) add a `physical_embryo_registry`
`PIPELINE_STEPS` row in `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` mirroring the
`frame_masks` row (stage="object_extraction", product_dir="physical_embryo_registry",
fanout=PER_WELL_THEN_MERGE, execution=EXECUTION_PER_WELL, per_well + merged artifact templates);
(2) add `build`/`validate`/`merge` task verbs in `pipeline_orchestrator/tasks.py` (pure dispatch to the
Stage-2 functions); (3) create `rules/physical_embryo_registry.smk` (copy `frame_masks.smk`/`frame_inventory.smk`
PER_WELL_THEN_MERGE template) and `include:` it in the Snakefile. Verify by running the build task verb
on a real per-well frame_masks shard (e.g. 20250912_B01) and confirming one row per (well_id, track_id).

**Open decisions:** none for Stage 3.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: physical_embryo_registry — Stage 1 identity validators)

**What shipped (Stage 1 of 4 — reusable identity validators):** the two reusable validators the rest of
the identity pipeline will lean on, per `physical_embryo_registry_world.md`. Files:
- `src/data_pipeline/shared/identifiers/validators.py` — ADDED `validate_physical_embryo_id(physical_embryo_id, *, well_id=None)`:
  the reusable STRING validator (next to `validate_well_id`). Delegates to `parse_physical_embryo_id`
  (already rejects `_e00`/index < 1) + `validate_well_id` on the embedded well; cross-checks the
  supplied `well_id` when given. Exported from `shared/identifiers/__init__.py`.
- `src/data_pipeline/segmentation/physical_embryo_registry/__init__.py` + `snip_identity_contract.py` —
  NEW package (home is **`segmentation/`**, beside the real `frame_masks` product — decision locked with
  mdcolon, NOT the spec's literal `segmentation_and_tracking/`). `snip_identity_contract.py` authors the
  grain-aware spine validator `validate_snip_grain_identity_columns(df, *, grain, physical_embryo_registry_df, check_sources, scope_label)`
  — the table-level enforcement of the LOCKED 5-clause identity-carrying law (a syntactically valid
  snip_id is NOT enough; IDs must agree). `check_sources` mirrors `validate_yx1_acquisition_inventory`.
- Tests: `tests/data_pipeline/shared/identifiers/test_validate_physical_embryo_id.py` (7) +
  `tests/data_pipeline/segmentation/physical_embryo_registry/test_snip_identity_contract.py` (12).

**Verified:** `pytest` over both new test files — **19 passed** under `segmentation_grounded_sam`. Package
imports clean (`PYTHONPATH=src`).

**Next concrete action (Stage 2):** in `src/data_pipeline/segmentation/physical_embryo_registry/`, mirror
the `frame_masks_contract.py`/`validate_frame_masks.py` split — create
`physical_embryo_registry_contract.py` (`*_REQUIRED_COLUMNS`, `*_UNIQUE_KEY`, `empty_*`),
`validate_physical_embryo_registry.py` (table validator, delegates row-wise to
`validate_physical_embryo_id`), and `build_physical_embryo_registry.py` (distinct `(well_id, track_id)`
from `frame_masks` — the DETECTED set, drop NA-track no-mask rows — → mint chain once per track). Add
`tests/data_pipeline/segmentation/physical_embryo_registry/test_physical_embryo_registry.py`. Verify:
`pytest tests/data_pipeline/segmentation/physical_embryo_registry/`.

**Open decisions:** none for Stage 2. (Staged build per mdcolon: verify + snapshot + commit at the end of
every stage; do not roll stages into one commit.)

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: model world — legacy VAE loading foundation)

**What shipped:** The model-loading *foundation* for the embeddings stage (Option A — no active
orchestration touched). Files:
- `specs/model_input_handoff_contract.md` — appended **§9 "Model loading & pipeline fit (2026-06-22)"**
  that supersedes the stale §5–§6 premises. Locks the boundary doctrine: **only files cross the
  3.9/3.10 env line, never model objects**; the embeddings command runs wholly in Python 3.9;
  `load_model_subprocess.py` is explicitly NOT to be built. The 3.9 interpreter is addressed by
  `env.yaml.runtime.model_python_executable` (preferred, direct path) with `model_python_env`
  (`conda run`) as fallback — never `config.yaml`.
- `env.example.yaml` (+ local `env.yaml`) — added `runtime.model_python_executable` +
  `runtime.model_python_env` + the `models_root/legacy/<model_name>/` convention note.
- `src/data_pipeline/features/legacy_embeddings/model_paths.py` — `resolve_legacy_model_dir(models_root,
  model_name)` (path-pure, nested `final_model/` tolerated, fail-loud naming the missing dir + fix).
- `src/data_pipeline/features/legacy_embeddings/load_model_smoke.py` — standalone 3.9 in-process
  load+report script (no model object crosses to 3.10).
- `tests/data_pipeline/features/legacy_embeddings/test_model_paths.py` — **4 passed**.

**Verified:** resolver test 4 passed under the main env; load-smoke under
`mseq_pipeline_py3.9` (Python 3.9.23) fails loud naming the missing
`…/models/legacy/20241107_ds_sweep01_optimum` (honest state — no legacy weights staged yet); the
non-3.9 guard refuses with exit 2 when run under 3.10. Note: test-tree dirs intentionally have **no**
`__init__.py` (matches sibling convention; pytest `prepend` import mode + `data_pipeline` namespace pkg).

**What's broken/half-done:** nothing within this pass. Deliberately deferred (the embeddings *product*
doesn't exist yet): `MODEL_RUN` Snakefile prefix, `tasks.py compute-embeddings` verb, the `embeddings`
`PIPELINE_STEPS` row, per-well/validate/merge embeddings rules, the encode-loop port (source images
from `snip_inventory.processed_snip_path`, not the broken `src.core.data` glob), and the
`analysis_ready` `embedding_calculated` flip.

**Next concrete action:** when staging the legacy VAE weights, drop them at
`models_root/legacy/20241107_ds_sweep01_optimum/` and re-run the load-smoke to confirm a real in-3.9
load + metadata print. Then start the NEXT pass: add the `embeddings` `PIPELINE_STEPS` row + `MODEL_RUN`
prefix + `compute-embeddings` verb + rule trio (spec §9.5). Doctrine: *rules come when the product exists.*

**Open decisions:** `model_name` default (legacy `20241107_ds_sweep01_optimum`); `use_snip` gating
default = "encode all, gate downstream in analysis_ready" (spec §9.6). Neither blocks the next pass.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: unet_snip backend — snip_auxiliary_masks Steps 1+2+viz)

**What shipped:**
- `src/data_pipeline/segmentation/backends/unet_snip/` — new backend package:
  - `snip_auxiliary_masks_contract.py` — contract, validator, `load_snip_auxiliary_masks()` reader
  - `run_unet_snip.py` — `AuxiliaryMaskPredictor` interface + `run_unet_for_snip_inventory()` runner
  - `model_loader.py` — `UNetSnipModelConfig`, `FishModelSnipPredictor`, `load_unet_snip_predictors()`
- `src/data_pipeline/models/unet.py` — `load_fish_unet_model()` (plain state dict + Lightning format)
- `src/data_pipeline/viz/render_snip.py` — `render_snip_auxiliary_masks()` + `render_snip_auxiliary_masks_contact_sheet()`
- 24 tests passing (17 unet + 7 viz). GPU smoke on real `20250912_B01_e01` snips: foreground and yolk firing correctly, via/focus/bubble silent on healthy in-focus embryos — visually confirmed.
- Commits: `a3ab55b4` (Steps 1+2), `caa6afd4` (loader + viz)

**What's broken/half-done:** nothing. `tests/improvements/snip_processing_smoke/output_real/snips/20250912_B01_e01/unet_auxiliary_masks_contact_sheet.jpg` is an untracked smoke artifact (not committed, intentionally).

**Next concrete action:** Step 3 — wire into the DAG. Add `tasks.py` dispatch entry (`snip-auxiliary-masks` subcommand) and `rules/snip_auxiliary_masks.smk` rule that calls `load_unet_snip_predictors(cfg)` + `run_unet_for_snip_inventory()`, consuming `snip_inventory` shard and emitting `<well_id>_snip_auxiliary_masks.csv`. Checkpoint paths come from `env.yaml` under a new `unet_snip` block pointing at `data_pipeline_output/models/segmentation/`.

**Open decisions:** none blocking Step 3.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 18:00 (session: viz package — contract-native overlay rendering)

**What shipped:** New `src/data_pipeline/viz/` package with 4 files:
- `config.py` — `RenderConfig` dataclass, `COLORBLIND_PALETTE`, `OVERLAY_COLORS`
- `overlay.py` — `draw_boxes()`, `draw_masks()`, `draw_banner()`, `color_for_key()` primitives (cv2, reads contract DataFrames per frame)
- `render_well.py` — `render_detection_video()`, `render_segmentation_video()`, `render_combined_video()` (iterate frame_inventory, write MP4)
- `__init__.py` — re-exports the three render functions

Test: `tests/data_pipeline/viz/test_render_well.py` — 4 tests, all pass. No GPU, synthetic frames.

**What's broken/half-done:** nothing. `segmentation/video_generation/` left untouched (legacy JSON-based eval videos). No Snakemake rule added yet (deliberate — this is a utility layer for now).

**Next concrete action:** Start Session C (real SAM2 backend invocation). See the previous snapshot below for the Session C scope. After Session C lands, consider wiring `render_combined_video` into a Snakemake QC rule so per-well overlay videos are emitted as DAG artifacts.

**Open decisions:** none blocking Session C.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: Session B — fake-predictor end-to-end segmentation contract)

**What shipped:** Session B is complete in four stage commits:
`939252d2` (`segmentation prompts: add SAM2 prompt detection vocabulary and validator`),
`559c1b7c` (`segmentation adapter: add SAM2 output adapter`),
`cb2abb69` (`segmentation fake predictor: add deterministic fake SAM2 predictor`),
`fecad307` (`segmentation session B: fake-predictor end-to-end integration test`).

New package: `src/data_pipeline/segmentation/backends/sam2_video/` containing:
- `prompt_detections.py` — `PROMPT_DETECTION_COLUMNS`, `validate_sam2_prompts`, `validate_frame_masks_against_sam2_prompts`
- `adapt_sam2_output.py` — `adapt_sam2_well_output(well_id, sam2_raw_output, model_frame_view, prompt_detections, *, model_id)` converting `dict[int, dict[int, np.ndarray]]` to `FRAME_MASKS_REQUIRED_COLUMNS`; SAM2_BACKEND_LABEL, SAM2_RLE_FORMAT constants
- `fake_predictor.py` — `FakePredictor(n_objects, mask_fill_fraction)` + `segment_one_well_fake` adapter

One well runs end-to-end through `run_sam2_video_for_wells` with a patched `load_sam2_video_model`
returning a `FakePredictor`; `validate_frame_masks(frame_masks, frame_inventory)` and
`validate_frame_masks_against_sam2_prompts(frame_masks, prompt_detections)` both pass. No GPU or
real SAM2 invoked. `prompt_seeds.py` untouched. Focused segmentation tests: **70 passed**.

**What's broken/half-done:** nothing within Session B. Session C remains deferred. No real SAM2 GPU
invocation, model/checkpoint path plumbing, `seed_selection.py`, `Sam2WellInput` rename, or
`prompt_seeds.py` cleanup was implemented.

**Next concrete action:** Start Session C by drafting a staged plan before coding. Session C scope:
real SAM2 backend invocation under `segmentation/backends/sam2_video/`, tiny frame-capped one-well
GPU smoke, model/config/checkpoint path handling, real-output validation, cleanup/retirement decision
for `prompt_seeds.py`, and `Sam2WellInput` rename/move if still desired.

**Open decisions:** none blocking Session C planning.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 23:06

**What shipped:** Session A implementation is complete in `morphseq-docs` through four stage commits:
`001adaf7` (`segmentation identifiers: add shared mask and track ID helpers`), `ab0e5a85`
(`segmentation masks: add RLE utility`), `e0fc7470` (`segmentation masks: add geometry utility`),
and `c2e4bafc` (`frame masks: migrate contract to canonical mask vocabulary`). A5 was intentionally
skipped: `src/data_pipeline/segmentation/prompt_seeds.py` was not edited. Focused segmentation
verification passed:
`/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python -m pytest tests/data_pipeline/segmentation`
reported `31 passed`.
**What's broken/half-done:** nothing within Session A. Session B/C remain deferred. No seed-selection
migration, SAM2 output adapter, fake predictor integration, real GPU SAM2 run, or prompt-seed cleanup
was implemented. The pre-existing `src/data_pipeline/segmentation/sam2_video/` package was not moved
or expanded in Session A.
**Next concrete action:** Start Session B by drafting a staged fake-predictor plan before coding.
First implementation target should be the deterministic fake-predictor end-to-end segmentation
contract: `seed_selection.py`, SAM2 prompt validator separation, `adapt_sam2_output.py`,
`segment_one_well`, and a fake predictor integration test that produces valid `frame_masks` through
the generic `validate_frame_masks(df, frame_inventory)` path.
**Open decisions:** none blocking Session B planning. Keep real GPU SAM2 smoke and prompt-seed
retirement for Session C.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 22:56

**What shipped:** Stage A4 is implemented in `morphseq-docs` commit `c2e4bafc`
(`frame masks: migrate contract to canonical mask vocabulary`). The frame-mask contract now uses
`prompt_detection_id`, `sam2_object_id`, constructor-minted `mask_id`, and constructor-minted
zero-based-compatible `track_id`; `seed_id` was removed from the generic frame-mask contract.
`adapt_legacy_mask_rle_to_frame_masks` now uses `build_mask_id` and `build_track_id` instead of
inline ID minting. Added `no_mask_frame_mask_row(...)` for explicit no-mask placeholders using
`build_no_mask_id(image_id)` and NA `track_id`. Generic validators are
`validate_frame_mask_block(df)`, `validate_frame_masks(df, frame_inventory)`, and
`valid_frame_masks(df)`. Prompt cross-checking is separated into
`validate_frame_masks_against_prompt_detections(...)` and is not required by generic validation.
Focused tests passed:
`/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python -m pytest tests/data_pipeline/segmentation`
reported `31 passed`.
**What's broken/half-done:** Stage A4 is complete. Session-B work remains deferred: no seed
selection migration, no SAM2 output adapter, no fake predictor integration, and no real SAM2 runner
changes. The pre-existing `src/data_pipeline/segmentation/sam2_video/` package was not introduced or
moved in this stage.
**Next concrete action:** Stage A5 from `segmentation_world_plan.md` — leave
`src/data_pipeline/segmentation/prompt_seeds.py` untouched by default, verify focused segmentation
tests/imports still pass, confirm no accidental new `segmentation/sam2_video/` migration work, and
record Session A as complete with Session B as the next staged plan.
**Open decisions:** none for A5. Session B still needs its own fake-predictor plan before coding.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 22:52

**What shipped:** Stage A3 is implemented in `morphseq-docs` commit `e0fc7470`
(`segmentation masks: add geometry utility`). Added
`src/data_pipeline/segmentation/masks/mask_geometry.py` with `mask_area_px`,
`mask_bounding_box_xyxy_px`, `mask_centroid_xy_px`, and `mask_geometry`, exported from
`src/data_pipeline/segmentation/masks/__init__.py`. Geometry reuses the A2 `validate_binary_mask`
guard, returns half-open pixel bounding boxes, and uses pixel-center centroid coordinates. Focused
tests passed:
`/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python -m pytest tests/data_pipeline/segmentation/masks/test_mask_geometry.py tests/data_pipeline/segmentation/masks/test_mask_rle.py`
reported `18 passed`.
**What's broken/half-done:** Stage A3 is complete. No frame-mask schema, generic validators, SAM2
prompt migration, seed selection, or output adaptation was touched. Stage A4-A5 remain deferred.
**Next concrete action:** Stage A4 from `segmentation_world_plan.md` — edit
`src/data_pipeline/segmentation/frame_masks_contract.py`,
`src/data_pipeline/segmentation/validate_frame_masks.py`, and the matching tests to enforce
constructor-minted `mask_id`/`track_id`, no-mask placeholders, duplicate `mask_id` rejection, and
generic `validate_frame_masks(df, frame_inventory)` behavior that does not require SAM2 prompt
inputs. Verify with the focused frame-mask pytest file, then commit as
`frame masks: migrate contract to canonical mask vocabulary`.
**Open decisions:** none for required generic A4 work. SAM2 prompt validators remain optional only
if the existing code can be migrated without introducing Session-B scope.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 22:50

**What shipped:** Stage A2 is implemented in `morphseq-docs` commit `ab0e5a85`
(`segmentation masks: add RLE utility`). Added `src/data_pipeline/segmentation/masks/mask_rle.py`
with `validate_binary_mask`, `encode_binary_mask_rle`, and `decode_binary_mask_rle`, exported from
`src/data_pipeline/segmentation/masks/__init__.py`. The RLE behavior is row-major, starts with the
background/False run, stores `{"shape": [height, width], "counts": [...]}`, decodes to bool masks,
accepts bool or integer 0/1 masks, and fails loud for non-2D masks, non-binary values, non-integer
dtypes, malformed shapes, negative counts, and counts that underfill/overflow the declared shape.
Focused tests passed:
`/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python -m pytest tests/data_pipeline/segmentation/masks/test_mask_rle.py`
reported `12 passed`.
**What's broken/half-done:** Stage A2 is complete. No frame-mask schema, generic validators, SAM2
prompt migration, seed selection, or output adaptation was touched. Stage A3-A5 remain deferred.
**Next concrete action:** Stage A3 from `segmentation_world_plan.md` — create
`src/data_pipeline/segmentation/masks/mask_geometry.py` and
`tests/data_pipeline/segmentation/masks/test_mask_geometry.py`, reuse the A2 binary-mask validation,
and test area/bounding-box/centroid semantics plus invalid-mask failure paths. Verify with the
focused mask geometry pytest file, then commit as `segmentation masks: add geometry utility`.
**Open decisions:** none for Stage A3.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 22:38

**What shipped:** Stage A1 is implemented in `morphseq-docs` commit `001adaf7`
(`segmentation identifiers: add shared mask and track ID helpers`). The shared identifier API now
lives under `src/data_pipeline/shared/identifiers/` with `build_mask_id`, `build_no_mask_id`,
`build_track_id`, `parse_mask_id`, and `parse_track_id` exported from the package. Mask IDs use
constructor-minted `<image_id>_m####` values, no-mask placeholders use `<image_id>_mask_none`, and
`parse_mask_id(mask_id)` returns exactly `(image_id, local_mask_index, is_no_mask)` with
`local_mask_index is None` for placeholders. Track IDs remain zero-based-compatible:
`build_track_id("WELL", 0) == "WELL_track0000"`. Focused identifier tests passed:
`/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python -m pytest tests/data_pipeline/shared/identifiers/test_identifiers.py`
reported `52 passed`.
**What's broken/half-done:** Stage A1 is complete. Session A2-A5 remain deferred; no mask RLE,
geometry, frame-mask contract, SAM2 prompt, seed-selection, or runner code was touched. The working
tree had pre-existing unrelated dirty files; the Stage A1 commit included only the shared identifier
files/tests.
**Next concrete action:** Stage A2 from `segmentation_world_plan.md` — create
`src/data_pipeline/segmentation/masks/mask_rle.py` and
`tests/data_pipeline/segmentation/masks/test_mask_rle.py`, covering empty masks, single-pixel masks,
multi-component masks, encode/decode round trip, and dtype/shape validation failure paths. Verify
with the focused mask RLE pytest file, then commit as `segmentation masks: add RLE utility`.
**Open decisions:** none for Stage A2.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-22 (session: segmentation Session A plan approved with path pins)

**What shipped (docs only):**
- Rewrote `specs/detect-seg-track/targets/segmentation_world_plan.md` from a one-shot scope note
  into a staged Session-A implementation contract.
- Session A is explicitly limited to shared mask/track ID helpers, segmentation mask utilities,
  generic frame-mask contract/validator vocabulary migration, and intentional retention of
  `prompt_seeds.py` for Session B.
- Identifier helpers are pinned to `src/data_pipeline/shared/identifiers/{constructors.py,
  parsers.py, __init__.py}` plus README updates as needed. Do not add identifier grammar under
  `segmentation/`.
- Mask utilities are pinned to `src/data_pipeline/segmentation/masks/mask_rle.py` and
  `src/data_pipeline/segmentation/masks/mask_geometry.py`.
- Each stage now has a functional artifact, focused synthetic tests, a commit boundary, and a
  required end-of-stage update back to this status doc.
- The plan pins the decisions that `parse_mask_id(mask_id)` returns
  `(image_id, local_mask_index, is_no_mask)`, no-mask placeholders use `local_mask_index is None`,
  `build_track_id("WELL", 0)` remains `WELL_track0000`, and negative `track_index` values are
  rejected.
- A4 is generic-first: required work is `frame_masks_contract.py`, `validate_frame_masks.py`, and
  `valid_frame_masks.py`; SAM2 prompt validators are optional only if existing tests/code can be
  migrated without introducing Session-B seed selection or output adaptation.
- Session-A actual-mask validation checks parseability and uniqueness. Placeholder rows must equal
  `build_no_mask_id(image_id)` and have NA `track_id`. Valid mask rows must have parseable
  `track_id`. Deterministic actual-mask `local_mask_index` assignment and SAM2-specific
  `track_id == build_track_id(...)` checks belong to the Session-B adapter/SAM2 validation unless
  already trivial and tested.
- `pipeline_file_philosophy.md` is now explicit implementation doctrine for Session A: no inline ID
  minting/parsing outside `shared/identifiers`, use named constants/helpers for grammar, keep
  explicit signatures/no haunted globals, put tests in the parallel `tests/data_pipeline/...` tree,
  keep validators authoritative beside their contracts, make validation errors name the fix, and review
  every file after editing for first-read clarity and organization.
- The downstream arc is now explicit: Session B is the fake-predictor end-to-end segmentation
  contract (`seed_selection.py`, SAM2 prompt validators, `adapt_sam2_output.py`, `segment_one_well`,
  fake predictor integration), while Session C is the real GPU SAM2 smoke plus prompt-seed cleanup /
  `Sam2WellInput` rename if still needed.

**What's broken/half-done:** no code changed. The implementation still needs to begin at Stage A1;
Session B/C remain deferred. Session B must stay deterministic and fake-predictor-backed even if GPU
is available; real SAM2 belongs to Session C. The next agent can assume GPU access when needed for
Session C and should try to carry the plan through the real SAM2 smoke, validation, and cleanup end
state rather than stopping at scaffolding.

**Next concrete action:** Stage A1 from `segmentation_world_plan.md` — edit
`src/data_pipeline/shared/identifiers/constructors.py`,
`src/data_pipeline/shared/identifiers/parsers.py`, and
`src/data_pipeline/shared/identifiers/__init__.py` to add `build_mask_id`, `build_no_mask_id`,
`build_track_id`, `parse_mask_id`, and `parse_track_id`, with synthetic round-trip and zero-based
track-ID tests. End the stage with its own commit and a new status-doc snapshot.

**Open decisions:** none for Stage A1. Stage A5 defaults to skip: do not touch `prompt_seeds.py`
unless imports force it; record that it is intentionally retained for Session B.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 (session: output tree doctrine flag-day rename)

**What shipped (commit 8c590559, 244 tests green):**
- `orchestration/paths.py`: all `PIPELINE_STEPS` stage values renamed to doctrine regimes
  (`experiment_metadata`→`acquisition`, `built_image_data`→`acquisition`,
  `detection`→`object_extraction`, `segmentation`→`object_extraction`).
- `product_dir` field added to `_experiment_step_dir` (new level between `<exp>/` and `per_well/`):
  `materialize_well` → `materialized_images/`, `frame_inventory` → `frame_inventory/`,
  `frame_detections` → `frame_detections/`, `frame_masks` → `frame_masks/`.
- Doctrine clarification landed in code: `materialize_well` owns only the `done` sentinel;
  `frame_inventory` owns the CSV contract path. `_materialize_well_inventory()` removed; all callers
  use `_frame_inventory_artifact()`.
- `Snakefile`: five doctrine-native constants (`ACQUISITION_DIR`, `OBJECT_EXTRACTION_DIR`, etc.);
  legacy names kept as transitional aliases with warning comments.
- `test_paths.py`: 39 tests (all green); new `TestOutputTreeDoctrine` class with 6 regime-pinning
  tests; `test_materialize_well_done_under_materialized_images` added.

**What's broken/half-done:** nothing. On-disk data still uses legacy folder names — wipe-and-rerun
or manual rename needed before running against live data on a machine that has existing outputs.

**Next concrete action:** Stage B from `keyence_wire_through.md` —
`image_materialization/scope/keyence/materialize_well_keyence.py` (per-well mosaic backend) +
route `_resolve_keyence*` in `scope_resolver_for_materialization_plan.py` + 2-way dispatch in
`run_materialize_well.py`. The acquisition inventory it consumes is live (Stage A). Open decisions
§8.1/§8.2 (Stage C stitch-map home, per-well-vs-batch) still need mdcolon's call before B/C land.

**Open decisions:** none new this session. Audit note: verify that `materialize_well` task writes
the frame inventory CSV to the path passed via `--frame-inventory-csv` (not internally derived
from `built_image_data_dir`). The `--built-image-data-dir` CLI arg name is a lie now (value is
`acquisition/`) — rename follow-up deferred to the tasks.py CLI cleanup session.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 18:45 (session: Stage A Keyence acquisition inventory BUILT + verified on real data; NEXT = Stage B mosaic backend)

**What shipped (code, all tests green):**
- NEW `metadata_ingest/scope/keyence/raw_plane_parsing.py` — the ONE Keyence filename grammar.
  Lifted `_extract_keyence_well_and_tile` / `_parse_keyence_time_and_z` / `_infer_keyence_stack_lookup`
  / `_well_from_w_index` out of the legacy materializer; ADDED `_parse_keyence_time_z_channel` which
  also captures the `CH#` integer (legacy regex matched `_CH\d+` but DROPPED it + hardcoded channel 0).
  Legacy `materialize_stitched_images.py` now imports these (no local redefs) → one grammar.
- NEW `metadata_ingest/scope/keyence/acquisition_inventory.py` — the Keyence twin of YX1's. One row
  per raw plane `(well, tile, z_index, channel_index, time_index)`, never collapsed; `source_tiff_path`
  PER ROW. Contract → Validation → Builder banners. Cell key
  `(well_id, position_index, z_index, channel_index, time_index_claimed)` with `position_index`
  TILE-UNIQUE (global enum over (well,tile)) so multi-tile wells don't falsely collide; uniqueness is
  FAIL-LOUD (overrode spec §8.4 warn-only per mdcolon — re-acquired wells fail here until Stage E).
  `assert_keyence_acquisition_sources_readable` is the TIFF twin of YX1's ND2 check, gated by
  `check_sources`. `x_um/y_um` were DROPPED (all-NaN placeholders fail the shared non-null check).
- **Channel decision (mdcolon, evolved over the session, web-confirmed):** Keyence XML is proprietary /
  channel NAME unreliable; the filename `CH#` index is the only reliable signal. So channel is anchored
  on the index via a SINGLE `KEYENCE_CHANNEL_INDEX_MAP` (`{1: "BF"}` today) — **MAP, fail-loud on
  unmapped, never default**. The old name-keyed `KEYENCE_CHANNEL_MAP` was DELETED and the legacy FF
  extractor's `_to_channel_id` rewired to also resolve channel_id from the `CH#` index → one mechanism
  everywhere. `raw_channel_name` falls back to the `CH#` token when no name scraped.
- Wired: `extract_scope_metadata.py` (+`acquisition_inventory_csv` param, builds from a RAW-PLANE scan
  not the collapsed FF rows), `tasks.py::cmd_extract_scope` (Keyence passes the arg), `Snakefile`
  `rule ingest_scope_metadata` (gate widened `YX1` → `("YX1","Keyence")`). Registry path already
  `{scope}`-templated → no `paths.py` edit.
- NEW `tests/data_pipeline/metadata_ingest/scope/keyence/test_acquisition_inventory.py`; updated the
  stale `tests/test_keyence_parsing_semantics.py` imports + `test_canonical_mapping.py`.

**Verification:** `195 passed, 1 pre-existing skimage warning` (`tests/data_pipeline/` +
`scope/tests/`). `snakemake -n` (Keyence config) declares `acquisition_inventory__keyence.csv` as an
output of `ingest_scope_metadata` with the `--acquisition-inventory-csv` arg. **Real-data run** on
`20230525` W057/P00001/T0027 (26-plane Z stack): rows==planes, cell key unique (26-Z stack NOT a
collision), channel_id=BF via CH1, real µm/px=5.66>0, elapsed finite/non-neg, all source_tiff_path
exist, `validate(check_sources=True)` PASS. Grep gates clean.

**Next concrete action:** Stage B — `image_materialization/scope/keyence/materialize_well_keyence.py`
(per-well mosaic backend, asserts `xy_composition=='mosaic'`) + route `_resolve_keyence*` in
`scope_resolver_for_materialization_plan.py` + make `run_materialize_well.py` a 2-way dispatch. The
inventory it consumes is now live. See `keyence_wire_through.md` Stage B. Open decisions §8.1/§8.2
(Stage C stitch-map home, per-well-vs-batch acceptability) still need mdcolon before B/C land.

**Note:** `materialize_stitched_images.py` is the LEGACY producer, mis-named per philosophy (named for
one op, not the stage; multi-microscope god-file). On the deletion path (strangle AFTER Stage D), so it
was NOT renamed — only had the Keyence parsers extracted.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 18:30 (session: YX1 front-half smoke green; frame_inventory shard + well_runner merge completed)

**What shipped:** front-end / microscope-boundary only. The YX1 materializer now writes derived
`well_id` and `image_id` columns into the per-well `frame_inventory` shard for downstream readability;
validation still recomputes them from atoms and fails loud on disagreement. `merge_frame_inventory`
now merges the materializer-emitted per-well shards rather than the old dead adapter path, and its
DAG-time artifact inputs are resolved through `well_runner.run_well_shard_paths(...)`. No
segmentation, feature, QC, or auxiliary-mask consumer code is part of the shipped change in this
snapshot; those remain beyond the front-half boundary.

**Verification:** interpreter sanity check used
`/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python`.
Focused tests: **80 passed** (`test_paths.py`, `test_well_runner.py`, `test_tasks_parser.py`,
`test_frame_inventory.py`, `test_frame_inventory_contract.py`). `git diff --check` clean. Actual
two-well YX1 front-half smoke: `snakemake front_half --configfile
config_smoke_front_half_20250912.yaml --cores 2 --rerun-triggers mtime --forcerun
materialize_well` completed **5/5 (100%)** for B01/C01 on CPU with `SMOKE_FRAME_CAP_ACTIVE` at 3
timepoints. Rewritten shards include atom columns plus derived `well_id`/`image_id`.
`snakemake .../experiment_metadata/20250912/20250912_frame_inventory.csv --configfile
config_smoke_front_half_20250912.yaml --cores 1 --rerun-triggers mtime` completed **1/1 (100%)**;
the merged view row-stacks the two validated materializer-emitted shards (6 rows total).

**What's broken/half-done:** nothing new in the YX1 front-half smoke. Downstream Beat 2 consumers
still read `frame_contract.csv` by design in this front-half-bounded session: segmentation,
auxiliary masks, feature extraction, focus/motion QC, and consolidation paths.

**Next concrete action:** stay in the front-end boundary and continue Step 7 producer cleanup:
remove the remaining YX1 legacy producer path (`materialize_stitched_images` →
`stitched_image_index.csv` → `build_frame_contract`) from the YX1 front-half DAG while preserving
legacy compatibility for downstream readers. Do not edit segmentation/features/QC. Verify with
`snakemake front_half --configfile config_smoke_front_half_20250912.yaml --cores 2 --rerun-triggers
mtime`, the merged frame-inventory target, and focused frame-inventory/path/well-runner tests.

**Open decisions:** none blocking the front-half producer cleanup. Beat 2 consumer migration remains
separate.

## ⭐ CURRENT SNAPSHOT — 2026-06-18 (session: Keyence wire-through spec + §8.3 resolved; NEXT = build Stage A acquisition inventory)

**What shipped (docs only, no code):**
- NEW spec `specs/front_end/keyence_wire_through.md` — the staged Keyence plan onto the per-well
  materialize interface (Stages A–E + 6 open decisions + reuse-vs-new file map). Indexed in
  `README.md` + `AGENT_QUICKSTART.md` pointer tables.
- **Decision §8.3 RESOLVED by inspection (raw Z, not pre-fused).** Raw Keyence data on disk is
  per-Z-plane/per-channel/per-tile TIFFs (`...XY##_NNNNN_Z###_CH#.tif`). Legacy materializer already
  discovers + focus-stacks them on the fly (LoG, same primitive as YX1). "Pre-fused FF tiles" was a
  misread of the legacy intermediate. Cascade: new backend focus-stacks raw Z; Stage-A inventory
  explodes one row per (well, tile, z_index, channel_index, time_index); reacquisition cell key is
  fully realizable.

**Verified state for context (NOT changed this session):** the YX1 per-well materialize interface
is BUILT + LIVE (`rule materialize_well` in `rules/frame_inventory.smk` → `image_materialization/`
backend → per-well frame_inventory shard; `rule front_half` drives it through validation). Beat 1
Step 6 effectively landed. Still in-flight: Beat 2 (segmentation/features/QC still read
`frame_contract.csv` — `rule segment_and_track_per_well`) + Step 7 strangle. The
`frame_inventory_contract.py` seam Keyence EMITS is frozen, so Keyence producer work does NOT chase
Beat 2.

**Next concrete action: build Stage A — Keyence acquisition inventory.** Mirror
`metadata_ingest/scope/yx1/acquisition_inventory.py`. CREATE
`metadata_ingest/scope/keyence/acquisition_inventory.py` (+ likely a shared
`scope/keyence/raw_plane_parsing.py`). DO NOT feed it from the current `extract_scope_metadata.py`
(it collapses to z_position=0); lift the parsing nucleus from the legacy materializer
(`_infer_keyence_stack_lookup`, `_parse_keyence_time_and_z`, `_extract_keyence_well_and_tile` in
`metadata_ingest/stitched_index/materialize_stitched_images.py`). Reuse the **canonical mapper
pattern** (`scope/shared/canonical_mapper.apply_canonical_mapping`) for scope-dialect→canonical
columns exactly as the extractor does for `channel` (via `KEYENCE_CHANNEL_MAP` + `VALID_CHANNEL_NAMES`)
— same glue, different dict+vocabulary; the applier is generic by design. Tier-1 core =
`metadata_ingest/scope/acquisition_inventory_contract.py`; reuse `scope/shared/acquisition_checks.py`
mechanics. Verify: `snakemake -n` declares `acquisition_inventory__keyence.csv`; clean experiment →
rows == #raw planes, multi-Z well NOT flagged a collision, every `source_tiff_path` exists.

**Open decisions still live (Stages B/C/E only, NOT blocking A):** §8.1 stitch-map home, §8.2
per-well-vs-batch acceptability, §8.4 warn-vs-fail collisions, §8.5 eligibility timing, §8.6
Snakefile scope-gating style. See `keyence_wire_through.md` §8.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 16:24 (session: canonical scope channel adapter — Beat 2 foundation refinement)

**The plan:** `~/.claude/plans/committed-working-tree-flickering-rocket.md` — canonical mapping
adapter now DONE. This completes the mapper/vocabulary/validator split that was deferred in the 14:52
snapshot: mappings translate scope dialect, the vocabulary defines canonical language, validators guard
contracts.

**What shipped:**
- `schemas/channel_normalization.py` now owns only the canonical channel language:
  `VALID_CHANNEL_NAMES`, `BRIGHTFIELD_CHANNELS`, and `validate_channel_id(channel_id)`.
- NEW `metadata_ingest/scope/shared/canonical_mapper.py` provides the generic exact-match applier:
  raw scope value + per-scope mapping + canonical vocabulary -> canonical token. It fails loud on both
  unmapped raw values and non-canonical mapping targets.
- Per-scope dialect DATA moved out of the schema module:
  - `scope/yx1/mappings.py`: `YX1_CHANNEL_MAP`
  - `scope/keyence/mappings.py`: `KEYENCE_CHANNEL_MAP`
- YX1 and Keyence extractors no longer use fuzzy `_normalize_channel_name` logic or silent raw
  pass-through. They call the shared applier through small `_to_channel_id(...)` wrappers.
- The YX1 acquisition inventory contract check delegates membership to `validate_channel_id(...)`
  instead of re-implementing the vocabulary set.
- Real-data smoke surfaced a previously implicit YX1 dialect label: the 20250912 ND2 reports raw
  channel `"Empty"`. It is now explicitly mapped in `YX1_CHANNEL_MAP` as `"Empty": "BF"` instead of
  being inferred by substring/fuzzy logic.
- Tests added for the shared mapper, unmapped failure, non-canonical target failure, per-scope map
  integrity, and `validate_channel_id`.

**Verification:**
- Interpreter sanity check used the required env:
  `/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python`.
- Focused adapter/acquisition tests: **24 passed**.
- Baseline: **188 passed, 1 existing skimage dtype warning**
  (`tests/data_pipeline/ src/data_pipeline/metadata_ingest/scope/tests/`).
- Grep gate: no `_normalize_channel_name` or `CHANNEL_NORMALIZATION_MAP` remains in
  `src/data_pipeline` / `tests/data_pipeline`.
- Front-half smoke with forced `ingest_scope_metadata` reminted
  `data_pipeline_output/experiment_metadata/20250912/acquisition_inventory__yx1.csv` through the new
  applier: 161,025 rows, `raw_channel_name=['Empty']`, `channel_id=['BF']`, `elapsed_time_s` present.
  Mapping and mapped-metadata validation completed; `discover_wells` found 95 wells.

**What's broken/half-done:** the full front-half smoke is **not green** in this session. It reached
forced `materialize_well` for `20250912_B01`, then the process was system-killed during CPU projection
after announcing `SMOKE_FRAME_CAP_ACTIVE`. This happened after the channel adapter path completed and
looks like the existing materializer/runtime memory issue, not a mapper failure.

**Next concrete action:** continue the parent Beat-2 cutover: Part 3 delete the legacy
`frame_contract` producer (clean-room cut; expect `rule all` to go intentionally red at the
segmentation/features seam while `front_half` remains the target to protect), then Part 4 repoint the
back half onto per-well frame-inventory shards. Separately, investigate the CPU materializer kill before
using the full 20250912 smoke as a green gate again.

**Open decisions:** none blocking Part 3. The materializer memory/runtime issue is a verification
blocker for the full smoke, not a design blocker for the channel adapter.

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
