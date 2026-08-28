# Agent briefs — pipeline-backed core model execution

**Updated:** 2026-08-27
**Dispatch authority:** lead/integration agent
**Binding inputs:** repository-root `AGENTS.md`, `docs/refactors/core-model/DECISIONS.md`, and
`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`.

These briefs are written to be passed directly to agents. Track A, Track D, and Track E may run now.
Track C briefs are present for planning but **must not be dispatched until the lead records Track A3
as accepted**.

## 1. Rules for every agent

Before editing:

```bash
cd /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq
git status --short --branch
git rev-parse --show-toplevel
git rev-parse HEAD
conda activate morphseq-env
python --version
```

Requirements:

1. Read root `AGENTS.md` completely. Do not cite `docs/refactors/core-model/_archive/` as current
   state.
2. Use Python 3.10 from `morphseq-env`. Do not implement compatibility for Python 3.9. Do not set or
   manipulate `PYTHONPATH` or `sys.path`.
3. Work only in the branch/worktree supplied by the lead. Do not merge or rebase another branch.
4. Stay inside the ownership fence in the assigned brief. Existing changes outside that fence belong
   to the user or another agent.
5. Treat IDs as opaque. Never parse, slice, or reconstruct them in core. Carry and join explicit
   identity columns.
6. Never glob the pipeline output tree to discover experiments or samples. Use the supplied explicit
   ordered experiment list and the pipeline path authority.
7. Prefer failing tests to weakening contracts. Failures name experiment, observation, asset, policy,
   or source field as applicable.
8. Add focused tests with the implementation. Run the assigned target tests and the complete
   `tests/core` suite before handoff.
9. Do not edit generated `docs/refactors/core-model/STATUS.md`.
10. Do not change production pipeline behavior unless the brief explicitly grants ownership. A
    contract discrepancy is reported to the lead, not patched around.
11. Commit only owned files. Use a concise commit message and provide the commit hash at handoff.
12. Run `git diff --check` before committing.

Every handoff uses this exact structure:

```text
Commit:
Branch/worktree:
Files changed:
Behavior implemented:
Tests run (exact commands and output summary):
Acceptance criteria satisfied:
Contract/source citations used:
Assumptions:
Known failures or incomplete work:
Decisions required from Nick/lead:
Files deliberately not touched:
```

An agent reports `ready_for_integration`; only the lead reports `integrated` or `accepted`.

## 2. Ownership map

| Role | Owns while active | Must not touch |
|---|---|---|
| Lead/integration | planning/contract docs; `src/core/run/**`; run provenance; root/model Hydra integration; E2E tests | A1/A2/D/E owned files while those agents are active |
| A1 manifest | new manifest/contracts modules; manifest configuration; manifest tests | datasets/transforms/loaders; run/loss code; production pipeline writers |
| A2 dataset | dataset classes, transforms, Lightning data loading, dataset tests | manifest construction; run/loss code; production pipeline |
| D surface QC | read-only study script, study report, derived report tables/figures | production QC code/config; core-model code |
| E stage audit | read-only study script, stage-state report, derived audit tables/figures | production stage/QC code; core-model code; legacy behavior |
| C1 relation | new metric mapping/relation modules and tests | dataset/pair sampler; loss implementation |
| C2 sampler | pairing/preflight modules; metric dataset pairing integration and tests | relation semantics; loss implementation |
| C3 loss | metric loss/config and loss tests/tuning harness | manifest and dataset ownership except integration requests |

The lead resolves any ambiguous file before work begins. Two active agents never edit the same file.

---

# Brief L — lead/integration agent

## Mission

Coordinate Track A and parallel Tracks D/E, own shared decisions/contracts, integrate A1 and A2, and
prove the pipeline-backed vanilla VAE works end to end. After vanilla acceptance, run one explicitly
test-only metric smoke. Do not absorb final metric science, surface-QC policy, or stage-estimator
selection into Track A.

## Read first

- root `AGENTS.md`
- `docs/refactors/core-model/README.md`
- `docs/refactors/core-model/PLAN.md`
- `docs/refactors/core-model/DECISIONS.md`
- `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`
- this entire brief
- `docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md`

## Owned files

- `docs/refactors/core-model/README.md`
- `docs/refactors/core-model/PLAN.md`
- `docs/refactors/core-model/DECISIONS.md`
- `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`
- `docs/refactors/core-model/plans/AGENT_BRIEFS_PHASE1.md`
- `src/core/run/**`
- new `src/core/run/provenance.py`
- `src/core/lightning/callbacks.py`
- root/model Hydra files needed to select the new data configuration, excluding the data-group files
  owned by A1
- `tests/core/test_pipeline_e2e.py`
- `tests/core/test_run_provenance.py`
- a shared synthetic fixture under `tests/core/fixtures/` if needed for coordination

Do not edit A1's manifest implementation or A2's dataset/loader implementation while those slices
are active. Request changes through a written interface discrepancy.

## Coordination sequence

1. Commit the ratified planning/contract changes before implementation branches fork, or provide the
   exact base commit all agents must use.
2. Supply A1 and A2 the same contract version and field names.
3. Ask A1 for an early fixture/interface commit if A2 needs executable table objects. Integrate that
   small commit before the larger implementations diverge.
4. Let A1 and A2 work in parallel against the frozen interface. Let D run independently.
5. Review A1 for source-contract accuracy, identity/join behavior, and real-data preflight evidence.
6. Review A2 for absence of pipeline reads, deterministic transforms, metadata carriage, and split
   isolation.
7. Integrate A1, rerun its tests, then integrate/reconcile A2 and run all core tests.
8. Implement run configuration/provenance and the A3 E2E harness without changing A1/A2 semantics.
9. Run the vanilla acceptance gate. Record exact commands/output in the handoff and generated status.
10. Only after A3 passes, add/run A4 with an unmistakably test-only metric policy.
11. Do not dispatch C1/C2/C3 until A3 is accepted. C agents may read the accepted interface then.

## Run/provenance requirements

Each accepted run writes locally even with W&B disabled:

- resolved configuration;
- ordered observation IDs;
- selected `(snip_id, snip_product_key, z_index)` keys;
- `physical_embryo_id` split assignments;
- experiment/product/z/QC/stage/covariate policies;
- source path/size/mtime/row-count/hash inventory;
- cohort report;
- mapping policy or explicit test-stub record;
- adapter git revision.

Do not write relative to Hydra's changing working directory. Use one configured
`run_artifacts_dir`. W&B is optional/mockable and receives the same bundle when enabled.

## Vanilla A3 acceptance test

Use a small explicit experiment list and real regenerated default-BF projection assets. Use a named
temporary smoke-test cohort policy. The test must prove:

1. observation and asset tables construct;
2. every selected row's IDs/product/path agree;
3. no physical embryo crosses splits;
4. batches have `[B, 1, 288, 128]` finite float32 tensors in `[0,1]`;
5. batch metadata includes temperature, elapsed time, time index, stage status/version, observation
   identity, and asset key;
6. vanilla `Trainer.fit` executes train and validation steps;
7. losses are finite and parameters change;
8. a tiny fixed subset demonstrates decreasing reconstruction objective;
9. checkpoint saves/reloads;
10. reloaded encode/reconstruct works on a held-out batch with finite expected shapes;
11. local provenance reconstructs the exact selected observation/asset keys;
12. the run succeeds with W&B disabled or offline.

Keep a fast synthetic/temporary-directory version suitable for `tests/core`. If the real-data
acceptance is too large or host-specific for routine pytest, provide a checked command plus a small
pytest that exercises the same path against generated files.

## A4 test-only metric smoke

- Config/policy name contains `test_only` or `dummy`.
- All fixture observations may use one group or another trivial explicit relation.
- Disable scientific stage gating or configure a declared broad test window.
- Preserve and exercise `self_stats`/`other_stats`.
- Run train, validation, backward, checkpoint, and reload.
- Mark the run provenance `scientific_policy: false` or an equivalently explicit field.
- Do not retune or interpret metric performance.

## Required commands before handoff

```bash
python -m pytest tests/core/test_run_provenance.py -q
python -m pytest tests/core/test_pipeline_e2e.py -q
python -m pytest tests/core -q
git diff --check
git status --short --branch
git log -5 --oneline --decorate
```

If a named test file is not created because an existing file is the clearer owner, report the exact
replacement and why.

---

# Brief A1 — observation/asset manifest adapter

## Mission

Build the sole adapter from explicit pipeline artifacts to the version-2 observation and asset
tables. Make temperature and elapsed acquisition time real loader-ready columns. Implement policy,
status, split, and reporting mechanics without choosing final science policy.

Suggested branch: `agent/core-track-a1-manifest`.

## Read first

- root `AGENTS.md`
- `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`
- `docs/refactors/core-model/DECISIONS.md`, especially D1, D4, D11, D18, D22, D24-D28, D31-D33
- `docs/refactors/core-model/PLAN.md`, Track A/A0/A1
- current pipeline writer/validator symbols for every source you declare

Do not use old reconnaissance column lists as writer authority. They are evidence about old stored
artifacts, not the current schema.

## Owned files

- new `src/core/data/pipeline_contracts.py`
- new `src/core/data/pipeline_manifest.py`
- optional new `src/core/data/manifest_types.py`
- `src/core/data/dataset_configs.py` only for manifest/policy configuration; coordinate any dataset
  constructor field with A2 before editing
- new `src/core/hydra_configs/data/**`
- `tests/core/test_manifest_contracts.py`
- `tests/core/test_pipeline_manifest.py`
- manifest-specific test fixtures, unless the lead owns the shared fixture

Do not touch:

- `src/core/data/dataset_classes.py`
- `src/core/data/data_transforms.py`
- `src/core/lightning/pl_wrappers.py`
- `src/core/run/**`
- `src/core/losses/**`
- production `src/data_pipeline` writers or pipeline output

## Required public result

Return a structured result with, at minimum:

- observation table;
- asset table;
- deterministic observation and asset order;
- source-artifact inventory;
- structured validation/schema report;
- per-filter cohort report;
- split assignments by `physical_embryo_id`;
- a selector-ready policy/config object.

Use typed dataclasses or another explicit typed boundary. Do not return a loose tuple whose elements
must be remembered positionally.

## Implementation requirements

### 1. Source contracts

Derive required/optional columns and dtypes from current writers/validators. Cite each source symbol
in code comments. Cover:

- product-aware snip inventory;
- frame inventory;
- stage predictions;
- snip QC;
- plate metadata;
- collection provenance when needed for start-age provenance.

Report schema variants. Do not fail on the first optional difference; collect a structured report.
Required identity/key failures still fail immediately and specifically.

### 2. Explicit source resolution

Inputs include:

- pipeline output root;
- explicit ordered `experiment_ids`;
- allowed product keys/z mode;
- validity policy;
- named QC policy;
- stage policy;
- covariate requirements;
- explicit `test_experiments`;
- split ratios/tolerance;
- metric mapping or explicit test-only stub.

Resolve artifacts through pipeline path/contract helpers. Imports from `src/data_pipeline` remain
confined to this adapter layer and limited to path/contract helpers. No globbing and no filename/ID
parsing.

### 3. Asset table

- Build one row per `(snip_id, snip_product_key, z_index)`.
- Normalize projection `z_index` to one consistent nullable integer representation.
- Treat duplicate null-z projection keys as real duplicates.
- Carry current explicit product, transform, path, encoding, output-grid, and scale columns.
- Do not derive product fields from paths.
- Do not copy observation QC into asset QC.
- Report the current z-snip writer gap; do not patch production rendering. Synthetic z rows are still
  required in contract tests.

### 4. Observation table

- Collapse asset identity fields to one row per `snip_id`, asserting all repeated biological parent
  fields agree.
- Join stage and current snip QC one-to-one on `snip_id`, reporting missing and extra IDs.
- Join plate metadata many-to-one on `well_id`, asserting exactly one matching source row.
- Preserve source spellings and nulls.
- Normalize booleans safely: strings `"True"` and `"False"` are not passed through `bool(...)`.

### 5. Temperature and time

- Map plate `temperature` to model-facing `incubation_temperature_c`; keep source provenance/status.
- Join canonical frame `elapsed_time_s` onto each observation.
- Validate all relevant frame products/planes for `(well_id, time_index)` agree on elapsed time.
- Preserve `time_index` separately.
- Carry correctly resolved start age plus `start_age_source`; collection-specific values must use
  declared provenance, not merged `time_index` guessing.
- Carry stage value/status/version separately from its raw inputs.
- Missing values remain missing with named status. A policy may require and fail on them.

### 6. QC and stage statuses

- `qc_status`: at least `evaluated`, `no_artifact`, `row_missing`.
- `stage_status`: literal source status or `unavailable` when the source schema lacks status.
- Absence is not failure.
- Carry every individual QC flag and applicability column present.
- A requested per-flag policy fails if the source schema lacks that flag; name experiment and flag.
- Basic mode can disable stage requirements.

### 7. Selection and splits

- Vanilla selection requires exactly one configured BF projection asset with null `z_index` per
  selected observation.
- Zero/multiple matches name `snip_id` and available assets.
- Filters are explicit and counted per experiment/reason.
- Explicit test experiments go entirely to test.
- Remaining `physical_embryo_id`s use deterministic `blake2b` assignment.
- Test group-disjointness, non-empty required splits, tolerance, reorder stability, and cohort-growth
  stability.

### 8. Inference switchability

QC filtering, stage requirements, split assignment, and metric mapping can each be disabled. Do not
bake train-only assumptions into table construction.

### 9. Real-data preflight

Run read-only against a small explicit set of currently available regenerated experiments supplied by
the lead. If regenerated outputs are incomplete, run against the current explicit artifacts and label
that preflight `plumbing_only`. Report:

- source paths and schema versions;
- row counts at each grain;
- product keys and null/non-null z behavior;
- join coverage;
- temperature/elapsed-time availability;
- all contract discrepancies;
- policy filter counts.

Do not silently force old counts such as 176,466 onto regenerated data.

## Required tests

At minimum:

1. observation key uniqueness;
2. asset compound-key uniqueness, including null z;
3. two products under one observation;
4. multiple ordered synthetic z planes;
5. orphan and conflicting identity failures;
6. selected product zero/multiple failures;
7. exact temperature and elapsed-time carriage;
8. conflicting frame times fail by well/time/product;
9. collection start-age provenance behavior;
10. safe string-boolean parsing;
11. one-to-one and many-to-one join coverage;
12. QC/stage three-state behavior;
13. missing flag under per-flag policy;
14. stable group-disjoint splits;
15. inference switches;
16. missing Parquet engine produces an actionable error rather than skipped QC;
17. deterministic output order.

## Required commands before handoff

```bash
python -m pytest tests/core/test_manifest_contracts.py -q
python -m pytest tests/core/test_pipeline_manifest.py -q
python -m pytest tests/core -q
git diff --check
git status --short --branch
git log -5 --oneline --decorate
```

The handoff must list every current-writer symbol used to define a source contract and every mismatch
between that contract and real artifacts.

---

# Brief A2 — manifest datasets, transforms, and loaders

## Mission

Replace `ImageFolder` and positional metadata with datasets/loaders consuming the frozen observation
and asset interface. Make the vanilla BF path work now without creating a one-file-per-`snip_id`
assumption that blocks z assets later.

Suggested branch: `agent/core-track-a2-dataset`.

## Read first

- root `AGENTS.md`
- `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`
- `docs/refactors/core-model/DECISIONS.md`, especially D6, D8, D11, D20, D25-D29
- `docs/refactors/core-model/PLAN.md`, A2/A3
- A1's public types/fixture if already integrated; otherwise use synthetic tables matching contract
  v2 and report any assumed type names

## Owned files

- `src/core/data/dataset_classes.py`
- `src/core/data/data_transforms.py`
- `src/core/lightning/pl_wrappers.py`
- optional new `src/core/data/asset_selection.py`
- `tests/core/test_pipeline_dataset.py`
- `tests/core/test_pipeline_loaders.py`

Coordinate before editing `src/core/data/dataset_configs.py`; A1 owns manifest/policy configuration.

Do not touch:

- `src/core/data/pipeline_manifest.py`
- `src/core/data/pipeline_contracts.py`
- `src/core/run/**`
- `src/core/losses/**`
- production `src/data_pipeline`
- planning/contract docs

## Implementation requirements

### 1. Plain manifest-backed datasets

- Delete the live `ImageFolder` training boundary; do not leave a fallback/config switch.
- Dataset indexing follows the resolved sample-view order.
- `__getitem__(i)` opens exactly the selected asset path for observation `i`.
- Do not search directories, parse filenames, or maintain a second positional metadata array.
- Preserve `DatasetOutput` keys expected by models/losses.

### 2. Vanilla asset selector

- Select the exact configured BF projection product and null `z_index`.
- Require one match per selected observation.
- Keep the selector API capable of returning asset-row lists later.
- Include synthetic tests where one `snip_id` has BF, RFP, and several z assets; vanilla must select
  BF deterministically and ignore sibling multiplicity without dropping it from the asset table.
- Do not choose the future z-model representation.

### 3. Deterministic image boundary

One documented path:

```text
open -> convert grayscale -> resize configured (H, W) -> float32 tensor [0,1]
```

Pin library, interpolation enum, antialias behavior, and whether resize occurs on PIL or tensor.
Match the legacy-embedding transform where intended and report the measured maximum deviation on a
small image fixture. Fix the live `contrastive_transform(target_size=...)` ignored-argument bug.

Vanilla default output is `[1, 288, 128]`. Metric two-view output remains compatible with the
existing model contract. Apply each view's augmentation independently after the deterministic size
boundary.

### 4. Batch metadata

Every vanilla item carries:

- `snip_id`;
- `physical_embryo_id`;
- `snip_product_key`;
- nullable `z_index`;
- split;
- `incubation_temperature_c` and status;
- `elapsed_time_s` and status;
- `time_index`;
- stage value/status/version;
- diagnostic asset path or stable asset-row reference.

The vanilla model ignores these fields; do not remove them. Preserve `self_stats`/`other_stats` for
the metric compatibility path. Do not confuse biological incubation temperature with contrastive
softmax temperature.

### 5. Worker memory and loaders

- Do not retain a large object-dtype DataFrame in each worker's hot path.
- Convert model-facing columns to compact arrays/categorical codes once.
- Use split-specific datasets/views rather than full-dataset positional samplers.
- Train/eval/test assets cannot cross their observation/embryo split.
- Let Lightning use a distributed sampler safely.
- Add an explicit test loader or explicitly fail if unsupported; never reuse eval pairing state.

### 6. Decode failure behavior

Implement D25 without cross-split leakage:

- record failed asset and `snip_id`;
- resample only from the same split;
- count failures and expose IDs for the cohort/run report;
- abort above a configured threshold;
- never silently switch products or z planes unless the configured selector permits it.

### 7. Metric plumbing only

Provide the minimum test-only paired path needed for A4 while preserving the accepted batch contract.
Do not implement final relation semantics, O(N)-scale pair selection, or metric loss changes; those
belong to Track C.

## Required tests

At minimum:

1. dataset row, pixel path, IDs, and metadata agree;
2. vanilla selector is exact with sibling BF/RFP/z assets present;
3. zero/multiple selected assets fail by `snip_id`;
4. output is finite float32 `[1,288,128]` in `[0,1]`;
5. deterministic pixels match the declared reference transform;
6. configured non-default size works in both basic and contrastive transforms;
7. two metric views are independently augmented after resizing;
8. temperature and elapsed time survive item and collate unchanged;
9. split-local loaders never cross physical embryos;
10. test loader behavior is explicit;
11. decode resampling stays within split and fails above threshold;
12. no `ImageFolder`, `make_seq_key`, positional split file, or second image-order authority remains
    reachable from the new presets;
13. worker-facing storage holds no source DataFrame reference in `__getitem__`.

## Required commands before handoff

```bash
python -m pytest tests/core/test_pipeline_dataset.py -q
python -m pytest tests/core/test_pipeline_loaders.py -q
python -m pytest tests/core -q
git diff --check
git status --short --branch
git log -5 --oneline --decorate
```

The handoff must state the exact interpolation/antialias settings, measured transform deviation,
metadata representation, worker-memory strategy, and the precise seam reserved for future z grouping.

---

# Brief D — surface-area QC investigation

## Mission

Determine whether the current lower surface-area gate rejects usable embryos, quantify the effect on
the prospective training cohort, and recommend a policy. This is a read-only study: do not change
production QC configuration or implementation.

Suggested branch: `agent/core-track-d-surface-qc`.

## Read first

- root `AGENTS.md`
- `docs/refactors/core-model/PLAN.md`, Track D
- `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`, QC sections
- current surface-area QC config, compute, reference, contracts, and snip-QC applicability code
- `docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md`
- non-archive reconnaissance reports, treating old counts as dated baselines

## Owned files

- new `scripts/recon/analyze_surface_area_qc.py` or another lead-approved single study script
- new `docs/refactors/core-model/reports/STUDY_surface_area_qc.md`
- new `docs/refactors/core-model/reports/surface_area_qc/**` derived CSV/figure artifacts

Do not touch:

- `src/data_pipeline/quality_control/**`
- pipeline configs/rules/tasks
- `src/core/**`
- manifest/plan/decision documents
- pipeline output artifacts

## Study requirements

### 1. Explicit sources and parity

Use an explicit ordered experiment list and declared artifact paths. Join, as available:

- surface-area QC;
- mask geometry;
- stage predictions;
- frame inventory/calibration;
- snip QC/applicability;
- plate metadata;
- segmentation backend/model provenance.

Recompute lower/upper reference bounds, direction, and normalized distance from the boundary. Verify
exact parity with stored flags before interpreting them. Report mismatches by ID and source version.

### 2. Failure decomposition

Report:

- too-small versus too-large failures;
- isolated surface-area-only failures;
- co-occurrence with every other QC flag;
- exclusion versus diagnostic-only applicability;
- current cohort delta if surface area does not exclude;
- results grouped by `physical_embryo_id`, not just rows.

### 3. Stratification

At minimum stratify by:

- experiment;
- developmental-stage bin;
- genotype/perturbation/control status where explicitly available;
- incubation temperature;
- source scope;
- calibration status/method;
- segmentation backend/model;
- snapshot versus time series when explicitly available;
- QC applicability.

Do not normalize genotype vocabulary or infer biology from IDs.
Treat `predicted_stage_hpf` as the live pipeline's nominal stage axis, not independently validated
biological truth. Record enough intermediate data for Track E to test stage-axis sensitivity, and
coordinate file exchange through the lead rather than editing Track E outputs.

### 4. Track persistence

For each physical embryo, distinguish persistent low area from isolated single-frame dips. Quantify
run length, neighboring-frame recovery, and distance below the lower bound. This tests the documented
pose/mask-confound hypothesis without assuming it is true.

### 5. Counterfactuals

Offline only:

- sweep `k_lower` from 0.70 through 0.90 with the upper multiplier fixed;
- report row and physical-embryo recovery/loss;
- report composition changes across the required strata;
- compare strict exclusion, diagnostic-only, and candidate shape-aware rules using already available
  aspect/circularity fields where valid.

Do not write counterfactual verdicts into pipeline artifacts.

### 6. Boundary review set

Produce an explicit balanced list of approximately 300 boundary examples, subject to data
availability, sampled across:

- below/above threshold;
- stage bins;
- control and perturbation;
- elongated/thin and compact shapes;
- persistent and isolated failures.

Record selection logic and asset keys. If visual review is not performed by the agent, provide the
review sheet/contact-sheet inputs without claiming labels.

### 7. Report standard

Separate:

- measured facts;
- hypotheses;
- unavailable determinations;
- recommendation.

Every claim cites a command, file:line, dated measurement, or tracked output table. Do not recommend
a production threshold solely from aggregate recovery counts; include boundary-review evidence and
group-composition effects.

## Required checks before handoff

```bash
python scripts/recon/analyze_surface_area_qc.py --help
# Run the study with the exact explicit experiment-list/config arguments recorded in the report.
git diff --check
git status --short --branch
git log -5 --oneline --decorate
```

The handoff reports runtime, environment, source roots, exact experiment-list authority, parity
result, missing artifacts, output table paths, and what could not be determined.

---

# Brief E — stage-estimation machinery and reliability audit

## Mission

Give Nick a current, plain-language account of the stage-estimation machinery: what is produced now,
what the legacy morphology-based path did, which fallbacks remain reachable, how stage values affect
QC and model code, what can and cannot be said about reliability, and which pieces should be
preserved, retired, or replaced. This is a read-only audit. Do not change a producer, consumer,
threshold, model, or artifact.

Suggested branch: `agent/core-track-e-stage-audit`.

## Read first

- root `AGENTS.md`
- `docs/refactors/core-model/PLAN.md`, Track E and O5/O8
- `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`, stage and covariate sections
- `docs/refactors/core-model/reports/STUDY_stage_lineage.md` in full
- `docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md`
- the live producer and contract under
  `src/data_pipeline/feature_extraction/stage_predictions/`
- `src/data_pipeline/feature_extraction/stage_inference.py`
- `src/data_pipeline/pipeline_orchestrator/rules/stage_predictions.smk`
- `src/build/infer_developmental_age.py`
- current core dataset, pairing, loss, analysis, and reporting consumers found by repository search

The existing lineage report is valid dated evidence, not a task to paraphrase. Re-run or update its
measurements only when current explicit sources permit a stronger result.

## Owned files

- new `scripts/recon/audit_stage_estimation.py` if repeatable measurements are needed
- new `docs/refactors/core-model/reports/STUDY_stage_estimation_state.md`
- new `docs/refactors/core-model/reports/stage_estimation/**` derived CSV/figure artifacts

Do not touch:

- `src/data_pipeline/**`
- `src/core/**`
- `src/build/**`
- planning, decision, contract, or generated-status documents
- pipeline or model output artifacts

## Required questions and evidence

### 1. Define the terms before evaluating them

Use distinct names for at least:

- declared start age;
- elapsed wall-clock time;
- incubation temperature;
- nominal temperature-adjusted clock stage (`predicted_stage_hpf` in the live pipeline);
- morphology-inferred stage (`inferred_stage_hpf_reg` or its exact legacy aliases);
- manually observed/anatomical stage, if any such source actually exists;
- stage status and method/model version;
- a missing value, a schema with no status column, and a code fallback/default.

Do not call two quantities equivalent because both are expressed in hpf. Do not use “ground truth,”
“default stage,” “prediction,” or “reliable” without naming the exact column, producer, reference, and
criterion.

### 2. Map the live machinery end to end

For the current pipeline, document with `file:line` citations:

1. source artifacts and join keys;
2. collection versus single-experiment start-age resolution;
3. elapsed-time resolution and aliases;
4. temperature source;
5. exact formula and method version;
6. output grain, schema, statuses, validation, and missingness behavior;
7. Snakemake/rule entrypoints, per-well shards, merge, and validation;
8. current artifact paths and schema variants for the explicit audited experiments;
9. all downstream consumers and whether each treats stage as metadata, ordering variable, QC
   conditioning axis, filter, metric-pair constraint, loss target, or plot axis.

The consumer inventory must explicitly cover surface-area QC, the planned core manifest, cohort
filters, metric pair sampling, metric loss, analysis-ready/reporting products, and any reachable
legacy core loader fallback. Search the repository; do not assume this list is exhaustive.

### 3. Map the legacy morphology path without preserving it by default

Document:

- the exact inputs and latent columns;
- how reference rows were selected;
- what target trained the `MLPRegressor`;
- experiment/temperature calibration and snapshot behavior;
- output columns and artifact paths;
- model/training-data/version provenance that is present and provenance that is missing;
- data leakage, circularity, domain-shift, and reproducibility risks supported by concrete code or
  artifact evidence;
- whether any current launch path still executes or consumes it.

Treat Nick's view that the legacy method may not merit preservation as a design prior, not proof.
Recommend retirement if the evidence supports it; do not spend effort making it operational.

### 4. Identify every “default” and fallback

Search code, configs, notebooks/scripts, and artifacts for default/fallback stage behavior. For each
one, report:

- triggering condition;
- selected column or value;
- whether the fallback is silent, warning-only, or a hard error;
- whether the path is reachable from the planned Track A presets;
- risk if nominal clock stage is mistaken for morphology stage.

In particular, inspect the warning/fallback in `src/core/data/dataset_utils.py:26-38`; do not assume
that the warning proves the planned manifest path uses it.

### 5. Evaluate reliability only against identifiable evidence

Start with input and contract integrity:

- coverage and status distribution by explicit experiment;
- finite/range checks;
- start-age source and collection provenance completeness;
- temperature and elapsed-time completeness;
- exact per-embryo monotonicity and discontinuities;
- duplicated/conflicting rows or method versions;
- snapshot versus time-series behavior.

Then inventory independent anchors: manual anatomical labels, fertilization timestamps, trusted
reference cohorts, or an explicit old/new identity crosswalk. If an anchor does not exist, say the
corresponding accuracy, bias, or agreement question is **not identifiable**. Internal monotonicity
is not accuracy. Agreement with the formula's own inputs is not independent validation.

If exact paired identities now exist, compare old and live values with counts, scatter/agreement
statistics, bias by experiment/temperature/stage range/acquisition mode, and track-level residuals.
If they do not, retain the existing `n=0` conclusion. Never parse or reconstruct IDs to create a
crosswalk.

### 6. Quantify downstream sensitivity

At minimum, determine:

- how much the surface-area reference bounds/flags change under plausible stage perturbations or
  alternate available stage axes, without changing production artifacts;
- how legal-positive counts and pair membership respond to the current metric sampler window;
- how the loss's separate `time_window + 1.5` rule responds;
- which cohort filters or reports change when stage is absent or status is unavailable;
- whether conclusions differ between snapshots, time series, temperatures, or experiments.

This is a sensitivity study, not validation of an alternative stage estimator. Coordinate output
tables with Track D through the lead; do not edit Track D files.

### 7. Deliver a decision-oriented recommendation

The report ends with a component-by-component table:

```text
Component | Current role | Evidence | Preserve / retire / replace / undecided | Immediate action | Blocks
```

It must answer plainly:

1. What can Track A safely carry and ignore for vanilla training?
2. What must be removed from or made unreachable in the new manifest-backed path?
3. Can nominal clock stage support stage-conditioned QC today, and with what caveat?
4. Can it support metric sampling/loss today, and how should windows be treated pending validation?
5. Is any part of the legacy morphology estimator worth preserving as data, benchmark, or code?
6. What is the smallest credible future validation or replacement study?
7. Which choices require Nick rather than an implementation agent?

Do not implement the recommendation. Do not invent a replacement architecture. Give two or three
bounded options with prerequisites and decision criteria when evidence cannot select one.

## Report and reproducibility standard

- Every factual claim carries a command, `file:line`, dated measurement, or tracked derived table.
- Separate measured facts, user reports, inference, and recommendations.
- Use an explicit ordered experiment list and explicit artifact paths; never glob pipeline output.
- Fingerprint source lists/artifacts used for corpus measurements.
- Record environment, current commit, branch, and exact commands.
- Report unavailable artifacts and non-identifiable questions, not plausible substitutes.
- Do not cite archived planning documents as current state.

## Required checks before handoff

```bash
# If a measurement script was needed:
python scripts/recon/audit_stage_estimation.py --help
# Run it with the exact explicit config/list arguments recorded in the report.
git diff --check
git status --short --branch
git log -5 --oneline --decorate
```

The handoff reports source roots, experiment-list authority, artifact fingerprints, current/legacy
coverage, exact crosswalk size, reachable fallbacks, consumer inventory, non-identifiable questions,
downstream sensitivity outputs, recommendation, and files deliberately not touched.

---

# Track C dispatch gate

Do not dispatch C1, C2, or C3 until the lead shows:

```text
A3 status: accepted
Integration commit: <hash>
Core tests: passing command/output
Vanilla E2E: passing command/output
Accepted observation/asset interface revision: <hash/version>
```

The C agents branch from that integration commit. They do not resurrect legacy positional metadata or
change the accepted observation/asset contract without lead approval.

C1/C2 mechanism work may proceed after this gate. No scientific metric run may claim a biologically
meaningful hpf window until Track E is accepted and Nick resolves O5/O8.

---

# Brief C1 — metric mapping and relation policy

## Mission

Replace positional `metric_array` semantics with a versioned, explicit mapping and one shared
class-to-class relation API. Implement mechanism and validation; Nick supplies/ratifies scientific
content.

Suggested branch: `agent/core-track-c1-relations`.

## Owned files

- new `src/core/metric/__init__.py`
- new `src/core/metric/mapping.py`
- new `src/core/metric/relations.py`
- metric-policy Hydra/config files assigned by the lead
- `tests/core/test_metric_mapping.py`
- `tests/core/test_metric_relations.py`

Do not touch dataset pairing or loss code.

## Requirements

- Consume explicit observation metadata and a required mapping artifact.
- Produce stable internal group codes without making codes the persisted scientific identity.
- Name all uncovered non-null source values and fail.
- Define exactly one API returning `positive`, `negative`, or `excluded` for two metric groups.
- Define and validate policy version, symmetry/asymmetry declaration, diagonal behavior, exclusions,
  and complete coverage.
- Ship a trivial test policy separately from the future scientific policy.
- Persist the exact mapping and relation-policy version in run provenance.
- Never infer mapping from fuzzy genotype/perturbation string matching.

## Tests

- complete/uncovered mapping;
- stable code assignment under row reorder;
- relation coverage;
- diagonal behavior;
- symmetry rule;
- excluded pairs;
- invalid/unknown group errors;
- test policy cannot load through a science preset;
- round-trip provenance.

Run targeted tests, `python -m pytest tests/core -q`, and `git diff --check` before handoff.

---

# Brief C2 — legal-positive preflight and pair sampler

## Mission

Build a split-local indexed pair-selection mechanism over accepted observation/asset rows. It must be
fast at cohort scale, reproducible across workers/ranks, and use C1's relation API without duplicating
relation semantics.

Suggested branch: `agent/core-track-c2-pairing`.

## Owned files

- new `src/core/metric/pairing.py`
- new `src/core/metric/pair_preflight.py`
- metric-pairing portions of `src/core/data/dataset_classes.py`, after A2 is integrated
- `tests/core/test_metric_pair_preflight.py`
- `tests/core/test_metric_pairing.py`

Do not edit C1 relation content or metric loss implementation.

## Requirements

- Index once per split by metric group, age/time bucket, and `physical_embryo_id` as needed.
- No O(N) full-length boolean construction inside `__getitem__`.
- Use explicit `physical_embryo_id`; never reconstruct it from `snip_id`.
- Preflight every anchor under the selected policy and name anchors with no legal positive.
- Never cross train/eval/test.
- Treat sibling products/z planes as assets of one observation, not independent biological animals.
- Make observation weighting explicit so embryos with more asset planes are not silently over-weighted.
- Separate same-embryo and different-embryo candidate policies.
- Make stage source, sampler window, and any loss-window buffer distinct configuration fields.
- Use deterministic seed derivation for worker and distributed rank; prove reproducibility.
- Preserve accepted `DatasetOutput`, `self_stats`, and `other_stats` structures.

## Tests

- every accepted anchor has a legal positive;
- missing-positive error names anchor and policy;
- no split leakage;
- no physical-embryo reconstruction;
- sibling-asset weighting behavior;
- window boundary behavior;
- relation exclusions honored;
- deterministic behavior under row reorder where policy permits;
- worker/rank seed reproducibility;
- indexed complexity demonstrated and no O(N) per-item scan.

Run targeted tests, `python -m pytest tests/core -q`, and `git diff --check` before handoff.

---

# Brief C3 — metric loss and tuning harness

## Mission

Make the metric loss consume C1's relation semantics, implement the ratified SupCon `L_out`, and
provide a reproducible tuning/acceptance harness. Do not change encoder/decoder architecture or claim
scientific completion before policy/window/weight tuning.

Suggested branch: `agent/core-track-c3-loss`.

## Owned files

- `src/core/losses/loss_functions.py`
- `src/core/losses/loss_configs.py`
- new loss helper modules approved by the lead
- metric-loss Hydra/config files assigned by the lead
- `tests/core/test_metric_loss_relations.py`
- `tests/core/test_metric_loss_lout.py`
- tuning harness/report path assigned by the lead

Do not change manifest or dataset selection semantics.

## Requirements

- Consume the same C1 relation API used by C2; no second target-matrix implementation with independent
  semantics.
- Implement SupCon `L_out`: average/sum positive log-probability terms outside the logarithm as
  ratified by D23.
- Preserve application to the intended biological latent subspace.
- Define behavior for excluded pairs and anchors with zero batch positives.
- Keep sampler window and loss target buffer separate and explicit.
- Record stage source/version, windows, metric weight, and contrastive temperature in provenance.
- Provide numerically stable computation and useful diagnostics.
- Provide a tuning harness for metric weight and contrastive temperature. Scientific tuning waits for
  Nick's mapping/relation and intended cohort.

## Tests

- hand-computed toy batches with one and several positives;
- `L_out` differs from legacy `L_in` on a multi-positive case;
- excluded pairs do not enter numerator/denominator as specified;
- zero-positive behavior is explicit;
- relation symmetry/asymmetry passes through correctly;
- biological-latent-only application;
- numerical stability at extreme distances/temperatures;
- gradients finite and non-zero where expected;
- sampler/loss window distinction pinned;
- integrated short metric run after C1/C2 merge.

Run targeted tests, `python -m pytest tests/core -q`, and `git diff --check` before handoff. Report
what remains untuned and do not label the metric overhaul scientifically complete without the final
policy and tuning evidence.
