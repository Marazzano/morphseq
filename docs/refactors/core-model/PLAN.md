# Plan — pipeline-backed core model

**Updated:** 2026-08-27
**Authority:** `DECISIONS.md` for ratified choices; `contracts/MANIFEST_SCHEMA.md` for the binding
data interface; `plans/AGENT_BRIEFS_PHASE1.md` for execution ownership; generated `STATUS.md` for
implementation/test state.

Every factual implementation/data claim below carries a source or is explicitly labeled as a dated
user report, inference, or plan.

## 1. Objective

Reach a real, pipeline-backed vanilla VAE training and validation run without redesigning the model,
then use that stable path to overhaul metric constraints. In parallel, investigate whether the
surface-area QC rule is rejecting usable embryos and audit what the current and legacy stage
machinery actually estimate.

The first accepted model path is deliberately simple:

```text
explicit experiments
  -> observation table + asset table
  -> one configured BF projection asset per observation
  -> deterministic train/eval/test loaders
  -> vanilla Trainer.fit
  -> validation
  -> checkpoint save/reload
  -> held-out encode/reconstruct
```

Metric relation semantics and developmental-age constraints are spoofed only after this vanilla path
works. Z-aware model architecture is not selected now, but Track A must make z assets representable
without changing biological IDs.

## 2. Planning baseline

The 2026-08-27 ground-truth audit found Phase 0 implemented and only five core tests, all Phase 0; it
found no manifest adapter, pipeline-backed dataset, Hydra data group, or full pipeline-output test
(`reports/GROUND_TRUTH_2026-08-27.md:151-178,688-729`). That audit is the implementation baseline;
generated `STATUS.md` is the authority after new work integrates.

The live pipeline has since become product-aware at snip grain: a `snip_id` may have several
`snip_product_key` rows (`src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/snip_identity_contract.py:198-223`).
The prior one-row-per-`snip_id` manifest specification was therefore replaced by contract version
2.0 on 2026-08-27.

Nick reported on 2026-08-27 that all pixel-dependent quantities are being regenerated. This plan
treats those outputs as an external input. The core refactor does not own renderer acceptance,
regeneration orchestration, or downstream pipeline production (D30).

## 3. Agreed architecture

### Observation table

One row per `snip_id`: biological identity, time, temperature, stage, plate metadata, and current
observation-level QC.

### Asset table

One row per `(snip_id, snip_product_key, z_index)`: pixel path, projection or z plane, scale,
transform, encoding, and future asset-level QC. `z_index` is null for projections.

### Resolved sample view

A configured selector maps observations to asset row(s). Vanilla Track A selects exactly one BF
projection asset. Future z modes may choose one plane, several planes, or an ordered stack without
changing `snip_id` or the two source tables.

The complete column, join, order, failure, and provenance rules are binding in
`contracts/MANIFEST_SCHEMA.md`.

## 4. Tracks and priority

### Track A — pipeline-backed vanilla model — P0

Track A is the critical path. It owns:

1. schema preflight against current pipeline writers;
2. observation/asset manifest adapter;
3. explicit cohort/product selection and identity-stable splits;
4. datasets, deterministic transforms, and split-local loaders;
5. temperature and elapsed-time metadata in every batch;
6. named basic and test-only metric presets;
7. local run provenance with optional W&B;
8. vanilla end-to-end acceptance;
9. one dummy metric-path smoke after vanilla acceptance.

Track A does not own final QC policy, real metric semantics, z-model architecture, or corpus
generation.

### Track C — metric constraint overhaul — starts after Track A acceptance

Track C owns:

1. versioned mapping from explicit metadata to `metric_group`;
2. one relation API returning positive/negative/excluded;
3. cohort-wide legal-positive preflight;
4. split-local, indexed, reproducible pair selection;
5. replacement of positional `metric_array` semantics;
6. ratified SupCon `L_out` implementation;
7. explicit stage source and independently configurable sampling/loss windows;
8. retuning of metric weight and contrastive temperature on the intended cohort.

Track A may exercise this machinery only through a visibly test-only stub (D31). The real mapping,
relation content, and age/window policy remain Nick decisions.

### Track D — surface-area QC investigation — parallel

Track D is read-only with respect to production policy. It owns:

1. exact reproduction of stored surface-area flags;
2. lower/upper and isolated-SA failure decomposition;
3. stratification by experiment, stage, biology, temperature, calibration, segmentation backend, and
   applicability;
4. embryo-track persistence analysis;
5. offline threshold counterfactuals;
6. a blinded boundary-review set;
7. a recommendation: retain, loosen, make diagnostic-only, or replace the lower gate.

The current lower rule is population/stage matched at `0.9 * p5`, not track-relative
(`src/data_pipeline/quality_control/surface_area_qc/config.py:20-31`;
`src/data_pipeline/quality_control/surface_area_qc/compute.py:36-48`). Production behavior changes
only after Nick reviews the study.

### Track E — stage-estimation state and reliability audit — parallel

Track E is read-only with respect to production and model behavior. It owns:

1. an updated map of live stage producers, contracts, orchestration, stored artifacts, and consumers;
2. a separate map of the legacy morphology-latent estimator and every fallback still reachable;
3. a plain-language distinction among nominal clock age, morphology-inferred developmental state,
   manually observed stage, and missing/default behavior;
4. evidence-backed reliability checks against whatever independent anchors actually exist;
5. sensitivity analysis for downstream stage users, especially metric windows and stage-conditioned
   surface-area QC;
6. a preserve/retire/replace recommendation and the minimum safe Track A/Track C interface.

The live pipeline value is a deterministic calculation from declared start age, elapsed time, and
temperature rather than image morphology
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:1-20,131-219`;
`src/data_pipeline/feature_extraction/stage_inference.py:15-24`). The legacy path fits an
`MLPRegressor` on VAE latent columns and then locally calibrates it
(`src/build/infer_developmental_age.py:46-100,107-156`). The 2026-08-24 lineage study could not
measure agreement because the exact old/new ID intersection was empty
(`reports/STUDY_stage_lineage.md:9-35`). Track E updates that evidence after regeneration and must not
manufacture a crosswalk by parsing IDs. Nick reported on 2026-08-27 that the default stages are not
reliable; the audit treats that as a hypothesis to characterize, not a conclusion to rubber-stamp.

Track E does not block A1-A3: Track A carries raw timing/temperature, stage value, status, and method
version and may ignore stage for vanilla training. Track E must finish before Nick ratifies O5 or a
scientific metric run claims that an hpf window has biological meaning.

### Corpus production — external, not a track

Regenerated pixel-dependent artifacts are an input to Track A. The adapter must preflight their
schema and report discrepancies. No core-model agent launches, repairs, or redefines the pipeline
rerun unless separately tasked.

## 5. Milestones and gates

| ID | Deliverable | Owner | Dependency | Gate |
|---|---|---|---|---|
| A0 | Contract v2 + current-writer preflight | lead + A1 | none | every source/grain discrepancy reported; synthetic observation/asset fixtures accepted |
| A1 | Observation/asset adapter | A1 | A0 | contract/join/filter/split tests pass; small real read-only preflight succeeds |
| A2 | Dataset, transforms, loaders | A2 | frozen v2 fixture; integrates after A1 | identity/path/metadata/shape/split tests pass; legacy `ImageFolder` path unreachable |
| A3 | Vanilla end-to-end | lead | A1 + A2 | full vanilla acceptance gate below passes |
| A4 | Test-only metric smoke | lead | A3 | paired batch, metric loss, backward, checkpoint under explicit stub |
| C1 | Mapping + relation API | C1 | A3 | exhaustive relation/mapping tests; uncovered values fail |
| C2 | Pair preflight + sampler | C2 | C1 | legal positives, no split leakage, indexed complexity, worker/rank reproducibility |
| C3 | `L_out` loss + tuning harness | C3 | C1 + stable batch contract | hand-checked loss tests and integrated metric run; no scientific claim before tuning |
| D1 | Surface-area study | D | none | reproducible report, tables, counterfactuals, and clearly separated evidence/hypotheses |
| E1 | Stage-estimation state/reliability audit | E | none | current/legacy machinery and consumers mapped; evidence limits stated; preserve/retire/replace recommendation delivered |

Agent completion is not milestone acceptance. A slice becomes:

1. `planned`;
2. `in_progress`;
3. `ready_for_integration` after its branch tests pass;
4. `integrated` after the lead merges and reruns tests;
5. `accepted` only after the milestone gate passes on the integration branch.

## 6. Track A details

### A0 — preflight and contract fixture

- Derive source contracts from current writer/validator symbols, not old reconnaissance schemas.
- Verify product-aware snip inventory, frame timing, stage, QC, plate, and collection-provenance paths.
- Record every mismatch between contract v2 and live artifacts.
- Publish one small synthetic observation table and asset table containing:
  - at least two observations;
  - two products for one `snip_id`;
  - projection rows with null `z_index`;
  - several ordered synthetic z rows;
  - temperature, elapsed time, stage status, QC status, and split.
- Do not make z rendering a Track A dependency; the z rows prove table/selector capacity only.

### A1 — adapter

- Accept an explicit ordered experiment list and explicit policies.
- Resolve artifacts only through the path authority.
- Build and validate observation and asset tables.
- Join frame timing and plate temperature without parsing IDs.
- Preserve missingness/status rather than converting absence to failure.
- Apply named product/QC/stage/covariate policies with counts.
- Assign and persist `physical_embryo_id` splits stable under cohort growth.
- Return structured source inventory and cohort report.
- Support inference by allowing QC, stage, split, and metric steps to be independently disabled.

### A2 — dataset and loaders

- Consume only the contract tables/resolved sample view; no direct pipeline reads.
- Select the configured vanilla BF asset and require exactly one match.
- Load exactly the row's path.
- Decode to grayscale, resize deterministically to configured `(H, W)`, and return finite float32 in
  `[0,1]`.
- Return identity, product, z, temperature, elapsed time, time index, and stage metadata.
- Keep observation metadata separate from asset multiplicity.
- Use split-local datasets/loaders and compact arrays/codes rather than a worker-copied object
  DataFrame.
- Preserve `DatasetOutput` and `self_stats`/`other_stats` compatibility.
- Leave a selector/grouping seam for future z assets; do not choose a z-model input shape.

### A3 — vanilla end-to-end acceptance

Use a small explicit set of real regenerated BF projection assets and a named smoke-test cohort
policy. Pass all of the following:

1. observation/asset tables build with a clean report or explicitly accepted warnings;
2. selected asset path and returned `snip_id`/product key agree for every tested row;
3. no `physical_embryo_id` crosses train/eval/test;
4. batches contain `[B, 1, 288, 128]` finite tensors and required metadata;
5. vanilla `Trainer.fit` executes training and validation optimizer steps with finite losses;
6. model parameters change after optimization;
7. a tiny fixed subset shows decreasing reconstruction objective under an overfit check;
8. checkpoint saves and reloads;
9. the reloaded model encodes and reconstructs a held-out batch with correct finite shapes;
10. local provenance reconstructs selected observation/asset keys;
11. the run succeeds with W&B disabled/offline.

This gate proves plumbing and vanilla model function. It does not validate final cohort science,
metric constraints, z modeling, or biological signal.

### A4 — test-only metric smoke

- Use a config name containing `test_only` or `dummy`.
- Use an explicit trivial mapping/relation and disabled or deliberately broad age gate.
- Exercise paired loading, `self_stats`/`other_stats`, metric loss, backward, validation, checkpoint,
  and reload.
- Mark provenance as a stub policy.
- Do not publish or describe the run as a scientific metric result.

## 7. Cohort policy in plain terms

The final cohort is the exact set of observations/assets admitted to a real training run. Choosing it
means choosing:

1. experiments;
2. product/z selector;
3. validity rule;
4. QC exclusion rule, including surface area;
5. behavior when QC is absent;
6. stage requirement and accepted statuses;
7. required covariates;
8. explicit test experiments and remaining split ratios.

Track A implements these controls and records their counts. It does not choose the final science
values. After regenerated QC arrives, Nick reviews the counts and ratifies the final policy. The run
then stores the exact selected keys and identity-based split assignments (D32/D33).

## 8. Run provenance and reproducibility

Every accepted run stores the D16/D17/D19/D33 bundle under one configured local root. W&B logging is
optional; local provenance is mandatory. Pair selection and decode resampling must be reproducible
across workers and distributed ranks. Source and selected-ID hashes are used; image content is not.

Decision-marker tests are added as their implementations land. `STATUS.md` remains generated and is
never edited to make progress look complete.

## 9. Coordination model

Agents are divided by owned files and interfaces, not one agent per thematic track. Track A is split
between adapter, dataset/loader, and lead integration. Tracks D and E run independently, with Track E
reporting stage-axis sensitivity relevant to Track D. After A3, Track C is split into relation,
sampler, and loss slices.

The lead agent alone owns shared planning/contract changes during parallel execution. Other agents
report a required interface change and stop at their file fence. The synthetic contract fixture is
the executable coordination boundary between A1 and A2.

Exact ownership, commands, and handoff templates are in `plans/AGENT_BRIEFS_PHASE1.md`.

## 10. Open decisions — Nick

| ID | Decision | Blocks |
|---|---|---|
| O1 | Final metric-group mapping and positive/negative/excluded relation policy | scientific Track C completion |
| O2 | Final named QC policy after regenerated-QC counts and Track D report | final cohort |
| O3 | Future z consumption: independent 2D planes, sampled planes, neighboring-plane channels, ordered stack, or auxiliary target | z-model implementation, not Track A |
| O4 | Final ordered experiment list and explicit test experiments | frozen science cohort |
| O5 | Stage source/status inclusion and separate sampler/loss windows | scientific metric training |
| O6 | Whether and how product/plane-specific QC should be produced | future multi-asset science cohort |
| O7 | Post-regeneration black-level handling: normalization, augmentation, conditioning, or no change | later science validity; not Track A |
| O8 | After Track E: treat nominal clock age as metadata only, rehabilitate a morphology estimator, build a new calibrated estimator, or use a hybrid/manual-anchor strategy | scientific stage-conditioned QC and metric interpretation |

## 11. Explicitly out of scope for the first vanilla gate

- renderer acceptance and corpus-rerun orchestration;
- final metric semantics or tuning;
- z-aware encoder/decoder changes;
- encoder/decoder redesign generally;
- latent biological/nuisance partition changes;
- native 576x256 model input;
- optical conditioning;
- segmentation-error mimicry;
- switching brightness augmentation defaults;
- implementing or training a replacement stage estimator;
- SeaHub inclusion unless explicitly selected;
- general pipeline packaging cleanup unrelated to reading the declared artifacts.
