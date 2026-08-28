# Study: stage-estimation machinery, evidence, and downstream sensitivity

**Measurement boundary.** This read-only audit ran on 2026-08-27 local / 2026-08-28 UTC at
`86f4af2782e591d2956bc9fc7184e17dcfea9d6f` on branch `agent/core-track-e-stage`, with Python
3.10.16, pandas 2.2.3, and NumPy 1.26.4
(`stage_estimation/run_summary.json`). It changed no producer, consumer, threshold, model, pipeline
artifact, or core-model file (repository diff; owned-file fence in
`docs/refactors/core-model/plans/AGENT_BRIEFS_PHASE1.md:701-899`).

**Bottom line.** The live `predicted_stage_hpf` producer is a deterministic nominal clock
calculation, not an image-morphology estimator. In the bounded current SeaHub snapshot, all 1,752
rows reproduce that formula exactly, but all are single observations and the collection-provenance
artifact required by the current entrypoint is absent for all 20 experiments
(`stage_estimation/run_summary.json`; `stage_estimation/stage_integrity_by_experiment.csv`). These
facts establish implementation consistency for the stored snapshot; they do not establish
anatomical-stage accuracy (inference from the absence of an independent anchor below).

The current and legacy morphology-inferred artifacts have zero exact shared `snip_id` and zero
exact shared `embryo_id` in this audit, so bias, correlation, and agreement remain not identifiable
(`stage_estimation/legacy_current_crosswalk.csv`). Track A can safely carry stage value, three-state
status, version, start age, elapsed time, and temperature as metadata while ignoring stage in
vanilla training; no scientific metric run should inherit the sampler's 1.5 hpf or the loss's 3.0
hpf gate as biologically validated windows (`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:94-127,239-274`;
`src/core/data/dataset_classes.py:99-127`; `src/core/losses/loss_functions.py:347-366`).

## 1. Terms: quantities that must not be conflated

| Term | Exact meaning in this repository | Source/status |
|---|---|---|
| Declared start age | For a single experiment, per-well `plate_metadata.start_age_hpf`; for a declared collection, per-source age from `start_age_by_source_ordinal`. It is an input declaration, not an observed morphology score. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:77-128` |
| Elapsed wall-clock time | Relative acquisition time resolved from `elapsed_time_s`, then `experiment_time_s`, then `time_s`; core's converged name is `elapsed_time_s`. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:34-45`; `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:71-79` |
| Incubation temperature | Per-well `plate_metadata.temperature`, called `incubation_temperature_c` at the model boundary to avoid collision with contrastive-loss temperature. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:159-213`; `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:101-114` |
| Nominal temperature-adjusted clock stage | `predicted_stage_hpf = start_age_hpf + elapsed_time_s/3600 * (0.055*temperature_c - 0.57)`. It reads declarations and clock time, not pixels. | `src/data_pipeline/feature_extraction/stage_inference.py:15-24` |
| Morphology-inferred stage | Legacy `inferred_stage_hpf_reg`: an `MLPRegressor` over VAE `z_mu*` columns followed by experiment- or temperature-conditioned linear calibration; snapshots copy nominal stage. | `src/build/infer_developmental_age.py:46-100,107-156` |
| Manually observed/anatomical stage | A legacy curation interface names `manual_stage_hpf`, but the exact declared file is absent; no populated manual/anatomical label artifact was identified in the explicit audit sources. SeaHub `stage_source_label` is parsed acquisition metadata, not a documented anatomical observation. | `src/build/build05_make_training_snips.py:21-53`; `src/data_pipeline/acquisition/seahub/reconciliation.py:320-333`; `stage_estimation/source_inventory.csv` |
| Stage status | Live source status is `predicted`, `missing_start_age_hpf`, or `missing_temperature`; planned core carries it as a non-null three-state-capable `stage_status`. | `src/data_pipeline/feature_extraction/stage_predictions/contract.py:18-25,49-65`; `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:119-127` |
| Method/model version | Live `model_version=kimmel1995_temp_rate_v1` identifies a formula/method, not trained weights; planned core carries it as `stage_model_version`. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:32,215-219`; `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:119-127` |
| Missing value | A row exists and its stage value is null, paired with a named missing-input status in live S01. | `src/data_pipeline/feature_extraction/stage_predictions/contract.py:58-65` |
| Status-column absence | Older S02 artifacts can contain finite stage values but no status column; core must call status `unavailable`, not failed. | `docs/refactors/core-model/reports/STUDY_stage_lineage.md:73-95`; `AGENTS.md:43-46` |
| Code fallback/default | Executable behavior chooses an alternate column/value or branch. It is distinct from both a missing value and a status-column absence. | `stage_estimation/fallback_inventory.csv` |

The terms “stage,” “prediction,” and “hpf” alone do not identify an estimand. This report therefore
uses the full names above throughout (terminology rule; `docs/refactors/core-model/plans/AGENT_BRIEFS_PHASE1.md:728-745`).

## 2. Live machinery, end to end

### Inputs, joins, and calculation

The filesystem entrypoint reads the configured snip inventory (the current rule supplies the
default-BF compatibility view), frame
inventory, plate metadata, physical-embryo registry, required collection provenance JSON, and an
optional acquisition inventory; it validates against the registry before writing CSV
(`src/data_pipeline/feature_extraction/stage_predictions/entrypoint.py:13-55`;
`src/data_pipeline/pipeline_orchestrator/rules/stage_predictions.smk:16-20,43-67`). The compute function
uses the explicit `snip_id` row as its output spine, joins frame timing by the row's explicit
`image_id`, joins plate facts by explicit `well_id`, and for collections joins `(well_id,
time_index)` to the acquisition inventory's explicit `source_ordinal`
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:37-74,147-164,178-198`). IDs are
never parsed or reconstructed in the audit script
(`scripts/recon/audit_stage_estimation.py:110-135,181-306`).

For a single experiment, absent/null start age yields null stage plus
`missing_start_age_hpf`; absent/null temperature yields null stage plus `missing_temperature`.
For a collection, missing source ordinal or missing age for its declared source is a structural
hard error rather than a nullable row (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:101-128,181-213`).
With complete inputs, the producer calls the exact formula in the definition table and emits
`predicted`, the method version, and the full snip feature spine
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:205-221`).

The contract grain is one row per `snip_id`. It requires the shared feature spine plus
`predicted_stage_hpf`, non-null `model_version`, and one of the three status literals; `predicted`
requires a non-null value and either missing-input status requires null
(`src/data_pipeline/feature_extraction/stage_predictions/contract.py:1-29,32-65`). The shared
feature-table validator supplies source/registry identity checks before the entrypoint writes the
artifact (`src/data_pipeline/feature_extraction/stage_predictions/entrypoint.py:44-55`).

### Orchestration and paths

The current rule builds one shard per well from validated inventory/registry plus frame, plate,
collection, and acquisition inputs; validates each shard; then merges the ordered schema with sort
keys `experiment_id, well_id, snip_id`
(`src/data_pipeline/pipeline_orchestrator/rules/stage_predictions.smk:16-68,71-110`). The tracked
path registry defines per-well `{well_id}_stage_predictions.csv` and merged
`{experiment_id}_stage_predictions.csv` under the feature-extraction stage
(`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:650-657`).

### Stored current artifacts and schema variants

This audit used the first 20 non-empty IDs from the explicit 92-ID authority
`/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiments.txt`
(SHA-256 `67e56843cd661e2a787b8edb8b228948f5cc1151cbc1e22c5a971ef4d5cd8677`) and the matching declared
bundle manifest (SHA-256 `bbdc1d634be90abe167c1d957a719c13cc757a9a4c19cf423847eda41b2f7a17`).
The exact ordered IDs and all source fingerprints are in
`stage_estimation/selected_experiments.txt` and `stage_estimation/source_inventory.csv`.

All 20 merged stage artifacts are S01 (explicit status), contain 1,752 unique `snip_id` rows, have
no duplicate `snip_id`, use only `predicted` and `kimmel1995_temp_rate_v1`, and have 1,752 finite
stages spanning 14–96 hpf (`stage_estimation/stage_integrity_by_experiment.csv`;
`stage_estimation/stage_status_distribution.csv`). Joining each row to the manifest-declared plate
and frame inputs reproduces all 1,752 stages with maximum absolute residual 0.0 hpf
(`stage_estimation/current_stage_formula_check.csv`; `stage_estimation/run_summary.json`).

This is nevertheless an **incoherent partial snapshot**, not regenerated-corpus acceptance
evidence. The canonical current-contract collection provenance is absent for all 20 selected
experiments, the current rerun logs document upstream failures, and the D seam has zero complete
`(snip_id, snip_product_key, z_index)` asset keys
(`docs/refactors/core-model/reports/STUDY_surface_area_qc.md:54-99`;
`stage_estimation/source_inventory.csv`; `stage_estimation/run_summary.json`). Formula agreement
does not repair missing provenance or validate anatomical accuracy (inference from those sources).

The broader dated 2026-08-24 study found S01 and S02 among 109 stage-bearing experiments: 511,118
rows were `predicted`, 898 were `missing_start_age_hpf`, and 20,219 finite S02 rows had status
`unavailable`; 24 inventory-bearing experiments had no stage artifact
(`docs/refactors/core-model/reports/STUDY_stage_lineage.md:73-95,194-230`). This audit does not replace
those broader dated measurements with the narrower SeaHub snapshot.

## 3. Downstream consumer inventory

The repeatable repository search scanned tracked `.py`, `.smk`, YAML, and shell files under
`src`, `scripts`, and `results` for the exact stage aliases and `time_window`. It records every
matching path and line, including historical launch scripts and definitions that are not active
consumers, in `stage_estimation/consumer_search_inventory.csv`
(`scripts/recon/audit_stage_estimation.py:646-711`). The confirmed load-bearing consumers are:

| Consumer | How stage is used | Evidence |
|---|---|---|
| Live surface-area QC | Conditioning axis: interpolate p5/p95 reference at `predicted_stage_hpf`; null stage becomes false flag with `not_applicable`. | `src/data_pipeline/quality_control/surface_area_qc/config.py:20-31`; `src/data_pipeline/quality_control/surface_area_qc/compute.py:68-113` |
| Surface-area report | Plot/gallery axis and reference-band lookup after joining stage by `snip_id`. | `src/data_pipeline/quality_control/surface_area_qc/report.py:25-40,73-87`; `src/data_pipeline/pipeline_orchestrator/rules/surface_area_qc.smk:122-143` |
| Planned Track A manifest | Metadata carried as nullable value plus three-state-capable status/version; basic cohort does not require it, require-stage policy is explicit. | `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:119-127,214-245,283-299` |
| Planned Track A cohort filters | Optional configured stage requirement/accepted statuses; absence is not failure. | `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:232-245`; `AGENTS.md:43-48` |
| Current legacy core sequence key | Chooses `inferred_stage_hpf` from training metadata, otherwise warns and renames `predicted_stage_hpf` to `stage_hpf`. | `src/core/data/dataset_utils.py:26-38` |
| Current legacy metric sampler | Pair constraint `abs(stage_i-stage_j) <= time_window`, default 1.5 hpf after loss-to-data config propagation. | `src/core/data/dataset_classes.py:99-135`; `src/core/losses/loss_configs.py:132-139`; `src/core/models/model_configs.py:121-128` |
| Current metric loss | Batch target gate `abs(stage_i-stage_j) <= time_window + 1.5`; at the current default this is 3.0 hpf, separate from the sampler's 1.5. | `src/core/losses/loss_functions.py:347-369` |
| Analysis-ready assembly | Stage table is a mandatory fan-in input and is carried as metadata by one-to-one `snip_id` join. | `src/data_pipeline/pipeline_orchestrator/rules/analysis_ready.smk:20-45`; `src/data_pipeline/analysis_ready/assemble.py:31-86` |
| Analysis-ready reports | PCA color, stage bins, alive-count and survival plot axes; `death_event_stage_hpf` is a separate direct-formula field. | `src/data_pipeline/analysis_ready/report.py:51-122,166-274` |
| Core/analyze/app utilities | Ordering, spline fitting, binning, coverage, visualization, and reporting axis across the exact paths enumerated in the search table. | `stage_estimation/consumer_search_inventory.csv` |
| Death-event QC | Computes `death_event_stage_hpf` independently from plate/frame inputs using the same pure formula; it does not consume the per-snip stage artifact. | `src/data_pipeline/quality_control/death_detection/death_event.py:1-17,62-99,127-145` |

A current-code nuance corrects an easy overstatement in the dated lineage report: `UrrDataConfig`
requires and merges `age_key.csv.inferred_stage_hpf_reg`, but `NTXentDataConfig` constructs its
`age_hpf_vec` from the pre-existing `seq_key.stage_hpf`, not from the merged `_reg` column
(`src/core/data/dataset_configs.py:46-66,211-237`). Thus the file is required by the current legacy
Hydra path, but its `inferred_stage_hpf_reg` values are not the sampler axis in this code. This is a
measured code-path correction, not a claim about older releases.

The planned manifest-backed path does not call `make_seq_key`; its binding contract consumes the
resolved sample table and makes stage policy explicit
(`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:214-245,276-299`). Therefore the warning in
`dataset_utils.py` is reachable from the current legacy Hydra data configs but is not evidence of a
planned Track A fallback (code-path comparison above).

## 4. Legacy morphology path

`infer_developmental_age` reads `embryo_stats_df.csv`, merges per-snip temperature, `embryo_id`, and
relative time, and then selects every column containing `z_mu` as the predictor block
(`src/build/infer_developmental_age.py:14-29,46-49,78-80`). The training target is the nominal
`predicted_stage_hpf`, and training rows are control/reference rows from either the caller's explicit
`reference_datasets` or four hard-coded date/control rules
(`src/build/infer_developmental_age.py:51-85`).

The fitted `MLPRegressor(random_state=1, max_iter=5000)` predicts morphology-derived stages for
reference/control rows. For each timelapse experiment, a second `LinearRegression` maps nearby
nominal stage to MLP stage; insufficient same-experiment references fall back to same-temperature
references. Snapshots copy nominal stage directly rather than using a morphology result
(`src/build/infer_developmental_age.py:83-100,107-156`).

The output carries `calc_stage_hpf`, `inferred_stage_hpf_reg`, elapsed hours, temperature,
experiment/embryo identity, and `train_dir`, model, and architecture names; the code writes
`metadata/age_key_df.csv`, whereas current core looks for `metadata/age_key.csv`
(`src/build/infer_developmental_age.py:31-40`; `src/core/data/dataset_configs.py:38-58`). The audited
on-disk `age_key.csv` has 83,476 unique finite rows, 40 experiment-date values, temperatures
22/29/30 C, and one stored tuple:
`20241008 / VAE_training_2024-10-08_21-40-44 / VAE_z100_ne250_base_model`
(`stage_estimation/legacy_provenance_summary.csv`). It does not persist the reference-dataset list,
model-weight checksum, or estimator hyperparameters
(`stage_estimation/legacy_provenance_summary.csv`; comparison with function arguments at
`src/build/infer_developmental_age.py:14,46`).

Concrete risks follow from that implementation:

- **Circular target risk:** the learned target is the nominal clock stage that the morphology output
  is later compared with, so agreement to that target is not independent anatomical validation
  (`src/build/infer_developmental_age.py:74-85`; inference).
- **Leakage risk:** experiment-local calibration uses reference rows from the same experiment and
  chooses neighbors using both nominal stage and clock time, without a persisted held-out protocol
  (`src/build/infer_developmental_age.py:107-153`; inference).
- **Domain-shift risk:** default MLP training is restricted to named dates/controls and fallback
  calibration pools by exact temperature, while the output spans other experiments/perturbations
  (`src/build/infer_developmental_age.py:51-70,107-153`; inference).
- **Reproducibility risk:** names survive, but the exact training rows, fitted weights, reference
  branch, and checksums do not (`stage_estimation/legacy_provenance_summary.csv`; inference).

Repository search found no current pipeline-orchestrator call to `infer_developmental_age`; the only
live call in that file is its hard-coded `__main__` block
(`src/build/infer_developmental_age.py:164-171`; `rg -n 'infer_developmental_age\('
src/data_pipeline src/core src/build`). Persisted legacy tables and old VAE/config code remain
readable, but the estimator is not a current pipeline launch path (same search; current
orchestration inventory in `stage_estimation/consumer_search_inventory.csv`).

## 5. Defaults and fallbacks

The complete audited fallback table gives trigger, selected value/column, signal strength,
reachability, and exact source citation in `stage_estimation/fallback_inventory.csv`. The most
consequential cases are:

- Live time aliases and the legacy collection age-map alias choose alternatives silently, then
  hard-fail if no usable value exists
  (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:34-45,107-128`).
- Live single-experiment missing start age or temperature is an explicit null/status state, not a
  fallback value (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:199-213`).
- Current legacy core warns and substitutes nominal `predicted_stage_hpf` when
  `inferred_stage_hpf` is absent; this path is not in planned Track A
  (`src/core/data/dataset_utils.py:26-38`; manifest contract cited above).
- Current surface-area QC treats a null stage as not applicable, while out-of-range finite stages
  silently clamp to the nearest reference endpoint through `numpy.interp`
  (`src/data_pipeline/quality_control/surface_area_qc/compute.py:87-103`;
  `src/data_pipeline/quality_control/surface_area_qc/reference.py:40-51`).
- Legacy morphology calibration silently changes estimands for snapshots by copying nominal stage,
  and changes reference scope when same-experiment matches are insufficient
  (`src/build/infer_developmental_age.py:123-156`).
- The manually launchable combined Build04 path creates literal stage 0.0 if the column is absent;
  this behavior is not in the current pipeline or planned Track A
  (`src/build/build04_perform_embryo_qc.py:1275-1300,1660-1678`).

Mistaking any fallback nominal clock stage for morphology stage would silently alter sampler
membership, loss targets, stage-conditioned QC bounds, and plots because each consumer uses numeric
deltas or interpolation without an estimand tag beyond the column name (inference from the consumer
table and cited implementations).

## 6. Reliability and identifiability

### Input and contract integrity

For the bounded SeaHub inputs, every stage row is finite, unique, status-consistent, version-
consistent, and exactly reconstructable from declared plate/frame values
(`stage_estimation/stage_integrity_by_experiment.csv`; `stage_estimation/current_stage_formula_check.csv`).
All 1,752 observations are at elapsed time 0 and 28.5 C, and D classified every physical embryo as
a one-observation snapshot (`docs/refactors/core-model/reports/STUDY_surface_area_qc.md:135-147,156-162`).
Accordingly, zero physical embryos have two observations and there are zero comparable adjacent
transitions: per-embryo monotonicity and discontinuities are **not assessable**, not “passing”
(`stage_estimation/run_summary.json`; `stage_estimation/stage_integrity_by_experiment.csv`).

Collection/start-age provenance completeness fails the current contract for all 20 experiments:
the formula can be reconstructed from plate values, but the current entrypoint requires a declared
collection-provenance JSON even for a one-source experiment
(`src/data_pipeline/feature_extraction/stage_predictions/entrypoint.py:13-35`;
`stage_estimation/source_inventory.csv`). This discrepancy is consistent with the failed current
reruns and is why these stored artifacts cannot stand in for regenerated acceptance
(`docs/refactors/core-model/reports/STUDY_surface_area_qc.md:54-99`).

### Independent anchors and paired identities

| Candidate anchor | Available result | Identifiable question |
|---|---|---|
| Exact old/new `snip_id` crosswalk | 0 rows | Correlation, bias, and agreement are not identifiable. |
| Exact old/new `embryo_id` crosswalk | 0 identities | Track-level residuals are not identifiable. |
| Populated manual/anatomical labels | Declared legacy curation candidate absent | Anatomical-stage error and bias are not identifiable. |
| Fertilization timestamps | No explicit timestamp field in selected stage/plate/frame sources | Absolute hpf accuracy is not identifiable. |
| Trusted reference cohort with independent scoring | No such scored artifact in explicit sources | Experiment/temperature/stage-range bias is not identifiable. |

Sources: `stage_estimation/legacy_current_crosswalk.csv`,
`stage_estimation/source_inventory.csv`, and the explicit current source columns in
`stage_estimation/current_stage_formula_check.csv`. SeaHub source stage labels and plate start age
are inputs to the formula, so agreement with them is not independent validation
(`src/data_pipeline/acquisition/seahub/integration.py:1253-1267`; inference).

The current 1,752-row crosswalk conclusion matches, but does not supersede, the dated broader
2026-08-24 result of zero exact paired rows across 83,476 legacy and 532,235 then-current stage rows
(`docs/refactors/core-model/reports/STUDY_stage_lineage.md:19-35,142-164`). IDs were not parsed,
sliced, or reconstructed in either study.

## 7. Downstream sensitivity

### Surface-area QC

The shared Track D seam has 1,752 rows and zero complete asset keys; it is explicitly the partial,
incoherent snapshot described above (`stage_estimation/run_summary.json`;
`docs/refactors/core-model/reports/STUDY_surface_area_qc.md:89-99`). On that seam, the stored nominal
axis produces 223 surface-area flags. Replacing it with the plate `stage_hpf` column changes zero
classifications because those two source values coincide here; this is not independent validation
(`stage_estimation/surface_area_stage_sensitivity.csv`).

Uniform diagnostic offsets change classifications as follows:

| Offset | Flagged rows | Rows whose classification changes |
|---:|---:|---:|
| -3.0 hpf | 218 | 93 |
| -1.5 hpf | 185 | 48 |
| +1.5 hpf | 326 | 119 |
| +3.0 hpf | 377 | 178 |

Source: `stage_estimation/surface_area_stage_sensitivity.csv`. These offsets are stress tests, not
alternative estimators. They demonstrate boundary sensitivity only; no false-positive rate or
anatomical accuracy is identifiable without usability/stage labels (inference).

### Metric sampler and loss

At the current 1.5 hpf sampler window, each single-observation embryo has one self option, and there
are 525,032 ordered **age-eligible** other-embryo pairs before split and relation filtering. The
loss's separate 3.0 hpf gate has the same age-eligible membership in this coarse seven-stage
snapshot; membership first changes at 4.0 hpf (+77,952 ordered pairs), 6.0 (+204,288), and 12.0
(+519,552) relative to 1.5 (`stage_estimation/metric_window_sensitivity.csv`).

These are not scientific legal-positive counts. Actual legality also requires split membership and
the positive/excluded relation mapping, which Track C/Nick have not selected. Therefore scientific
legal-positive counts and membership are **not identifiable** from this seam; the table deliberately
labels its results `not_identifiable_without_relation_mapping_and_split`
(`stage_estimation/metric_window_sensitivity.csv`;
`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:259-274`).

Uniform offsets and the coincident plate axis change no sampler/loss pair membership because pair
deltas cancel a uniform shift; they can still change the nonlinear surface-area reference lookup
(`stage_estimation/metric_axis_sensitivity.csv` versus
`stage_estimation/surface_area_stage_sensitivity.csv`). Scale or experiment-dependent stage errors
could change pair membership, but their effect size is not identifiable without paired anchors
(inference from the missing-crosswalk result).

### Missing stage

The planned basic Track A policy retains rows with `stage_status=unavailable`; an explicit
require-stage policy rejects them. Current surface-area QC retains them as `not_applicable`, while
the legacy metric dataset produces no age-matched self/other options and reaches an empty random
choice. Stage-axis reports have no finite stage rows to bin or plot
(`stage_estimation/stage_absence_sensitivity.csv`; source implementations in the consumer table).
Absence must therefore remain a named state and policy input, never a boolean failure.

### Stratification limits

The current seam contains only snapshots, one temperature (28.5 C), one source scope, one
calibration status, and one segmentation model; no within-embryo trajectories exist
(`docs/refactors/core-model/reports/STUDY_surface_area_qc.md:135-162`). Consequently sensitivity
differences between snapshots/time series, temperatures, source scopes, calibration states, or
segmentation versions are not identifiable in this current audit. Per-experiment and nominal-stage
composition is retained in the row-level seam and integrity table, but composition is not accuracy
(`stage_estimation/current_stage_formula_check.csv`; inference).

## 8. Decision-oriented recommendation

| Component | Current role | Evidence | Preserve / retire / replace / undecided | Immediate action | Blocks |
|---|---|---|---|---|---|
| Declared start age, elapsed time, temperature | Formula inputs and model metadata | Exact reconstruction for 1,752 rows; provenance incomplete | Preserve | Carry raw values plus source/status; never impute in core | None for vanilla |
| `predicted_stage_hpf` | Nominal clock metadata; QC/report axis | Formula-consistent, anatomical accuracy not identifiable | Preserve as metadata; scientific role undecided | Name estimand/version in provenance; vanilla may ignore | O5/O8 science use |
| Stage status and method version | Absence/method provenance | Live S01 contract plus historical S02 | Preserve | Keep three-state-capable status; never map absence to failure | None for vanilla |
| Current collection provenance seam | Required live source authority | Missing for 20/20 current selected artifacts | Replace incomplete artifacts through external rerun, not Track E code | Fail regenerated-corpus preflight until declared artifact exists | Corpus acceptance |
| Legacy `dataset_utils` nominal fallback | Current positional loader fallback | Reachable from legacy Hydra; not planned manifest path | Retire from new path | Ensure manifest dataset never calls it | Track A integration check |
| Legacy morphology estimator code | Manual learned/calibrated path | Missing training/reference/weight provenance; no paired validation | Retire from operational path; preserve code/data only as historical evidence pending Nick | Do not rehabilitate by default | O8 |
| Legacy `age_key.csv` | Historical morphology-stage artifact | 83,476 finite rows, provenance tuple but no current crosswalk | Preserve as immutable candidate benchmark; usefulness undecided | Require authoritative crosswalk before comparison | O8/identity authority |
| Stage-conditioned surface-area QC | Current interpolation axis | Mechanically sensitive; D seam partial and unlabeled | Undecided | Keep this audit diagnostic; no production threshold/policy change | Independent labels + regenerated corpus |
| Metric sampler/loss windows | 1.5 hpf sampler; 3.0 hpf loss gate | Different gates; current seam cannot establish scientific legality | Replace inherited assumptions with separately configured/validated windows | Track C records source and two windows; no biological claim yet | O5 + relation policy + validation |
| Manual/anatomical labels | Potential independent anchor | Interface exists; declared candidate artifact absent | Undecided | Locate or collect explicit labels with opaque-ID joins | O8 validation design |

Every evidence cell above cites this report's derived tables or the source sections that precede it;
the preserve/retire/replace column is a recommendation, not an implementation change.

Plain answers:

1. **Track A:** safely carry start age, elapsed time, incubation temperature, nominal stage,
   stage status, and method version; ignore stage in vanilla training under the explicit basic
   policy (`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:214-245,283-299`).
2. **New manifest path:** make the positional `make_seq_key` fallback, filename parsing, and
   implicit age-key requirement unreachable; require explicit IDs and named stage policy
   (`src/core/data/dataset_utils.py:26-75`; `AGENTS.md:30-38`).
3. **Stage-conditioned QC today:** nominal clock stage can mechanically index the current reference,
   but it cannot be described as anatomically validated; the current SeaHub sensitivity is
   diagnostic only (`stage_estimation/surface_area_stage_sensitivity.csv`; missing-anchor table).
4. **Metric sampling/loss today:** code can consume the numeric axis, but the sampler and loss
   windows must be separately configured and treated as unvalidated until O5 and a validation study
   are resolved (`src/core/data/dataset_classes.py:121-127`;
   `src/core/losses/loss_functions.py:347-366`).
5. **Legacy estimator:** no current evidence supports restoring it operationally. Preserve the
   artifact as a possible historical benchmark only if an authoritative crosswalk and missing
   provenance can be supplied; otherwise retire the operational code (recommendation from legacy
   risks and zero-crosswalk evidence).
6. **Smallest credible study:** obtain an explicit opaque-ID crosswalk plus independently scored
   anatomical labels or fertilization-time anchors, then evaluate held-out-by-experiment and
   held-out-by-temperature bias/agreement and downstream pair/QC stability. This is a study design,
   not a replacement architecture.
7. **Nick decisions:** O5 selects acceptable stage statuses/source and separate windows; O8 selects
   metadata-only, rehabilitation, new calibration, or hybrid/manual-anchor direction
   (`docs/refactors/core-model/PLAN.md:296-304`).

Bounded options when current evidence cannot choose:

- **Metadata-only:** keep nominal clock stage for ordering/description and do not use it in
  scientific metric relations; prerequisite is only complete source/status provenance.
- **Validate nominal clock stage:** add independent timestamps/anatomical labels and accept it only
  if held-out bias and downstream membership stability meet Nick's predeclared criteria.
- **Evaluate morphology or hybrid staging:** only after a labeled, identity-crosswalked dataset and
  leakage-proof experiment/temperature splits exist; no architecture is selected here.

## 9. Reproducibility and output index

The exact command was:

```bash
cd /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-core-e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate morphseq-env
python scripts/recon/audit_stage_estimation.py \
  --experiment-list /net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiments.txt \
  --experiment-limit 20 \
  --experiment-manifest /net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiment_manifest.csv \
  --pipeline-output-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output \
  --legacy-age-key /net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/age_key.csv \
  --surface-area-seam docs/refactors/core-model/reports/surface_area_qc/stage_axis_sensitivity_input.csv \
  --surface-area-source-inventory docs/refactors/core-model/reports/surface_area_qc/source_inventory.csv \
  --surface-area-reference src/data_pipeline/quality_control/surface_area_qc/references/surface_area_reference_v1.csv \
  --manual-curation-csv /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/combined_metadata_files/curation/curation_df.csv \
  --output-dir docs/refactors/core-model/reports/stage_estimation
```

The script selects a bounded prefix from the explicit list, verifies manifest order, constructs
stage/provenance paths only from declared IDs and the tracked path contract, and never searches the
pipeline output tree (`scripts/recon/audit_stage_estimation.py:90-135,181-306,714-760`).

Important fingerprints are in `stage_estimation/source_inventory.csv`: legacy age key
`4fa7473b0d70255fd796a8b0dd3d23023c1f60c15847e06f65a4e3312cc608f6`, Track D seam
`f10dde5cc668683b0572dfba022cc34785c577b21f2bd5b0d39925518e572dde`, Track D source inventory
`628de8a8ccc2fbb1c4ae7f42a638c195050ba4cd4e31c038bb8b4bf2c9867858`, and surface-area reference
`3bffedb6a9ef96ec58e7e85b1e37ffb4983b2120d1a7607563a58ae64bb6c960`.

Outputs:

- `run_summary.json`: environment, revision, authority, runtime, coverage, crosswalk, and seam
  limitations.
- `source_inventory.csv` and `selected_experiments.txt`: explicit authority and source
  fingerprints, including missing artifacts.
- `current_stage_formula_check.csv`, `stage_integrity_by_experiment.csv`, and
  `stage_status_distribution.csv`: row-level reconstruction and current contract integrity.
- `legacy_provenance_summary.csv` and `legacy_current_crosswalk.csv`: legacy coverage/provenance and
  exact-ID agreement boundary.
- `surface_area_stage_sensitivity.csv`, `metric_window_sensitivity.csv`,
  `metric_axis_sensitivity.csv`, and `stage_absence_sensitivity.csv`: downstream sensitivity.
- `fallback_inventory.csv` and `consumer_search_inventory.csv`: repeatable fallback and repository
  consumer inventories.

Unavailable artifacts and non-identifiable questions are reported rather than replaced with parsed
IDs, filename proxies, experiment averages, or assumed policies (study procedure; script and tables
above).
