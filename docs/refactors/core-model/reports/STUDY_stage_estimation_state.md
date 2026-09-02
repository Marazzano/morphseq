# Study: stage-estimation machinery and reliability state

**Agent:** refactor Agent E

**Completion state:** `ready_for_integration`

**Measurement date:** 2026-08-27 America/Los_Angeles (the final audit JSON records its UTC timestamp)

**Audited code base:** `core-model-refactor` at `d830391f9f7132d8086a53b5470129847e07e9d2`
(`git rev-parse HEAD`, 2026-08-27)

**Scope:** read-only audit; no producer, QC rule, threshold, core model, legacy implementation, or
source artifact was changed. The executable added is the read-only measurement script, and all
generated files are under `reports/stage_estimation/` (`git diff --name-only --
scripts/recon/audit_stage_estimation.py
docs/refactors/core-model/reports/STUDY_stage_estimation_state.md
docs/refactors/core-model/reports/stage_estimation`, 2026-08-27).

## Executive answer

The live pipeline's `predicted_stage_hpf` is a **nominal temperature-adjusted clock**, not an image
or morphology estimate. It deterministically combines a declared start age, elapsed wall-clock
time, and incubation temperature using one formula; its method literal is
`kimmel1995_temp_rate_v1` (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:32-45,
77-128,131-219`; `src/data_pipeline/feature_extraction/stage_inference.py:15-24`).

The audited artifacts support an input/contract-integrity conclusion, not an accuracy conclusion.
There are 109 stage files and 532,235 rows; 511,118 rows declare `predicted`, 898 declare
`missing_start_age_hpf`, and 20,219 finite rows have no source status column and are therefore
`unavailable`, not failed. The exact current/legacy `snip_id` crosswalk remains zero and the audited
training metadata contains zero non-null `manual_stage_hpf` values. Absolute accuracy, biological
bias, and current-versus-legacy agreement are therefore **not identifiable** from these artifacts
([audit_summary.json](stage_estimation/audit_summary.json),
[exact_crosswalk.csv](stage_estimation/exact_crosswalk.csv), and
[legacy_axis_summary.csv](stage_estimation/legacy_axis_summary.csv), measured 2026-08-27 by the exact
command in “Reproducibility”).

The legacy morphology-latent MLP should not be restored as a production default. It was trained on
nominal clock stage rather than independent anatomical labels, calibrated partly on the same
experiment or temperature, copied nominal stage unchanged for snapshots, and did not persist its
reference-row identity (`src/build/infer_developmental_age.py:46-100,107-156`). Its frozen CSV is
useful as a lineage artifact or retrospective benchmark, but the executable path lacks the evidence
needed for rehabilitation. This is a recommendation based on the cited implementation and artifact
provenance, not an assertion that morphology contains no developmental information.

The more immediate core risk is different: the integrated core path at the audited commit prefers
the legacy `inferred_stage_hpf` column, not `inferred_stage_hpf_reg`; the former has extreme finite
values from `-7.7063e19` to `4.2266e18` in the current training metadata. If that column is absent,
the loader prints a warning and falls back to nominal `predicted_stage_hpf`
(`src/core/data/dataset_utils.py:26-38`;
[legacy_axis_summary.csv](stage_estimation/legacy_axis_summary.csv), measured 2026-08-27). Both paths
must be unreachable from the planned manifest-backed Track A presets; basic Track A explicitly does
not require stage (`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:215-227`).

## 1. Terms used in this report

| Term | Exact meaning here | Not equivalent to |
|---|---|---|
| Declared start age | `start_age_hpf` attached to a well for a single experiment, or to a source ordinal for a collection. It is an input declaration, not an observed per-snip outcome (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:77-128`). | Fertilization timestamp; morphology; anatomical stage at imaging. |
| Elapsed wall-clock time | First non-null value in `elapsed_time_s`, `experiment_time_s`, then `time_s`, joined from frame inventory by `image_id` (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:34-45,147-178`). | Developmental progress. |
| Incubation temperature | Per-well `plate_metadata.temperature`, used in both single and collection branches (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:159-177,205-213`). | Contrastive-loss temperature. |
| Nominal temperature-adjusted clock stage | Live `predicted_stage_hpf = start_age_hpf + elapsed_time_s/3600 * (0.055 * temperature_c - 0.57)` (`src/data_pipeline/feature_extraction/stage_inference.py:15-24`). | Morphology-inferred or manually observed stage. |
| Morphology-latent MLP stage | Legacy `inferred_stage_hpf_reg`, initially predicted from VAE `z_mu*` columns and then locally calibrated (`src/build/infer_developmental_age.py:46-100,107-153`). | Live `predicted_stage_hpf`; legacy area-based `inferred_stage_hpf`. |
| Legacy area-inferred stage | `inferred_stage_hpf`, produced by a surface-area reference and experiment-level interpolation/regression in the legacy build (`src/build/build04_perform_embryo_qc.py:657-674,711-803`). | The latent MLP field despite the similar name. |
| Independently observed anatomical stage | A morphology label assigned independently of the formula/model being evaluated, with an exact row/animal identity. No usable anchor was found in the audited artifacts (zero manual values in [legacy_axis_summary.csv](stage_estimation/legacy_axis_summary.csv), measured 2026-08-27). | A declared start age or the MLP's nominal training target. |
| Stage status | Live literal `predicted`, `missing_start_age_hpf`, or `missing_temperature`; for an old schema with no status column this report records `unavailable` (`src/data_pipeline/feature_extraction/stage_predictions/contract.py:18-25,49-65`). | A Boolean success/failure flag. |
| Method/model version | Live `model_version`, which is formula provenance, or the legacy training/model/architecture tuple (`src/data_pipeline/feature_extraction/stage_predictions/contract.py:1-6`; `src/build/infer_developmental_age.py:31-40`). | Evidence of accuracy. |
| Missing value | Null/non-finite stage in a row. | A schema that never declared a status column. |
| Status unavailable | A finite stage from the two S02 files whose schema lacks `stage_prediction_status` ([stage_status_counts.csv](stage_estimation/stage_status_counts.csv), measured 2026-08-27). | Failed staging. |
| Code fallback/default | A branch that selects another column, literal, or alias because an expected input is absent. | A scientifically validated substitute. |

## 2. Audit boundary and reproducibility

The ordered experiment authority is the first occurrence of each `experiment_id` in
`docs/refactors/core-model/reports/recon_tables/availability_schema.csv`: 148 IDs, SHA-256
`64f5f36ca0d76e971364ab71fd45c56d87112884203734d89550b1937e0f9409`. The exact ordered array is in
`audit_summary.json`; expected paths were constructed from each explicit ID and the path registry,
and the pipeline output tree was never globbed
(`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:135-166,294-310,411-440,559-568,
646-688,903-912,1103-1114,1170-1214`;
[source_fingerprints.csv](stage_estimation/source_fingerprints.csv), measured 2026-08-27).

This corrects one fingerprint typo in the dated 2026-08-24 lineage report: that report prints the
legacy age-key digest as the experiment-authority digest at
`docs/refactors/core-model/reports/STUDY_stage_lineage.md:267-277`. Its row counts and `n=0`
crosswalk conclusion reproduce; only the authority digest was wrong
([audit_summary.json](stage_estimation/audit_summary.json) and
[source_fingerprints.csv](stage_estimation/source_fingerprints.csv), measured 2026-08-27).

The source roots were:

- pipeline output: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output`;
- legacy training metadata:
  `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/embryo_metadata_df_train.csv`;
- legacy age key:
  `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/age_key.csv`;
- legacy relation key:
  `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/metric_key.csv`; and
- packaged surface reference:
  `src/data_pipeline/quality_control/surface_area_qc/references/surface_area_reference_v1.csv`
  ([source_fingerprints.csv](stage_estimation/source_fingerprints.csv), measured 2026-08-27).

The audit ran under
`/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python`, Python 3.10.16
(`which python`; `python --version`, 2026-08-27). A direct `python -c 'import data_pipeline'` from the
repository root failed with `ModuleNotFoundError`, and importing through `src.data_pipeline` reached
an orchestration initializer that still imports top-level `data_pipeline`; no `PYTHONPATH` or
`sys.path` workaround was used (commands run 2026-08-27). The audit therefore read the declared path
templates statically and source artifacts directly; it did not weaken packaging checks.

Exact measurement command:

```bash
source /net/trapnell/vol1/home/nlammers/micromamba/etc/profile.d/conda.sh
conda activate morphseq-env
python scripts/recon/audit_stage_estimation.py \
  --repo-root /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq \
  --pipeline-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output \
  --experiment-authority /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/docs/refactors/core-model/reports/recon_tables/availability_schema.csv \
  --training-metadata /net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/embryo_metadata_df_train.csv \
  --legacy-age-key /net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/age_key.csv \
  --metric-key /net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/metric_key.csv \
  --surface-reference /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/data_pipeline/quality_control/surface_area_qc/references/surface_area_reference_v1.csv \
  --output-dir /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/docs/refactors/core-model/reports/stage_estimation \
  --membership-sample-size 256
```

The run is fail-loud on duplicate `snip_id`, conflicting per-image elapsed times, incomplete metric
group coverage, missing required keys, and absent required source files
(`scripts/recon/audit_stage_estimation.py`; `python scripts/recon/audit_stage_estimation.py --help`,
2026-08-27).

## 3. Live producer: end-to-end map

### Inputs, joins, and collection handling

| Input | Grain and join | Live behavior | Evidence |
|---|---|---|---|
| Snip inventory | One source row per `snip_id`; loop driver carrying the feature spine, `well_id`, `image_id`, and `time_index`. | Produces exactly one output row for every input snip. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:131-157,215-221`; contract grain at `contract.py:1-6`. |
| Frame inventory | `image_id` lookup. | Resolves elapsed time in the order `elapsed_time_s`, `experiment_time_s`, `time_s`; no available alias is a hard, ID-specific error. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:34-45,147-178`. |
| Plate metadata | `well_id` lookup. | Missing well is a hard error; singles take `start_age_hpf` here and all modes take `temperature` here. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:148-177,199-213`. |
| Collection provenance | Experiment JSON, required by the current entrypoint. | `is_collection=false` uses the plate age. `is_collection=true` uses `start_age_by_source_ordinal`, with `start_age_by_time_index` accepted only as a legacy artifact alias. | `src/data_pipeline/feature_extraction/stage_predictions/entrypoint.py:13-50`; `compute.py:77-128`. |
| Acquisition inventory | `(well_id, time_index) -> source_ordinal`. | Consulted for collections; conflicting source ordinals or an uncovered collection source fail loudly. | `src/data_pipeline/feature_extraction/stage_predictions/compute.py:48-74,181-198`. |
| Physical-embryo registry | Source validator input, not a formula input. | Output is validated with `check_sources=True`. | `src/data_pipeline/feature_extraction/stage_predictions/entrypoint.py:27-51`. |

For a collection, missing source ordinal or source-specific start age is structural failure; it does
not become a nullable status. For a single experiment, missing `start_age_hpf` becomes a null stage
with `missing_start_age_hpf`; missing temperature becomes a null stage with
`missing_temperature` (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:181-213`).

The current collection key is correctly `source_ordinal`, not merged `time_index`. One source may
span many merged times, so using time index as source identity could assign another acquisition's
declared age (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:85-100`). The
Snakemake helper comment still says a collection reads start age by `time_index`; that comment is
stale relative to the implementation and task CLI
(`src/data_pipeline/pipeline_orchestrator/rules/stage_predictions.smk:31-34` versus
`src/data_pipeline/feature_extraction/stage_predictions/compute.py:107-128` and
`src/data_pipeline/pipeline_orchestrator/tasks.py:1810-1823`).

### Formula, schema, and validation

```text
elapsed_hours = elapsed_time_s / 3600
developmental_rate = 0.055 * temperature_c - 0.57
predicted_stage_hpf = declared_start_age_hpf + elapsed_hours * developmental_rate
```

The formula reads no pixels or morphology, fits no weights, and always writes method literal
`kimmel1995_temp_rate_v1` (`src/data_pipeline/feature_extraction/stage_inference.py:15-24`;
`src/data_pipeline/feature_extraction/stage_predictions/compute.py:28-34,212-218`). The repository
labels it “Kimmel et al. (1995),” but this audit verified the executed coefficients, not that
external scientific attribution (`src/data_pipeline/feature_extraction/stage_inference.py:1-5`).

The output is the snip feature spine plus nullable `predicted_stage_hpf`, non-null `model_version`,
and status. Validation enforces the columns, feature-table/source contract, allowed status literals,
non-null version, and status/value consistency. It deliberately performs no biological range check
(`src/data_pipeline/feature_extraction/stage_predictions/contract.py:18-65`).

`stage_inference.py` also contains two non-live surfaces: `infer_stage_from_area` returns two NaNs,
and a batch helper uses different defaults, raises on missing values, accepts only `time_s` as a time
fallback, and emits a constant `stage_confidence=1.0`. Repository search found no caller outside the
definitions (`src/data_pipeline/feature_extraction/stage_inference.py:27-70`;
`rg -n 'infer_stage_from_area|compute_stage_predictions_batch' --glob '*.py'`, 2026-08-27).

### Orchestration and paths

The rule builds per-well shards from the default-BF compatibility snip inventory, frame inventory,
plate metadata, physical-embryo registry, collection JSON, and acquisition inventory; validates each
shard; then concatenates in deterministic `experiment_id`, `well_id`, `snip_id` order
(`src/data_pipeline/pipeline_orchestrator/rules/stage_predictions.smk:16-110`). The task CLI makes
collection provenance required and acquisition inventory optional for singles
(`src/data_pipeline/pipeline_orchestrator/tasks.py:724-751,1810-1823`).

The declared merged stage path is
`{root}/feature_extraction/{experiment_id}/stage_predictions/{experiment_id}_stage_predictions.csv`;
per-well files are under `per_well/{well_id}` and named `{well_id}_stage_predictions.csv`
(`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:646-658,1103-1114,1170-1214`).

## 4. Stored-artifact integrity and limits

Measured presence over the 148 explicit experiments was: 109 stage files, 133 merged snip
inventories, 144 plate tables, 3 merged frame inventories, 9 collection-provenance JSONs, 133 mask
geometry files, 3 surface-area-QC files, and 120 optional analysis-ready Parquets. These are presence
counts, not claims that each artifact was generated by the current commit
([source_inventory.csv](stage_estimation/source_inventory.csv) and
[audit_summary.json](stage_estimation/audit_summary.json), measured 2026-08-27).

| Source schema/status | Experiments | Rows | Finite stage | Interpretation |
|---|---:|---:|---:|---|
| Status-bearing S01, `predicted` | 107 status-bearing files include these rows | 511,118 | 511,118 | Formula result. |
| Status-bearing S01, `missing_start_age_hpf` | 9 partially/fully unresolved experiments | 898 | 0 | Missing declared age, not failed morphology. |
| Status-bearing S01, `missing_temperature` | 0 observed rows | 0 | 0 | Allowed contract state, absent in this corpus. |
| No-status S02, normalized here to `unavailable` | 2 | 20,219 | 20,219 | Status absence; not a failed row. |

All counts come from [stage_by_experiment.csv](stage_estimation/stage_by_experiment.csv) and
[stage_status_counts.csv](stage_estimation/stage_status_counts.csv), measured 2026-08-27. The two
no-status files are `20260417_irx_pilot` and `20260418_irx_pilot`; this reproduces the dated lineage
result at `docs/refactors/core-model/reports/STUDY_stage_lineage.md:79-92`.

Every stage file had unique `snip_id`, one method version, and a contract-recognized status after
schema normalization; the one method literal covered all 532,235 rows. Observed finite values ranged
from 7.0 to 135.1094 hpf. Across 521,471 finite consecutive within-embryo steps sorted by
`time_index`, zero were negative and zero exactly flat; the largest positive step was 79.4121 hpf in
`20251121`. Monotonicity establishes formula/order consistency, not developmental accuracy, and the
large steps should be inspected as acquisition/start-age boundaries rather than silently called
biological continuity ([stage_by_experiment.csv](stage_estimation/stage_by_experiment.csv), measured
2026-08-27; no range check at `stage_predictions/contract.py:1-6,32-65`).

Seventy-two stage files are snapshots by the observed maximum of one row per physical embryo (6,127
rows, 5,229 finite); 37 are time series (526,108 rows, all finite). “Snapshot” here describes stored
row multiplicity, not a producer branch—the formula is identical
([stage_by_experiment.csv](stage_estimation/stage_by_experiment.csv), measured 2026-08-27).

Only three explicit experiments have the stored stage, merged frame inventory, and plate table
needed to reconstruct the formula. All 20,316 comparable rows reproduce the stored value to floating
precision, with complete elapsed time, start age, and temperature. None has the collection JSON now
required by the entrypoint, so these are historical formula reconstructions, not proof that the
current producer can regenerate them
([formula_reconstruction.csv](stage_estimation/formula_reconstruction.csv), measured 2026-08-27;
`src/data_pipeline/feature_extraction/stage_predictions/entrypoint.py:13-20,31-50`).

Nine explicit collection JSONs were present; all declared `is_collection=false`, each listed one
source, and none overlapped the three formula-reconstruction experiments. Collection-mode
start-age resolution therefore has code evidence but no stored end-to-end example here
([collection_provenance.csv](stage_estimation/collection_provenance.csv) and
[formula_reconstruction.csv](stage_estimation/formula_reconstruction.csv), measured 2026-08-27).

Agreement with the formula's own inputs is an integrity check only. The other 106 stage experiments
lack a merged frame inventory at the declared path, so their input completeness cannot be
reconstructed without substituting a different source
([source_inventory.csv](stage_estimation/source_inventory.csv), measured 2026-08-27).

## 5. Downstream consumers

The complete static token-hit inventory contains every repository `.py`, `.smk`, YAML, shell, and
notebook file that mentions the audited stage fields, with category and treatment labels. It is a
search inventory, not proof that every hit is reachable; runtime conclusions below come from source
inspection ([consumer_inventory.csv](stage_estimation/consumer_inventory.csv), generated with `rg -l`
by `audit_stage_estimation.py`, 2026-08-27).

### Pipeline consumers

| Consumer | Stage treatment | Missing/unavailable behavior | Evidence |
|---|---|---|---|
| Surface-area QC | QC conditioning axis. Interpolates a packaged p5/p95 area reference at `predicted_stage_hpf`; flags outside `0.9*p5` and `1.4*p95`. | Missing stage row is a hard error. A present null stage emits `sa_outlier_flag=False`, applicability `not_applicable`. Single-z may be `diagnostic_only`; legacy/native is exclusion. | `src/data_pipeline/quality_control/surface_area_qc/config.py:20-32`; `compute.py:36-48,51-113,116-175`. |
| Analysis-ready assembler | Metadata column in an optional one-to-one left join. | Missing stage source row becomes null columns; the authoritative snip-QC base row remains. | `src/data_pipeline/analysis_ready/assemble.py:1-12,31-50,53-86`; optional rule at `pipeline_orchestrator/rules/analysis_ready.smk:1-8,20-45`. |
| Analysis-ready report | Continuous PCA color, survival ordering/grid, 2-hpf binning, and plot axis. | Stage-binned well summaries drop null stage; survival grids use non-null unique stage. | `src/data_pipeline/analysis_ready/report.py:51-85,104-133,156-177,209-267`. |
| Death-event product | Recalculates nominal stage at the called-death frame from plate start age, frame elapsed time, and temperature. | Missing age/temperature returns NaN; structural join ambiguity fails. It does not consume collection provenance. | `src/data_pipeline/quality_control/death_detection/death_event.py:1-9,55-100,104-145`. |
| Planned manifest adapter | Carries nullable stage value, non-null three-state-capable status, model version, start-age source, elapsed time, and incubation temperature. | Basic Track A does not require stage; a non-basic cohort configures stage requirement and accepted statuses. | `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:77-125,215-245,269-298`. |

The death-event calculation is semantically aligned for single experiments but lacks the live
producer's collection/source-ordinal branch. Whether to replace it with shared resolved
stage-at-frame machinery is undecided; no stored collection example exists to quantify the error
(`death_event.py:104-145` versus `stage_predictions/compute.py:77-128`;
[collection_provenance.csv](stage_estimation/collection_provenance.csv)).

Beyond these direct pipeline consumers, 44 `src/analyze`, `src/morphseq`, or app files use a stage
field as a filter, grouping/order variable, regression covariate, classifier target, or plot axis;
121 token-bearing notebooks are exploratory and not launch-path evidence. File-by-file paths and
static treatment labels are in [consumer_inventory.csv](stage_estimation/consumer_inventory.csv),
measured 2026-08-27.

### Integrated core at the audited commit

At `d830391f`, both Basic and NT-Xent data configs call legacy `split_train_test`, which starts with
`make_seq_key`; NT-Xent then constructs `age_hpf_vec` from `seq_key["stage_hpf"]`
(`src/core/data/dataset_configs.py:46-68,111-150,211-227`). The metric Hydra preset selects
`NTXentDataset` (`src/core/hydra_configs/model/metric_vae_timm.yaml:1,40-45`). Model initialization
calls `make_metadata`, and checkpoint loading constructs a `BaseDataConfig` and calls it again
(`src/core/run/run_utils.py:330-346,459-493`). Thus legacy stage selection is reachable in training
and checkpoint-evaluation paths at the audited commit.

`make_seq_key` selects `inferred_stage_hpf` whenever that column exists, silently renames it to
`stage_hpf`, and only when absent prints a warning before selecting `predicted_stage_hpf`. It then
globs image folders and reconstructs embryo identity by slicing `snip_id`
(`src/core/data/dataset_utils.py:26-38,40-75`). The age-key merge later adds
`inferred_stage_hpf_reg`, but `make_metadata` still selects the already-existing `stage_hpf`; the MLP
field is not selected for `age_hpf_vec`
(`src/core/data/dataset_configs.py:50-58,219-227`).

This narrows a statement in the prior lineage report. It was correct that current core reads and
merges `age_key.csv` (`STUDY_stage_lineage.md:126-129`), but “consumed this output” is too broad for
the metric age axis: the MLP column is merged and then unused by the sampler. The selected default is
the separate area-derived `inferred_stage_hpf`.

The integrated NT-Xent sampler admits same- or relation-positive candidates at
`abs(age_i-age_j) <= time_window`; the loss labels pairs positive at
`<= time_window + 1.5`. Default `time_window` is 1.5 and model config copies it into data config
(`d830391f:src/core/data/dataset_classes.py:99-135`;
`src/core/losses/loss_functions.py:347-352`; `src/core/losses/loss_configs.py:132-139`;
`src/core/models/model_configs.py:121-128`). The sampler self set includes the anchor, so a finite
anchor always has at least one self candidate; a NaN anchor has none and can reach
`np.random.choice` with an empty option set. The latter is an inference from the cited Boolean logic,
not a measured training crash.

Track A is not integrated at the audited base commit; the ground-truth report establishes that the
earlier claimed Phase 1 commits did not exist
(`docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md:15-29`). In-flight uncommitted Track A
files in the shared worktree belong to other agents and were not treated as accepted behavior (`git
status --short --branch`, 2026-08-27). Under the binding planned contract, basic Track A is stage
optional and consumes deterministic manifest rows, so `dataset_utils.make_seq_key`, its warning
fallback, image globs, age key, and ID slicing must be unreachable—not repaired for the new preset
(`MANIFEST_SCHEMA.md:195-227,269-298`).

## 6. Legacy estimators and provenance

### Morphology-latent MLP (`inferred_stage_hpf_reg`)

The producer reads a VAE run's `figures/embryo_stats_df.csv`, merges temperature, explicit
`embryo_id`, and `Time Rel (s)` from training metadata by `snip_id`, and selects every column whose
name contains `z_mu` as predictors (`src/build/infer_developmental_age.py:14-29,46-49,78-80`). It
fits `MLPRegressor(random_state=1, max_iter=5000)` to **nominal** `predicted_stage_hpf`, not an
independent anatomical target (`infer_developmental_age.py:74-85`).

Reference eligibility is wild type or control. An explicit `reference_datasets` list restricts by
experiment; otherwise code hard-codes `20240411`, `20240626`, early `20230620`, and WT `20231218`
(`infer_developmental_age.py:51-70`). Source comments claim manual agreement for some experiments,
but no row-level anatomical labels or persisted reviewer evidence are read
(`infer_developmental_age.py:59-66`).

After predicting reference/control rows, time-series experiments fit a linear mapping from nominal
stage to MLP stage using same-experiment rows within 2.5 hpf and 2.5 clock hours when at least five
exist; otherwise they use nearby same-temperature references. The fit includes an intercept.
Snapshot experiments bypass calibration and copy nominal stage directly into
`inferred_stage_hpf_reg` (`infer_developmental_age.py:88-156`).

The function writes `age_key_df.csv` with `calc_stage_hpf`, `inferred_stage_hpf_reg`, absolute clock
time, perturbation, and training/model/architecture names (`infer_developmental_age.py:31-40`). The
stored consumer artifact is instead named `age_key.csv`. It has 83,476 unique rows and one tuple:
`train_dir=20241008`, `model_name=VAE_training_2024-10-08_21-40-44`,
`architecture_name=VAE_z100_ne250_base_model`; all rows match current training metadata by exact
`snip_id` ([legacy_age_key_provenance.csv](stage_estimation/legacy_age_key_provenance.csv) and
[source_fingerprints.csv](stage_estimation/source_fingerprints.csv), measured 2026-08-27).

The artifact has 24 time-series experiment groups with 82,808 rows and 16 snapshot groups with 668
rows. Every snapshot experiment's median MLP-minus-nominal delta is zero, as code requires;
time-series experiment median deltas range from -10.0179 to +5.5641 hpf
([legacy_axis_by_experiment.csv](stage_estimation/legacy_axis_by_experiment.csv), measured
2026-08-27). These are differences between model-derived axes, not errors against anatomy.

Persisted provenance does **not** identify the exact reference rows/list, latent column ordering,
VAE checkpoint hash, input metadata hash, scikit-learn version, serialized MLP weights, local
regression coefficients, or the rename/copy from `age_key_df.csv` to `age_key.csv`. Repository
search found no current caller except the hard-coded `__main__`; current core reads only the already
produced CSV (`infer_developmental_age.py:164-171`;
`rg -n 'infer_developmental_age|get_embryo_age_predictions' --glob '*.py'`, 2026-08-27).

Reliability risks supported by the implementation are:

1. **Target circularity:** the MLP is trained on nominal stage, so nominal agreement does not validate
   anatomy (`infer_developmental_age.py:74-85`).
2. **Reference leakage:** same-experiment nominal/time-near rows can calibrate the evaluated
   experiment, with no held-out experiment protocol (`infer_developmental_age.py:107-153`).
3. **Snapshot non-estimation:** 668 snapshot rows copy nominal stage
   (`infer_developmental_age.py:123-156`; `legacy_axis_by_experiment.csv`).
4. **Domain shift:** calibration branches by experiment or exact temperature, but no held-out
   temperature/acquisition evaluation persists (`infer_developmental_age.py:107-153`;
   `legacy_age_key_provenance.csv`).
5. **Reproducibility gap:** VAE run names persist, but MLP/calibration state and training-row identity
   do not (`infer_developmental_age.py:31-40`).

These risks support retiring it as a default. They do not establish that its values are always
wrong; that accuracy question lacks independent anchors.

### Legacy area stage (`inferred_stage_hpf`)

The default core field comes from a different legacy path. `infer_embryo_stage` loads
`metadata/stage_ref_df.csv`, maps surface area to stage, identifies experiment reference rows by
WT/control status, and either directly interpolates snapshot areas or fits a bounded polynomial from
nominal stage to area-interpolated stage for time series
(`src/build/build04_perform_embryo_qc.py:657-674,681-737,739-803`). The combined Build04 path invokes
it before writing the next metadata table (`build04_perform_embryo_qc.py:1497-1502`).

In current training metadata, `inferred_stage_hpf` is non-null on 129,413 of 130,315 rows, but its
finite extrema are approximately `-7.706e19` and `4.227e18` hpf; nominal stage is non-null on 130,299
rows and ranges 9.0–79.8018 hpf. This is a numerical-integrity failure for a field consumed as hpf
even without an anatomical anchor
([legacy_axis_summary.csv](stage_estimation/legacy_axis_summary.csv), measured 2026-08-27). It should
neither be treated as the latent MLP nor carried into Track A.

### Manual and SeaHub sources

Legacy Build05 can merge `manual_stage_hpf` from an exact-`snip_id` curation CSV, but current
training metadata has zero non-null manual values; the producer initializes the curation column to
null (`src/build/build05_make_training_snips.py:33-53`;
`src/build/build04_perform_embryo_qc.py:1517-1539`;
[legacy_axis_summary.csv](stage_estimation/legacy_axis_summary.csv), measured 2026-08-27).

SeaHub preserves source morphology labels and converts recognized cell/epiboly/somite/prim labels to
hpf with a declared crosswalk (`src/data_pipeline/acquisition/seahub/stages.py:10-43,62-112`;
`src/data_pipeline/acquisition/seahub/reconciliation.py:223-270,307-332`). Integration also writes
that converted `stage_hpf` as `start_age_hpf`, making it a nominal-formula input, not an independent
downstream target (`src/data_pipeline/acquisition/seahub/integration.py:1253-1266`). Such labels
could become anchors only if their observation process and exact identities are shown independent
and retained for comparison; this audit did not find that crosswalk.

## 7. Default and fallback inventory

| Path | Trigger | Selected value | Signal | Reachable now / planned Track A | Disposition |
|---|---|---|---|---|---|
| `src/core/data/dataset_utils.py:26-38` preferred branch | `inferred_stage_hpf` exists. | Legacy area field renamed `stage_hpf`. | Silent. | Reachable from integrated Basic, metric, and checkpoint paths; must be unreachable from Track A. | Retire from new path. |
| `src/core/data/dataset_utils.py:34-38` fallback | Preferred column absent. | Nominal `predicted_stage_hpf` renamed `stage_hpf`. | Printed warning. | Reachable now; must be unreachable from Track A. | Retire implicit fallback. |
| `src/core/data/dataset_configs.py:53-58` | `age_key.csv` exists/missing. | Merges MLP field; missing file raises misleading `Stage key provided!`. | Silent merge / hard error. | Read now but not selected for `age_hpf_vec`; no Track A role. | Remove dependency from new path. |
| `stage_predictions/compute.py:107-112` | Canonical collection map absent. | Legacy `start_age_by_time_index`. | Silent alias; missing coverage later hard-fails. | Live producer. | Preserve temporarily with provenance. |
| `stage_inference.py:35-70` batch helper | Default columns absent. | `time_s`/temperature aliases; confidence 1.0. | Hard errors. | No caller found; not Track A. | Keep unreachable or retire. |
| `build03A_process_images.py:1627-1674` | Legacy calculation invoked. | Nominal formula if columns/types work. | Missing/type errors silent unless verbose. | Legacy build only. | Retire from new path. |
| `build04_perform_embryo_qc.py:1279-1303` | Legacy columns absent. | Stage, area, time set to `0.0`; temperature null. | Silent literals. | Legacy build only. | Retire unsafe defaults. |
| `src/legacy/vae/models/vae/vae_config.py:39-50` and duplicate `src/vae/...` | Empty age-key path. | Constant MLP stage `1`. | Silent. | Legacy packages only. | Retire except exact checkpoint emulation. |
| Surface-area QC null stage | Stage row exists, value null. | False flag plus `not_applicable`. | Explicit applicability. | Live QC; Track A carries result. | Preserve three-state semantics. |
| Analysis-ready/report | Stage source absent/null. | Left-joined null; stage plots/bins omit nulls. | No assembler row drop; plot omission. | Optional live product. | Preserve row and report denominator. |

The 121 token-bearing notebooks include experimental stage analyses and defaults, but no notebook is
a supported core or pipeline launch path. Paths remain in
[consumer_inventory.csv](stage_estimation/consumer_inventory.csv); no notebook result was treated as
production validation.

## 8. Identifiability and reliability verdict

| Question | Verdict | Identifiable evidence |
|---|---|---|
| Does stored stage implement the formula? | **Yes, for 20,316 historical rows in three experiments.** | Same-input recomputation; max absolute residual `7.11e-15` hpf ([formula_reconstruction.csv](stage_estimation/formula_reconstruction.csv)). |
| Is nominal stage accurate against independently observed anatomy? | **Not identifiable.** | Zero manual anchors; SeaHub label is reused as start age; no independent comparison table (`legacy_axis_summary.csv`; SeaHub citations above). |
| Is nominal stage unbiased by experiment, temperature, acquisition mode, or range? | **Not identifiable.** | The inputs span those variables, but no independent outcome exists. Agreement with inputs is circular. |
| Does track monotonicity establish accuracy? | **No.** | Zero negative formula steps establish order/integrity only (`stage_by_experiment.csv`). |
| Does the MLP agree with live stage on exact animals? | **Not identifiable (`n=0`).** | Exact `snip_id` and exact current-physical-versus-legacy-embryo intersections are zero (`exact_crosswalk.csv`). |
| Is the MLP accurate against anatomy? | **Not identifiable.** | Its target is nominal clock and no manual labels are present. |
| Is the current core default numerically fit as hpf? | **No.** | `inferred_stage_hpf` has finite magnitudes around `1e19` hpf (`legacy_axis_summary.csv`). This is numeric failure, not an anatomical error estimate. |
| Is `unavailable` status failure? | **No.** | It is absence of the status column in two finite-stage S02 files (`stage_status_counts.csv`). |

The exact crosswalk remains `n=0`, reproducing the 2026-08-24 result
(`STUDY_stage_lineage.md:20-27,131-148`). No ID was parsed, sliced, or reconstructed by this audit.

## 9. Downstream sensitivity

### Surface-area QC

The audit recomputed the production formula on the only three experiments with stage, geometry, and
stored surface-QC artifacts. At zero shift, all 20,316 stored flags matched. This validates the audit
calculation against historical flags, not biological validity
(`src/data_pipeline/quality_control/surface_area_qc/compute.py:36-48`;
[surface_area_stage_sensitivity.csv](stage_estimation/surface_area_stage_sensitivity.csv), measured
2026-08-27).

| Additive stage perturbation | Flagged rows | Flagged share | Flag changes vs baseline | Flip share |
|---:|---:|---:|---:|---:|
| -3.0 hpf | 8,922 | 43.92% | 1,783 | 8.78% |
| -1.5 hpf | 9,543 | 46.97% | 1,118 | 5.50% |
| -0.5 hpf | 10,151 | 49.97% | 446 | 2.20% |
| 0.0 hpf | 10,477 | 51.57% | 0 | 0% |
| +0.5 hpf | 10,791 | 53.12% | 448 | 2.21% |
| +1.5 hpf | 11,333 | 55.78% | 1,096 | 5.39% |
| +3.0 hpf | 12,145 | 59.78% | 1,986 | 9.78% |

Values are from [surface_area_stage_sensitivity.csv](stage_estimation/surface_area_stage_sensitivity.csv),
measured 2026-08-27. At ±1.5 hpf, experiment effects differed: the 97-row
`20250612_24hpf_ctrl_atf6` snapshot changed 0 flags at -1.5 and 8 at +1.5; the two IRX time series
changed 161/318 and 957/770
([surface_area_stage_sensitivity_by_experiment.csv](stage_estimation/surface_area_stage_sensitivity_by_experiment.csv)).
The interpolator clamps out-of-range stages to endpoints, so an extreme finite stage does not itself
fail QC (`src/data_pipeline/quality_control/surface_area_qc/reference.py:40-52`).

No alternate morphology axis exists on the same live rows because the exact old/new crosswalk is
empty. Additive perturbations are local sensitivity, not an empirically calibrated error model.

### Metric sampler and loss windows

The audit applied the existing metric relation matrix without changing its content. Counts are
pooled over the exact 83,476-row age-key/metadata join and are **pre-split upper bounds**; they do not
replace Track C's split-local legal-positive preflight
([metric_window_sensitivity.csv](stage_estimation/metric_window_sensitivity.csv), measured
2026-08-27).

| Axis on legacy rows | Gate | Finite anchors | Median same-embryo candidates, anchor included | Relation-positive different-embryo median | Zero relation-positive anchors |
|---|---:|---:|---:|---:|---:|
| Current core default `inferred_stage_hpf` | 1.5 | 82,808 | 508 | 502 | 138 |
| Current core default `inferred_stage_hpf` | 3.0 | 82,808 | 1,022 | 1,010 | 137 |
| Nominal clock `predicted_stage_hpf` | 1.5 | 83,476 | 549 | 543 | 87 |
| Nominal clock `predicted_stage_hpf` | 3.0 | 83,476 | 1,096 | 1,085 | 86 |
| Legacy MLP `inferred_stage_hpf_reg` | 1.5 | 83,476 | 542 | 533 | 86 |
| Legacy MLP `inferred_stage_hpf_reg` | 3.0 | 83,476 | 1,121 | 1,106 | 86 |

The 1.5 rows emulate the sampler; 3.0 emulates the default loss gate. Doubling the gate roughly
doubles median pools but does not resolve all uncovered anchors. Pool size is not biological
validation (`dataset_classes.py:121-135`; `loss_functions.py:347-352`;
`metric_window_sensitivity.csv`).

On 128 deterministic time-series anchors finite on both axes, median candidate-set Jaccard overlap
between the current default and nominal clock was 0.530 at 1.5 and 0.686 at 3.0; overlap with the MLP
was 0.500 and 0.659. Temperature-stratified medians vary, but describe membership sensitivity, not
accuracy ([metric_membership_strata.csv](stage_estimation/metric_membership_strata.csv), measured
2026-08-27).

All 668 legacy snapshot rows have nominal and MLP values but lack the current core default
`inferred_stage_hpf`; comparison against that default is unavailable for snapshots, and the default
axis excludes those anchors before split policy
([metric_axis_coverage_strata.csv](stage_estimation/metric_axis_coverage_strata.csv), measured
2026-08-27). This differs from time series and is a coverage defect, not evidence favoring an axis.

### Cohorts and reports

Basic Track A does not filter on stage. Later cohorts make stage requirements and accepted statuses
configuration; `unavailable` remains distinct from a named missing-input state
(`MANIFEST_SCHEMA.md:215-245`). Surface QC preserves null-stage rows as `not_applicable`, while
analysis-ready preserves rows but stage plots/bins omit nulls (`surface_area_qc/compute.py:87-113`;
`analysis_ready/assemble.py:31-50,74-86`; `analysis_ready/report.py:64-85,104-133`). Thus stage policy
affects QC applicability, metric eligibility, and plot denominators through different mechanisms.

## 10. Bounded recommendations

### Track A

**Recommendation:** carry declared timing/temperature provenance, nullable nominal stage, explicit
status, and method version through the observation table, but ignore stage in vanilla model inputs.
Make all legacy `dataset_utils`, ImageFolder/glob, `age_key.csv`, and inferred-column fallbacks
unreachable from manifest presets. This follows the stage-optional basic contract and does not
require deciding nominal accuracy (`MANIFEST_SCHEMA.md:77-125,195-227,269-298`).

### Stage-conditioned QC

Nominal clock stage can support the **existing deterministic area lookup mechanically**: all 20,316
comparable stored flags reproduced. It cannot support an accuracy claim, and ±1.5-hpf shifts changed
5.4–5.5% of flags. Preserve stage status and QC applicability, label the axis nominal, and leave
threshold/role changes to Nick after Track D/E review. This is a recommendation based on
`surface_area_stage_sensitivity.csv`, not a production threshold change.

### Track C metric use

Nominal stage may be used for plumbing or an explicitly provisional clock-matched analysis, but a
science run must not inherit “1.5 hpf” as validated morphology equivalence. Track C should persist
sampler and loss windows separately and name the source/status policy. Until validation Nick has two
bounded options:

1. keep biological metric sampling disabled and use the trivial test policy for mechanism checks; or
2. authorize a clearly labeled nominal-clock analysis with prespecified sensitivity windows and no
   claim of anatomical matching.

The option, final windows, and admissible `unavailable` status are O5 and require Nick
(`docs/refactors/core-model/PLAN.md:292-303`).

### Legacy morphology estimator

Preserve frozen `age_key.csv`, its checksum, and source as historical lineage. Retire MLP and area
columns as defaults and do not spend Track A effort operationalizing them. Reconsider a learned
estimator only as a new anchor-trained study with group-disjoint evaluation. Whether to retain the
artifact indefinitely or move it to an explicit historical archive requires Nick/lead policy.

### Smallest credible future validation or replacement study

The smallest credible study must create independent exact-ID anchors; experiment averages are
insufficient. Two bounded paths exist:

1. **Existing-label validation, if feasible.** Verify whether SeaHub anatomical labels were recorded
   independently before the formula and map to exact `snip_id` or explicit `physical_embryo_id`
   without parsing. Cover multiple experiments, temperatures, stage ranges, and snapshot/time-series
   acquisition. Proceed only if label provenance and exact identities are complete.
2. **Prospective blinded anchor panel.** Record fertilization/start time separately; have blinded
   scorers stage exact identified observations under a preregistered rubric; cover temperature,
   early/mid/late stage, snapshot/time-series, and acquisition mode; hold out whole experiments.
   Determine final sample size by pilot/power analysis rather than an arbitrary row count.

First evaluate nominal clock bias and inter-rater uncertainty by experiment, temperature, range, and
mode. Only if Nick selects O8's estimator option should a morphology candidate be compared on the
same held-out anchors. Replacement requires lower held-out error and stable metric/QC decisions, not
monotonicity or correlation with nominal clock. O8 belongs to Nick (`PLAN.md:300-303`).

## 11. Component decision table

| Component | Current role | Evidence | Preserve / retire / replace / undecided | Immediate action | Blocks |
|---|---|---|---|---|---|
| Live nominal formula and `kimmel1995_temp_rate_v1` | Temperature-adjusted clock metadata per snip. | `stage_inference.py:15-24`; 20,316-row reconstruction. | **Preserve** as nominal metadata. | Carry value/status/version; do not relabel morphology. | No Track A block; O8 for interpretation. |
| Collection source-ordinal age | Source-specific declared age for collections. | `stage_predictions/compute.py:77-128`; no stored collection example. | **Preserve code; validation undecided.** | Add regenerated collection fixture in owning track. | Collection scientific use. |
| Live status contract | Predicted/missing input; adapter adds unavailable. | `contract.py:18-65`; status table. | **Preserve.** | Keep three-state-capable policy/provenance. | Cohort O5. |
| Range/discontinuity validation | No live bound; stored max 135.109 and step 79.412. | `stage_by_experiment.csv`; `contract.py:32-65`. | **Undecided.** | Investigate acquisition boundaries before a bound. | Nick/producer owner. |
| Surface-QC stage axis | Conditions area band on nominal stage. | `surface_area_qc/compute.py:36-113`; perturbation table. | **Undecided scientific role; preserve mechanics.** | Keep applicability; Nick reviews D/E. | O2/O8. |
| Death-event stage | Repeats single formula at event time. | `death_event.py:104-145`; no collection branch. | **Replace/share, undecided.** | Design shared resolution after collection evidence. | Collection reporting. |
| Analysis-ready stage | Optional metadata/report axis. | `analysis_ready/assemble.py:31-86`; report source. | **Preserve.** | Report missing/status denominators. | None for Track A. |
| Planned manifest fields | Carry timing/value/status/version; basic optional. | `MANIFEST_SCHEMA.md:77-125,215-245`. | **Preserve.** | Implement only in owning Track A slices. | Track A acceptance. |
| Core `dataset_utils` selection | Area default, nominal warning fallback. | `dataset_utils.py:26-38`; numeric extremes. | **Retire from manifest path.** | Make unreachable; do not repair. | Track A safety. |
| Core age-key MLP merge | Reads MLP but current age vector selects prior field. | `dataset_configs.py:50-58,219-227`. | **Retire dependency.** | Remove from new path; historical emulation only. | None for vanilla. |
| Area `inferred_stage_hpf` | Actual current core default. | Build04 `:657-803`; extrema near `1e19`. | **Retire.** | Preserve evidence; prohibit as preset axis. | Legacy-run trust. |
| Latent MLP code | VAE latents fit to nominal stage and calibrated. | `infer_developmental_age.py:46-156`; provenance gaps. | **Retire default; preserve source historically.** | Do not operationalize in Track A. | O8 if rehabilitation requested. |
| Frozen `age_key.csv` | Historical MLP output, 83,476 rows. | Provenance and fingerprint tables. | **Preserve as lineage data.** | Record checksum and historical status. | Retention policy. |
| Constant/zero fallbacks | Old pipelines continue with stage 1 or 0. | Legacy VAE `:39-46`; Build04 `:1279-1300`. | **Retire.** | Keep unreachable from supported presets. | None. |
| Sampler 1.5 gate | Filters positive candidates. | `d830391f:dataset_classes.py:121-135`; sensitivity. | **Replace with explicit Track C policy.** | Disabled/provisional until O5. | O5, C1/C2. |
| Loss `time_window + 1.5` | Effective 3.0 default positive target. | `loss_functions.py:347-352`; sensitivity. | **Replace implicit coupling.** | Configure/validate separately after A3. | O5, C3. |
| Independent anatomical anchors | None usable. | Zero manual and zero exact crosswalk. | **Replace absence with study.** | Existing-label audit or blinded panel. | O8 and accuracy claims. |
| SeaHub labels | Potential observed labels, also formula start age. | `seahub/stages.py:10-112`; `integration.py:1253-1266`. | **Undecided as anchors.** | Verify independence/exact identity. | Smallest study/O8. |
