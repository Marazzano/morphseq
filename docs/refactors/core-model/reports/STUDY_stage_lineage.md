# Study: developmental-stage lineage and metric-window compatibility

_Read-only study measured 2026-08-24. No `src/` file was modified. The requested branch
`study/stage-lineage` did not exist and the supplied worktree was already dirty on
`core-model-refactor`, so the study was performed without switching branches; HEAD was
`3ecc4818994de5c2ab9db1cafa85f1af60c64044` (`git branch --show-current`; `git rev-parse HEAD`,
2026-08-24)._

## Conclusion

**`predicted_stage_hpf` and legacy `inferred_stage_hpf_reg` are not interchangeable as a drop-in
age axis.** The new field is a deterministic nominal-age calculation from start age, elapsed clock
time, and temperature; it reads no image and fits no learned model
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:1-16,32-75`;
`src/data_pipeline/feature_extraction/stage_inference.py:15-24`). The legacy field is produced from
VAE morphology latents by an `MLPRegressor`, followed by experiment- or temperature-conditioned
linear calibration (`src/build/infer_developmental_age.py:46-100,107-156`). They therefore name
different estimands even though both are expressed in hpf.

Empirical interchangeability cannot be estimated from the artifacts currently on disk. The legacy
key has 83,476 unique `snip_id` rows; the new stage artifacts have 532,235 unique `snip_id` rows;
their exact `snip_id` intersection is **zero**. Exact `embryo_id` intersection is also zero. Eighteen
experiment IDs occur in both sources (24,215 legacy rows and 44,185 new rows), but there is no
explicit row/animal crosswalk. Consequently Pearson correlation, mean bias, and Bland-Altman limits
of agreement are all **not estimable (`n=0`)**, not zero (measured 2026-08-24 from the sources and
protocol under “Measurement provenance”). IDs were not parsed or reconstructed to manufacture a
proxy match, per `docs/refactors/core-model/AGENTS.md:14-16`.

**Implication for metric learning:** the current core default `time_window=1.5` was introduced into
a stack that loaded `inferred_stage_hpf_reg`, more than a year before the new stage artifact existed
(`git blame -L 138,138 src/core/losses/loss_configs.py`; `git log --follow -- src/data_pipeline/feature_extraction/stage_predictions/compute.py`). Carrying `1.5` onto
`predicted_stage_hpf` would therefore be an unvalidated semantic change. Metric training should not
claim legacy compatibility until an explicit crosswalk permits a paired delta-agreement study and
the window is either validated or retuned. This recommendation is an inference from the code
lineage and the measured absence of paired rows.

## 1. New pipeline lineage: `predicted_stage_hpf`

### Producer and meaning

The producer is not an image-stage regressor. For each snip it looks up `well_id` and `image_id`,
then reads:

- `start_age_hpf` from the well's `plate_metadata` row
  (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:40-58`);
- `temperature` from the same row
  (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:61-69`); and
- the first non-null frame-timing field among `elapsed_time_s`, `experiment_time_s`, and `time_s`
  (`src/data_pipeline/feature_extraction/stage_predictions/compute.py:18-29`).

It then evaluates the following deterministic formula
(`src/data_pipeline/feature_extraction/stage_inference.py:15-24`):

```text
elapsed_hours = elapsed_time_s / 3600
developmental_rate = 0.055 * temperature_c - 0.57
predicted_stage_hpf = start_age_hpf + elapsed_hours * developmental_rate
```

The code calls this “Kimmel et al. (1995),” but this study did not independently validate that
scientific attribution or the coefficients; it verified only what the repository executes
(`src/data_pipeline/feature_extraction/stage_inference.py:1-5,20-24`). There are no learned weights,
image inputs, or training data for this producer. `model_version` is therefore method/formula
provenance, not a trained-model identifier
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:16,32-38,73-75`;
`src/data_pipeline/feature_extraction/stage_predictions/contract.py:3-6`).

Across all 109 present stage artifacts in the explicit 148-experiment reconnaissance list, the only
value is `kimmel1995_temp_rate_v1` on all **532,235** rows (measured 2026-08-24; protocol below).

### Missingness semantics

Current code emits `missing_start_age_hpf` or `missing_temperature` and a null stage when the
corresponding plate value is absent; otherwise it emits `predicted`
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:56-75`). The validator requires
exactly that relationship and allows those three literals
(`src/data_pipeline/feature_extraction/stage_predictions/contract.py:23-25,49-65`).

The observed source-row distribution is below. Every count and percentage was measured 2026-08-24
over 532,235 present stage rows using the explicit experiment list described below.

| normalized source status | rows | share of present stage rows | finite stage? |
|---|---:|---:|---|
| `predicted` | 511,118 | 96.032% | yes |
| `unavailable` (S02 has no status column) | 20,219 | 3.799% | yes |
| `missing_start_age_hpf` | 898 | 0.169% | no |
| `missing_temperature` | 0 | 0.000% | no |

`unavailable` is absence of source-schema status, not a failure. It consists exactly of
`20260417_irx_pilot` (4,968 rows) and `20260418_irx_pilot` (15,251 rows); both files contain finite
stages and `model_version` but omit `stage_prediction_status`
(`docs/refactors/core-model/reports/PIPELINE_RECON.md:181-182,789-795`; measured 2026-08-24).

## 2. Legacy lineage: `inferred_stage_hpf_reg`

The legacy code still exists at `src/build/infer_developmental_age.py`; a last-containing commit is
therefore not applicable. The path is:

1. Read `embryo_stats_df.csv` from a trained VAE output and merge per-snip temperature,
   `embryo_id`, and relative clock time (`src/build/infer_developmental_age.py:14-29`).
2. Select all columns whose names contain `z_mu`; these VAE morphology latents are the predictors
   (`src/build/infer_developmental_age.py:46-49,78-80`).
3. Fit `sklearn.neural_network.MLPRegressor(random_state=1, max_iter=5000)` to nominal
   `predicted_stage_hpf` on reference/control rows (`src/build/infer_developmental_age.py:51-85`).
4. Predict morphology-derived stage on reference/control rows, then for each experiment fit a local
   linear mapping from nominal calculated stage to the MLP-predicted stage. The preferred references
   are same-experiment rows close in both nominal stage and clock time; the fallback is same-
   temperature reference rows (`src/build/infer_developmental_age.py:88-153`). The linear fit has an
   intercept specifically to accommodate start-frame stage-assignment errors
   (`src/build/infer_developmental_age.py:150-153`).
5. Snapshot experiments bypass the morphology calibration and copy `predicted_stage_hpf` directly
   into `inferred_stage_hpf_reg` (`src/build/infer_developmental_age.py:123-156`).
6. Write an age key carrying nominal stage renamed to `calc_stage_hpf`, the inferred field, clock
   time, and run/model/architecture names (`src/build/infer_developmental_age.py:31-40`).

The training-row selection is not completely recoverable for the on-disk key. The function accepts
an explicit `reference_datasets`; without one it uses hard-coded control/reference subsets from
`20240411`, `20240626`, early `20230620`, and `20231218`
(`src/build/infer_developmental_age.py:46-70`). The on-disk CSV does not persist which branch or
reference list was used. It does persist one provenance tuple on all 83,476 rows:
`train_dir=20241008`, `model_name=VAE_training_2024-10-08_21-40-44`, and
`architecture_name=VAE_z100_ne250_base_model` (measured 2026-08-24 from the legacy key; protocol
below). Exact reference-training data are therefore a provenance gap, not something this study can
reconstruct.

Legacy core demonstrably consumed this output: `UrrDataConfig` reads `metadata/age_key.csv`, selects
only `snip_id` and `inferred_stage_hpf_reg`, and merges them into the sequence key
(`src/core/data/dataset_configs.py:38-56`). The older `MorphIAFVAEConfig` does the same
(`src/legacy/vae/models/morph_iaf_vae/morph_iaf_vae_config.py:109-122,160-166`).

## 3. Direct agreement test

| requested statistic | result | reason |
|---|---:|---|
| exact paired rows | 0 | no shared `snip_id` |
| Pearson correlation | not estimable | `n=0` |
| mean bias (`new - legacy`) | not estimable | `n=0` |
| SD of paired differences | not estimable | `n=0` |
| Bland-Altman 95% limits (`bias ± 1.96 SD`) | not estimable | `n=0` |

Every cell was measured 2026-08-24 by one-to-one exact-ID intersection of the two declared source
fields. The legacy source contained 83,476 rows with no duplicate `snip_id` and no null
`inferred_stage_hpf_reg`; the concatenated new source contained 532,235 rows with no duplicate
`snip_id`. Eighteen experiment IDs were common, but their 24,215 legacy and 44,185 new rows still
shared neither a `snip_id` nor an `embryo_id` (measurement protocol below).

No correlation of experiment averages, nominal clock stages, filenames, or reconstructed IDs is
reported. Any of those would substitute a proxy for the paired comparison requested in the brief.

## 4. `time_window` provenance and unit risk

The current core has two age thresholds:

- The dataset sampler admits a positive when `abs(age_i - age_j) <= time_window`
  (`src/core/data/dataset_classes.py:99-127`).
- The loss target uses `abs(age_i - age_j) <= time_window + 1.5`
  (`src/core/losses/loss_functions.py:347-352`).

The loss config's current default is **1.5 hpf**
(`src/core/losses/loss_configs.py:132-139`), and model initialization copies that value into the data
config (`src/core/models/model_configs.py:121-128`). The older `MorphIAFVAEConfig` is internally
ambiguous: its annotated field default is 1.5, but its explicit constructor default is 2.0 and the
constructor assigns that value (`src/legacy/vae/models/morph_iaf_vae/morph_iaf_vae_config.py:43-45,
48-73,99-106`). This study found no repository record of an empirical window sweep or tuning study;
`rg -n time_window` found definitions/usages but no metric-window calibration artifact (command run
2026-08-24).

What history does establish is the applicable unit lineage:

- `time_window: float = 1.5` entered the current core loss config in commit `13152768d` on
  2025-04-28 (`git blame -L 138,138 src/core/losses/loss_configs.py`; `git show --format=fuller
  13152768d`). That same commit's data config selected `inferred_stage_hpf_reg` from `age_key.csv`
  (`git show 13152768d:src/data/dataset_configs.py`, lines 51-54 in that blob).
- The new `stage_predictions` producer first appeared in commit `72931d2ad` on 2026-07-06
  (`git log --follow --format='%H %ad %s' --date=iso -- src/data_pipeline/feature_extraction/stage_predictions/compute.py`).

Thus there is no evidence that 1.5 was tuned at all, but it was unquestionably specified in the
legacy inferred-stage stack and could not have been tuned against an artifact introduced in 2026.

An additive bias shared by every row would cancel in a within-pair delta, but a scale difference or
experiment-dependent calibration would change which pairs cross the threshold. The legacy producer
explicitly fits experiment-conditioned mappings with intercepts, while the new producer does not
(`src/build/infer_developmental_age.py:107-153` versus
`src/data_pipeline/feature_extraction/stage_inference.py:15-24`). Therefore silently retaining 1.5
can change both positive sampling and loss targets. This paragraph is an inference from the cited
implementations; the missing paired data prevent an effect-size estimate.

## 5. Corpus coverage by experiment

The explicit 148-experiment list yielded 133 readable inventories totaling 699,505 rows. Of those,
109 experiments have stage artifacts with 532,235 rows; every present stage artifact is a complete
one-to-one `snip_id` match to its inventory. There are 531,337 finite stages (75.959% of all inventory
rows), while 24 inventory-bearing experiments totaling 167,270 rows have no stage artifact
(measured 2026-08-24; protocol below). The exact 133-row per-experiment table is
[`staging_by_experiment.csv`](recon_tables/staging_by_experiment.csv); its aggregate counts were
independently reproduced on 2026-08-24.

Coverage partitions the 133 inventory-bearing experiments as follows (measured 2026-08-24):

- **91 experiments:** 100% finite-stage coverage.
- **9 experiments:** partial finite-stage coverage, all due to `missing_start_age_hpf`:

| experiment | finite / inventory | coverage |
|---|---:|---:|
| `20260324_cep290_24hpf_plate02` | 69 / 101 | 68.317% |
| `20260414_b9d2_30hpf_plate02` | 63 / 72 | 87.500% |
| `20260415_b9d2_30to48hpf_plate02_t02` | 51 / 58 | 87.931% |
| `20260415_cep290_18hpf_plate03` | 56 / 81 | 69.136% |
| `20260415_cep290_30to48hpf_plate02_t01` | 60 / 93 | 64.516% |
| `20240718` | 64 / 99 | 64.646% |
| `20260319_cilia_crispant_24hpf` | 58 / 99 | 58.586% |
| `20260319_cilia_crispant_30hpf` | 58 / 102 | 56.863% |
| `20260416_cep290_30to48hpf_plate02_t02` | 59 / 70 | 84.286% |

- **9 experiments:** a stage artifact exists but has zero finite stages:
  `20240510`, `20250703_chem3_28C_T00_1325`, `20250703_chem3_34C_T00_1131`,
  `20250703_chem3_34C_T01_1457`, `20250703_chem3_35C_T00_1101`,
  `20250703_chem3_35C_T01_1437`, `20260724_hotfish_24hpf_plate01`,
  `20260724_hotfish_30hpf_plate01`, and `20260724_hotfish_36hpf_plate01`. All 661 rows are
  `missing_start_age_hpf` (measured 2026-08-24).
- **24 experiments:** no stage artifact:
  `20230831`, `20231110`, `20231206`, `20231218`, `20240306`, `20240307`, `20240404`,
  `20240411`, `20240418`, `20240509`, `20240522`, `20240530`, `20240626`, `20240812`,
  `20241022`, `20241023`, `20250126`, `20250215`, `20250305`, `20250415`, `20250425`,
  `20260724_hotfish_24hpf_plate02`, `20260724_hotfish_30hpf_plate02`, and
  `20260724_hotfish_36hpf_plate02` (167,270 inventory rows; measured 2026-08-24).

The remaining 15 experiments in the explicit list have no readable inventory, so stage coverage is
not defined: `20240813_extras`, `20260320_cilia_crispant_48hpf`,
`20260324_cep290_18hpf_24hpf_plate02`, `20260324_cep290_18hpf_plate01`,
`20260331_b9d2_18hpf_plate01`, `20260414_b9d2_14hpf_plate02`, `20230525`,
`20250623_chem_35C_T02_1204`, `20260320`, `20250416`, `20251104`, `20251106`, `20251113`,
`20260223`, and `20260224` (measured 2026-08-24).

## Prior documentation treatment

| document | what it establishes | unresolved stage-semantic issue |
|---|---|---|
| `docs/core/metric_loss_overview.md` | Defines the loss age gate as `|a_i-a_j| <= time_window + 1.5` and calls `time_window` an age tolerance (`:216-226,319-328`). | It does not identify the source or estimand of `a_i`, and it documents the loss buffer but not the sampler's narrower gate. |
| `docs/architecture/training_guide.md` | Shows a metric-VAE launch and exposes metric weight and temperature (`:82-92`). | It names neither `time_window` nor the stage source; the default table at `:46-63` omits both. |
| `docs/data_pipeline/METADATA_AUDIT.md` | Treats `start_age_hpf` and `temperature` as required plate tabs and inventories missing inputs (`:3-7,13-44`). | It does not define the output formula or distinguish nominal clock stage from morphology-inferred stage. |

The documents do not make a false equivalence; they leave the equivalence question unstated. That
omission is material because the metric equation treats its age axis as semantically stable. This
last sentence is an inference from the cited documentation and implementations.

## Recommendation

1. Do not label a run legacy-compatible if it pairs on `predicted_stage_hpf` with the inherited 1.5
   window. Require the stage source/method version and `time_window` to be explicit run provenance.
   This is an inference/recommendation from the lineage above.
2. Produce an authoritative explicit legacy-to-pipeline row crosswalk at the identity boundary; do
   not derive it by parsing or rewriting IDs. On that paired set, report Pearson/Spearman correlation,
   regression slope, mean and median bias, SD and 95% limits of agreement, and results by experiment,
   temperature, stage range, and snapshot/timelapse mode. This is a recommendation.
3. Retune the sampler threshold on the intended new-stage cohort, and separately validate the
   loss's `+1.5` buffer. The acceptance target should be stability of legal-positive availability and
   pair membership, not merely correlation of absolute ages. This is a recommendation.
4. Keep S02 status as `unavailable`, not failed; whether its 20,219 finite rows are admissible is an
   explicit cohort-policy decision independent of stage-scale calibration
   (`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:39-48`).

## Measurement provenance

All corpus measurements above were performed read-only on 2026-08-24.

- New pipeline root:
  `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output`.
- Explicit experiment authority: the first-occurrence ordered `experiment_id` values in
  `docs/refactors/core-model/reports/recon_tables/availability_schema.csv` (148 IDs; SHA-256
  `64f5f36ca0d76e971364a8b0dd3d23023c1f60c15847e06f65a4e3312cc608f6`). Expected inventory and
  stage paths were constructed directly from each explicit ID and the path contract; the pipeline
  output tree was not globbed
  (`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:395-406,554-565`).
- Legacy key:
  `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/models/metadata/age_key.csv`
  (SHA-256 `4fa7473b0d70255fd796a8b0dd3d23023c1f60c15847e06f65a4e3312cc608f6`).
- Coverage was computed by reading explicit inventory `snip_id`s, explicit stage rows, and checking
  exact set equality per experiment. Status was the literal `stage_prediction_status` when present
  and `unavailable` when the entire column was absent. Finite means pandas numeric coercion followed
  by non-null. Exact-ID comparison used set intersection and a one-to-one merge; no ID was parsed,
  sliced, or reconstructed (measurement command: an in-memory `python`/pandas script, 2026-08-24).
- Existing comparison table fingerprint:
  `docs/refactors/core-model/reports/recon_tables/staging_by_experiment.csv` SHA-256
  `a8bdace2f8d7a40d57ee6f37ca4c5902ca113d90f9b6c15b783d76d7f1332fd3`.
