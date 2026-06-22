# Feature World - computed feature targets

**Status:** planning spec, 2026-06-22. This doc lists the computed features and feature-derived QC
targets we can start, their dependencies, and the acceptance bar for each.

**Doctrine:** features quantify validated objects. QC consumes feature tables and emits flags or
QC-specific annotations over the same `snip_id` universe. Neither layer mints object identity,
chooses segmentation prompts, or performs analysis-ready joins.

---

## Shared Feature And QC Stage Pattern

Every computed feature and feature-derived QC stage should use the same WellRunner-compatible
per-well stage recipe:

- one noun-like registry row in the pipeline-wide path registry;
- one compute function with explicit inputs and no hidden globals;
- one thin `tasks.py` verb or entrypoint that parses arguments and delegates;
- one templated per-well rule;
- one focused contract/validator for the feature or QC table it owns;
- tests in the parallel `tests/data_pipeline/...` tree.

Feature and QC paths must come from the orchestration registry, not domain-local path registries or
raw strings in rules. Code may import identity constructors/parsers when it needs IDs, but it must
not mint or split IDs inline. The per-well shard paths must be compatible with `well_runner`: DAG-time
helpers declare expected per-well outputs, runtime collectors inspect finished shards.

Each file written for feature work must pass the `pipeline_file_philosophy.md` first-read check:
clear names, explicit signatures, flow-order organization, and fail-loud errors that name the fix.

---

## Code Organization Pattern

Do not add a `stages/` folder under `feature_extraction` or `quality_control`. Those package names
already name the pipeline world; the child folder should name the product directly.

Computed feature products live here:

```text
src/data_pipeline/feature_extraction/
  mask_geometry/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  curvature_metrics/
    contract.py
    compute.py
    skeletonization.py
    entrypoint.py
    __init__.py
  fraction_alive/
    contract.py
    compute.py
    via_masks.py
    entrypoint.py
    __init__.py
  stage_predictions/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  consolidated_features/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  io/
    loaders.py
    writers.py
  shared/
    feature_table_utils.py
```

Feature-derived QC products mirror that shape:

```text
src/data_pipeline/quality_control/
  death_detection/
    contract.py
    compute.py
    persistence.py
    alignment.py
    entrypoint.py
    __init__.py
  surface_area_qc/
    contract.py
    reference_contract.py
    reference.py
    compute.py
    entrypoint.py
    __init__.py
    references/
      surface_area_reference_v1.csv
      README.md
  motion_qc/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  focus_qc/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  snip_qc/
    contract.py
    inputs.py
    build.py
    entrypoint.py
    __init__.py
  io/
    loaders.py
    writers.py
  shared/
    qc_table_utils.py
```

Per product folder:

- `contract.py` defines the table truth: grain, required columns, nullable columns, and validator;
- `compute.py` owns pure feature/QC logic and returns dataframes or values;
- `build.py` is allowed when the product assembles a verdict rather than measuring a feature;
- `inputs.py` is allowed when a product needs small, explicit in-memory assembly from registered upstream artifacts;
- `config.py` owns product-local defaults when the product has thresholds, model choices, or policies;
- `entrypoint.py` is the thin CLI/task adapter that loads inputs, resolves config, calls compute/build, validates, writes;
- extra `*.py` files are product-specific helpers only.

Domain-level `io/` and `shared/` are for boring mechanics reused by multiple products. `shared/`
is earned: keep helpers product-local until a second product needs them. Existing legacy entrypoints
may remain as temporary wrappers that import the product-local `entrypoint.main`.

Doctrine: domain package names the world; product folder names the table; `contract.py` defines it;
`compute.py` makes it; `entrypoint.py` touches the filesystem. No `stages/` inside stages.

Code package and output stage intentionally differ: source code lives under `feature_extraction/`,
while feature artifacts land under the `features/` output stage. Do not add a new
`data_pipeline.features` package for future feature work.

---

## Contract Naming Pattern

Inside a single `contract.py`, generic local names are acceptable. Public exports must name the
product so imports stay readable when multiple contracts are in scope.

Examples:

- `MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS` and `validate_mask_geometry_features(df)`
- `FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS` and `validate_fraction_alive_features(df)`
- `DEATH_DETECTION_QC_REQUIRED_COLUMNS` and `validate_death_detection_qc(df)`
- `SURFACE_AREA_QC_REQUIRED_COLUMNS` and `validate_surface_area_qc(df)`
- `SNIP_QC_REQUIRED_COLUMNS` and `validate_snip_qc(df)`

Avoid exported generic names like `REQUIRED_COLUMNS` from product contracts.

---

## Stage Table Pattern

**One-row grain:** default grain is one row per `snip_id`. A feature or QC stage may use a different
grain only if its contract names that grain explicitly.

**Feature tables:** features are measured or predicted values.

- required key: `snip_id`;
- feature columns may be continuous, categorical, or numeric predictions;
- feature columns must not be boolean inclusion/exclusion decisions;
- feature columns should not end in `_flag`.

Examples: `area_um2`, `centroid_x_um`, `mean_curvature_per_um`, `speed_um_per_s`,
`fraction_alive`, `predicted_stage_hpf`.

**QC tables:** QC tables are judgments over the feature universe.

- required key: `snip_id`;
- every boolean QC output column must end in `_flag`;
- every `*_flag` column must be non-null boolean dtype;
- annotation columns are allowed, but they are not flags;
- stage-specific QC tables emit one or more `*_flag` columns;
- `snip_qc` computes `use_snip` and pipe-delimited `qc_fail_reasons` from selected exclusion flags.

Examples: `dead_flag`, `sa_outlier_flag`, `motion_flag`, `focus_flag`, `edge_flag`,
`discontinuous_mask_flag`; annotations include `death_inflection_time_int` and
`death_predicted_stage_hpf`.

**WellRunner compatibility:** every feature and QC stage should produce per-well shards first, then
merge. The stage contract must identify whether the shard is per-well or merged, and the registry row
must be the single source of path truth for both Snakemake and Python entrypoints.

---

## Feature Universe

The feature universe is the one-row-per-`snip_id` table that feature and QC products align to.
The canonical source is validated `snip_inventory`; feature and QC products consume this universe, they do not
create it.

Required identity columns:

- `snip_id`
- `embryo_id`
- `physical_embryo_id`
- `experiment_id`
- `well_id`
- `image_id`
- `time_index`
- `channel_id`

Most feature and QC products either compute directly at `snip_id` grain or compute at an upstream
object/mask grain and then explicitly project to `snip_id` grain through validated `snip_inventory`. No
feature or QC product may silently change the universe. Missing, duplicate, or extra `snip_id` rows
must be documented by the product contract and fail loud unless the product explicitly defines a
different behavior.

Tiny doctrine: validated snip inventory defines the universe; feature tables measure it; QC
tables judge it; `snip_qc` summarizes the verdict; `analysis_ready` applies the verdict.

---

## Product Overview

| Product | Domain | Grain | Primary input | Output product |
|---|---|---|---|---|
| `mask_geometry` | `feature_extraction` | `snip_id` | validated `snip_inventory` + `frame_masks` | `mask_geometry_features` |
| `consolidated_features` | `feature_extraction` | `snip_id` | validated feature tables | `consolidated_features` |
| `curvature_metrics` | `feature_extraction` | `snip_id` | `mask_geometry` + canonical masks | `curvature_features` |
| `pose_kinematics` | `feature_extraction` | `snip_id`; track operations internal | `mask_geometry` + `frame_inventory` | `pose_kinematics_features` |
| `stage_predictions` | `feature_extraction` | `snip_id` | morphology/timing features | `stage_prediction_features` |
| `fraction_alive` | `feature_extraction` | `snip_id` | embryo masks + auxiliary/VIA masks | `fraction_alive_features` |
| `surface_area_qc` | `quality_control` | `snip_id` | `mask_geometry` + feature universe + packaged reference | `surface_area_qc` |
| `motion_qc` | `quality_control` | `snip_id` | `pose_kinematics` + feature universe | `motion_qc` |
| `death_detection` | `quality_control` | `snip_id` | `fraction_alive` + feature universe | `death_detection_qc` |
| `focus_qc` | `quality_control` | `snip_id` after declared inheritance | named focus metric + feature universe | `focus_qc` |
| `metadata_completeness_qc` | `quality_control` | `snip_id` | validated `snip_inventory` + required metadata contracts | `metadata_completeness_qc` (deferred) |
| `snip_qc` | `quality_control` | `snip_id` | feature universe + selected QC flag inputs | `snip_qc` |

Any product not actually `snip_id` grained must say so in its contract and must also name the
projection step that returns to the feature universe.

---

## Feature Table Provenance

Feature tables carry minimal method provenance, not biological metadata. Add provenance columns when
multiple methods, formulas, references, configs, or models could produce the same product.

Allowed or required where relevant:

- `feature_backend` or `method_name`
- `feature_version` or `model_version`
- `reference_version` for reference-backed QC or features
- `config_name` or `config_hash` when config materially changes output values

Do not join genotype, condition, perturbation, or `use_snip` into feature products. Those joins belong
in `analysis_ready`, after feature and QC contracts are validated.

---

## Product Path Acceptance

Every product-specific Done When inherits this path acceptance bar:

- per-well output paths come from `artifact_path(...)` with registry step/artifact keys;
- validated sentinels come from `validated_path(...)`;
- output directories come from `step_dir(...)` or `per_well_step_dir(...)`;
- no raw output path strings appear in rules, `tasks.py`, or entrypoints;
- packaged static references, such as `surface_area_qc/references/surface_area_reference_v1.csv`,
  are source assets and are not `paths.py` artifacts.

---

## Config Pattern

Every feature or QC product must declare its config surface before implementation. Any threshold,
method choice, reference version, smoothing window, missing-data policy, or output policy is a config
knob, not a hidden literal inside `compute.py`.

Target convention:

- run-level overrides live under the product key in pipeline config, for example
  `feature_extraction.mask_geometry` or `quality_control.death_detection`;
- product defaults live beside the product code when the product has real knobs, usually in
  `config.py`;
- `entrypoint.py` resolves defaults plus run overrides, then passes the resolved config or explicit
  keyword values into `compute.py`;
- `compute.py` never imports global run config and never reads thresholds or reference paths by
  itself;
- each contract test should include at least one non-default config case when a product has knobs.

A product with no knobs should still say that explicitly in its mini-spec. Do not leave a blank space
where a future agent has to guess.

---

## Output Location Pattern

All feature steps land under the `features/` stage on disk. Each step gets its own `product_dir`
— the named product family within the stage. This is how `PIPELINE_STEPS` in `orchestration/paths.py`
constructs the path:

```python
"latent_embeddings": {
    "stage": "features",          # top-level folder
    "product_dir": "latent_embeddings",  # product family within it
    "fanout": PER_WELL_THEN_MERGE,
    "artifacts": { ... }
}
```

Which gives you, for free via `artifact_path(...)`:

```
features/<exp>/latent_embeddings/per_well/<well_id>/<well_id>_latents.parquet
features/<exp>/mask_geometry/per_well/<well_id>/<well_id>_mask_geometry.parquet
features/<exp>/consolidated/per_well/<well_id>/<well_id>_consolidated_features.parquet
```

The output tree is declarative: stage names the phase; `product_dir` names what the artifact is.
`product_dir` uses the artifact type, not the method — `latent_embeddings` not `legacy_vae`,
`mask_geometry` not `sam2_geometry`. Method provenance belongs in code and config, not paths.

No caller ever types a raw path string. Every path is resolved through `artifact_path()`,
`validated_path()`, or `step_dir()` with the step name and artifact key from the registry.

---

## Priority Order

1. **Stage table pattern and contracts** - lock `snip_id` grain, feature-vs-QC schema rules,
   `_flag` naming, and WellRunner-compatible shard/merge paths.
2. **Mask geometry** - first computed feature; smallest dependency surface after segmentation
   Session A/B and proves canonical mask consumption.
3. **Consolidated features v1** - merge one feature table by `snip_id`; proves the feature universe
   and collision/duplicate policy early.
4. **Surface area QC** - follows mask geometry and proves the feature-derived QC pattern early.
5. **Curvature metrics** - follows mask geometry after mask decode and centerline behavior are stable.
6. **Pose and kinematics** - follows stable `track_id`, temporal ordering, and frame timing.
7. **Motion QC** - follows pose/kinematics and proves temporal QC flags.
8. **Stage predictions** - follows morphology features and consolidation.
9. **Fraction alive** - follows auxiliary/VIA mask product clarity.
10. **Death detection QC** - follows `fraction_alive` and the feature universe; first viability-derived
   QC flag table.
11. **Focus QC** - follows a named focus/input-quality feature or image-quality contract; do not
   invent ad hoc focus inputs.
12. **Metadata completeness QC** - deferred; follows the metadata completeness contract and may later feed `snip_qc` as `missing_metadata`.
13. **Snip QC** - follows MVP exclusion QC tables and builds the final `use_snip` verdict.
14. **Latent embeddings** - follows validated snip inventory; has its own spec
   (`legacy_embeddings.md`) due to the 3.9 env boundary and batch-execution constraints.
   Feeds `analysis_ready` on `snip_id`.

---

## Mask Geometry

**Product folder:** `src/data_pipeline/feature_extraction/mask_geometry/`

> Note: the source code package is `feature_extraction/`; the on-disk output stage is `features/`.
> These are intentionally different — the code package name describes what the code does; the output
> stage name describes what the artifact is. All paths come from the registry, so no caller conflates
> the two.

**Key functions:**

- `contract.py::MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_mask_geometry_features(df)`
- `compute.py::compute_mask_geometry_features(snip_inventory_df, frame_masks_df, frame_inventory_df, *, mask_decoder)`
- `compute.py::compute_mask_geometry_for_mask(mask, pixel_size_um)`
- `entrypoint.py::main()`

**Direct dependencies:** canonical validated `snip_inventory`, `frame_masks`, `frame_inventory`, segmentation
mask decode/geometry helpers, and shared `snip_id` identity.

**Config surface:** `feature_extraction.mask_geometry`

- `pixel_size_source`: column/contract source for micron calibration, default `frame_inventory`;
- `placeholder_policy`: how no-mask placeholders are handled, default `exclude`;
- `mask_decoder`: decoder name or injected decoder policy, default canonical mask RLE decoder;
- `output_units`: fixed to micron-aware outputs where calibration exists;
- `min_valid_area_px`: smallest non-empty mask accepted for feature computation.

**Grain:** one row per `snip_id`.

Mask Geometry uses validated `snip_inventory` as the object-to-snip handoff:

```text
snip_id -> mask_id / track_id / image_id
```

It does not scan all `frame_masks` independently and invent a feature universe. Frame masks are
looked up through the snip inventory row.

**What it computes:** area, perimeter, centroid, width/height-style geometry, and calibration-aware
micron-scale measurements for each snip-selected object mask.

**Depends on:**

- validated `snip_inventory` rows with one row per `snip_id` and a documented mask handoff;
- validated `frame_masks` rows with parseable `mask_id` / `track_id`;
- pixel calibration from `frame_inventory`;
- mask RLE/decode and geometry helpers from segmentation Session A;
- no-mask placeholders filtered out or handled explicitly before feature computation.

**Needs before coding:**

- feature table contract: one row per `snip_id`;
- source of pixel size named in the input contract;
- decision that this feature reads canonical masks, not backend-native SAM2 outputs;
- documented join from `snip_inventory` to `frame_masks` by `mask_id`, `track_id`, and/or `image_id`.

**Done when:**

- synthetic masks produce deterministic geometry metrics;
- invalid or placeholder masks fail loud or are excluded by contract;
- per-well output validates and can be merged without experiment-grain assumptions.

---

## Curvature Metrics

**Product folder:** `src/data_pipeline/feature_extraction/curvature_metrics/`

**Key functions:**

- `contract.py::CURVATURE_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_curvature_features(df)`
- `compute.py::compute_curvature_features(mask_geometry_df, frame_masks_df, *, mask_decoder)`
- `compute.py::compute_curvature_for_mask(mask, pixel_size_um)`
- `skeletonization.py::extract_centerline_points(mask)`
- `entrypoint.py::main()`

**Direct dependencies:** mask geometry feature rows, canonical masks, pixel calibration, and deterministic
centerline extraction.

**Config surface:** `feature_extraction.curvature_metrics`

- `skeletonization_method`: centerline extraction method;
- `min_centerline_points`: minimum points required before curvature is computed;
- `smoothing_window_points`: optional centerline smoothing window;
- `resample_spacing_um`: optional centerline resampling interval;
- `low_information_policy`: return documented null metrics or fail loud for tiny/empty masks.

**What it computes:** centerline length, centerline point count, and curvature summaries from valid
embryo masks.

**Depends on:**

- mask geometry inputs and calibration;
- stable binary mask decode/read policy;
- centerline/skeletonization behavior that is deterministic on synthetic masks.

**Needs before coding:**

- explicit behavior for masks with too few centerline points;
- clear units: curvature in inverse microns, lengths in microns;
- tests for straight, curved, tiny, and empty masks.

**Done when:**

- low-information masks return documented null metrics instead of silent bad numbers;
- synthetic fixtures pin centerline and curvature behavior;
- output joins cleanly by `snip_id` with mask geometry.

---

## Pose And Kinematics

**Product folder:** `src/data_pipeline/feature_extraction/pose_kinematics/`

**Key functions:**

- `contract.py::POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_pose_kinematics_features(df)`
- `compute.py::compute_pose_kinematics_features(mask_geometry_df, frame_inventory_df)`
- `compute.py::compute_pose_features_for_mask(mask, pixel_size_um)`
- `compute.py::compute_kinematics_for_track(track_df)`
- `entrypoint.py::main()`

**Direct dependencies:** mask geometry feature rows, `track_id`, `snip_id`, frame timing from
`frame_inventory`, and deterministic ordering within each track.

**Config surface:** `feature_extraction.pose_kinematics`

- `orientation_method`: how object orientation is computed;
- `time_column`: frame timing column, default `elapsed_time_s`;
- `first_frame_kinematics_policy`: null behavior for first row in each track;
- `max_allowed_time_gap_s`: optional fail-loud guard for track gaps;
- `coordinate_units`: expected position units after calibration.

**What it computes:** orientation, bounding box dimensions, displacement, speed, coordinate deltas,
and elapsed-time deltas for each tracked object.

**Depends on:**

- valid object masks and mask geometry;
- stable `track_id` from the segmentation contract;
- frame time carried through `frame_inventory`;
- deterministic ordering within each `track_id`.

**Needs before coding:**

- segmentation Session B fake-predictor path proving deterministic `track_id` and mask rows;
- clear first-frame behavior for displacement/speed nulls;
- validation that time deltas are positive within each track.

**Done when:**

- synthetic two-frame and three-frame tracks produce expected displacement/speed;
- missing or non-monotonic time fails loud with the track and frame named;
- first observation per track has documented null kinematics.

---

## Consolidated Features

**Product folder:** `src/data_pipeline/feature_extraction/consolidated_features/`

**Key functions:**

- `contract.py::CONSOLIDATED_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_consolidated_features(df)`
- `compute.py::consolidate_feature_tables(feature_tables, *, key="snip_id")`
- `compute.py::assert_feature_table_compatible(df, *, key="snip_id", feature_name)`
- `entrypoint.py::main()`

**Direct dependencies:** one or more validated feature shards, `snip_id` uniqueness, and the registry
paths for per-well and merged feature products.

**Config surface:** `feature_extraction.consolidated_features`

- `join_key`: default `snip_id`;
- `feature_tables`: ordered list of feature products to merge;
- `column_collision_policy`: fail loud unless collisions are explicitly allowed;
- `required_core_features`: minimum columns expected in the consolidated output;
- `missing_feature_policy`: fail, allow-null, or skip for optional feature products.

**What it computes:** the merged per-object/per-snip feature table used by downstream model/QC and
analysis-ready stages.

**Depends on:**

- one or more validated per-well feature shards;
- shared identity keys across feature tables;
- registry-supported per-well and merged artifact paths.

**Needs before coding:**

- canonical join key, expected to be `snip_id` or the post-segmentation object identity chosen by
  the object contract;
- collision policy for overlapping columns;
- explicit list of required core feature columns for the consolidated contract.

**Done when:**

- shard merge is one-to-one on the chosen key;
- duplicate keys and column collisions fail loud;
- downstream QC can read the consolidated contract without feature-specific path knowledge.

---

## Stage Predictions

**Product folder:** `src/data_pipeline/feature_extraction/stage_predictions/`

**Key functions:**

- `contract.py::STAGE_PREDICTION_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_stage_prediction_features(df)`
- `compute.py::compute_stage_prediction_features(feature_df, *, model_version)`
- `compute.py::predict_stage_hpf(feature_row)`
- `entrypoint.py::main()`

**Direct dependencies:** morphology/size feature inputs, stable feature units, and model/version
provenance if persisted.

**Config surface:** `feature_extraction.stage_predictions`

- `model_name`: default stage model name, for example `kimmel1995_temp_rate`;
- `model_version`: persisted model/formula version;
- `temperature_rate_slope` and `temperature_rate_intercept`: formula coefficients if using the
  temperature-rate model;
- `stage_min_hpf` and `stage_max_hpf`: optional clipping bounds;
- `required_input_columns`: morphology/timing columns required by the model;
- `missing_input_policy`: fail loud unless a documented fallback exists.

**What it computes:** developmental stage prediction from morphology/size features.

**Depends on:**

- consolidated or selected geometry feature inputs;
- stage inference model/rule already present in `feature_extraction`;
- stable feature names and units.

**Needs before coding:**

- feature inputs required by the stage model;
- model/version provenance fields if predictions become a persisted contract;
- null behavior when required morphology features are missing.

**Done when:**

- deterministic fixture rows produce expected stage predictions;
- missing feature inputs fail loud;
- output can be merged with consolidated features without changing upstream feature contracts.

---

## Fraction Alive

**Product folder:** `src/data_pipeline/feature_extraction/fraction_alive/`

**Key functions:**

- `contract.py::FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_fraction_alive_features(df)`
- `compute.py::compute_fraction_alive_features(frame_masks_df, auxiliary_masks_df, *, mask_decoder)`
- `compute.py::compute_fraction_alive_for_masks(embryo_mask, via_mask)`
- `via_masks.py::build_via_mask_lookup(auxiliary_masks_df)`
- `entrypoint.py::main()`

**Direct dependencies:** canonical embryo masks, auxiliary/VIA mask product, `snip_id`, `image_id`, and
feature-table alignment rules.

**Config surface:** `feature_extraction.fraction_alive`

- `auxiliary_mask_type`: named auxiliary mask product to consume;
- `join_key`: default `image_id` or documented object key;
- `missing_auxiliary_mask_policy`: fail loud, return null, or configured fallback;
- `empty_embryo_mask_policy`: fail loud or documented null behavior;
- `fraction_clip_min` and `fraction_clip_max`: expected output range, default 0.0 to 1.0;
- `overlap_mode`: pixel-overlap rule for embryo mask versus auxiliary/VIA mask.

**What it computes:** continuous viability fraction from embryo masks and auxiliary/VIA masks.

**Depends on:**

- canonical embryo/object masks;
- auxiliary/VIA mask product contract and paths;
- clear join between object mask rows and auxiliary mask rows by `image_id` or object identity.

**Needs before coding:**

- auxiliary mask world or QC-world decision on VIA mask ownership;
- explicit behavior when VIA masks are missing;
- tests for empty embryo mask, full dead tissue mask, no overlap, and partial overlap.

**Done when:**

- feature computation uses canonical mask products, not legacy path guesses;
- missing auxiliary masks fail loud with the expected source named;
- output joins cleanly into consolidated features.

---

## Death Detection QC

**Product folder:** `src/data_pipeline/quality_control/death_detection/`

**Key functions:**

- `contract.py::DEATH_DETECTION_QC_REQUIRED_COLUMNS`
- `contract.py::validate_death_detection_qc(df)`
- `compute.py::compute_death_detection_flags(fraction_alive_df, snip_universe_df, *, thresholds)`
- `persistence.py::find_inflection_candidates(embryo_fraction_alive_df, *, thresholds)`
- `persistence.py::validate_death_persistence(embryo_fraction_alive_df, inflection_time, *, thresholds)`
- `alignment.py::align_death_flags_to_snip_universe(death_flags_df, snip_universe_df)`
- `entrypoint.py::main()`

**Direct dependencies:** `fraction_alive` feature rows, feature universe keyed by `snip_id`,
per-embryo temporal ordering, and QC defaults/thresholds.

**Config surface:** `quality_control.death_detection`

- `persistence_threshold`: required post-inflection dead fraction, current default 0.80;
- `lead_time_hr`: lead time applied to the inferred death time, current default 4.0;
- `decline_rate_threshold`: minimum decline rate for candidate inflections, current default 0.05;
- `dead_fraction_threshold`: fraction-alive cutoff for dead-state evidence, current default 0.90;
- `min_timepoints`: minimum observations per embryo before death detection is attempted;
- `time_column`: default `time_index` unless the contract moves to elapsed hours;
- `stage_column`: optional stage annotation column, default `predicted_stage_hpf`;
- `smoothing_window`: optional smoothing window for noisy fraction-alive traces;
- `transient_decline_policy`: reject transient dips unless persistence passes;
- `missing_fraction_policy`: fail loud or documented skip behavior;
- `output_alignment_policy`: output must align one-to-one to the feature universe.

**What it decides:** whether a snip should be flagged dead, plus the inferred death inflection time
and stage annotation for flagged rows.

**Classification:** QC, not feature extraction. It consumes `fraction_alive` and the feature universe,
then emits flags/annotations. It does not compute a new measured morphology feature.

**Depends on:**

- `fraction_alive` feature rows with `snip_id`, `embryo_id`, `time_index`, and `fraction_alive`;
- a feature universe table with exactly one row per `snip_id`;
- optional `predicted_stage_hpf` if death stage annotations are required;
- stable per-embryo temporal ordering.

**Input contract:**

- `fraction_alive` input columns:
  - `snip_id`
  - `embryo_id`
  - `time_index`
  - `fraction_alive`
  - optional `predicted_stage_hpf`
- feature universe input:
  - `snip_id`
  - any extra feature columns needed only to prove the universe and downstream joins

QC should align to the feature universe. Missing, duplicate, or extra `snip_id` rows must fail loud
before flags are written.

**Output contract:**

- `snip_id`
- `dead_flag`
- `death_inflection_time_int`
- `death_predicted_stage_hpf`

`dead_flag` is a required non-null boolean. The death timing/stage columns are annotations, not
flags. `death_inflection_time_int` is the current output field name even though target input time
identity uses `time_index`; rename only through an explicit contract migration. They are required
when `dead_flag` is true and nullable when `dead_flag` is false.

**Algorithm shape:**

- group by `embryo_id`;
- sort by `time_index`;
- find sustained `fraction_alive` decline candidates;
- validate post-inflection persistence;
- apply configured lead time;
- align output back to the feature universe by `snip_id`.

**Done when:**

- synthetic time series cover alive, clearly dead, transient decline, and too-few-timepoints cases;
- output validates against the stage-specific death-detection schema;
- consolidated QC can merge death flags without rereading masks or segmentation outputs.

---

## Surface Area QC

**Product folder:** `src/data_pipeline/quality_control/surface_area_qc/`

**Packaged reference location:**

```text
src/data_pipeline/quality_control/surface_area_qc/references/
  surface_area_reference_v1.csv
  README.md
```

A curated static default reference lives beside the code because it is part of the QC product and
should be versioned/reviewed with the logic that consumes it. `paths.py` does not need a row for
this packaged reference because it is source data, not a pipeline artifact.

**Key functions:**

- `contract.py::SURFACE_AREA_QC_REQUIRED_COLUMNS`
- `contract.py::validate_surface_area_qc(df)`
- `reference_contract.py::SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS`
- `reference_contract.py::validate_surface_area_reference(df)`
- `reference.py::load_packaged_surface_area_reference(version="v1")`
- `reference.py::select_surface_area_reference(mask_geometry_df, surface_area_reference_df, *, thresholds)`
- `compute.py::compute_surface_area_qc_flags(mask_geometry_df, snip_universe_df, surface_area_reference_df, *, thresholds)`
- `compute.py::compute_surface_area_flag(area_um2, reference_row, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** `mask_geometry` feature rows, feature universe keyed by `snip_id`, validated
surface-area reference rows, and configured threshold policy.

**Config surface:** `quality_control.surface_area_qc`

- `reference_version`: packaged static reference version, default `v1`;
- `area_column`: default `area_um2`;
- `reference_group_columns`: columns used to select comparable reference rows;
- `k_upper`: upper multiplier/threshold, current default 1.4;
- `k_lower`: lower multiplier/threshold, current default 0.7;
- `missing_reference_policy`: fail loud or documented fallback;
- `missing_area_policy`: fail loud or documented flag behavior.

**What it decides:** whether a snip has a suspicious area measurement for downstream analysis.

**Reference policy:**

- the default curated reference is packaged in `surface_area_qc/references/`;
- packaged reference files are static source assets, not runtime outputs;
- `paths.py` does not know about the packaged reference;
- `compute.py` never secretly reads the reference source;
- `entrypoint.py` loads the packaged default through `reference.py`, validates it, then passes
  `surface_area_reference_df` into `compute.py`;
- if a future generated reference becomes a true pipeline output, that future generated artifact gets
  a path-registry row, but the packaged default still stays beside this code.

**Depends on:**

- `mask_geometry` rows with `snip_id` and `area_um2`;
- a feature universe table with exactly one row per `snip_id`;
- a validated surface-area reference table;
- threshold/reference configuration owned by QC config, not hardcoded in the compute function.

**Output contract:**

- `snip_id`
- `sa_outlier_flag`

`sa_outlier_flag` is a required non-null boolean. Any threshold annotations must be named as
annotations, not flags.

**Done when:**

- packaged `surface_area_reference_v1.csv` validates with `reference_contract.py`;
- fixture rows cover low, normal, high, missing, and duplicate `snip_id` cases;
- missing area or missing reference values fail loud or map to documented QC behavior;
- the output aligns one-to-one with the feature universe.

---

## Motion QC

**Product folder:** `src/data_pipeline/quality_control/motion_qc/`

**Key functions:**

- `contract.py::MOTION_QC_REQUIRED_COLUMNS`
- `contract.py::validate_motion_qc(df)`
- `compute.py::compute_motion_qc_flags(pose_kinematics_df, snip_universe_df, *, thresholds)`
- `compute.py::compute_motion_flag(track_or_snip_row, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** `pose_kinematics` feature rows, feature universe keyed by `snip_id`, frame
timing, and motion thresholds.

**Config surface:** `quality_control.motion_qc`

- `ncc_min_threshold`: image-pair similarity cutoff when NCC inputs exist, current default 0.85;
- `bad_pair_frac_threshold`: maximum bad-pair fraction, current default 0.10;
- `speed_um_per_s_max`: optional maximum plausible speed from pose/kinematics;
- `displacement_jump_um_max`: optional maximum plausible frame-to-frame jump;
- `min_track_length`: minimum observations before temporal QC is meaningful;
- `first_frame_policy`: how first-frame null kinematics are treated;
- `missing_motion_metric_policy`: fail loud or documented neutral flag behavior.

**What it decides:** whether a snip has implausible or analysis-breaking motion/temporal behavior.

**Depends on:**

- pose/kinematics rows with `snip_id`, `track_id`, time deltas, and speed/displacement columns;
- a one-row-per-`snip_id` feature universe;
- explicit policy for first-frame rows where kinematic deltas are null by contract.

**Output contract:**

- `snip_id`
- `motion_flag`

`motion_flag` is a required non-null boolean. Generic validation must check the flag shape only;
track-specific deterministic checks belong in the motion compute tests.

**Done when:**

- synthetic tracks cover stationary, plausible motion, impossible jumps, and missing-time cases;
- first-frame null kinematics follow the documented contract;
- output aligns to the feature universe without adding or dropping `snip_id` rows.

---

## Focus QC

**Product folder:** `src/data_pipeline/quality_control/focus_qc/`

**Key functions:**

- `contract.py::FOCUS_QC_REQUIRED_COLUMNS`
- `contract.py::validate_focus_qc(df)`
- `compute.py::compute_focus_qc_flags(focus_feature_df, snip_universe_df, *, thresholds)`
- `compute.py::compute_focus_flag(focus_metric, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** a named focus/input-quality feature table or image-quality contract,
feature universe keyed by `snip_id`, and configured thresholds.

**Config surface:** `quality_control.focus_qc`

- `focus_metric_product`: named upstream product that owns the focus metric;
- `focus_metric_column`: metric column to threshold;
- `focus_min_threshold` or `focus_max_threshold`: configured cutoff, depending on metric direction;
- `inheritance_grain`: whether focus is per snip, per image, or inherited from frame-level QC;
- `missing_focus_policy`: fail loud or documented neutral flag behavior.

**What it decides:** whether a snip should be flagged for focus or image-quality failure.

**Depends on:**

- a prior contract that defines the focus metric input;
- one row per `snip_id` in the feature universe;
- QC thresholds from config.

**Needs before coding:**

- name the input focus metric product and its schema;
- decide whether focus is computed per image, per object/snip, or inherited from a frame-level
  quality product;
- avoid reading raw images from this QC stage unless the focus metric contract explicitly says so.

**Output contract:**

- `snip_id`
- `focus_flag`

`focus_flag` is a required non-null boolean.

**Done when:**

- the input focus metric contract exists;
- fixture rows cover good focus, bad focus, missing focus, and duplicate keys;
- output aligns to the feature universe.

---

## Snip QC

**Product folder:** `src/data_pipeline/quality_control/snip_qc/`

**Files:**

```text
src/data_pipeline/quality_control/snip_qc/
  contract.py
  inputs.py
  build.py
  entrypoint.py
  __init__.py
```

**Purpose:** build the final per-snip QC verdict from already-computed stage-specific QC flags.

Doctrine: QC products find problems; `snip_qc` builds the verdict; `analysis_ready` applies the
verdict. Flags are facts. Reasons are verdict prose. `use_snip` is the switch.

**Grain:** one row per `snip_id`.

**Output artifact:**

- step/product: `snip_qc`
- artifact key: `verdict`
- per-well file: `<well_id>_snip_qc.parquet`
- merged file: `<experiment_id>_snip_qc.parquet`

**Output columns:**

- `snip_id`
- `use_snip`
- `qc_fail_reasons`

`qc_fail_reasons` is a non-null pipe-delimited string. Empty string means pass. Examples: `""`,
`"dead"`, `"surface_area_outlier"`, `"dead|surface_area_outlier"`.

**Key functions and constants:**

- `contract.py::SNIP_QC_REQUIRED_COLUMNS`
- `contract.py::SNIP_QC_EXCLUSION_REASONS`
- `contract.py::validate_snip_qc(df, *, source="")`
- `inputs.py::load_snip_qc_flag_inputs(...)`
- `build.py::build_snip_qc_verdict(snip_universe_df, qc_flags_df, *, exclusion_reasons)`
- `entrypoint.py::main()`

**Contract policy:**

```python
SNIP_QC_REQUIRED_COLUMNS = [
    "snip_id",
    "use_snip",
    "qc_fail_reasons",
]

SNIP_QC_EXCLUSION_REASONS = {
    "dead": "dead_flag",
    "surface_area_outlier": "sa_outlier_flag",
}
```

The `snip_qc` contract owns the verdict policy: reason name maps to source flag column. For MVP, do
not add a source-product registry to the contract. Future hook: `missing_metadata` may map to
`metadata_missing_flag` after `metadata_completeness_qc` exists, but do not include it in
`SNIP_QC_EXCLUSION_REASONS` for the MVP.

**Input assembly:**

`inputs.py` owns the small, explicit in-memory assembly of QC flag columns needed to build the final
verdict. It imports public path-registry helpers from `orchestration.paths`; it does not
define a second QC source registry.

```python
from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    artifact_path,
    known_artifacts,
    validated_path,
)
```

Use registry helpers. Do not inspect registry internals unless no helper exists. Once the back-half
QC rows are added to `paths.py`, `inputs.py` resolves paths with `artifact_path(...)`, optionally
checks sentinels with `validated_path(...)`, and reads only the requested flag columns. It must not
construct raw paths, discover arbitrary QC artifacts, or duplicate source step/artifact filename
mappings locally.

`load_snip_qc_flag_inputs(...)` takes explicit registered source step/artifact pairs from
`entrypoint.py`, for example the MVP sources `death_detection_qc` and `surface_area_qc`. The
entrypoint should pass artifact keys explicitly. `inputs.py` may infer an artifact key with
`known_artifacts(step)` only when a source step has exactly one registered artifact, and must fail
loud if a source step has multiple artifacts and no key was passed. The function returns one row per
`snip_id` with only `snip_id` plus the requested flag columns. Missing registry rows, missing source
artifacts, missing flag columns, duplicate `snip_id`, null flags, and non-boolean flags fail loud.
MVP must not treat missing flags as pass.

**Build behavior:**

`build.py` is pure verdict logic. It does not import `paths.py`, know product artifact names, or load
files. It consumes `snip_universe_df` and an already-assembled `qc_flags_df`.

- start from `snip_universe_df[["snip_id"]]`;
- require `snip_universe_df` has one row per `snip_id`;
- require `qc_flags_df` has one row per `snip_id` for the relevant universe;
- require every flag column named by `SNIP_QC_EXCLUSION_REASONS` is present in `qc_flags_df`;
- require those flag columns are non-null boolean;
- for each snip, build `qc_fail_reasons` from reasons whose flag column is true;
- set `use_snip = qc_fail_reasons == ""`;
- return only `SNIP_QC_REQUIRED_COLUMNS`.

**Validation:**

- `snip_id` is present, non-null, and unique;
- `use_snip` is present and non-null boolean;
- `qc_fail_reasons` is present, non-null string;
- empty `qc_fail_reasons` means the snip passed QC;
- non-empty `qc_fail_reasons` is a pipe-delimited list of known reason keys;
- all reasons are known keys in `SNIP_QC_EXCLUSION_REASONS`;
- `use_snip` is true iff `qc_fail_reasons == ""`;
- `use_snip` is false iff `qc_fail_reasons != ""`.

**Entrypoint behavior:**

- load the feature universe from validated `snip_inventory`;
- call `load_snip_qc_flag_inputs(...)` for current MVP exclusion flags;
- call `build_snip_qc_verdict(...)`;
- call `validate_snip_qc(...)`;
- write the `verdict` artifact via `artifact_path(...)`;
- write the validation marker via `validated_path(...)`.

**Explicit non-goals:**

- no `flag_contract.py`;
- no `exclusion_reasons.py`;
- no source-product registry in `contract.py`;
- no local duplicate of `paths.py` registry data in `inputs.py`;
- no direct `PIPELINE_STEPS` import or inspection unless a registry helper is genuinely missing;
- no `nullifies_columns`;
- no `snip_qc_flags` wide artifact in MVP;
- no feature nullification;
- no biological metadata joins;
- no genotype/condition/`use_snip` mutation in feature products.

**Boundary:**

Stage-specific QC tables remain the source of detailed QC facts. `snip_qc` is the final operational
verdict table. `analysis_ready` decides whether to filter rows, expose rows with `use_snip=false`,
or null selected outputs.

Tiny doctrine: paths locate artifacts; inputs assemble columns; build resolves verdicts; contract
guards meaning.

---

## Not This World

- Detection, segmentation, tracking, and prompt adaptation live in detect/seg/track specs.
- Analysis-ready joins live after features and `snip_qc`.
- GPU SAM2 validation lives in segmentation Session C, not feature computation.
