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
    compute.py
    entrypoint.py
    __init__.py
  focus_qc/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  consolidated_qc/
    contract.py
    compute.py
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
- `entrypoint.py` is the thin CLI/task adapter that loads inputs, calls compute, validates, writes;
- extra `*.py` files are product-specific helpers only.

Domain-level `io/` and `shared/` are for boring mechanics reused by multiple products. `shared/`
is earned: keep helpers product-local until a second product needs them. Existing legacy entrypoints
may remain as temporary wrappers that import the product-local `entrypoint.main`.

Doctrine: domain package names the world; product folder names the table; `contract.py` defines it;
`compute.py` makes it; `entrypoint.py` touches the filesystem. No `stages/` inside stages.

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
- consolidated QC computes `use_snip` from exclusion flags.

Examples: `dead_flag`, `sa_outlier_flag`, `motion_flag`, `focus_flag`, `edge_flag`,
`discontinuous_mask_flag`; annotations include `death_inflection_time_int` and
`death_predicted_stage_hpf`.

**WellRunner compatibility:** every feature and QC stage should produce per-well shards first, then
merge. The stage contract must identify whether the shard is per-well or merged, and the registry row
must be the single source of path truth for both Snakemake and Python entrypoints.

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
12. **Consolidated QC** - follows at least one stage-specific QC table and defines `use_snip`.
13. **Latent embeddings** - follows validated snip inventory; has its own spec
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

- `contract.py::validate_mask_geometry_features(df)`
- `compute.py::compute_mask_geometry_features(frame_masks_df, frame_inventory_df, *, mask_decoder)`
- `compute.py::compute_mask_geometry_for_mask(mask, pixel_size_um)`
- `entrypoint.py::main()`

**Direct dependencies:** canonical `frame_masks`, `frame_inventory`, segmentation mask decode/geometry
helpers, and shared `snip_id` identity.

**What it computes:** area, perimeter, centroid, width/height-style geometry, and calibration-aware
micron-scale measurements for each valid object mask.

**Depends on:**

- validated `frame_masks` rows with parseable `mask_id` / `track_id`;
- pixel calibration from `frame_inventory`;
- mask RLE/decode and geometry helpers from segmentation Session A;
- no-mask placeholders filtered out or handled explicitly before feature computation.

**Needs before coding:**

- feature table contract: one row per valid mask/object observation;
- source of pixel size named in the input contract;
- decision that this feature reads canonical masks, not backend-native SAM2 outputs.

**Done when:**

- synthetic masks produce deterministic geometry metrics;
- invalid or placeholder masks fail loud or are excluded by contract;
- per-well output validates and can be merged without experiment-grain assumptions.

---

## Curvature Metrics

**Product folder:** `src/data_pipeline/feature_extraction/curvature_metrics/`

**Key functions:**

- `contract.py::validate_curvature_features(df)`
- `compute.py::compute_curvature_features(mask_geometry_df, frame_masks_df, *, mask_decoder)`
- `compute.py::compute_curvature_for_mask(mask, pixel_size_um)`
- `skeletonization.py::extract_centerline_points(mask)`
- `entrypoint.py::main()`

**Direct dependencies:** mask geometry feature rows, canonical masks, pixel calibration, and deterministic
centerline extraction.

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
- output joins cleanly by object identity with mask geometry.

---

## Pose And Kinematics

**Product folder:** `src/data_pipeline/feature_extraction/pose_kinematics/`

**Key functions:**

- `contract.py::validate_pose_kinematics_features(df)`
- `compute.py::compute_pose_kinematics_features(mask_geometry_df, frame_inventory_df)`
- `compute.py::compute_pose_features_for_mask(mask, pixel_size_um)`
- `compute.py::compute_kinematics_for_track(track_df)`
- `entrypoint.py::main()`

**Direct dependencies:** mask geometry feature rows, `track_id`, `snip_id`, frame timing from
`frame_inventory`, and deterministic ordering within each track.

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

- `contract.py::validate_consolidated_features(df)`
- `compute.py::consolidate_feature_tables(feature_tables, *, key="snip_id")`
- `compute.py::assert_feature_table_compatible(df, *, key="snip_id", feature_name)`
- `entrypoint.py::main()`

**Direct dependencies:** one or more validated feature shards, `snip_id` uniqueness, and the registry
paths for per-well and merged feature products.

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

- `contract.py::validate_stage_prediction_features(df)`
- `compute.py::compute_stage_prediction_features(feature_df, *, model_version)`
- `compute.py::predict_stage_hpf(feature_row)`
- `entrypoint.py::main()`

**Direct dependencies:** morphology/size feature inputs, stable feature units, and model/version
provenance if persisted.

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

- `contract.py::validate_fraction_alive_features(df)`
- `compute.py::compute_fraction_alive_features(frame_masks_df, auxiliary_masks_df, *, mask_decoder)`
- `compute.py::compute_fraction_alive_for_masks(embryo_mask, via_mask)`
- `via_masks.py::build_via_mask_lookup(auxiliary_masks_df)`
- `entrypoint.py::main()`

**Direct dependencies:** canonical embryo masks, auxiliary/VIA mask product, `snip_id`, `image_id`, and
feature-table alignment rules.

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

- `contract.py::validate_death_detection_qc(df)`
- `compute.py::compute_death_detection_flags(fraction_alive_df, snip_universe_df, *, thresholds)`
- `persistence.py::find_inflection_candidates(embryo_fraction_alive_df, *, thresholds)`
- `persistence.py::validate_death_persistence(embryo_fraction_alive_df, inflection_time, *, thresholds)`
- `alignment.py::align_death_flags_to_snip_universe(death_flags_df, snip_universe_df)`
- `entrypoint.py::main()`

**Direct dependencies:** `fraction_alive` feature rows, feature universe keyed by `snip_id`,
per-embryo temporal ordering, and QC defaults/thresholds.

**What it decides:** whether a snip should be flagged dead, plus the inferred death inflection time
and stage annotation for flagged rows.

**Classification:** QC, not feature extraction. It consumes `fraction_alive` and the feature universe,
then emits flags/annotations. It does not compute a new measured morphology feature.

**Depends on:**

- `fraction_alive` feature rows with `snip_id`, `embryo_id`, `time_int`, and `fraction_alive`;
- a feature universe table with exactly one row per `snip_id`;
- optional `predicted_stage_hpf` if death stage annotations are required;
- stable per-embryo temporal ordering.

**Input contract:**

- `fraction_alive` input columns:
  - `snip_id`
  - `embryo_id`
  - `time_int`
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
flags. They are required when `dead_flag` is true and nullable when `dead_flag` is false.

**Algorithm shape:**

- group by `embryo_id`;
- sort by `time_int`;
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

**Key functions:**

- `contract.py::validate_surface_area_qc(df)`
- `compute.py::compute_surface_area_qc_flags(mask_geometry_df, snip_universe_df, *, thresholds)`
- `compute.py::compute_surface_area_flag(area_um2, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** `mask_geometry` feature rows, feature universe keyed by `snip_id`, and
configured surface-area threshold/reference values.

**What it decides:** whether a snip has a suspicious area measurement for downstream analysis.

**Depends on:**

- `mask_geometry` rows with `snip_id` and `area_um2`;
- a feature universe table with exactly one row per `snip_id`;
- threshold/reference configuration owned by QC config, not hardcoded in the compute function.

**Output contract:**

- `snip_id`
- `sa_outlier_flag`

`sa_outlier_flag` is a required non-null boolean. Any threshold annotations must be named as
annotations, not flags.

**Done when:**

- fixture rows cover low, normal, high, missing, and duplicate `snip_id` cases;
- missing area values fail loud or map to a documented QC behavior;
- the output aligns one-to-one with the feature universe.

---

## Motion QC

**Product folder:** `src/data_pipeline/quality_control/motion_qc/`

**Key functions:**

- `contract.py::validate_motion_qc(df)`
- `compute.py::compute_motion_qc_flags(pose_kinematics_df, snip_universe_df, *, thresholds)`
- `compute.py::compute_motion_flag(track_or_snip_row, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** `pose_kinematics` feature rows, feature universe keyed by `snip_id`, frame
timing, and motion thresholds.

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

- `contract.py::validate_focus_qc(df)`
- `compute.py::compute_focus_qc_flags(focus_feature_df, snip_universe_df, *, thresholds)`
- `compute.py::compute_focus_flag(focus_metric, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** a named focus/input-quality feature table or image-quality contract,
feature universe keyed by `snip_id`, and configured thresholds.

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

## Consolidated QC

**Product folder:** `src/data_pipeline/quality_control/consolidated_qc/`

**Key functions:**

- `contract.py::validate_consolidated_qc(df)`
- `compute.py::consolidate_qc_tables(qc_tables, snip_universe_df, *, key="snip_id")`
- `compute.py::compute_use_snip_flag(qc_row, *, exclusion_flags)`
- `entrypoint.py::main()`

**Direct dependencies:** validated stage-specific QC tables, the feature universe keyed by `snip_id`,
and the canonical list of exclusion versus informational flags.

**What it decides:** the final QC table for downstream joins, including `use_snip`.

**Depends on:**

- at least one validated QC stage table;
- a one-row-per-`snip_id` feature universe;
- explicit `SNIP_EXCLUSION_FLAGS` / informational flag lists owned by the QC schema or contract.

**Output contract:**

- `snip_id`
- stage-specific `*_flag` columns included in the configured QC universe;
- `use_snip`

All `*_flag` columns, including `use_snip`, are required non-null booleans in the consolidated
output. Annotation columns may be nullable.

**Done when:**

- duplicate or missing `snip_id` rows fail loud before merge;
- `use_snip` is computed only from documented exclusion flags;
- consolidated QC can be joined with consolidated features without path or schema knowledge from
  individual QC stages.

---

## Not This World

- Detection, segmentation, tracking, and prompt adaptation live in detect/seg/track specs.
- Analysis-ready joins live after feature and QC consolidation.
- GPU SAM2 validation lives in segmentation Session C, not feature computation.
