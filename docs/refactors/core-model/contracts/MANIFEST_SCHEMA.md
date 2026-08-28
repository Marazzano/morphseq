# Core model data boundary — observation and asset tables

**Version:** 2.0
**Ratified:** 2026-08-27
**Status:** Binding interface for Track A. If live pipeline writers disagree with this contract,
report the discrepancy and update this document through the lead agent; do not improvise a private
schema inside a consumer.

This contract supersedes version 1.0's one-row-per-`snip_id` manifest. The live snip inventory is
product-aware: one embryo-time may have several rendered products, and uniqueness is
`(snip_id, snip_product_key)` today
(`src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/snip_identity_contract.py:198-223`).
The model boundary therefore separates biological observations from raster assets.

## 1. Non-negotiable identity model

### Observation table

One row per `snip_id`.

`snip_id` identifies an embryo-time observation. It is not a file identifier and must not be parsed,
sliced, or rebuilt in core. Every row carries the explicit parent columns `physical_embryo_id`,
`embryo_id`, `well_id`, and `experiment_id`.

### Asset table

One row per:

```text
(snip_id, snip_product_key, z_index)
```

`z_index` is null for projection assets and a non-negative integer for an individual z-plane asset.
Null is a real member of the compound key for projection rows: two rows with the same `snip_id`,
`snip_product_key`, and null `z_index` are duplicates. Product or z semantics are never encoded into
`snip_id`.

The current frame grammar already represents z planes explicitly with `z_index` and a z-aware
`image_id` (`src/data_pipeline/shared/identifiers/constructors.py:78-100`). The current snip renderer
does not yet expose a complete z-plane asset seam: its source lookup constructs a z-less image ID
(`src/data_pipeline/object_extraction/snip_processing/entrypoints/run_snip_processing.py:297-322`).
Track A must preserve the z-capable table shape but must not claim end-to-end z rendering works.
Vanilla mode uses projection assets only.

### Relationship

```text
ObservationTable 1 ---- N AssetTable
                    snip_id
```

Every asset must match exactly one observation. An observation may have zero assets before product
selection, but a selected vanilla observation must resolve to exactly one requested asset. Orphan
assets, duplicate observation IDs, conflicting parent identity, zero matches, and multiple selected
matches are named errors.

## 2. Source artifacts

The adapter resolves sources for an explicit ordered experiment list. It never discovers experiments
or pixels by globbing the output tree.

| Source | Grain | Used for |
|---|---|---|
| product-aware `snip_inventory` | asset candidate | identity spine, pixel/mask paths, product, transform, scale, validity |
| `frame_inventory` | frame product/plane | canonical `elapsed_time_s`, source timing and product/plane validation |
| `stage_predictions` | observation | predicted stage, stage status, method version |
| `snip_qc` | observation under the current default-BF QC pipeline | verdict, individual flags, applicability |
| `plate_metadata` | well | incubation temperature and biological metadata |
| collection provenance, when applicable | source acquisition | source-specific start-age provenance used by the stage producer |

`elapsed_time_s` is a required frame-inventory column
(`src/data_pipeline/acquisition/image_materialization/frame_inventory_contract.py:44-59`). The adapter
must join it explicitly; the old manifest contract omitted frame inventory and therefore could not
make elapsed time available to the loader. For one `(well_id, time_index)`, all relevant
channel/product/plane rows must agree on elapsed time or the adapter fails with the offending rows.

The stage producer accepts `elapsed_time_s`, `experiment_time_s`, then `time_s`, in that order
(`src/data_pipeline/feature_extraction/stage_predictions/compute.py:34-45`). Core standardizes only
the converged `elapsed_time_s`; it does not expose a scope-specific fallback under the same name.

## 3. Observation table schema

### Required identity and time columns

| Column | Requirement | Meaning |
|---|---|---|
| `snip_id` | non-null, unique | embryo-time observation identity |
| `embryo_id` | non-null | explicit parent identity |
| `physical_embryo_id` | non-null | split/group identity |
| `experiment_id` | non-null | explicit experiment authority |
| `well_id` | non-null | plate join key |
| `image_id` | non-null | segmentation/origin frame provenance; not necessarily every asset's source image ID |
| `time_index` | non-null integer | acquisition sequence index |
| `channel_id` | non-null | identity/segmentation channel carried by the snip spine; asset source channel comes from the product key |
| `elapsed_time_s` | finite, non-negative when available | converged relative acquisition time |
| `elapsed_time_status` | non-null | `available` or a named source error/absence state |

The adapter validates identity agreement through the pipeline contract helper. Downstream core code
uses explicit columns and never parses identifiers.

### Biological and acquisition covariates

| Column | Requirement | Source/handling |
|---|---|---|
| `incubation_temperature_c` | carry with status | `plate_metadata.temperature`; the model-facing name avoids collision with contrastive-loss temperature |
| `temperature_status` | non-null | `available`, `missing`, or another named source state |
| `start_age_hpf` | carry when correctly resolvable | per-well plate value for single acquisitions; source-aware value for collections |
| `start_age_source` | non-null | `plate_metadata`, `collection_provenance`, or `unavailable` |
| `genotype` | carry | never normalized in core |
| `medium` | carry | nullable only if source is null |
| `strain` | carry when present | preserve source spelling |
| `chem_perturbation` | carry when present | preserve source spelling |

Missing covariates remain missing with an explicit status; the adapter does not derive, default, or
impute them. A run policy may require them and fail with the affected observation IDs.

### Stage columns

| Column | Requirement | Meaning |
|---|---|---|
| `predicted_stage_hpf` | carry, nullable | pipeline stage value |
| `stage_status` | non-null, three-state-capable | source status, `unavailable`, or a named missing-input status |
| `stage_model_version` | carry | method/formula provenance |

Pipeline `predicted_stage_hpf` is a temperature-adjusted nominal clock stage, not the legacy
morphology-inferred age (`docs/refactors/core-model/reports/STUDY_stage_lineage.md:9-35`). Track A may
carry or spoof stage for plumbing but may not claim legacy semantic equivalence.

### Observation-level QC columns

The current `snip_qc` product is joined one-to-one on `snip_id` and describes the current default-BF
observation path. Carry:

- `use_snip`
- `qc_fail_reasons`
- every individual flag present, including `sa_outlier_flag`
- every applicability column present
- `qc_status`
- `qc_schema_version` or equivalent schema provenance

`qc_status` distinguishes at least `evaluated`, `no_artifact`, and `row_missing`. Absence is not
failure. A per-flag policy applied to a schema that lacks the requested flag fails by experiment and
flag name; it never falls back silently to `use_snip`. The audited old corpus had 28
inventory-bearing experiments with no QC artifact
(`docs/refactors/core-model/reports/PIPELINE_RECON.md:893-905`); regenerated counts are not assumed to
match that measurement.

Current observation QC must not be relabeled or blindly copied as asset-specific QC. A future
product/plane QC table belongs on the asset grain and must declare its scope.

## 4. Asset table schema

### Required key and identity columns

| Column | Requirement | Meaning |
|---|---|---|
| `snip_id` | non-null FK | parent observation |
| `snip_product_key` | non-null | source image product plus snip recipe |
| `z_index` | nullable integer | null projection; non-negative z plane |
| `processed_snip_path` | non-null for valid assets | authoritative pixel path |
| `is_valid_snip` | non-null | materialization validity, not QC acceptance |
| `error_message` | carry | rendering/decode provenance |

### Product, geometry, and replay columns

Carry every available explicit column from the current product-aware inventory, including:

- `source_image_product_key`
- `image_path`
- `embryo_mask_snip_path`
- `snip_transform_id`
- `output_grid_id`
- `source_micrometers_per_pixel`
- `snip_micrometers_per_pixel`
- source/output shape
- crop bounds in pixels and micrometers
- orientation policy/source
- centering
- image and mask interpolation
- realized scale
- `pixel_dtype`
- `resolved_transform_chain_json`

The pipeline declares these product and construction fields in
`src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/snip_identity_contract.py:241-366`.
The adapter reports absent optional fields by source schema; it never recreates them from paths or
identifiers.

### Asset-level QC

Version 2 defines the location but does not invent a producer. If product- or plane-specific QC is
available, it joins on the full asset key and uses names distinct from observation QC, such as
`asset_qc_status`. Until then these fields are absent, not copied from observation QC.

## 5. Deterministic order and resolved sample view

The two source tables have deterministic order:

1. observation order follows the explicit experiment order and stable source-row order;
2. asset order follows observation order, then configured product order, then ascending `z_index`
   with projection nulls in a documented fixed position.

No consumer re-sorts, globs, or creates a parallel positional metadata array.

Datasets consume a derived **resolved sample view**, not a third independent authority. A selector
maps each observation to asset row(s):

- vanilla selector: exactly one asset row;
- future single-plane selector: exactly one configured/sampled z row;
- future multi-plane selector: an ordered list of asset-row indices for one observation.

The resolved sample-view order is dataset order. Every returned batch carries the observation and
asset keys used to create it.

## 6. Vanilla Track A selector

The first pipeline-backed run is explicit and narrow:

- one configured BF projection `snip_product_key` (the current compatibility default is
  `BF__projection__focus_stack__clahe_blend`, declared at
  `src/data_pipeline/object_extraction/snip_processing/snip_product_keys.py:35-38`);
- `z_index` must be null;
- `is_valid_snip` required;
- a named temporary or final QC policy;
- stage not required in basic mode;
- explicit experiment list;
- explicit train/eval/test split configuration.

Zero or multiple matching assets for a selected observation is a hard error naming the
`snip_id` and available asset keys.

## 7. Cohort policies

A cohort policy is configuration, not code hidden in the adapter. Every filter reports rows in,
rows out, and reason per experiment.

At minimum configure:

- ordered `experiment_ids`;
- allowed `snip_product_key` values and z-selection mode;
- validity requirement;
- named/versioned QC policy;
- stage requirement and accepted statuses;
- covariate requirements;
- explicit test experiments;
- hash split ratios for the remaining embryos.

The final science cohort is chosen after regenerated pixel-dependent QC is available (D32). Track A
uses a clearly named temporary smoke-test policy and records its exact selected IDs. Old-corpus
counts such as 176,466 remain historical regression evidence, not an acceptance target for the
regenerated corpus.

## 8. Splits

Splits are assigned and persisted by `physical_embryo_id`. Explicit `test_experiments` go wholly to
test; the remaining embryos use deterministic `blake2b` content hashing. Adding rows, products,
planes, or experiments must not move an existing embryo between splits. All sibling assets and all
timepoints of one physical embryo stay in the same split.

Assert:

- no physical embryo crosses splits;
- required splits are non-empty;
- achieved ratios are within configured tolerance on the unpinned pool;
- assignments are stable under row reordering and cohort growth.

## 9. Metric plumbing stub

Basic mode does not require stage or metric grouping. Track A's metric smoke uses an unmistakably
test-only policy, for example `test_only_single_group`, and either disables the scientific age gate
or uses a declared test window. It preserves the existing `self_stats`/`other_stats` batch contract
but cannot be selected by production/science presets.

The real metric mapping, relation function, pair policy, and stage windows are Track C deliverables.
Core never guesses mapping semantics from genotype or plate strings.

## 10. Dataset batch contract

Every vanilla item returns at least:

- image tensor;
- `snip_id`;
- `physical_embryo_id`;
- `snip_product_key`;
- `z_index`;
- split;
- `incubation_temperature_c` and its status;
- `elapsed_time_s` and its status;
- `time_index`;
- stage value/status/version;
- source asset path or stable asset-row reference for diagnostics.

The model may ignore metadata, but the loader must not discard it. Metric items additionally
preserve `self_stats` and `other_stats` during compatibility work.

## 11. Inference and failure behavior

The adapter is usable outside training. QC filtering, stage requirements, split assignment, and
metric mapping are independently switchable. Missing artifacts, unreadable Parquet, ambiguous joins,
uncovered mappings, missing selected assets, and decode failures name the experiment and relevant
identity.

Decode failures follow D25: report and resample within the same split, record counts and IDs, and
abort only above the configured threshold. No failure silently changes product, split, QC policy, or
metadata.

## 12. Run provenance

Every accepted run writes locally, even when W&B is disabled:

- fully resolved configuration;
- ordered observation IDs and selected asset keys;
- split assignments keyed by `physical_embryo_id`;
- exact experiment, product, z, QC, stage, and covariate policies;
- metric-group map or explicit test-stub declaration;
- source artifact paths, sizes, mtimes, row counts, and hashes;
- per-filter cohort report;
- adapter revision.

Image contents are not hashed. Source artifacts and selected identity lists are sufficient for this
provenance contract (D19).

## 13. Required contract tests

Before Track A acceptance, tests cover:

1. observation uniqueness and asset compound-key uniqueness, including null projection `z_index`;
2. two products for one `snip_id` are valid;
3. several ordered z planes for one `snip_id` are valid in a synthetic asset table;
4. orphan assets, conflicting parent identities, and duplicate keys fail by ID;
5. product selection returns exactly one vanilla asset or fails by ID;
6. temperature and elapsed time reach the dataset item unchanged;
7. frame rows disagreeing on elapsed time fail by well/time/product;
8. three-state QC and stage absence remain distinct from failure;
9. string booleans are parsed safely;
10. split group-disjointness and stability under growth;
11. inference switches do not accidentally apply training filters;
12. provenance reconstructs the selected observation and asset keys.
