# New-pipeline to core model integration audit

**Date:** 2026-08-13  
**Status:** Decision-support audit; no model or pipeline implementation changes made  
**Scope:** The shortest robust route from `src/data_pipeline` image/metadata outputs into the
loading, metric-pairing, and training machinery in `src/core`

## Executive conclusion

The right phase-one boundary is a **manifest-backed dataset in `src/core`**. It should build one
resolved training table from a small set of canonical, experiment-level pipeline products, then use
that table as the sole authority for image paths, metadata, splits, and row order.

The recommended sources are:

1. `snip_inventory` for image paths and stable identity;
2. `stage_predictions` for developmental age;
3. `snip_qc` for the operational inclusion verdict;
4. `plate_metadata` for genotype, perturbation, and other biological annotations; and
5. a separately configured, labeled metric-compatibility matrix plus an explicit rule or mapping
   that creates one canonical `metric_group` column.

This route does **not** require changing the VAE architecture, latent-space layout, reconstruction
loss, or the metric loss's batch interface. The phase-one model input should remain grayscale
`[1, 288, 128]`; the new pipeline's grayscale `576 x 256` PNGs should be deterministically resized
at the dataset boundary before training augmentation.

Three prerequisites are part of the minimum viable refactor, not optional cleanup:

- repair the current `src.core` package/entrypoint wiring so training imports the real core data
  configuration rather than the separate inference shim in `src/data`;
- replace `ImageFolder` and filename parsing with a manifest-indexed `Dataset`; and
- create split assignments at `physical_embryo_id` grain and persist IDs, not positional indices.

`analysis_ready` is not the recommended training boundary. It still has to be joined to
`snip_inventory` to locate pixels, it is optional, and its current build depends on running the
legacy embedding model and all feature branches. Making new-model training depend on old-model
inference would add unnecessary coupling.

## Recommended boundary at a glance

```text
pipeline output root + explicit experiment IDs
        |
        +-- snip_inventory.csv  ------- pixels, identity, validity, relative path
        +-- stage_predictions.csv ----- predicted_stage_hpf + status
        +-- snip_qc.parquet ----------- use_snip + exclusion reasons
        +-- plate_metadata.csv -------- well-level biology
        +-- metric mapping/matrix ----- explicit training semantics
        |
        v
core manifest builder + contract validation
        |
        +-- resolved training manifest (one row / snip_id)
        +-- stable split assignment (one split / physical_embryo_id)
        |
        v
ManifestDataset / MetricPairDataset
        |
        v
existing DatasetOutput keys -> existing model -> existing loss
```

## Audit basis

This audit followed the live execution paths rather than treating directory names as contracts.
It covered:

- core data configuration, transforms, datasets, split generation, Lightning dataloaders,
  model forwarding, metric loss, run initialization, and run metadata callbacks;
- pipeline identity constructors and validators, the artifact path registry, snip writing and path
  resolution, stage prediction, QC, plate metadata, legacy embedding input, and analysis-ready
  assembly; and
- representative artifacts already present under the configured pipeline output tree.

A live `20240813_24hpf` artifact was inspected as a format check:

- the merged inventory has 48 data rows and 22 columns;
- `processed_snip_path` is relative to the pipeline output root;
- a referenced file is a non-interlaced, 8-bit grayscale PNG, `256 x 576` in image-file `(W, H)`
  convention, corresponding to pipeline `(H, W) = (576, 256)`; and
- the matching experiment has merged stage-prediction CSV, snip-QC Parquet, plate-metadata CSV,
  and analysis-ready Parquet artifacts.

That live inventory predates the current writer contract: it lacks the newer
`source_micrometers_per_pixel` and `snip_micrometers_per_pixel` columns. This is a useful warning:
the adapter must validate the artifacts it actually receives and report schema/version gaps, not
assume every existing run was rebuilt after the latest code change.

No training run was launched during the audit. The existing working tree was clean before this
document was created.

## The pipeline contract relevant to training

### Canonical artifact locations

`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` is the authoritative path registry.
For an explicit `experiment_id`, the training-relevant merged artifacts resolve to:

| Source | Canonical path under `output_root` | Grain | Format |
|---|---|---:|---|
| Snip inventory | `object_extraction/{experiment}/snips/{experiment}_snip_inventory.csv` | snip | CSV |
| Stage predictions | `feature_extraction/{experiment}/stage_predictions/{experiment}_stage_predictions.csv` | snip | CSV |
| Snip QC | `quality_control/{experiment}/snip_qc/{experiment}_snip_qc.parquet` | snip | Parquet |
| Plate metadata | `acquisition/{experiment}/ingest_metadata/plate_metadata.csv` | well | CSV |
| Analysis ready | `analysis_ready/{experiment}/analysis_ready/{experiment}_analysis_ready.parquet` | snip | Parquet |

The model config should accept `pipeline_output_root` and an **explicit ordered list of experiment
IDs**. It should not recursively glob the output tree: pipeline outputs are experiment-grain, and
uncontrolled discovery would make the training cohort change when unrelated experiments finish.

### Identity is already better than the core's legacy inference

The pipeline carries a stable identity spine:

```text
experiment_id -> well_id -> physical_embryo_id -> embryo_id -> snip_id
```

It also carries `image_id`, `time_index`, and `channel_id` as frame provenance. The canonical ID
grammar distinguishes the physical animal from its channel and timepoint:

- `physical_embryo_id = {well_id}_e{one_based_index}`;
- `embryo_id = {physical_embryo_id}_{channel_id}`; and
- `snip_id = {embryo_id}_t{time_index}`.

The pipeline validates agreement between these columns. Core should consume the explicit
`physical_embryo_id` rather than recreate embryo identity with `snip_id[:-6]`. It should also treat
IDs as opaque strings instead of parsing them locally.

### Image contract

The current base pipeline configuration declares:

- canonical snip frame `(H, W) = (576, 256)`;
- grayscale processed PNG output;
- target physical scale `6.5 um/pixel`; and
- a legacy morphology-model doorway of `(H, W) = (288, 128)`, one grayscale channel.

`snip_inventory.processed_snip_path` is intentionally relative to `output_root` when possible.
`src/data_pipeline/object_extraction/snip_processing/io.py` already owns path resolution, and
`feature_extraction/legacy_embeddings/snip_source.py` demonstrates the correct manifest-based
loading seam. The new core adapter should reuse or factor that resolver rather than inventing a
second relative-path rule.

The legacy embedding branch also contains the correct deterministic tensor contract in
`feature_extraction/legacy_embeddings/transforms.py`: open the image, convert to `L`, resize using
`(H, W)`, then convert to a float32 tensor in `[0, 1]`. This is a strong regression oracle for the
new core transform.

Physical scale and intensity preprocessing remain scientific inputs, even when array shape is
correct. Current code says `6.5 um/pixel`, but older artifacts and run overlays may differ. The
manifest preflight should reject or explicitly partition mixed pixel scales and should record the
observed scale/provenance. A small raster-distribution report is also warranted before launching a
large refit, because a previous internal handoff documented scale/saturation differences between
some legacy and new-pipeline snips.

### Stage, QC, and plate metadata contracts

`stage_predictions` supplies `predicted_stage_hpf` plus
`stage_prediction_status`. Allowed unresolved states include missing start age and missing
temperature. Metric training therefore must require `status == "predicted"` and finite age. A
basic VAE may allow unstaged images if its cohort policy says so.

`snip_qc` supplies a non-null Boolean `use_snip` and a pipe-delimited `qc_fail_reasons`. Pipeline
validity and QC are distinct gates:

- `is_valid_snip` means the crop was successfully materialized;
- `use_snip` means the crop passed the configured biological/technical QC policy.

Both should be visible in the resolved manifest. The default training cohort should require both;
science-specific exceptions should be explicit named policies and saved with the run, not embedded
as silent loader logic.

The plate contract guarantees only `genotype`, `start_age_hpf`, `temperature`, and `medium` in
addition to experiment/well identity, and those biological fields may be null. Extra workbook
fields such as `chem_perturbation` are carried opportunistically. It does **not** guarantee the
legacy core column `short_pert_name`.

### Why `analysis_ready` should not be the primary boundary

There are several reasons not to point the training loader directly at it:

1. It does not own `processed_snip_path`, so the inventory is still required.
2. Its assembly is based on `snip_qc`, then joins all configured features, legacy latent
   embeddings, and plate metadata.
3. The pipeline labels it optional, even though the current default Snakemake target does build it.
4. Requiring legacy embeddings creates an unnecessary old-model prerequisite for training the next
   model.
5. It is Parquet, while the documented `morphseq-env` lacks a Parquet engine.

It can be supported later as a convenience metadata source, but it is not the lean or stable
training API. There is also documentation drift: `analysis_ready/__init__.py` still calls the
subsystem an unwired stub, while `assemble.py`, the task entrypoint, registry, and Snakefile are now
wired. Consumers should trust the actual product contracts and validators, not that stale module
docstring.

## What the current core machinery assumes

### Data root and ordering

`src/core/data/dataset_utils.py::make_seq_key` assumes a legacy training bundle:

```text
root/
  images/<ImageFolder class>/*.jpg
  metadata/embryo_metadata_df_train.csv
  metadata/age_key.csv
  metadata/metric_key.csv
  metadata/perturbation_train_key.csv   # optional
```

It scans exactly one class-directory level and only `*.jpg`, strips the last four filename
characters to recover `snip_id`, and inner-joins that list to the metadata CSV. This is incompatible
with pipeline PNGs, nested per-well/per-embryo paths, and relative manifest paths.

Both `BasicDataset` and `NTXentDataset` inherit `torchvision.datasets.ImageFolder`. In the metric
case, image lookup uses `ImageFolder.samples[index]` while age, embryo, perturbation, and split
vectors use `seq_key[index]`. There is no assertion that these independently produced orderings
match. A silent row-order mismatch would attach one embryo's metric metadata to another embryo's
pixels, which is more serious than a loader crash.

### Metadata translation

The core expects `short_pert_name`, `experiment_date`, and either inferred or predicted stage in
`embryo_metadata_df_train.csv`. It creates integer perturbation and embryo IDs from the observed
strings. The separate merge of `age_key.inferred_stage_hpf_reg` is not used to replace
`stage_hpf`, so that file is currently required without affecting pair selection.

The pipeline supplies a stronger identity model and a different metadata vocabulary. The bridge
must translate into a small internal schema rather than make the pipeline emit legacy CSVs.

### Split generation

The split function says it operates at embryo level, but it concatenates each embryo's frames and
then cuts the resulting list at frame-count thresholds. An embryo spanning a boundary can therefore
be present in two splits; the source comment explicitly allows this. Time-window constraints can
also put out-of-window frames from an embryo into test while other frames from the same embryo go to
train/eval. `test_perturbations` additionally indexes a DataFrame as a NumPy array and references a
misspelled column, so that branch is broken.

Only positional integer arrays are persisted to `split_indices.pkl`. Those indices change meaning
if manifest ordering changes, images are added, or filters change.

For the new data, `physical_embryo_id` is the correct default grouping unit. All rows for one
physical embryo must receive one split unless a deliberately named temporal-extrapolation protocol
is selected. Split assignments should be materialized by ID and included as a manifest column.

### Metric pairing and compatibility matrix

`NTXentDataset` returns the batch structure the current metric model/loss expects:

- `data`: two views, `[2, C, H, W]` per dataset item;
- `self_stats`: embryo ID, age, perturbation ID for the anchor; and
- `other_stats`: the same for the selected positive partner.

It chooses a same-embryo neighbor or a different embryo within `time_window` whose perturbation
matrix entry is `1`. The metric loss later constructs batch-wide positives/exclusions with matrix
values `1`, `0`, and `-1`. It applies the contrastive term only to biological latent dimensions.

The old production `metric_key.csv` inspected during this audit is a labeled `48 x 48` matrix. Its
row/column labels agree, its values are limited to `-1/0/1`, it is symmetric, and its diagonal is
`1`. The current loader does not validate those properties or report missing perturbation labels.
It reorders the matrix through generated integer IDs, which can produce invalid ordering when a
selected label is absent.

The loss and sampler also use slightly different age criteria: pair sampling uses
`delta <= time_window`, while the batch-wide target matrix uses `delta <= time_window + 1.5`.
That is existing model semantics, not an image-format issue. It should be documented and tested
before anyone decides whether to change it, but it need not block phase-one loading.

### Transform and model shape

`basic_transform` resizes only when `target_size` is supplied. More importantly,
`contrastive_transform(target_size=...)` accepts the argument but ignores it. New `576 x 256`
images would therefore reach a model configured for `[1, 288, 128]`.

The model architecture and all current Hydra model configs expect `[1, 288, 128]`. The
reconstruction loss also hardcodes its pixel normalization to `128 * 288`. Keeping the phase-one
model doorway at `288 x 128` avoids an unrelated architecture/loss-scale change.

### Training and reproducibility wiring

The Lightning wrapper creates a new full dataset for train and validation and applies
`SubsetRandomSampler` to positional indices. It has no test dataloader, and a non-train item in the
metric dataset always uses the eval candidate mask. The interaction of the custom sampler with
Lightning's automatic distributed sampling is not asserted or tested.

Run metadata currently saves only positional split indices. It does not save the selected snip IDs,
resolved source manifest, QC cohort, metric label mapping/matrix, or input transform contract.

The trainer is also hardcoded to GPU, FP16, all devices, and DDP. This is not a data-format blocker,
but trainer hardware should become configurable enough to run a one-batch CPU integration test.

## Blocking package and entrypoint drift

Before any new dataset code can be trusted, core needs one authoritative import route.

`src/core/models/model_configs.py` imports:

```python
from data.dataset_configs import BaseDataConfig, NTXentDataConfig
```

That is not `src.core.data.dataset_configs`. From the repository root it fails because `data` is
not a top-level package. With `src` placed on `PYTHONPATH`, it resolves to
`src/data/dataset_configs.py`, which describes itself as a minimal downstream-analysis shim;
its `NTXentDataConfig` is only a placeholder and cannot train the metric model.

The run surfaces are similarly stale:

- `src/core/run/training.py` uses an absolute Hydra path `/src/core/hydra_configs`;
- `training_cluster.py` points at a cluster path ending in `src/hydra_configs`, while the live
  configs are under `src/core/hydra_configs`; and
- scripts under `src/core/run` invoke modules such as `src.run.training_cluster`, not
  `src.core.run.training_cluster`.

This should be phase 0: make `src.core` imports explicit, select one supported entrypoint, resolve
Hydra configuration relative to that package, and add an import/config smoke test. Otherwise a
successful loader implementation can remain unreachable or, worse, bind to the wrong data class.

## Minimum robust implementation route

### Phase 0 — restore one executable core stack

1. Change core model configuration imports to `src.core.data...` (or consistently package the repo
   as `core...`; do not mix styles).
2. Choose one training entrypoint and make its Hydra path package-relative.
3. Update or retire stale run scripts so they invoke that entrypoint.
4. Add a smoke test that imports the model config, composes the metric Hydra config, and verifies
   the resulting data config is the training implementation.

This phase should not alter model behavior.

### Phase 1A — introduce a core-owned pipeline manifest adapter

A focused module such as `src/core/data/pipeline_manifest.py` should:

1. accept `pipeline_output_root`, explicit `experiment_ids`, channel selection, QC policy, and
   metric-label configuration;
2. resolve source files through the pipeline path registry or a thin centralized path adapter;
3. read each experiment's merged inventory as the base table;
4. validate the inventory contract actually present, normalize Booleans safely, require
   `is_valid_snip`, and resolve `processed_snip_path` against `output_root`;
5. join stage and QC one-to-one on `snip_id` with coverage reports;
6. join plate metadata many-to-one on `well_id`;
7. explicitly filter channel(s), validity, QC, and metric-stage eligibility;
8. materialize `stage_hpf` and `metric_group` as the only model-facing biological fields;
9. create stable, group-disjoint split assignments by `physical_embryo_id`; and
10. return and save the final table plus a structured validation summary.

Imports from `src/data_pipeline` should be limited to contract/path/path-resolution helpers in this
adapter. Datasets and training code should consume the resolved table, not reach back into pipeline
internals.

If core and pipeline environments remain separated, materialize the resolved manifest as CSV in an
environment that can read `snip_qc.parquet`, then train from that CSV. The current documented
environment state is:

- `morphseq-env`: no PyArrow;
- `vae-env-cluster`: PyArrow available.

The adapter should fail at preflight with a specific Parquet-engine message. Silently omitting QC
because its reader is unavailable is not an acceptable default.

### Phase 1B — define one internal training-manifest schema

Recommended columns are:

| Column | Requirement | Purpose |
|---|---|---|
| `snip_id` | always | unique sample and provenance key |
| `processed_snip_path` | always | portable path relative to pipeline root |
| `experiment_id`, `well_id` | always | source and plate joins/holdouts |
| `physical_embryo_id` | always | leakage-safe split and sequential pairing group |
| `embryo_id`, `channel_id`, `time_index` | always | channel/time provenance |
| `is_valid_snip` | always | materialization gate |
| `use_snip`, `qc_fail_reasons` | default cohort | auditable QC gate |
| `stage_hpf`, `stage_prediction_status` | metric mode | temporal pairing constraint |
| `metric_group` | metric mode | labeled compatibility-matrix lookup |
| `split` | always | `train`, `eval`, or `test`, assigned by group |
| pixel-scale/shape provenance | when available | prevent silent mixed input regimes |

The table's row order becomes the dataset order. No second glob, ImageFolder index, filename slice,
or parallel NumPy vector should exist.

### Phase 1C — replace ImageFolder with manifest datasets

Implement a plain `torch.utils.data.Dataset` that indexes manifest rows and opens exactly the path
in that row. A metric variant may share the same table and pairing index. Preserve the existing
`DatasetOutput` keys so the model and loss remain unchanged during the compatibility phase.

For train/eval/test, prefer split-specific dataset views over full-dataset positional samplers.
That makes candidate selection naturally split-local and allows Lightning to apply a distributed
sampler safely. Add a true test dataloader or explicitly declare test inference out of phase-one
scope; do not let test rows silently use eval pairing state.

The current pair search builds full-length Boolean arrays for every item, making selection `O(N)`
per sample. It may be acceptable for an initial smoke, but large multi-experiment training should
pre-index candidates by split, physical embryo, metric group, and age bins. Whatever implementation
is chosen must fail with the anchor ID and selection policy when no legal positive exists.

### Phase 1D — make image normalization explicit

Use a common deterministic decoder:

```text
open -> convert("L") -> resize to model (H, W) -> float32 tensor in [0, 1]
```

The basic dataset stops there. The metric-training dataset applies its random affine, flips, and
brightness augmentation around that explicit model-size contract. Fix the current contrastive
transform so `target_size` is not ignored.

For the compatibility phase, set model input to `(1, 288, 128)` and regression-test the
deterministic tensor against the pipeline legacy-embedding transform on the same PNG. Changing the
model to native `576 x 256` should be a later model-design decision because it affects architecture,
memory, loss normalization, and checkpoint compatibility.

### Phase 1E — make metric semantics labeled and validated

The pipeline does not guarantee `short_pert_name`, so core must not infer it from filenames or
silently concatenate plate fields. Configuration should require one of:

- a named plate column to use as `metric_group`; or
- an explicit mapping table that assigns `metric_group` at well or sample grain.

The labeled compatibility matrix should remain a separate, versioned science artifact. Validate
before constructing integer IDs:

- square matrix, unique labels, identical row/column label sets;
- allowed values exactly `-1`, `0`, and `1`;
- symmetry and diagonal policy (the current matrix is symmetric with diagonal `1`);
- every selected non-null `metric_group` is covered; and
- every anchor has at least one legal positive under its split/age policy.

Extra matrix labels not present in a run may be allowed with a warning. Missing selected labels
must fail. Save the exact matrix and label-to-integer map with the run.

### Phase 1F — make runs reproducible by identity

Replace or extend `split_indices.pkl` with:

- `training_manifest.csv` (or a portable equivalent);
- `split_assignments.csv` keyed by `physical_embryo_id` and/or `snip_id`;
- the selected metric matrix and metric-group mapping;
- source artifact paths, sizes/hashes or validated sentinels, experiment list, and pipeline root
  alias;
- cohort/filter counts and reasons; and
- input decode/resize/augmentation configuration.

Absolute paths may be useful at runtime, but the saved source of truth should retain pipeline-root
relative paths so `/net/...` and `/media/...` mounts do not create different datasets.

## Validation gates and tests

The first implementation should not be considered complete until the following pass.

### Contract/preflight tests

- Required artifact exists for every configured experiment, with a clear per-experiment error.
- `snip_id` is unique in the base inventory and in every one-row-per-snip source.
- Identity spine columns are non-null and internally consistent.
- `processed_snip_path` resolves under the configured root (or is an explicitly permitted absolute
  path), exists, and opens successfully.
- Channel filtering is explicit; morphology training does not silently mix BF with other channels.
- Boolean parsing distinguishes the strings `"False"` and `"True"` correctly.
- Stage/QC joins report missing and extra IDs rather than hiding them through an inner join.
- Every selected well has exactly one plate row.
- Metric-mode rows have finite stage and a covered, non-null metric group.
- All post-filter counts are reported by experiment and exclusion reason.

### Dataset and split tests

- Manifest row `i`, loaded pixel path, and returned metadata all refer to the same `snip_id`.
- Basic tensors are `[1, 288, 128]`, float32, finite, and in `[0, 1]`.
- Deterministic transform output matches the pipeline legacy-embedding transform.
- Metric items are `[2, 1, 288, 128]` and retain the current `self_stats`/`other_stats` contract.
- No `physical_embryo_id` appears in more than one split.
- Positive pairs never cross splits and satisfy the declared age/matrix policy.
- Split assignments are identical when manifest row order changes.
- Missing positive candidates produce an actionable error, not a NumPy sampling exception.

### End-to-end acceptance tests

- A tiny synthetic pipeline output tree builds a manifest and both basic and metric dataloaders.
- One basic VAE train/validation step runs on CPU.
- One metric VAE train/validation step runs on CPU and produces finite reconstruction, KL, and
  metric terms.
- A real small experiment such as `20240813_24hpf` passes preflight and yields expected path,
  channel, shape, and stage coverage summaries.
- A run snapshot can reconstruct the exact selected snip-ID set without relying on old positional
  indices.

## Alternatives considered

### Copy or symlink PNGs into a legacy ImageFolder tree

This produces the smallest superficial diff but is not robust. It duplicates or mirrors pipeline
layout, requires synthetic class folders and legacy CSVs, retains independent ordering, and loses
the pipeline's identity/QC/path contracts. It should be rejected.

### Teach the current `make_seq_key` to glob PNGs recursively

Adding `**/*.png` would address only file discovery. It would retain filename parsing, parallel
ordering, positional splits, missing QC, and the missing `short_pert_name` problem. It is not a
sufficient bridge.

### Join `analysis_ready` to the snip inventory

This is viable for an exploratory loader and requires fewer joins in core, but it imports optional
features and legacy-model inference into the training dependency chain and still requires Parquet.
It is not recommended as the production phase-one boundary.

### Add a first-class pipeline `training_manifest` product

This is a clean long-term option, especially if multiple consumers need the same cohort. It is more
cross-stack work now and would force training-specific concepts such as metric groups and splits
into the data pipeline before their contract is settled. Start with the core-owned adapter; promote
its stable, model-agnostic portion into a pipeline product later if reuse justifies it.

## Suggested file-level change surface

The implementation can remain narrow:

- `src/core/models/model_configs.py`: correct the data-config import.
- `src/core/data/pipeline_manifest.py` (new): source resolution, joins, validation, cohort, split.
- `src/core/data/dataset_configs.py`: add/select a pipeline-manifest config without keeping the
  legacy root assumptions on the new path.
- `src/core/data/dataset_classes.py`: add manifest-backed basic and metric datasets.
- `src/core/data/data_transforms.py`: enforce resize/model tensor contract for both modes.
- `src/core/lightning/pl_wrappers.py`: use split-specific datasets/loaders and add test behavior.
- `src/core/lightning/callbacks.py`: save identity-based manifest/split provenance.
- `src/core/run/{training.py,training_cluster.py,run_utils.py}` and run scripts: repair entrypoint
  paths and make trainer hardware minimally configurable for tests.
- `src/core/hydra_configs/model/*.yaml`: add pipeline root, experiment cohort, channel, QC, metric
  mapping, matrix, input size, and split policy.
- `tests/core/...` (new): contract, transform, split, pair, and integration coverage.

The phase-one change should avoid modifying:

- encoder/decoder architecture;
- latent biological/nuisance partitioning;
- reconstruction/KL/GAN/LPIPS/NT-Xent formulas;
- the pipeline's canonical artifact schemas; or
- the pipeline's legacy embedding/analysis-ready branches.

## Decisions needed before implementation

Only three science/policy choices materially affect the bridge design:

1. **Metric group source.** Which plate column or curated mapping defines the compatibility-matrix
   label for new experiments? Recommendation: require an explicit `metric_group` mapping and do not
   default to `genotype` alone.
2. **QC cohort policy.** Should phase-one training use strict `is_valid_snip & use_snip`, or a named
   project-specific override? Recommendation: strict by default, with any override versioned and
   saved in the manifest.
3. **Holdout policy.** Is the initial evaluation a random physical-embryo group split, or should
   whole experiments/perturbations be held out? Recommendation: implement group-disjoint physical
   embryo splits first, while allowing explicit experiment-level test holds without changing the
   dataset layer.

Everything else can proceed without revisiting model design. In particular, the data bridge can
keep `288 x 128` model inputs now and leave native-resolution training for the later morphology
model plan.

## Recommended phase-one completion definition

Phase one is complete when an explicit list of new-pipeline experiments can be validated, converted
to an identity-stable manifest, split without physical-embryo leakage, loaded as basic or metric
batches of the expected shape, and used for finite train/validation steps by the existing models;
the exact IDs, filters, metric semantics, and source artifacts must be recoverable from the run
directory.

