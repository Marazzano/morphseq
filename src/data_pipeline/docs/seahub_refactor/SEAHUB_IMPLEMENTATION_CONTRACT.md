# SeaHub implementation contract

This document records the owner-reviewed decisions that supersede conflicts in
`SEAHUB_INTEGRATION_DESIGN.md` and `SEAHUB_INTEGRATION_WORKPLAN.md`.

## Identity and execution grouping

- SeaHub has first-class modality support. It is not represented as a native microscope.
- One detected embryo is one canonical well. This is non-negotiable.
- The 96-well grid is used only to satisfy the existing identity grammar and batch work. The
  operational experiment is named
  `{operational_date}_seahub_{source_experiment}_shard{index:03d}`.
- A shard has no plate-level biological meaning. All biology is per well.
- `source_embryo_id = seahub_{source_fov_id}_p{position:02d}` is the stable source identity and
  remains valid if operational packing changes.
- The front end does not pre-mint `physical_embryo_id`; detection/segmentation owns that identity.
- Use operational date `20260723`. Preserve the actual collection date from metadata when present;
  it is nullable and never gates inclusion.

## Reconciliation and inclusion

- Remediate only existing `unmatched_stage` and `unmatched_condition` rows.
- Structured `stage_collected` and `stage_addition` fields take precedence. Collection-name stages
  are validation/fallback values.
- Apply the owner-supplied morphological crosswalk. In particular, `12s`/`12-somite = 15.0 hpf`.
- Normalize condition separators/order and recognize forms such as `6hpf-treatment`.
- When an unambiguous collection-name `GENE#`/`CHEM#` prefix disagrees with `expt`, use the prefix
  for matching and preserve original/effective/correction audit fields.
- Pass `unmatched_stage` and `unmatched_condition` through after remediation. Preserve the status,
  nullable values, and `reconciliation_failure_passed_through`.
- Exclude wrong roles, abandoned/not-used paths, explicit not-collected/not-sequenced records,
  unreadable records, missing experiments, duplicate/ambiguous metadata matches, and contract-invalid
  rows.
- Detection must produce exactly positions 1 through 8. Otherwise exclude the complete FOV; never
  emit a partial set of wells.
- Every exclusion is written to `dropped_fovs.csv` or `detection_failures.csv` with a reason.

## Image and modality contract

- Crop using the reviewed GroundingDINO boxes, convert to grayscale, apply canonical polarity, and
  center-pad without downscaling.
- Compute one corpus-wide canvas from maximum accepted box width/height, rounded up to 32 pixels.
- Route the frame mechanically through `BF__projection__focus_stack` with `time_index=0` and null
  `z_index`.
- Declare acquisition truth independently:

  - `source_scope = seahub`
  - `image_kind = single_z`
  - `z_position = null`
  - `calibration_status = placeholder`

- Infer one provisional source scale per `source_fov_id`, and broadcast that exact value to all
  eight embryos cropped from the FOV. Never infer scale independently per embryo; doing so would
  normalize away biological size phenotypes.
- The packaged `seahub_scale_reference_v1.csv` is the canonical nine-stage prior/reference table.
  For a FOV with at least four valid, finite, positive masks, compute:

  ```text
  raw_um_per_px = sqrt(strict_WT_stage_p50_area_um2 / median_valid_mask_area_px)
  regularized_um_per_px = stage_prior_um_per_px
                          + 0.50 * (raw_um_per_px - stage_prior_um_per_px)
  ```

- Constrain the regularized estimate first to +/-20% of its stage prior and then to the absolute
  range 3.5--9.0 um/px. With fewer than four valid masks, use the stage prior alone. For an
  unresolved stage or a numeric stage absent from the v1 table, use 7.8 um/px and record the
  corresponding global-fallback status. Do not silently interpolate or drop these pass-through
  FOVs.
- This is morphology-based inference, not physical metrology. The shared acquisition field remains
  `calibration_status = placeholder`; `scale_estimation_status` distinguishes mask-regularized,
  stage-prior, and 7.8-global-fallback cases.
- Persist the complete FOV audit in `integration/fov_scale_calibration.csv`: source/stage identity,
  valid-mask support and median area, strict-WT reference version/source/count/p50, stage prior,
  raw and regularized estimates, regularization weight, relative and absolute bounds, clipping
  flags, final scale, estimation status/method, and fallback issue. Carry the applicable provenance
  into well, frame, plate, and downstream snip products.
- Standardize canonical processed snips to 6.5 um/px. Record both
  `source_micrometers_per_pixel` and `snip_micrometers_per_pixel` in the snip inventory. The legacy
  VAE resizes the 576x256 canonical snip to 288x128 (2x linear reduction), so its intended effective
  input scale is 13 um/px. A modest source-to-6.5 upsample is permitted; the source scale and
  resampling provenance must remain explicit.

## QC semantics

- Focus QC remains computable but is diagnostic-only for `single_z`:
  `focus_qc_applicability = diagnostic_only`.
- Motion-blur QC requires adjacent z planes and is not applicable:
  metrics are null, `motion_blur_flag = false`, and
  `motion_blur_qc_applicability = not_applicable`.
- Surface-area QC remains computable for a resolved `single_z` snip, but is diagnostic-only:
  `surface_area_qc_applicability = diagnostic_only`. An unresolved stage produces a nullable
  `predicted_stage_hpf` with `stage_prediction_status=missing_start_age_hpf`; because no
  stage-binned comparison can be made, it emits `sa_outlier_flag=false` with
  `surface_area_qc_applicability=not_applicable`.
- Death evidence is retained for audit on `single_z`, but both viability and persistence flags are
  diagnostic-only: `death_detection_qc_applicability = diagnostic_only`. A single snapshot must
  never be excluded solely by a temporal death heuristic.
- `snip_qc` only adds a flag to `qc_fail_reasons` when its applicability is `exclusion`.
- Flags and applicability columns are carried through `snip_qc` to `analysis_ready`.

## Drop-in bundle

Each operational shard contains:

```text
experiments/{experiment_id}/
  images/{well_id}/BF/projection/focus_stack/{image_id}.jpg
  dropin_frame_inventory.csv
  dropin_frame_inventory.csv.validated
  plate_metadata.csv
  plate_metadata.csv.validated
  runtime_config.yaml
```

The integration directory contains `fov_scale_calibration.csv`, `embryo_ingest.csv`,
`well_provenance.csv`, `dropped_fovs.csv`, `detection_failures.csv`, `experiment_manifest.csv`, and
`experiments.txt`. The runtime drop-in config requires all of `frame_inventory_csv`,
`plate_metadata_csv`, and `image_root`.

Every production attempt starts from a new, empty bundle/output root. `build-bundle` refuses a
non-empty root; old acquisition, frame-inventory, or validation products must be quarantined rather
than reused across runs. Materialized `image_path` values and manifest/runtime paths are absolute,
must resolve inside the selected fresh experiment root, and must exist. Before the standard
pipeline is submitted, the shard preflight reopens every image and reruns strict frame-inventory
source validation with no fallback `image_root`. A relative, missing, external, or stale path is a
hard preflight failure and must not consume a GPU.

Full-corpus GroundingDINO and materialization runs are cluster work. Do not run them on a login node.

## Entrypoint

Reconcile without materializing:

```bash
conda run -n morphseq-env --no-capture-output env PYTHONPATH=src \
  python -m data_pipeline.acquisition.seahub reconcile \
  --image-reconciliation-csv <image_metadata_reconciliation.csv> \
  --collection-metadata-xlsx <collection_metadata.xlsx> \
  --output-dir <integration_dir>
```

After the existing GroundingDINO workflow has produced the complete detection manifest and the
source-FOV SAM2 pass has produced the complete mask-area manifest, build the bundle on the cluster:

```bash
conda run -n morphseq-env --no-capture-output env PYTHONPATH=src \
  python -m data_pipeline.acquisition.seahub build-bundle \
  --reconciled-fovs-csv <integration_dir/reconciled_fovs.csv> \
  --detection-manifest-csv <full_embryo_manifest.csv> \
  --scale-mask-manifest-csv <full_source_fov_mask_manifest.csv> \
  --output-root <bundle_root>
```

The scale-mask manifest may identify its source FOV as `source_fov_id` or `image_id`. Finite,
positive `mask_area_px` values are usable when no explicit `is_valid_mask` column is present. No
mask-score cutoff is silently imposed; `--min-scale-mask-score` is an explicit optional review
choice.

`--plan-only` exercises reconciliation, inclusion, detection completeness, identity packing,
metadata, and runtime-config generation without opening or writing source images. Each row of
`integration/experiment_manifest.csv` points to the per-shard runtime config used with the standard
Snakefile.

For a cluster array, plan once and materialize one immutable shard per task:

```bash
conda run -n morphseq-env --no-capture-output env PYTHONPATH=src \
  python -m data_pipeline.acquisition.seahub materialize-shard \
  --bundle-root <bundle_root> \
  --experiment-id <experiment_id>
```

This command reads the persisted `well_provenance.csv` and frame inventory, writes only the selected
shard's images, and runs strict source validation. Existing correctly shaped images may be reused
only while resuming that same fresh bundle; cross-run reuse is prohibited and `--overwrite-images`
is explicit.

Preflight every materialized shard before submitting its standard-pipeline job:

```bash
conda run -n morphseq-env --no-capture-output env PYTHONPATH=src \
  python -m data_pipeline.acquisition.seahub preflight-shard \
  --bundle-root <bundle_root> \
  --experiment-id <experiment_id>
```
