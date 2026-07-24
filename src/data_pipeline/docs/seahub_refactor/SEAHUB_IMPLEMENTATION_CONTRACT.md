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

- Use `image_micrometers_per_pixel = 7.8`. This is intentionally flagged as unverified and must be
  revisited before physical-size measurements are treated as calibrated.

## QC semantics

- Focus QC remains computable but is diagnostic-only for `single_z`:
  `focus_qc_applicability = diagnostic_only`.
- Motion-blur QC requires adjacent z planes and is not applicable:
  metrics are null, `motion_blur_flag = false`, and
  `motion_blur_qc_applicability = not_applicable`.
- An unresolved stage produces a nullable `predicted_stage_hpf` with
  `stage_prediction_status=missing_start_age_hpf`. Stage-binned surface-area QC then emits
  `sa_outlier_flag=false` with `surface_area_qc_applicability=not_applicable`.
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

The integration directory contains `embryo_ingest.csv`, `well_provenance.csv`,
`dropped_fovs.csv`, `detection_failures.csv`, `experiment_manifest.csv`, and `experiments.txt`.
The runtime drop-in config requires all of `frame_inventory_csv`, `plate_metadata_csv`, and
`image_root`.

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

After the existing GroundingDINO workflow has produced the complete detection manifest, build the
bundle on the cluster:

```bash
conda run -n morphseq-env --no-capture-output env PYTHONPATH=src \
  python -m data_pipeline.acquisition.seahub build-bundle \
  --reconciled-fovs-csv <integration_dir/reconciled_fovs.csv> \
  --detection-manifest-csv <full_embryo_manifest.csv> \
  --output-root <bundle_root>
```

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
shard's images, and runs strict source validation. Existing correctly shaped images are reused;
`--overwrite-images` is explicit.
