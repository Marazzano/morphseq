# Output Tree Doctrine — target data pipeline output structure

**Status:** LOCKED doctrine (2026-06-21). Migration not yet executed. Current on-disk layout uses
legacy names (`experiment_metadata/`, `built_image_data/`, `detection/`). This file is the target.

---

## The regime river

```
acquisition → object_extraction → features → quality_control → analysis_ready
```

In plain English:

1. Acquire frames and establish canonical frame truth
2. Extract objects (find and delineate things in frames)
3. Compute features (quantify and represent those objects)
4. Judge quality (trust/evaluation layer, spans all regimes)
5. Export the usable joined table

---

## Target layout

```
data_pipeline_output/
  acquisition/
    <experiment_id>/
      scope_metadata/
      acquisition_inventory/
      materialized_images/
        <well_id>/
          projection/
            BF/
              <well_id>_BF_t0000.png
      frame_inventory/
        per_well/
          <well_id>/
            <well_id>_frame_inventory.csv
            <well_id>_frame_inventory.csv.validated
        <experiment_id>_frame_inventory.csv
        <experiment_id>_frame_inventory.csv.validated

  object_extraction/
    <experiment_id>/
      frame_detections/
        per_well/
          <well_id>/
            <well_id>_frame_detections.csv
            <well_id>_frame_detections.csv.validated
        <experiment_id>_frame_detections.csv
        <experiment_id>_frame_detections.csv.validated
      frame_masks/
        per_well/
          <well_id>/
            <well_id>_frame_masks.csv
            <well_id>_frame_masks.csv.validated
        <experiment_id>_frame_masks.csv
      tracks/
        per_well/
          <well_id>/
            <well_id>_tracks.csv

  features/
    <experiment_id>/
      frame_features/
      embryo_features/
      well_features/
      embeddings/

  quality_control/
    <experiment_id>/
      acquisition_qc/
      detection_qc/
      mask_qc/
      track_qc/
      feature_qc/
      plots/
      reports/

  analysis_ready/
    <experiment_id>/
      <experiment_id>_analysis_ready.parquet
      <experiment_id>_analysis_ready.csv
```

---

## Regime definitions

| Regime | What lives here | What does NOT live here |
|---|---|---|
| `acquisition/` | Canonical acquisition/frame truth: scope metadata, acquisition inventory, materialized pixel files, frame inventory | Anything derived from a model; QC judgments |
| `object_extraction/` | Object-level products from model inference: detections, masks, tracks | Raw features; QC tables; embeddings |
| `features/` | Measured or learned representations derived from objects/frames/embryos: geometry, kinematics, embeddings | Raw model outputs; QC; final joined tables |
| `quality_control/` | Trust and evaluation artifacts across all regimes: QC tables, plots, reports | Pipeline products; analysis outputs |
| `analysis_ready/` | Final joined products for downstream analysis. Terminal — no subfolders should accumulate here | Intermediate products of any kind |

---

## Internal per-product convention

Every product that has per-well shards AND a merged experiment view uses this shape consistently:

```
<product>/
  per_well/
    <well_id>/
      <well_id>_<product>.csv
      <well_id>_<product>.csv.validated
  <experiment_id>_<product>.csv
  <experiment_id>_<product>.csv.validated
```

This is the same shape `paths.py` already enforces via `PATH_MODE_PER_WELL` / `PATH_MODE_MERGED`.
The migration is a folder rename, not a structural change.

---

## What does NOT live in the output tree

**Model weights and configs** (`models_root` in `env.yaml`) are inputs the pipeline reads but never
writes. They are orthogonal to the experiment-scoped output tree and stay outside it entirely.
`models_root` remains a separate env.yaml pointer.

**Raw microscope files** (`input_root` in `env.yaml`) are never written by the pipeline.

---

## Interface with `paths.py`

The `PIPELINE_STEPS` registry in `orchestration/paths.py` is the code mirror of this doctrine.
Each step's `"stage"` key is the top-level regime folder. The migration requires:

| Current `"stage"` value | Target `"stage"` value |
|---|---|
| `"experiment_metadata"` | `"acquisition"` (for ingest/inventory steps) |
| `"built_image_data"` / `"materialize_well"` | `"acquisition"` |
| `"detection"` | `"object_extraction"` |
| `"segmentation"` | `"object_extraction"` |
| `"features"` | `"features"` |
| `"quality_control"` | `"quality_control"` |

**Migration scope** (one dedicated session, flag-day):

1. Update `"stage"` values in `PIPELINE_STEPS` registry (`orchestration/paths.py`)
2. Update top-level directory variables in `Snakefile` (`EXPERIMENT_METADATA_DIR`, `BUILT_IMAGE_DATA_DIR`, etc.)
3. Update `frame_inventory.smk` and `frame_detections.smk` output paths
4. Update `env.yaml` path comments
5. Update any tests that assert output paths
6. Rename on-disk directories if migrating live data (or wipe and re-run from a clean state)

**Do not** migrate piecemeal — all path changes must land in one commit or the DAG will have
mixed-regime paths and Snakemake will re-run everything.

---

## Doctrine in one sentence per regime

- **Acquisition** makes frames trustworthy.
- **Object extraction** finds things in frames.
- **Features** quantify things.
- **Quality control** judges trust.
- **Analysis-ready** exports the usable table.

Do not create `analysis/` as a broader folder. If `analysis_ready/` starts accumulating subfolders,
the discipline has broken down — push back.
