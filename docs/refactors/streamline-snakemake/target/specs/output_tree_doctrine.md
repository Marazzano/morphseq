# Output Tree Doctrine — target data pipeline output structure

**Status:** LIVE (2026-06-26). Layout matches the current on-disk output produced by the pipeline.
Migration from legacy names (`experiment_metadata/`, `built_image_data/`, `detection/`) is complete.

---

## The regime river

```
acquisition → object_extraction → features → quality_control → analysis_ready
```

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
      ingest_metadata/
        scope_metadata__<scope>.csv          # raw scope metadata (per-scope suffix)
        acquisition_inventory__<scope>.csv   # per-plane acquisition inventory
        scope_metadata_mapped.csv            # joined (scope + position_well_mapping)
        scope_metadata_mapped.csv.validated
        keyence_stitch_map__keyence.json     # experiment-grain mosaic tile coords (Keyence only)

      well_identities/
        position_well_mapping.csv
        position_well_mapping.csv.provenance.json
        discovered_wells.txt

      materialized_images/
        <well_id>/
          <channel_id>/                      # channel-first (BF, GFP, …)
            projection/
              focus_stack/
                <image_id>.png
                focus_index_map/
                  <image_id>.npz             # focus_index_map + z_indices (provenance)
            z_stack/
              <image_id>_z<ZZZZ>.png         # one PNG per z-plane

      frame_inventory/
        product_inventories/
          per_well/
            <well_id>/
              <well_id>_<product_key>_frame_inventory.csv        # e.g. BF__projection__focus_stack
              <well_id>_<product_key>_frame_inventory.csv.validated
        resolved_product_plans/
          per_well/
            <well_id>/
              <product_key>_resolved_product_plan.json
        available_products/
          per_well/
            <well_id>/
              <well_id>_available_products.csv
        per_well/
          <well_id>/
            <well_id>_frame_inventory.csv          # assembled (all products for this well)
            <well_id>_frame_inventory.csv.validated
        <experiment_id>_frame_inventory.csv         # merged across wells
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
            <well_id>_prompt_seeds.csv
        <experiment_id>_frame_masks.csv

      physical_embryo_registry/
        per_well/
          <well_id>/
            <well_id>_physical_embryo_registry.csv
            <well_id>_physical_embryo_registry.csv.validated
        <experiment_id>_physical_embryo_registry.csv

      snips/
        per_well/
          <well_id>/
            <well_id>_snip_inventory.csv
            <snip_id>/
              <snip_id>_<channel_id>_t<TTTT>.png   # processed snip crop (snip-frame grid)
              <snip_id>_embryo.png                  # embryo mask on snip-frame grid

      snip_auxiliary_masks/
        per_well/
          <well_id>/
            <well_id>_snip_auxiliary_masks.csv
            <well_id>_snip_auxiliary_masks.csv.validated
            <snip_id>/
              <snip_id>_<channel_id>_t<TTTT>/       # per-snip per-frame mask dir
                via.png
                yolk.png
                focus.png
                bubble.png

      tracks/
        per_well/
          <well_id>/
            <well_id>_tracks.csv

  feature_extraction/
    <experiment_id>/
      latent_embeddings/
      mask_geometry/
      curvature_metrics/
      pose_kinematics/
      stage_predictions/
      fraction_alive/
      consolidated_features/

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
| `object_extraction/` | Object-level products from model inference: detections, masks, physical embryo registry, snips, snip auxiliary masks, tracks | Raw features; QC tables; embeddings |
| `feature_extraction/` | Measured or learned representations derived from objects/frames/embryos: geometry, kinematics, embeddings | Raw model outputs; QC; final joined tables |
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

This is the shape `paths.py` enforces via `PATH_MODE_PER_WELL` / `PATH_MODE_MERGED`.

### Product key convention

The `<product_key>` used in frame_inventory filenames encodes the full product identity:

```
<channel_id>__<image_product_type>__<projection_method>
# e.g.  BF__projection__focus_stack
#        BF__z_stack
```

Double-underscore (`__`) is the separator. Single-underscore is allowed within a component.

---

## Notable conventions

**`ingest_metadata/` suffix grammar** — scope-specific files use `__<scope>` suffix
(e.g. `scope_metadata__keyence.csv`, `acquisition_inventory__keyence.csv`). The joined
`scope_metadata_mapped.csv` is scope-agnostic (no suffix).

**`materialized_images/` is channel-first** — `{well_id}/{channel_id}/{product_type}/...`.
The `focus_index_map/` provenance dir lives inside the product dir it explains
(`projection/focus_stack/focus_index_map/`). The `.npz` is never in `ALLOWED_IMAGE_SUFFIXES`.

**`snips/` grain** — snip images live at per-snip-per-frame resolution on the snip-frame
grid (configured by `snip_frame_shape` in `config.yaml`). The embryo mask (`.png`) is the
cropped `mask_cropped` RLE from `frame_masks`, converted to the snip-frame grid.

**`snip_auxiliary_masks/` grain** — keyed by `snip_id`, with one subdirectory per
`(snip_id, channel_id, time_index)`. The four UNet families (via/yolk/focus/bubble) are
separate `.png` files in that dir, all on the snip-frame grid.

---

## What does NOT live in the output tree

**Model weights and configs** (`models_root` in `env.yaml`) are inputs the pipeline reads
but never writes. They stay outside the experiment-scoped output tree entirely.

**Raw microscope files** (`input_root` in `env.yaml`) are never written by the pipeline.

**`.snakemake/`** — Snakemake's own bookkeeping; not part of the doctrine tree.

---

## Interface with `paths.py`

The `PIPELINE_STEPS` registry in `orchestration/paths.py` is the code mirror of this doctrine.
Each step's `"stage"` key is the top-level regime folder. All steps already use the target
names (`"acquisition"`, `"object_extraction"`, `"feature_extraction"`, `"quality_control"`).

---

## Doctrine in one sentence per regime

- **Acquisition** makes frames trustworthy.
- **Object extraction** finds things in frames.
- **Features** quantify things.
- **Quality control** judges trust.
- **Analysis-ready** exports the usable table.

Do not create `analysis/` as a broader folder. If `analysis_ready/` starts accumulating
subfolders, the discipline has broken down — push back.
