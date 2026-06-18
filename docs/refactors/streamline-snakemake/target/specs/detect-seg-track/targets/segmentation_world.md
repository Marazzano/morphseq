# Segmentation / Frame Masks World

**Status:** target-planning draft. This doc captures the current agreed target model before folder
layout hardens.

---

## Role

Segmentation answers:

> Given trusted frames and selected seed prompts, which pixels belong to each object, and what track
> identity did the segmenter/backend assign?

For the current SAM2 path, segmentation and tracking are physically coupled. SAM2 propagates masks
and carries object IDs through time in the same operation. The target contract should not invent a
second tracking algorithm at this seam.

The target data object is:

```text
frame_masks
```

A `frame_masks` row means:

> On this `image_id`, here is one mask for one tracked object-like thing.

Tracking quality is not judged here. Track coverage, jumps, swaps, gaps, and other biological or
temporal quality checks belong to later mask/track QC.

---

## Current Target River

```text
frame_inventory[well]
  -> model_frame_view[well]
  -> frame_detections[well]
  -> seed_selection[well]
  -> frame_masks[well]
  -> mask/track QC later
```

`seed_selection` is the filtering/selection seam from detections into SAM prompts. Detection can keep
all detector candidates for audit, but segmentation consumes selected seeds.

`frame_masks` includes masks and track identity because SAM gives us both in the same operation. We
keep track provenance so a future backend can differ.

Naming:

```text
segment_masks  = stage / action
frame_masks    = artifact / table
```

Do not use `segmentation_tracking` as the source contract for this world. That is the downstream
legacy table name. The target segmentation-and-tracking product is `frame_masks`.

---

## Inputs

Primary inputs:

```text
model_frame_view[well]
seed_selection[well]
```

The stage also consumes backend config/model files. SAM2 may need a temporary model-specific frame
layout such as sequential `00000.jpg` symlinks. That layout is a consumer view, not a pipeline
contract.

---

## Required Frame Identity Block

`frame_masks` starts with the same frame identity block as `frame_detections`:

```text
experiment_id
well_id
image_id
time_index
z_index
channel_id
source_image_path
image_width_px
image_height_px
```

Notes:

- This block should come from the frame inventory contract owner, not be redefined in the masks
  package.
- `time_index` is the target pipeline frame atom. `frame_index` may be carried as a temporary
  compatibility alias while current SAM2 code still indexes frames that way.
- The segmentation stage joins backend frame output back to `model_frame_view`; it does not
  reinterpret frame identity.

---

## Seed Selection Handoff

`seed_selection` answers:

> Which detection evidence should initialize mask propagation, and how should it be presented to the
> segmenter?

Core seed fields:

```text
seed_id
seed_image_id
seed_time_index
detection_id
prompt_type
prompt_x_min_px
prompt_y_min_px
prompt_x_max_px
prompt_y_max_px
selection_score
selection_reason
is_selected
```

`seed_id` is the handoff ID from detection into segmentation. It is a pipeline ID for the selected
prompt/object seed, not a mask ID and not an embryo ID.

Preferred seed ID shape:

```text
{seed_image_id}_seed{selected_seed_index:04d}
```

For SAM2, the adapter must preserve this mapping:

```text
sam2_obj_id -> seed_id
```

That mapping lets every propagated mask row point back to the seed that initialized the SAM object.

---

## Frame Masks Contract

Core mask and track fields:

```text
mask_id
track_id
seed_id
mask_rle
mask_rle_format
area_px
bbox_x_min_px
bbox_y_min_px
bbox_x_max_px
bbox_y_max_px
centroid_x_px
centroid_y_px
mask_confidence
is_valid_mask
```

Backend/provenance fields:

```text
segmentation_backend
segmentation_model_id
tracking_backend
track_id_source
```

For SAM2:

```text
segmentation_backend = sam2_video
tracking_backend     = sam2_video
track_id_source      = sam2_obj_id
track_id             = stable value derived from SAM2 object id
```

This is the agreed middle ground:

```text
SAM/SAM2 provides object identity.
The pipeline records that identity as track_id.
This stage does not validate track quality or repair track IDs.
```

Optional backend audit columns may include:

```text
sam2_obj_id
sam2_frame_index
propagation_direction
mask_logit_mean
mask_logit_max
empty_mask_reason
filter_reason
```

`sam2_obj_id` can be useful as raw provenance, but it is not required as a first-class shared ID if
`track_id` and `track_id_source` already preserve the identity source.

---

## Core Column Derivations

`frame_masks` is derived from three inputs:

```text
model_frame_view / frame_inventory
seed_selection
SAM2 propagation output
```

Each output row represents one valid or placeholder mask for one `image_id` and one `track_id`.

### Frame Identity Columns

These come directly from `model_frame_view` after mapping the backend frame index back to pipeline
frame identity:

```text
experiment_id
well_id
image_id
time_index
z_index
channel_id
source_image_path
image_width_px
image_height_px
```

They are not recalculated by the SAM adapter.

### `mask_id`

`mask_id` is a pipeline-generated row ID for one mask on one frame.

`image_id` is not enough because one frame can contain multiple masks. Preferred shape:

```text
mask_id = {image_id}_m{local_mask_index:04d}
```

`local_mask_index` should be deterministic within the frame, for example after sorting by `track_id`.

### `track_id`

`track_id` is the temporal object identity accepted from the segmentation backend.

For SAM2:

```text
track_id = build_track_id(well_id, sam2_obj_id)
```

Example shape:

```text
{well_id}_track{sam2_obj_id:04d}
```

The raw backend source is recorded separately:

```text
track_id_source = sam2_obj_id
tracking_backend = sam2_video
```

### `seed_id`

`seed_id` comes from `seed_selection`.

For SAM2, each prompted object should map back to one selected seed:

```text
seed_id = selected_seeds_by_obj_id[sam2_obj_id]
```

The SAM adapter therefore needs an explicit mapping:

```text
sam2_obj_id -> seed_id
```

### `mask_rle` / `mask_rle_format`

These are derived from the binary mask output by SAM2:

```text
mask_rle = encode_mask_to_rle(binary_mask)
mask_rle_format = coco_rle
```

### Geometry

These are derived from mask pixels, not from the seed/detection box:

```text
area_px = mask.sum()
bbox_* = bbox_from_mask(mask)
centroid_* = centroid_from_mask(mask)
```

### `mask_confidence`

`mask_confidence` is backend-specific and optional. Current SAM2 code can use mean positive logit.
Treat this value as an opaque segmenter score, not as a cross-backend calibrated confidence.

### `is_valid_mask`

`is_valid_mask` is structural, not QC.

For normal SAM2 masks:

```text
is_valid_mask = true if mask has positive area and valid RLE
```

For placeholders or empty backend outputs:

```text
is_valid_mask = false
```

---

## Empty / Missing Masks

An empty `frame_masks.csv` is not the target shape when segmentation ran. The artifact should make it
clear what happened to each selected seed.

Policy:

```text
if a selected seed produces no mask on any frame:
  write one placeholder row anchored on seed_image_id
  frame identity columns populated from the seed frame
  mask_id = {seed_id}_mask_none
  track_id populated if the backend object id is known
  seed_id populated
  is_valid_mask = false
  geometry/RLE columns may be NA

if an object propagates but has no valid mask on a particular frame:
  write a frame-level placeholder row
  mask_id = {image_id}_m{local_mask_index:04d}
  track_id populated
  seed_id populated when known
  is_valid_mask = false
  empty_mask_reason populated
  geometry/RLE columns may be NA
```

Rejected or empty masks are different from missing execution. A failed segmenter run should fail the
stage; a completed run with no usable mask should be represented in the table.

---

## Consumer View

Downstream consumers that need usable masks should consume a shared valid view:

```python
def valid_frame_masks(df: pd.DataFrame) -> pd.DataFrame:
    validate_frame_masks(df)
    return df[df["is_valid_mask"]].copy()
```

This is a structural filter only. It is not a track-quality gate.

---

## Validators

Layer validators rather than writing one SAM2-specific check:

```text
validate_frame_identity_block(df, frame_inventory)
validate_seed_selection(df, frame_detections, model_frame_view)
validate_frame_mask_block(df)
validate_frame_masks(df, model_frame_view, seed_selection)
```

### Seed Selection Validator

The seed validator checks:

```text
required seed columns are present
seed_id is unique within the well
seed_image_id values exist in model_frame_view
detection_id values exist in kept_frame_detections when detector-backed
prompt_type is an allowed value
box prompts are finite and inside image bounds
is_selected is boolean
at least one selected seed exists before segment_masks runs
```

### Frame Masks Validator

The frame masks validator checks structural coherence only:

```text
required frame mask columns are present
mask_id is unique within the well
image_id values exist in model_frame_view
seed_id values exist in selected seed rows when known
segmentation_backend is non-empty
segmentation_model_id is non-empty
tracking_backend is non-empty
track_id_source is non-empty
track_id is present for valid masks
no duplicate image_id + track_id rows for valid masks
is_valid_mask is boolean
valid mask rows have structurally valid RLE
RLE dimensions match image_width_px / image_height_px
bbox coordinates are finite and inside image bounds
bbox_x_min_px < bbox_x_max_px for non-empty masks
bbox_y_min_px < bbox_y_max_px for non-empty masks
area_px is finite and non-negative
centroid coordinates are finite and inside image bounds for non-empty masks
```

This validator should not check:

```text
track jumps
identity swaps
coverage quality
biological plausibility
split/merge suspicion
long-gap repair
```

Those belong to downstream mask/track QC.

---

## Core Utilities

Target folder layout:

```text
src/data_pipeline/
  segmentation/
    frame_masks_contract.py
    validate_frame_masks.py
    valid_frame_masks.py
    build_frame_masks.py
    seed_selection.py

    masks/
      mask_rle.py
      mask_geometry.py
      mask_ids.py

    backends/
      sam2_video/
        run_sam2_video.py
        adapt_sam2_output.py
        sam2_frame_view.py
```

Keep detection outside `segmentation/`. Detection owns `frame_detections`; segmentation consumes
`frame_detections` to choose prompts and build `frame_masks`.

`seed_selection.py` stays directly under `segmentation/` for now. It is the segmentation ingest step
that turns kept detections into prompts. It does not need its own subdirectory unless multiple seed
selection policies grow enough code to justify one.

### Mask Utility Modules

`masks/mask_rle.py` should own RLE encoding/decoding/validation:

```text
encode_binary_mask_to_rle(mask) -> rle
decode_rle_to_binary_mask(rle) -> mask
validate_mask_rle(rle, image_width_px, image_height_px)
```

`masks/mask_geometry.py` should own geometry derived from binary masks:

```text
compute_mask_area_px(mask) -> int
compute_mask_centroid_px(mask) -> tuple
compute_mask_bbox_xyxy_px(mask) -> tuple
compute_mask_geometry_table(masks) -> DataFrame
validate_bbox_inside_image(bbox, image_width_px, image_height_px)
```

Do not start with separate `mask_bbox.py` and `mask_centroid.py` files. Split them later only if
`mask_geometry.py` becomes genuinely crowded.

`masks/mask_ids.py` should own mask/track ID helpers:

```text
build_mask_id(image_id, local_mask_index) -> mask_id
build_track_id(well_id, backend_obj_id) -> track_id
validate_mask_id_uniqueness(frame_masks)
```

`build_seed_id` probably belongs in `seed_selection.py`, not `masks/mask_ids.py`, because seed IDs
are the detection-to-segmentation prompt handoff, not a mask primitive. If a broader shared
identifier module appears later, seed IDs can move there.

### Stage Utilities

The stage-level utilities are:

```text
selected_seed_selection(seed_selection) -> selected seeds
valid_frame_masks(frame_masks) -> structurally valid masks
map_seed_ids_to_backend_obj_ids(selected_seeds) -> dict
adapt_sam2_output_to_frame_masks(...) -> frame_masks
```

The most load-bearing utility is:

```text
adapt_sam2_output_to_frame_masks(
    sam2_results,
    model_frame_view,
    seed_selection,
    backend_provenance,
) -> frame_masks
```

That adapter centralizes all derivation instead of repeating it in `run_per_well`.

---

## SAM2-Specific Note

Current SAM2 code physically does three things in one pass:

```text
box prompts from a seed frame
mask propagation forward/backward through a temporary frame directory
object ID propagation through SAM2 obj_id
```

Do not split that GPU call just to satisfy a conceptual boundary. The first strangler pass can keep
one fused SAM2 execution and emit:

```text
frame_masks sidecar
```

The important contract move is:

```text
seed_id -> sam2_obj_id -> track_id -> mask_id
```

SAM2's object ID is accepted as the track identity for this stage, with provenance recorded through
`tracking_backend` and `track_id_source`. Quality control on whether that track is trustworthy
happens later.
