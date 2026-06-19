# Segmentation / Frame Masks World

**Status:** target-planning draft. This doc captures the current agreed target model before folder
layout hardens.

---

## Role

Segmentation answers:

> Given trusted frames and kept detection prompts, which pixels belong to each object, and what track
> identity did the segmenter/backend assign?

For the current SAM2 path, segmentation and tracking are physically coupled. SAM2 propagates masks
and carries object IDs through time in the same operation. The target contract should not invent a
second tracking algorithm at this seam.

For the first target pass, this world does not need much more conceptual machinery than that. The
contract is:

```text
frame_detections
  -> kept_frame_detections(frame_detections.is_kept == true)
  -> SAM2 box prompts
  -> SAM2 mask propagation + object IDs
  -> frame_masks
```

Everything else in this doc exists to make that handoff auditable: how `detection_id` becomes a
prompt/seed, how `sam2_obj_id` becomes `track_id`, and how SAM2 frame indices map back to pipeline
`image_id`.

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
  -> kept_frame_detections[well]
  -> prompt_seeds[well]
  -> frame_masks[well]
  -> mask/track QC later
```

`is_kept` is the filtering seam from detections into SAM prompts. Detection can keep all detector
candidates for audit, but segmentation consumes only `kept_frame_detections(...)`.

`prompt_seeds` may be an in-memory view or a small persisted sidecar. It is not a new filtering
decision in the MVP; it is the deterministic conversion of kept detection boxes into SAM2 prompts and
seed IDs.

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
frame_detections[well]
```

The stage also consumes backend config/model files. SAM2 may need a temporary model-specific frame
layout such as sequential `00000.jpg` symlinks. That layout is a consumer view, not a pipeline
contract.

---

## SAM2 Frame View

SAM2 requires a directory of sequential image names:

```text
00000.jpg
00001.jpg
00002.jpg
...
```

The pipeline should satisfy that requirement with a backend-local frame view, not by renaming or
copying the canonical frame artifacts.

Execution policy:

```text
model_frame_view sorted by time_index
  -> list[source_image_path]
  -> temporary sam2_frame_view directory of symlinks
  -> predictor.init_state(video_path=sam2_frame_view)
  -> remove temporary directory after SAM2 finishes
```

The symlink names are SAM2-local indices only. The adapter must keep an explicit mapping:

```text
sam2_frame_index -> image_id
sam2_frame_index -> time_index
```

`frame_masks` rows are built by joining SAM2 output back through that mapping. No downstream product
should infer frame identity from `00000.jpg` names.

The live pipeline already follows this shape for execution with `sam2_frame_context(...)`: it creates
a `sam2_frames_*` temp directory, symlinks chronological source frames as `00000.jpg`, `00001.jpg`,
and so on, passes that directory to SAM2, then cleans it up in a `finally` block.

In the target package layout, this belongs inside the SAM2 backend, for example:

```text
segmentation/backends/sam2_video/sam2_frame_view.py
```

That module should own arranging the temporary folder, naming symlinks, cleanup, and the
`sam2_frame_index` mapping. Shared segmentation code should only receive the mapping and adapted
`frame_masks`; it should not know SAM2's folder naming rules.

Current implementation note: the repo still has `segmentation/backends.py`, so the first concrete
helpers live under `segmentation/sam2_video/` until the backend package name is cleared.

Optional persisted debug/QC views are allowed, for example:

```text
artifacts/raw_frames/{image_id}.jpg
artifacts/sam2_frames/00000.jpg
```

Those views are for inspection or rerun convenience. They are not contract inputs and validators
should not require them.

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

## Detection Handoff / Prompt Seeds

`kept_frame_detections(...)` answers:

> Which detection evidence is allowed to initialize mask propagation?

The shared helper is intentionally simple:

```python
def kept_frame_detections(df: pd.DataFrame) -> pd.DataFrame:
    validate_frame_detections(df)
    return df[df["is_kept"]].copy()
```

Segmentation then derives SAM2 prompt seeds from those rows. Core prompt-seed fields:

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
```

`seed_id` is the handoff ID from detection into segmentation. It is a pipeline ID for the selected
prompt/object seed, not a mask ID and not an embryo ID.

Preferred seed ID shape:

```text
{seed_image_id}_seed{prompt_seed_index:04d}
```

For SAM2, the adapter must preserve this mapping:

```text
sam2_obj_id -> seed_id
```

That mapping lets every propagated mask row point back to the seed that initialized the SAM object.

The MVP does not need to invent a second selection contract here. If a later pass adds seed-frame
choice, de-duplication, or prompt ranking, those policies can extend this section. The current
contract is simply `frame_detections.is_kept` followed by deterministic prompt conversion.

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
prompt_seeds derived from kept_frame_detections
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

`seed_id` comes from the prompt seed derived from a kept detection row.

For SAM2, each prompted object should map back to one prompt seed:

```text
seed_id = prompt_seeds_by_obj_id[sam2_obj_id]
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
clear what happened to each prompt seed.

Policy:

```text
if a prompt seed produces no mask on any frame:
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
validate_prompt_seeds(prompt_seeds, frame_detections, model_frame_view)
validate_frame_mask_block(df)
validate_frame_masks(df, model_frame_view, prompt_seeds)
```

### Prompt Seed Validator

The prompt seed validator checks:

```text
required seed columns are present
seed_id is unique within the well
seed_image_id values exist in model_frame_view
detection_id values exist in kept_frame_detections
prompt_type is an allowed value
box prompts are finite and inside image bounds
at least one prompt seed exists before segment_masks runs
```

### Frame Masks Validator

The frame masks validator checks structural coherence only:

```text
required frame mask columns are present
mask_id is unique within the well
image_id values exist in model_frame_view
seed_id values exist in prompt seed rows when known
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
    prompt_seeds.py

    masks/
      mask_rle.py
      mask_geometry.py
      mask_ids.py

    backends/
      sam2_video/
        model_loader.py
        run_sam2_video.py
        adapt_sam2_output.py
        sam2_frame_view.py
```

Keep detection outside `segmentation/`. Detection owns `frame_detections`; segmentation consumes
`frame_detections` to choose prompts and build `frame_masks`.

`prompt_seeds.py` stays directly under `segmentation/` for now. It is the segmentation ingest step
that turns kept detections into SAM2 prompts. It does not need its own subdirectory unless multiple
prompt/seed policies grow enough code to justify one.

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

`build_seed_id` probably belongs in `prompt_seeds.py`, not `masks/mask_ids.py`, because seed IDs
are the detection-to-segmentation prompt handoff, not a mask primitive. If a broader shared
identifier module appears later, seed IDs can move there.

### Stage Utilities

The stage-level utilities are:

```text
kept_frame_detections(frame_detections) -> kept detections
build_prompt_seeds(kept_frame_detections) -> prompt seeds
valid_frame_masks(frame_masks) -> structurally valid masks
map_seed_ids_to_backend_obj_ids(prompt_seeds) -> dict
adapt_sam2_output_to_frame_masks(...) -> frame_masks
```

The most load-bearing utility is:

```text
adapt_sam2_output_to_frame_masks(
    sam2_results,
    model_frame_view,
    prompt_seeds,
    backend_provenance,
) -> frame_masks
```

That adapter centralizes all derivation instead of repeating it in `run_per_well`.

### Backend Model Loading

The SAM2 backend should own model loading:

```text
sam2_video/model_loader.py
  parse_sam2_video_model_config(...)
  load_sam2_video_model(...)
```

Shared segmentation routing should not import SAM2 directly or know the model root/config/checkpoint
path rules. The backend loader may delegate to a lower-level shared model utility, but the backend
is the stable callsite.

SAM2 should be loaded once for the current run-well batch, not once per well. Use the orchestrator's
well-runner helpers to resolve the `run_wells` set and shard paths, then hand the selected well
inputs to the SAM2 backend runner:

```text
run_well_ids_for_experiment(...)
  -> run_well_shard_paths(...)
  -> load_sam2_video_model(...) once
  -> for each run well:
       build SAM2 frame view
       run propagation
       adapt to frame_masks shard
  -> concat_well_shards_to_file(...) for the merged frame_masks artifact
```

This preserves the per-well contract/shard model while avoiding repeated model construction.

---

## SAM2-Specific Note

Current SAM2 code physically does three things in one pass:

```text
box prompts from a seed frame
mask propagation forward/backward through a temporary frame directory
object ID propagation through SAM2 obj_id
```

Use SAM2's native video propagation direction controls for this. The current pipeline already seeds
the true seed frame once, then calls `propagate_in_video(..., start_frame_idx=seed_idx,
reverse=False)` for forward propagation and `propagate_in_video(..., start_frame_idx=seed_idx,
reverse=True)` for backward propagation on the same inference state. Do not rebuild the older
workaround that split/reversed frame directories just to propagate backward.

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
