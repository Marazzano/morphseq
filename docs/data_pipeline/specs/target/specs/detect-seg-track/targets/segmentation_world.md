# Segmentation / Frame Masks World

**Status:** target-planning draft. This doc captures the target contract for the mask-producing
stage before folder layout hardens.

---

## Role

Segmentation answers:

> Given trusted frame detections, which selected detection prompts initialize segmentation, which
> pixels belong to each propagated object, and what track identity did the backend assign?

The target doctrine is:

```text
Detections propose.
Prompts initialize.
SAM2 propagates.
Masks record the realized object.
```

Naming:

```text
segment_and_track  = stage / action
frame_masks        = artifact / table
sam2_video         = backend / implementation
```

Stages name operations. Artifacts name tables. Backends name implementations.

Do not use `segment_masks` as the stage name; masks are the artifact, not the action. Do not use
`run_sam2_segmentation_and_tracking` as the stage name; SAM2 is the backend implementation, not the
shared stage.

The target data object is:

```text
frame_masks
```

A `frame_masks` row means:

> On this `image_id`, here is either one actual mask instance for one propagated object, or one
> structural placeholder proving the frame was processed and no mask was produced.

Tracking quality is not judged here. Track coverage, jumps, swaps, gaps, and other biological or
temporal quality checks belong to later mask/track QC.

---

## Current Target River

The shared pipeline river is:

```text
frame_inventory[well]
  -> frame_detections[well]
  -> kept_frame_detections(frame_detections, frame_inventory)
  -> SAM2 prompts[well]                      # backend-local table/view
  -> segment_and_track[well]
  -> frame_masks[well]
  -> mask/track QC later
```

The SAM2 backend-local river is:

```text
frame_detections + frame identity
  -> kept detections
  -> prompts
  -> sam2_frame_view
  -> SAM2 propagation
  -> adapted frame_masks
```

`frame_detections[well]` is the required direct input to `segment_and_track`. `frame_inventory[well]`
is the reference validation input. The stage validates the detection rows against the trusted frame
inventory, then consumes only the kept detection view.

`model_frame_view` should not be a universal shared product in this world. SAM2 needs a sequential
frame view, but that view is backend-local and belongs inside the SAM2 backend as `sam2_frame_view`.
Likewise, prompt selection is a SAM2 backend policy in the first pass, not a shared pipeline
artifact.

Do not use `segmentation_tracking` as the source contract for this world. The source contract is
`frame_masks`. Any downstream presentation view can be derived later.

---

## Inputs

Primary inputs:

```text
frame_detections[well]
frame_inventory[well]                 # reference validation input
```

Backend inputs:

```text
segmentation backend config
segmentation model config/checkpoint files
```

`frame_detections` already carries the shared frame identity block, including `source_image_path`,
`image_width_px`, and `image_height_px`. From a data-engineering point of view, that makes it the
right direct input for this stage. `frame_inventory` remains the source of truth used to validate
that the frame identity carried through detection has not drifted.

The stage must not hand-roll `df[df["is_kept"]]`. It must consume detections through the shared
helper:

```python
def kept_frame_detections(
    df: pd.DataFrame,
    reference_frame_inventory: pd.DataFrame,
) -> pd.DataFrame:
    validate_frame_detections(df, reference_frame_inventory)
    return df[df["is_kept"]].copy()
```

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
copying canonical frame artifacts and not by creating a fake universal `model_frame_view` product.

Execution policy inside the SAM2 backend:

```text
validated frame rows sorted by time_index
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

The live pipeline already follows this execution shape with `sam2_frame_context(...)`: it creates a
`sam2_frames_*` temp directory, symlinks chronological source frames as `00000.jpg`, `00001.jpg`,
and so on, passes that directory to SAM2, then cleans it up in a `finally` block.

In the target package layout, this belongs inside the SAM2 backend:

```text
segmentation/backends/sam2_video/sam2_frame_view.py
```

That module should own arranging the temporary folder, naming symlinks, cleanup, and the
`sam2_frame_index` mapping. Shared segmentation code should receive the mapping and adapted
`frame_masks`; it should not know SAM2's folder naming rules.

Optional persisted debug/QC views are allowed:

```text
artifacts/raw_frames/{image_id}.jpg
artifacts/sam2_frames/00000.jpg
```

Those views are for inspection or rerun convenience. They are not contract inputs and validators
should not require them.

---

## Required Frame Identity Block

Every frame-level model product starts with the shared frame identity block. `frame_masks` starts
with the same prefix as `frame_detections`:

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
- The segmentation stage validates frame identity against `frame_inventory`; it does not reinterpret
  frame identity.

---

## Detection Handoff / SAM2 Prompts

Prompt selection is a SAM2-backend-local policy for the first pass. It should live here:

```text
src/data_pipeline/segmentation/backends/sam2_video/seed_selection.py
```

Do not put the first implementation here:

```text
src/data_pipeline/segmentation/seed_selection.py
```

The internal seed-selection function should have this shape:

```text
select_sam2_prompts(
    kept_detections,
    reference_frame_inventory,
    config,
) -> prompts
```

`prompts` is an internal SAM2 backend table/view for the first pass. It should be returned from a
pure function and tested, but it is not yet a required Snakemake artifact. Optional debug export can
come later as `sam2_prompts.csv`.

`prompt_detection_id` is the prompt provenance column:

```text
prompt_detection_id = frame_detections.detection_id used to initialize this SAM2 prompt
```

For the MVP, one kept detection becomes one SAM2 box prompt. Later policies may support:

```text
one detection -> multiple prompts
multiple detections -> one prompt
manual prompt
example prompt
re-prompted object
```

MVP seed-selection policy mirrors current behavior:

```text
1. group kept detections by image_id
2. choose the seed frame with the maximum kept detection count
3. tie-break by earliest time_index
4. tie-break by image_id lexical order
5. create one box prompt per kept detection on that seed frame
6. assign prompt_order as deterministic zero-based order of prompts submitted to SAM2
```

Seed frame selection and prompt ordering must be deterministic and must not depend on input row
order. For MVP prompt ordering, sort by `prompt_detection_id` unless legacy code requires a different
order; if it does, write that rule down and test it.

Do not rank or filter kept detections again in the MVP. Filtering already happened in detection via
`is_kept`; adding another hidden filter here would create a second detector policy inside SAM2.

Later SAM2 seed-selection modes may include:

```text
single_seed_frame
multi_seed_frame
manual_seed_frame
best_confidence_frame
midpoint_seed_frame
```

Later seed-selection config may include:

```text
max_seeds_per_frame
min_prompt_confidence
deduplicate_iou_threshold
```

Those are out of scope for the first pass.

Core prompt fields:

```text
prompt_detection_id
prompt_image_id
prompt_time_index
prompt_order
prompt_type
prompt_bbox_x_min_px
prompt_bbox_y_min_px
prompt_bbox_x_max_px
prompt_bbox_y_max_px
prompt_bbox_format
```

Prompt box columns are prefixed with `prompt_bbox_*` because unprefixed `bbox_*` in `frame_masks`
belongs to realized mask geometry.

For SAM2, the adapter must preserve this mapping:

```text
sam2_object_id -> prompt_detection_id
```

That mapping lets every propagated mask row point back through:

```text
frame_detections.detection_id -> prompt_detection_id -> sam2_object_id -> track_id -> mask_id
```

---

## Frame Masks Contract

Core mask and track fields:

```text
mask_id
track_id
prompt_detection_id
sam2_object_id
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
track_id_source      = "sam2_object_id"
track_id             = stable value derived from sam2_object_id
```

This is the agreed middle ground:

```text
SAM2 provides object identity.
The pipeline records that raw backend identity as sam2_object_id.
The pipeline derives track_id from sam2_object_id.
This stage does not validate track quality or repair track IDs.
```

Optional backend audit columns may include:

```text
sam2_frame_index
propagation_direction
mask_logit_mean
mask_logit_max
empty_mask_reason
filter_reason
```

`sam2_object_id` is the raw SAM2 object ID assigned during propagation. Do not normalize it into a
pipeline-flavored string; `track_id` is the pipeline-derived stable ID. `sam2_object_id` is required
for actual SAM2 mask rows. Placeholder no-mask rows set it to NA.

---

## Core Column Derivations

`frame_masks` is derived from three inputs:

```text
frame_detections validated against frame_inventory
prompts derived from kept_frame_detections
SAM2 propagation output
```

Each output row represents one actual mask for one `image_id` and one `track_id`, or one structural
no-mask placeholder row for a processed frame.

### Frame Identity Columns

These come from validated detection/frame rows after mapping the backend frame index back to pipeline
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

### Identifier Policy

All identifier minting and decomposition must go through shared identifier constructors and parsers.
The shapes shown in this doc are target shapes for readability, not permission to hand-build strings
inside stage code.

Allowed:

```text
build_mask_id(image_id, local_mask_index)
build_no_mask_id(image_id)
build_track_id(well_id, local_track_index)
parse_image_id(image_id)
parse_mask_id(mask_id)
parse_track_id(track_id)
```

Not allowed outside `shared/identifiers`:

```text
f"{image_id}_m{...}"
f"{image_id}_mask_none"
f"{well_id}_track{...}"
string splits on "_m", "_mask_none", "_track"
regex parsing of mask_id / track_id
```

Constructors mint. Parsers reveal. Validators compare. Everyone else treats IDs as opaque.

Target home:

```text
src/data_pipeline/shared/identifiers/constructors.py
src/data_pipeline/shared/identifiers/parsers.py
```

### `mask_id`

`mask_id` is a pipeline-generated row ID for one mask row on one frame.

Actual mask rows are minted with:

```text
mask_id = build_mask_id(image_id, local_mask_index)
```

No-mask placeholder rows are minted with:

```text
mask_id = build_no_mask_id(image_id)
```

`local_mask_index` should be deterministic within the frame, for example after sorting by `track_id`.

### `track_id`

`sam2_object_id` is the backend object identity. `track_id` is the pipeline temporal object identity
derived from `sam2_object_id`.

For SAM2:

```text
local_track_index = sam2_object_id
track_id = build_track_id(well_id, local_track_index)
```

`local_track_index` is an object-extraction/backend identity and may be zero-based. Physical embryo
indexing is a snip-world concern and must not be inferred in `frame_masks`.

Example shape:

```text
{well_id}_track{sam2_object_id:04d}
```

The raw backend source is recorded separately:

```text
track_id_source = "sam2_object_id"
tracking_backend = sam2_video
```

### `prompt_detection_id`

`prompt_detection_id` references the kept detector candidate used to initialize the SAM2 prompt:

```text
prompt_detection_id = frame_detections.detection_id
```

There is no `prompt_id` in the MVP because one kept detection maps to one SAM2 prompt.
`prompt_detection_id` is the stable prompt provenance key.

For SAM2, each propagated object should map back to one prompt detection in the MVP:

```text
prompt_detection_id = prompt_detections_by_object_id[sam2_object_id]
```

### `mask_rle` / `mask_rle_format`

These are derived from the binary mask output by SAM2:

```text
mask_rle = encode_mask_to_rle(binary_mask)
mask_rle_format = coco_rle
```

### Geometry

These are derived from mask pixels, not from the prompt/detection box:

```text
area_px = mask.sum()
bbox_* = bbox_from_mask(mask)
centroid_* = centroid_from_mask(mask)
```

Unprefixed `bbox_*` in `frame_masks` always means realized mask geometry. Prompt boxes use
`prompt_bbox_*`.

### `mask_confidence`

`mask_confidence` is backend-specific and optional. Current SAM2 code can use mean positive logit.
Treat this value as an opaque segmenter score, not as a cross-backend calibrated confidence.

### `is_valid_mask`

`is_valid_mask` is structural, not QC:

```text
is_valid_mask = true   -> this row contains an actual mask payload
is_valid_mask = false  -> structural placeholder / no mask row
```

It does not mean good embryo, biologically plausible, accepted by QC, or trustworthy track.

For valid mask rows:

```text
mask_rle is non-null
area_px > 0
track_id populated
prompt_detection_id populated
sam2_object_id populated
```

For placeholder rows:

```text
mask_rle is NA
area_px is NA
track_id is NA
prompt_detection_id is NA
sam2_object_id is NA
```

---

## Empty / Missing Masks

An empty `frame_masks.csv` is not the target shape when `segment_and_track` ran. The artifact should
make it clear which frames were processed.

Policy:

```text
if a processed frame has no masks:
  write one placeholder row for that image_id
  frame identity columns populated
  mask_id = build_no_mask_id(image_id)
  track_id = NA
  prompt_detection_id = NA
  sam2_object_id = NA
  is_valid_mask = false
  mask/geometry/confidence fields may be NA

if a frame has one or more actual masks:
  write one row per mask
  frame identity columns populated
  mask_id = build_mask_id(image_id, local_mask_index)
  track_id populated
  prompt_detection_id populated
  sam2_object_id populated
  is_valid_mask = true
  mask/geometry fields populated
```

No-mask placeholders are different from failed execution. A failed segmenter run should fail the
stage; a completed run with no masks on a frame should be represented in the table.

---

## Consumer View

Downstream consumers that need actual mask payloads should consume a shared valid view:

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
validate_sam2_prompts(prompts, kept_frame_detections, frame_inventory)
validate_frame_mask_block(df)
validate_frame_masks(df, frame_inventory, prompts)
```

### Prompt Validator

The prompt validator checks:

```text
required prompt columns are present
prompt_detection_id values exist in kept_frame_detections.detection_id
prompt_image_id values exist in frame_inventory
prompt_time_index agrees with frame_inventory
prompt_order is unique within the selected prompt frame
prompt_order is zero-based and contiguous
prompt_type is an allowed value
prompt_bbox coordinates are finite and inside image bounds
prompt_bbox_x_min_px < prompt_bbox_x_max_px
prompt_bbox_y_min_px < prompt_bbox_y_max_px
prompt_bbox_format is an allowed value
at least one prompt exists before segment_and_track runs
```

### Frame Masks Validator

The frame masks validator checks structural coherence only:

```text
required frame mask columns are present
mask_id is unique within the artifact
image_id values exist in frame_inventory
frame identity columns agree with frame_inventory
prompt_detection_id values exist in SAM2 prompt rows for valid masks
segmentation_backend is non-empty
segmentation_model_id is non-empty
tracking_backend is non-empty
track_id_source is non-empty for valid masks
track_id is present for valid masks
sam2_object_id is present for valid SAM2 mask rows
for valid SAM2 rows, track_id == build_track_id(well_id, sam2_object_id)
no duplicate image_id + track_id rows for valid masks
is_valid_mask is boolean
valid mask rows have non-null, structurally valid RLE
RLE dimensions match image_width_px / image_height_px
bbox coordinates are finite and inside image bounds for valid masks
bbox_x_min_px < bbox_x_max_px for valid masks
bbox_y_min_px < bbox_y_max_px for valid masks
area_px > 0 for valid masks
centroid coordinates are finite and inside image bounds for valid masks
actual mask rows have mask_id minted by build_mask_id(image_id, local_mask_index)
placeholder rows have mask_id = build_no_mask_id(image_id)
placeholder rows have track_id / prompt_detection_id / sam2_object_id as NA
placeholder rows have mask_rle / geometry / confidence fields as NA
placeholder rows have is_valid_mask = false
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
  shared/
    identifiers/
      constructors.py
      parsers.py

  segmentation/
    frame_masks_contract.py
    validate_frame_masks.py
    valid_frame_masks.py
    build_frame_masks.py

    masks/
      mask_rle.py
      mask_geometry.py

    backends/
      sam2_video/
        model_loader.py
        run_sam2_video.py
        seed_selection.py
        adapt_sam2_output.py
        sam2_frame_view.py
```

Keep detection outside `segmentation/`. Detection owns `frame_detections` and
`kept_frame_detections(...)`; the SAM2 backend consumes the kept view to build backend-local prompt
seeds and `frame_masks`.

`seed_selection.py` stays under `segmentation/backends/sam2_video/` for the first pass because the
policy is SAM2-specific. Promote prompt selection to a shared `segmentation/` module only after
another backend needs the same abstraction.

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

Canonical ID constructors and parsers belong in `shared/identifiers`, not under `segmentation/masks/`:

```text
shared/identifiers/constructors.py
  build_mask_id(image_id, local_mask_index) -> mask_id
  build_no_mask_id(image_id) -> mask_id
  build_track_id(well_id, local_track_index) -> track_id

shared/identifiers/parsers.py
  parse_mask_id(mask_id) -> image_id, local_mask_index | no-mask sentinel
  parse_track_id(track_id) -> well_id, local_track_index
```

Prompt ordering belongs with `sam2_video/seed_selection.py`, not `masks/mask_ids.py`, because prompt
ordering is part of the detection-to-SAM2 handoff, not a mask primitive.

### Stage Utilities

The stage-level utilities are:

```text
kept_frame_detections(frame_detections, frame_inventory) -> kept detections
select_sam2_prompts(kept_detections, frame_inventory, config) -> prompts
valid_frame_masks(frame_masks) -> actual mask rows
map_sam2_object_ids_to_prompts(prompts, sam2_object_ids) -> dict
adapt_sam2_output_to_frame_masks(...) -> frame_masks
```

The most load-bearing utility is:

```text
adapt_sam2_output_to_frame_masks(
    sam2_results,
    frame_detections,
    frame_inventory,
    prompts,
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
       validate frame_detections against frame_inventory
       build kept detections
       select SAM2 prompts
       build SAM2 frame view
       run propagation
       adapt to frame_masks shard
  -> concat_well_shards_to_file(...) for the merged frame_masks artifact
```

This preserves the per-well contract/shard model while avoiding repeated model construction.

---

## Storage

Keep the first target artifact set simple:

```text
<well_id>_frame_masks.csv
<well_id>_frame_masks.csv.validated
<experiment_id>_frame_masks.csv.validated
```

Additional provenance/debug sidecars can be added after the contract stabilizes. The first priority
is a clear per-well `frame_masks` table and a validation marker proving it passed structural checks.

---

## Snip Handoff

Snip processing consumes:

```text
valid_frame_masks(frame_masks)
```

That view supplies actual mask payload rows only. Snip processing should resolve track identity into
the embryo and snip identifiers owned by the snip world. The likely target chain is:

```text
track_id -> local_embryo_index -> physical_embryo_id -> embryo_id -> snip_id
```

This doc does not own that full identifier chain. It owns `frame_masks` and the structural guarantee
that valid rows have mask payloads, `prompt_detection_id`, `sam2_object_id`, and `track_id`.

Track quality is not judged in `frame_masks`.

---

## SAM2-Specific Note

Current SAM2 code physically does three things in one pass:

```text
box prompts from selected SAM2 prompts
mask propagation forward/backward through a temporary frame directory
object ID propagation through SAM2 object IDs
```

Use SAM2's native video propagation direction controls for this. The current pipeline already seeds
the true seed frame once, then calls `propagate_in_video(..., start_frame_idx=seed_idx,
reverse=False)` for forward propagation and `propagate_in_video(..., start_frame_idx=seed_idx,
reverse=True)` for backward propagation on the same inference state. Do not rebuild the older
workaround that split/reversed frame directories just to propagate backward.

Do not split that GPU call just to satisfy a conceptual boundary. The first strangler pass can keep
one fused SAM2 execution and emit:

```text
frame_masks table
```

The important contract move is:

```text
frame_detections.detection_id -> prompt_detection_id -> sam2_object_id -> track_id -> mask_id
```

SAM2's object ID is accepted as the track identity source for this stage, with provenance recorded
through `sam2_object_id`, `tracking_backend`, and `track_id_source`. Quality control on whether that
track is trustworthy happens later.
