# Detection World

**Status:** target-planning draft. Keep this doc simple until the contract is stable.

---

## Role

Detection answers:

> For each trusted frame, where are candidate embryos, and how confident is the detector?

It does not decide masks. It does not decide temporal embryo identity. It produces candidate object
evidence for the mask-producing stage.

---

## Core Rule

Every post-`frame_inventory` model product starts with the same required frame identity block. Each
product then adds its own product-specific block. Each block gets its own validator.

For detection:

```text
frame_inventory[well]
  -> detector backend adapter
  -> frame_detections.csv
  -> kept_frame_detections(...) view
  -> frame_masks
```

Backend-specific code diverges before `frame_detections`; all downstream code consumes the shared
contract.

---

## Required Frame Identity Block

These columns are default for frame-level model products:

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

- This block is stated here for planning clarity, but implementation should import it from the
  frame inventory contract owner: `src/data_pipeline/image_materialization/frame_inventory_contract.py`.
  If that module does not yet export the exact downstream block, add it there first. Do not create a
  second source of truth in detection.
- `time_index` is the target pipeline frame atom. Avoid `frame_index` in this contract unless a
  compatibility alias is temporarily required.
- `z_index` is present even when null for projection/BF MVP rows.
- These columns come from the validated per-well `frame_inventory`; detection does not reinterpret
  them.

---

## Required Detection Block

`frame_detections` adds the detector-specific minimum:

```text
detection_id
detector_backend
detector_model_id
class_label
confidence
bbox_x_min_px
bbox_y_min_px
bbox_x_max_px
bbox_y_max_px
bbox_format
is_kept
```

This is one row per **canonicalized detector candidate**, not only accepted detections.

`is_kept` is the shared seam between backend-specific filtering and downstream segmentation:

```text
Filtering logic is backend-specific.
Filtering outcome is shared.
Segmentation obeys the outcome.
```

The table therefore supports two views:

```text
audit view:        all frame_detections rows
segmentation view: kept_frame_detections(frame_detections)
```

The shared table should stay model-neutral. Backend details like text prompts, raw logits, threshold
settings, NMS details, or phrases belong in optional audit columns, provenance, or backend-specific
sidecars unless they become genuinely shared semantics.

MVP assumes one `frame_detections.csv` artifact is produced by one detector backend/config run. The
artifact-level provenance sidecar records how the detections were made; rows do not need a
`detection_run_id` unless a future table intentionally mixes multiple detector runs.

Allowed optional backend audit columns may include:

```text
filter_reason
raw_phrase
raw_box_score
raw_text_score
nms_rank
suppressed_by_detection_id
```

The shared validator should allow unknown backend-specific columns but only hard-validate the shared
required blocks.

---

## Validators

Layer the validators rather than writing one large detector-specific check:

```text
validate_frame_identity_block(df, frame_inventory)
validate_frame_detection_block(df)
validate_frame_detections(df, frame_inventory)
```

`validate_frame_detections` composes the first two.

### Frame Identity Validator

The general frame identity validator checks:

```text
required frame identity columns are present
rows belong to one experiment_id / well_id
image_id values exist in frame_inventory
time_index / channel_id / z_index agree with frame_inventory
source_image_path agrees with frame_inventory
image_width_px / image_height_px agree with frame_inventory
z_index may be null for projection rows
```

This validator should be reusable for `frame_detections`, `frame_masks`, and later frame-level
products. Its column definitions and reference checks should route through the frame inventory
contract owner, because `frame_inventory` creates the trusted frame rows.

### Detection Validator

The detection validator checks:

```text
required detection columns are present
detection_id is unique within the well
detector_backend is non-empty
detector_model_id is non-empty
confidence is finite and in the allowed range
bbox coordinates are finite
bbox coordinates are inside image bounds
bbox_x_min_px < bbox_x_max_px
bbox_y_min_px < bbox_y_max_px
bbox_format is an allowed value
is_kept is boolean
```

It should allow zero kept detections on some frames. Whether a well can proceed to mask generation is
a later seed/mask concern, not a detector-schema concern.

### Kept Detection View

Any downstream stage that acts on detections must consume the kept view through a helper, not hand-roll
row filtering:

```python
def kept_frame_detections(df: pd.DataFrame) -> pd.DataFrame:
    validate_frame_detections(df)
    return df[df["is_kept"]].copy()
```

Segmentation should call this helper before creating prompts or masks.

---

## Backend Region

Use the same shape as the front-end scope/router pattern:

```text
segmentation/detection/
  frame_detections_contract.py        # shared product contract
  validate_frame_detections.py        # shared validators
  run_frame_detections.py             # router / stage entry
  backends/
    groundingdino/
      adapter.py
      config.py
      raw_output.py?                  # only if useful
    detectron2/
      adapter.py
      config.py
      raw_output.py?
```

The shared contract owns `frame_detections`. Backend adapters translate native model output into
that contract:

```text
GroundingDINO native output -> backend filtering + canonicalization -> frame_detections rows
Detectron2/Facebook output  -> backend filtering + canonicalization -> frame_detections rows
```

Do not add GroundingDINO-only columns to the shared table unless they are needed by every detector
backend.

Rows describe detections. Provenance describes how detections were made. Raw sidecars preserve
backend weirdness.
