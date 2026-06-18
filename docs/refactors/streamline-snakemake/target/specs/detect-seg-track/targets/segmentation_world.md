# Segmentation World

**Status:** target-planning draft.

---

## Role

Segmentation answers:

> Given trusted frames and seed prompts, which pixels belong to each candidate object?

It may use SAM2, a future SAM variant, or another mask-producing backend. It should not be responsible
for the final downstream `segmentation_tracking.csv` presentation contract.

---

## Inputs

Primary inputs:

```text
model_frame_view[well]
segmentation_seed_selection[well]
```

The segmentation stage may need a temporary model-specific frame layout such as sequential
`00000.jpg` symlinks for SAM2. That is a **consumer view**, not a new pipeline contract.

---

## Seed Selection Product

`segmentation_seed_selection` is the seam from detection into segmentation.

Candidate columns:

| Column | Meaning |
|---|---|
| `experiment_id` | experiment atom |
| `well_id` | global well id |
| `seed_image_id` | image used to initialize segmentation |
| `seed_frame_index` / `seed_time_index` | temporal coordinate |
| `seed_id` | one row per prompt/object seed |
| `detection_id` | source detection row, if detector-backed |
| `prompt_type` | `box`, `point`, `mask`, etc. |
| `bbox_x_min`, `bbox_y_min`, `bbox_x_max`, `bbox_y_max` | prompt box when used |
| `selection_score` | score used by seed policy |
| `selection_reason` | short policy label |

Validator should ensure seed rows reference real detections and real frame rows.

---

## Output Contract: `mask_instances`

One row per frame/object mask before final track presentation.

Candidate columns:

| Column | Meaning |
|---|---|
| `experiment_id` | experiment atom |
| `well_id` | global well id |
| `image_id` | frame identity |
| `frame_index` / `time_index` | temporal coordinate |
| `segmentation_object_id` | backend-local object id |
| `seed_id` | seed that initialized this object |
| `segmentation_backend` | backend name |
| `segmentation_model_id` | model/checkpoint label or hash |
| `mask_rle` | encoded binary mask |
| `area_px` | mask area |
| `bbox_x_min`, `bbox_y_min`, `bbox_x_max`, `bbox_y_max` | mask-derived box |
| `mask_confidence` | backend score if available |

Avoid embedding final `embryo_id` semantics here unless the tracking world explicitly adopts the
backend object IDs as tracks.

---

## Validator

`validate_mask_instances_for_well` should prove:

- masks reference valid `image_id`s from the model frame view;
- masks decode, or at least have structurally valid RLE;
- mask dimensions match frame dimensions;
- bbox and area agree with the mask within tolerance;
- object IDs are unique within a frame;
- every seed has expected coverage, or missing propagation is explicitly represented.

Level-2 content checks can be phased: start with structural RLE/dim checks, then add decode-heavy
checks when runtime cost is acceptable.

---

## SAM2-Specific Note

SAM2's video predictor also carries object IDs through time. That looks like tracking, but target
architecture should treat it as segmentation-backend output until the tracking world accepts or
repairs those identities.

