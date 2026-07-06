# Detection / Segmentation / Tracking Specs

**Status:** active planning space, created 2026-06-18. This folder is the downstream companion to
`specs/front_end/`: the front-end specs define the validated per-well `frame_inventory` handoff;
this space defines what happens next, from trusted frames to tracked embryo/snips.

**Current purpose:** not implementation yet. First we define the target worlds, their seams, their
contracts, their validators, and the adapter shape that lets different model backends plug into the
same pipeline.

---

## Why This Space Exists

The current pipeline stage name, `segment_and_track_per_well`, hides three different jobs:

```text
validated frame_inventory[well]
  -> detection
  -> segmentation
  -> tracking
  -> segmentation_tracking.csv
  -> snip_processing
```

Today those jobs are bundled behind GroundingDINO + SAM2. That is useful operationally, but it is
not the architecture we want. Detection, segmentation, and tracking have different failure modes,
different validators, different model choices, and different replacement timelines. They need
separate contract seams before we swap models.

The immediate model pressure is detection: we want to move from GroundingDINO to a Detectron2/Facebook model.
The target is not "hardcode a new detector." The target is an adapter seam where detector backends
can change without rewriting segmentation, tracking, or downstream snip processing.

---

## Current Contract Debt

The frame identity block used by detection, frame masks, and later frame-level products should not
be redefined in this folder long term. Per `pipeline_file_philosophy.md`, contracts live beside the
data product that creates them.

The source of truth should be:

```text
src/data_pipeline/image_materialization/frame_inventory_contract.py
```

That module already owns the frame inventory handoff columns, derived `well_id` / `image_id`
helpers, and consistency checks. The remaining debt is to export a downstream reusable frame identity
block from there, then import it into detect/segment/track contracts.

Until that exists, specs in this folder may spell out the block for discussion, but implementation
should route through the frame inventory contract rather than duplicating the list.

---

## Organizing Rule

Use the same discipline as the front-half refactor:

```text
one world -> one product contract -> one validator -> one adapter seam -> one routed stage
```

Do not split by library name. Split by pipeline meaning:

| World | Question | Example backend today | Future backend pressure |
|---|---|---|---|
| Detection | Where are candidate embryos in a frame or seed frame? | GroundingDINO | Facebook detector |
| Segmentation | What pixels belong to each detected object over frames? | SAM2 video predictor | SAM2 variants / future segmenters |
| Tracking | Which masks across time are the same embryo? | currently implicit in SAM2 object propagation | explicit tracker / post-SAM identity repair |
| Presentation | What does downstream snip processing consume? | `segmentation_tracking.csv` | stable contract |

Model packages are backend details. The pipeline contracts are the stable layer.

---

## Reading Order

| Open this | When |
|---|---|
| `targets/overall_goal_and_plan.md` | Start here. Defines the phased plan and the intended river. |
| `targets/adapter_seams.md` | When deciding how backends plug in and what model-specific code may return. |
| `targets/detection_world.md` | When discussing detector inputs/outputs, seed frames, boxes, scores, prompts. |
| `targets/segmentation_world.md` | When discussing SAM2/mask products, frame views, mask encodings, content checks. |
| `targets/tracking_world.md` | When discussing embryo identity through time and track-level validation. |
| `working/open_questions.md` | Back-and-forth parking lot for unresolved decisions. |
| `working/current_audit_notes.md` | Notes from the current live code and prior docs; update as audits deepen. |

---

## Boundary Assumption

This space assumes the front-half Beat 1 handoff exists:

```text
materialize_well[well_id]
  -> {well_id}_frame_inventory.csv
  -> validate_frame_inventory_for_well
  -> .validated
```

Detection/segmentation/tracking begins **after** that seam. It should read the validated per-well
`frame_inventory` shard, not experiment-level `frame_contract.csv`.

---

## Naming Guardrails

- `frame_inventory` is the trusted input to this world.
- `frame_detections` is a detector product, not a segmentation product.
- `frame_masks` is the frame-level mask product; SAM/SAM2 may already carry object identity here.
- `segmentation_tracking` is legacy/downstream presentation vocabulary, not the target source
  contract for new model products.
- Avoid `video_id` in target contracts. The well-level video unit is `well_id`.
