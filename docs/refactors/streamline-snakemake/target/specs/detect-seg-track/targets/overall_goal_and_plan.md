# Detect / Segment / Track — Overall Goal and Plan

**Status:** target-planning draft. This doc names the goal and migration strategy before implementation
details harden.

---

## Goal

Turn the current fused `segment_and_track_per_well` stage into a readable per-well river:

```text
validated frame_inventory[well]
  -> prepare_model_frame_view[well]
  -> detect_embryos[well]
  -> choose_segmentation_seeds[well]
  -> segment_masks[well]
  -> link_tracks[well]
  -> build_segmentation_tracking[well]
  -> snip_processing[well]
```

The final `segmentation_tracking.csv` contract remains the stable handoff to snip processing. The
new intermediate products make detection, segmentation, and tracking independently inspectable and
replaceable.

---

## What We Are Solving

Current reality hides separate concerns inside one function:

- frame filtering and ordering from a manifest;
- GroundingDINO model load;
- per-frame detection;
- detection filtering/NMS;
- seed frame selection;
- conversion of boxes to SAM2 prompts;
- SAM2 frame-directory construction;
- SAM2 bidirectional propagation;
- mask decoding/RLE encoding;
- embryo identity minting;
- final `segmentation_tracking.csv` construction.

That makes the first model swap too risky. A detector replacement should have to satisfy one detector
contract, not understand SAM2 propagation, tracking IDs, snip IDs, and final table formatting.

---

## Target Products

The target products should be per-well shards. Exact filenames can still change when they are added
to `paths.py`, but the conceptual slots are:

| Product | Grain | Owner | Meaning |
|---|---|---|---|
| `model_frame_view` | one row per frame | segmentation ingest/view | ordered, validated view of frame_inventory for model input |
| `frame_detections` | one row per detection | detection world | boxes/scores/classes/prompts from a detector backend |
| `segmentation_seed_selection` | one row per selected seed/prompt | detection->segmentation seam | the boxes/points/frame chosen to initialize segmentation |
| `mask_instances` | one row per frame/object mask | segmentation world | masks from a segmentation backend, before final track contract |
| `track_instances` | one row per frame/object identity | tracking world | temporal identity and track metadata |
| `segmentation_tracking` | one row per snip | presentation contract | stable downstream table consumed by snip processing |

`segmentation_tracking` may initially be built directly from SAM2 propagation output while the
intermediate products are written as sidecars. Once the sidecars are stable, make them proper DAG
artifacts.

---

## Migration Strategy

Use the same strangler doctrine as the front-half work, but adapted for model stages:

1. **Repoint the input seam.**
   `segment_and_track_per_well` reads the validated per-well `frame_inventory` shard instead of
   experiment-level `frame_contract.csv`. Output remains unchanged.

2. **Write intermediate products inside the fused runner.**
   Keep one GPU execution job, but emit `frame_detections`, `segmentation_seed_selection`,
   `mask_instances`, and `track_instances` beside the final `segmentation_tracking.csv`. This gives
   auditability without changing scheduling or reloading models multiple times.

3. **Add validators for each product.**
   Validators prove the contract at each seam: detector boxes are in-bounds; selected seeds exist in
   detections; mask instances match the frame view; tracks are internally coherent; the final
   presentation table is stable for snip processing.

4. **Introduce backend adapters.**
   Detection gets the first adapter seam because that is the first planned model swap. The adapter
   returns the detector contract, not arbitrary model-native JSON.

5. **Promote sidecars into DAG stages one by one.**
   Split the Snakemake rules only after the contracts are real. Avoid a rule split that just moves
   in-memory coupling into temporary files without stable meaning.

6. **Strangle the fused runner.**
   Once the split path reproduces `segmentation_tracking.csv` for one well, then a second well, retire
   the fused `segment_and_track_per_well` implementation.

---

## Verification Gates

Before promotion, compare fused-vs-split output on a small well set:

- `segmentation_tracking.csv` required columns and `snip_id` uniqueness;
- frame count and image_id coverage;
- number of detections per frame;
- selected seed frame and seed boxes;
- mask area/bbox tolerances;
- track continuity and embryo_id stability;
- optional overlay videos for visual review when mask pixels differ.

Byte identity is not required for mask outputs if a backend changes. Contract equivalence and visual
acceptance are the promotion gate.

---

## Non-Goals For The First Draft

- Do not design every future detector backend.
- Do not move snip processing yet.
- Do not make the model-frame view a second frame-inventory contract.
- Do not teach `well_runner` about model semantics.
- Do not keep `video_id` as a target concept; use `well_id`.

