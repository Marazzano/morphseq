# Adapter Seams — Model Backends Without Contract Drift

**Status:** target-planning draft. This doc defines the shape of backend routing before choosing the
first replacement detector.

---

## Principle

Backends return pipeline contracts, not backend-native structures.

```text
backend-specific model output
  -> adapter normalizes
  -> contract dataframe / record set
  -> contract validator
  -> downstream stage
```

The adapter is allowed to know a model library. Downstream stages are not.

---

## Analogy To Scope Backends

This should feel like the front-end scope routing:

```text
scope backend:      raw microscope format -> acquisition/frame products
model backend:      model-native output    -> detection/segmentation products
```

Both have the same discipline:

- shared stage name;
- backend selected by config;
- backend-specific code isolated under a backend folder;
- shared contract validator after the backend;
- downstream code consumes the shared contract only.

---

## Proposed Package Shape

Names are provisional, but the boundary is the target:

```text
src/data_pipeline/segmentation/
  model_frame_view.py
  contracts/
    frame_detections_contract.py
    seed_selection_contract.py
    mask_instances_contract.py
    track_instances_contract.py
    segmentation_tracking_contract.py
  detection/
    detect_embryos.py                  # shared stage/router
    backends/
      groundingdino.py                 # current adapter
      facebook_detector.py             # future adapter
  segmentation_masks/
    segment_masks.py                   # shared stage/router
    backends/
      sam2_video.py                    # current adapter
  tracking/
    link_tracks.py                     # shared stage/router
    backends/
      sam2_object_ids.py               # current implicit tracking adapter
      explicit_tracker.py              # future
  presentation/
    build_segmentation_tracking.py
```

The exact module names can change, but the rule is fixed: backend modules do not own the pipeline
contract; contract modules own the pipeline contract.

---

## Router Shape

Each shared stage should have one small router:

```text
detect_embryos_for_well(config, model_frame_view) -> frame_detections
segment_masks_for_well(config, seed_selection, model_frame_view) -> mask_instances
link_tracks_for_well(config, mask_instances) -> track_instances
```

The router chooses a backend from config and delegates. It should not contain model math or pandas
cleanup beyond calling validators.

---

## Detection Router Hypothesis

Detection gets the first concrete router because detector replacement is the immediate pressure.
The shared file should be a thin sequencer, not a smart detector:

```text
run_frame_detections.py
  load validated frame_inventory / frame identity rows
  load detection config
  resolve detector backend
  call backend detection code
  call backend adapter -> canonical frame_detections rows
  validate_frame_detections(...)
  write frame_detections.csv
  write .frame_detections.validated
  write .frame_detections.provenance.json
  optionally write backend raw/debug sidecar
```

The important boundary is:

```text
Backend detects.
Adapter canonicalizes.
Shared validator judges.
Router writes.
```

The router should not know what a GroundingDINO phrase, text logit, or NMS suppression reason means.
It should only know that the adapted dataframe must satisfy the shared `frame_detections` contract.

### Backend Interface Sketch

The exact Python shape can wait, but the interface should feel like this:

```python
class DetectorBackend(Protocol):
    backend_name: str

    def detect(
        self,
        *,
        frames: pd.DataFrame,
        config: Mapping[str, Any],
    ) -> Any:
        """Return backend-native raw detections."""
```

Each backend then owns its native output and adapter:

```text
backends/groundingdino/
  run_groundingdino_detection.py
  adapt_groundingdino_detections.py
  config.py
  raw_output.py?
```

The adapter translates native output to the canonical table. The shared validator is the strong
boundary check after translation.

### Provenance

MVP assumes one `frame_detections.csv` artifact comes from one detector backend/config run. The
router writes artifact-level provenance beside the table:

```text
frame_detections.csv
.frame_detections.validated
.frame_detections.provenance.json
groundingdino_raw_detections.parquet   # optional debug sidecar
```

The provenance should record backend/model/config/thresholds/adapter version. It does not need a
row-level `detection_run_id` unless a future table mixes multiple detector runs.

---

## Config Shape

Target config should make backend choice explicit:

```yaml
detection:
  backend: groundingdino
  groundingdino:
    text_prompt: individual embryo
    box_threshold: 0.35
    text_threshold: 0.25
    confidence_threshold: 0.45
    iou_threshold: 0.5
  facebook_detector:
    model_name: TBD
    checkpoint_path: TBD

segmentation_masks:
  backend: sam2_video
  sam2_video:
    tracking_mode: bidirectional
    config_path: configs/sam2.1/sam2.1_hiera_l.yaml
    checkpoint_path: ...

tracking:
  backend: sam2_object_ids
```

The existing `segmentation_and_tracking:` block can remain as compatibility during the strangler
period, but target docs should use the separated config vocabulary.

---

## Adapter Contract

A backend adapter must:

- take explicit paths/config objects, not hidden globals;
- return or write the stage contract it owns;
- include backend provenance columns or a sidecar where useful;
- fail loud when required model files/config are missing;
- not mint global IDs with string hacks;
- not write downstream presentation contracts directly unless it is the presentation stage.

Backends may include model-native metadata as optional columns only if the contract validator allows
them and downstream ignores them.
