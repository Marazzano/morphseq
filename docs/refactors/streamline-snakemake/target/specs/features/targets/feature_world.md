# Feature World - computed feature targets

**Status:** planning spec, 2026-06-22. This doc lists the computed features we can start, their
dependencies, and the acceptance bar for each. QC is intentionally out of scope here; it should be a
sibling world that cross-links this one because the per-well stage pattern is shared.

**Doctrine:** features quantify validated objects. They do not mint object identity, choose
segmentation prompts, decide QC flags, or perform analysis-ready joins.

---

## Shared Feature Stage Pattern

Every computed feature should use the same per-well stage recipe:

- one noun-like registry row in the pipeline-wide path registry;
- one compute function with explicit inputs and no hidden globals;
- one thin `tasks.py` verb or entrypoint that parses arguments and delegates;
- one templated per-well rule;
- one focused contract/validator for the feature table it owns;
- tests in the parallel `tests/data_pipeline/...` tree.

Feature paths must come from the orchestration registry, not `feature_extraction/io/paths.py` or
raw strings in rules. Feature code may import identity constructors/parsers when it needs IDs, but it
must not mint or split IDs inline.

Each file written for feature work must pass the `pipeline_file_philosophy.md` first-read check:
clear names, explicit signatures, flow-order organization, and fail-loud errors that name the fix.

---

## Start Order

1. **Mask geometry** - first computed feature to start once segmentation Session A/B contracts exist.
2. **Curvature metrics** - follows mask geometry after mask decode/read policy is stable.
3. **Pose and kinematics** - follows stable `track_id` and temporal ordering from segmentation
   Session B.
4. **Consolidated features** - follows at least one per-well feature shard and defines the merge
   contract.
5. **Stage predictions** - follows geometry/consolidation because it consumes morphology features.
6. **Fraction alive** - blocked until auxiliary/VIA mask products are clearly specified.

---

## Mask Geometry

**What it computes:** area, perimeter, centroid, width/height-style geometry, and calibration-aware
micron-scale measurements for each valid object mask.

**Depends on:**

- validated `frame_masks` rows with parseable `mask_id` / `track_id`;
- pixel calibration from `frame_inventory`;
- mask RLE/decode and geometry helpers from segmentation Session A;
- no-mask placeholders filtered out or handled explicitly before feature computation.

**Needs before coding:**

- feature table contract: one row per valid mask/object observation;
- source of pixel size named in the input contract;
- decision that this feature reads canonical masks, not backend-native SAM2 outputs.

**Done when:**

- synthetic masks produce deterministic geometry metrics;
- invalid or placeholder masks fail loud or are excluded by contract;
- per-well output validates and can be merged without experiment-grain assumptions.

---

## Curvature Metrics

**What it computes:** centerline length, centerline point count, and curvature summaries from valid
embryo masks.

**Depends on:**

- mask geometry inputs and calibration;
- stable binary mask decode/read policy;
- centerline/skeletonization behavior that is deterministic on synthetic masks.

**Needs before coding:**

- explicit behavior for masks with too few centerline points;
- clear units: curvature in inverse microns, lengths in microns;
- tests for straight, curved, tiny, and empty masks.

**Done when:**

- low-information masks return documented null metrics instead of silent bad numbers;
- synthetic fixtures pin centerline and curvature behavior;
- output joins cleanly by object identity with mask geometry.

---

## Pose And Kinematics

**What it computes:** orientation, bounding box dimensions, displacement, speed, coordinate deltas,
and elapsed-time deltas for each tracked object.

**Depends on:**

- valid object masks and mask geometry;
- stable `track_id` from the segmentation contract;
- frame time carried through `frame_inventory`;
- deterministic ordering within each `track_id`.

**Needs before coding:**

- segmentation Session B fake-predictor path proving deterministic `track_id` and mask rows;
- clear first-frame behavior for displacement/speed nulls;
- validation that time deltas are positive within each track.

**Done when:**

- synthetic two-frame and three-frame tracks produce expected displacement/speed;
- missing or non-monotonic time fails loud with the track and frame named;
- first observation per track has documented null kinematics.

---

## Consolidated Features

**What it computes:** the merged per-object/per-snip feature table used by downstream model/QC and
analysis-ready stages.

**Depends on:**

- one or more validated per-well feature shards;
- shared identity keys across feature tables;
- registry-supported per-well and merged artifact paths.

**Needs before coding:**

- canonical join key, expected to be `snip_id` or the post-segmentation object identity chosen by
  the object contract;
- collision policy for overlapping columns;
- explicit list of required core feature columns for the consolidated contract.

**Done when:**

- shard merge is one-to-one on the chosen key;
- duplicate keys and column collisions fail loud;
- downstream QC can read the consolidated contract without feature-specific path knowledge.

---

## Stage Predictions

**What it computes:** developmental stage prediction from morphology/size features.

**Depends on:**

- consolidated or selected geometry feature inputs;
- stage inference model/rule already present in `feature_extraction`;
- stable feature names and units.

**Needs before coding:**

- feature inputs required by the stage model;
- model/version provenance fields if predictions become a persisted contract;
- null behavior when required morphology features are missing.

**Done when:**

- deterministic fixture rows produce expected stage predictions;
- missing feature inputs fail loud;
- output can be merged with consolidated features without changing upstream feature contracts.

---

## Fraction Alive

**What it computes:** continuous viability fraction from embryo masks and auxiliary/VIA masks.

**Depends on:**

- canonical embryo/object masks;
- auxiliary/VIA mask product contract and paths;
- clear join between object mask rows and auxiliary mask rows by `image_id` or object identity.

**Needs before coding:**

- auxiliary mask world or QC-world decision on VIA mask ownership;
- explicit behavior when VIA masks are missing;
- tests for empty embryo mask, full dead tissue mask, no overlap, and partial overlap.

**Done when:**

- feature computation uses canonical mask products, not legacy path guesses;
- missing auxiliary masks fail loud with the expected source named;
- output joins cleanly into consolidated features.

---

## Not This World

- Detection, segmentation, tracking, and prompt adaptation live in detect/seg/track specs.
- QC flags live in a future QC world.
- Analysis-ready joins live after feature and QC consolidation.
- GPU SAM2 validation lives in segmentation Session C, not feature computation.
