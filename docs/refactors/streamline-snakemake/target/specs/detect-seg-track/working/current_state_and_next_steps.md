# Current State & Next Steps — detect-seg-track STATUS doc

**Status:** the "where is detect-seg-track RIGHT NOW" anchor. The top `⭐ CURRENT SNAPSHOT` block is
the live truth. Design lives in `targets/` (the world specs); unresolved decisions live in
`working/open_questions.md`; the folder map is `README.md`. This is the detect-seg-track twin of the
front-half `target/current_state_and_next_steps.md` — keep the two separate, do not merge them.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-21 (first slice landed: standalone `frame_detection` stage, CSV, no DAG change)

**What shipped (code, no DAG change):** the standalone `detection/` stage now exists BESIDE the live
inline path — first slice per the 2026-06-18 plan. New `src/data_pipeline/detection/`:
`frame_detections_contract.py` (imports `DOWNSTREAM_FRAME_IDENTITY_BLOCK` from the inventory contract;
defines the detection block, `ALLOWED_BBOX_FORMATS=("xyxy_px_abs",)`, `CONFIDENCE_RANGE`,
`_det_none` helpers), `validate_frame_detections.py` (layered: `validate_frame_detection_block` schema
layer + composed `validate_frame_detections`), `kept_frame_detections.py`
(`kept_frame_detections(df, reference_frame_inventory=None)` + `read_frame_detections_csv`),
`run_frame_detection.py` (backend-agnostic router: `run_frame_detection_df` core +
`run_frame_detection` CSV path wrapper; `BACKENDS` registry), and `backends/groundingdino/` (config +
adapter REUSING `detect_embryos`/`filter_detections` verbatim, model INJECTED) +
`backends/detectron2/` (`NotImplementedError` stub, signature-conformant). The `is_kept`
flag-not-drop seam is implemented: rejected candidates AND `_det_none` no-candidate placeholders are
RETAINED. Output is CSV. The frame-identity block was added to the contract owner
(`image_materialization/frame_inventory_contract.py`): `DOWNSTREAM_FRAME_IDENTITY_BLOCK`,
`FRAME_IDENTITY_NULLABLE={"z_index"}`, and reusable `validate_frame_identity_block(df,
reference_frame_inventory, *, context=...)` — single source of truth, imported by detection. `z_index`
is synthesized NA (nullable), NOT added to inventory atoms.

**Verified state for context (NOT changed this session):** the live inline detection path in
`segmentation_and_tracking/pipelines/segmentation_and_tracking.py` is UNTOUCHED and still runs;
nothing in the DAG references `data_pipeline.detection` (grep-confirmed). Tests:
`tests/data_pipeline/detection/` (contract, layered validators, kept view + CSV round-trip,
groundingdino adapter with stubbed inference, router flag-not-drop + BF-only + backend routing,
detectron2 stub) plus extended `test_frame_inventory_contract.py` for the identity validator — all
green (`125 passed` across detection + image_materialization). Test helpers are inlined per file
(no shared `_helpers.py`, no detection-local conftest, `tests/` is NOT a package) — house-consistent
with `test_frame_inventory.py`.

**Naming decisions locked this session (mdcolon):** new detection/identity validators take the trusted
inventory as a read-only positional `reference_frame_inventory` (validators do NOT read files) and an
error-prefix kwarg `context=` — NOT `scope_label` ("scope" collides with microscope-scope YX1/Keyence).
Two-layer doctrine: schema = "right bones"; reference = "bones belong to the right body".

**What's broken/half-done:** nothing broken. The stage is dormant — built and tested but unwired. No
Snakemake rule, no `PIPELINE_STEPS` entry, no `tasks.py` verb; the segmentation runner still produces
its own `frame_detections.parquet` inline. (Pre-existing unrelated working-tree edits to `Snakefile`
and `tasks.py` are from earlier front-half sessions, not this slice.)

**Next concrete action:** wire the stage into the DAG (next slice): add the Snakemake rule +
`PIPELINE_STEPS` registry entry + `tasks.py` verb that runs `run_frame_detection` per validated
frame_inventory; then flip the segmentation runner to consume the validated CSV via
`kept_frame_detections()` and retire the inline parquet path. Detectron2 real backend remains deferred.

**Open decisions:** first Detectron2 model name/config/weights (stub interface is ready);
boxes-vs-masks-vs-points; per-frame vs seed-only detection; provenance sidecar vs columns. See
`working/open_questions.md`.

---

## ⭐ CURRENT SNAPSHOT — 2026-06-18 (planning complete; NEXT = wire standalone `frame_detection`)

**What shipped (docs only, no code):** the target world specs for this space are written —
`targets/detection_world.md`, `targets/segmentation_world.md`, `targets/tracking_world.md`,
`targets/adapter_seams.md`, and `targets/overall_goal_and_plan.md`. No detect-seg-track *code* has
been refactored yet; this is the planning → build handoff point.

**Verified state for context (NOT changed this session):** detection currently runs **inline** inside
`src/data_pipeline/segmentation_and_tracking/pipelines/segmentation_and_tracking.py` (≈ lines
185–222). It loops frames → calls the GroundingDINO ingestor
(`segmentation_and_tracking/ingestors/gdino_ingestor.py`, which wraps `detect_embryos` /
`filter_detections` in `segmentation/grounded_sam2/gdino_detection.py`) → normalizes via
`segmentation_and_tracking/normalizers/normalize_frame_detections.py` → writes
`frame_detections.parquet` to the per-well shard `contracts/` dir, then does seed selection + SAM2
propagation in the same function. The live schema
(`src/data_pipeline/schemas/segmentation.py::REQUIRED_COLUMNS_FRAME_DETECTIONS`) DIFFERS from the
target in `detection_world.md`:

| Aspect | Live (inline) | Target (`detection_world.md`) |
|---|---|---|
| temporal col | `time_int` | `time_index` |
| bbox cols | `box_x_min_abs` … | `bbox_x_min_px` … + `bbox_format` |
| detection id | `detection_index` + `detection_instance_id` | `detection_id` (`{image_id}_det{idx:04d}`) |
| keep seam | none (rejected/empty rows dropped) | `is_kept` + `_det_none` placeholder rows |
| class label | none | `class_label` |
| backend prov | `source_backend`/`source_model`/`model_release` | `detector_backend`/`detector_model_id` |
| artifact | parquet | CSV |
| validators | none for frame_detections | layered: identity + detection blocks, composed |
| consumer view | hand-rolled filter | `kept_frame_detections(df)` helper |

The frame-identity block is owned by
`src/data_pipeline/image_materialization/frame_inventory_contract.py` (atoms + derived `well_id` /
`image_id` + `frame_inventory_image_ids` membership helper). `detection_world.md` says the downstream
identity block should be imported from there, adding it to that module first if absent.

**What's broken/half-done:** nothing new — planning is done, no build started.

**Next concrete action:** wire the standalone `frame_detection` stage to the `detection_world.md`
contract. Recommended **first slice** (safe to land alone, no DAG change): new `detection/` module +
`frame_detections_contract.py` + layered validators (`validate_frame_identity_block` exported from the
frame_inventory contract owner, composed with a detection-block validator) + `kept_frame_detections()`
consumer view + a GroundingDINO backend that **reuses** `detect_embryos` (do not rewrite inference)
but routes filtering through a flag-not-drop `is_kept` seam and emits CSV with rejected + no-candidate
placeholder rows + tests — all **without** yet ripping detection out of the segmentation runner. Later
slices: Snakemake rule + `PIPELINE_STEPS` registry entry + `tasks.py` verb; flip the segmentation
runner to consume the validated CSV via `kept_frame_detections()`; Detectron2 backend stub.

**Locked decisions (mdcolon, 2026-06-18):**
1. **Detection module home = `src/data_pipeline/detection/`** — a sibling of `segmentation/`, per
   `targets/segmentation_world.md` "keep detection outside `segmentation/`". The literal
   `segmentation/detection/` path in `detection_world.md`'s Backend Region is superseded; treat that
   sketch as relative shape, not the path.
2. **First slice = contract + validators + tests, CSV only** — build `detection/`,
   `frame_detections_contract.py`, layered validators, `kept_frame_detections()`, and a GroundingDINO
   backend that **reuses** `detect_embryos` (no inference rewrite) emitting spec CSV with rejected +
   no-candidate placeholder rows + tests. Do **NOT** touch the live segmentation runner or Snakemake
   this pass — nothing in the DAG changes; fully revertible. Snakemake rule, segmentation flip, and
   the back-half cutover are later, separate slices.
3. **Detectron2 = stub now, implement later** — create the `detection/backends/detectron2/` folder as
   a `NotImplementedError` stub that conforms to the adapter interface (returns rows with `is_kept` +
   `bbox_format`), only to prove the router is backend-agnostic. Real implementation deferred until the
   Facebook model name/config/weights are chosen.

**`z_index` carve-out (not a decision, a fact to honor):** `z_index` is not currently a
`frame_inventory` column (NA on BF/projection rows). The downstream identity block/validator must
**synthesize `z_index` as NA** for the MVP rather than hard-require it; do **not** add it to the
frame_inventory atoms in this work.

**Still open** (see `working/open_questions.md`): first Detectron2 backend name/config;
boxes-vs-masks-vs-points; per-frame vs seed-only detection; provenance sidecar vs columns.

---

### How to use this log

Each session, prepend a new dated `⭐ CURRENT SNAPSHOT — <date> (<one-line session theme>)` block at
the TOP (newest-first), using the same five fields: **What shipped**, **Verified state for context
(NOT changed this session)**, **What's broken/half-done**, **Next concrete action**, **Open
decisions**. Keep it lean — this log tracks detect-seg-track only; front-half state lives in the
top-level `target/current_state_and_next_steps.md`.
