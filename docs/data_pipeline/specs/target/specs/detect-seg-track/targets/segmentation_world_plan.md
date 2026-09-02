# Segmentation World Plan - narrowed Session A

**Status:** approved narrowed Session A, mdcolon 2026-06-22. Do not code the full segmentation
world plan in one session. This document is the stage contract for the first functional slice only.

**Session A goal:** establish shared identifier grammar, segmentation mask utilities, and generic
frame-mask contract/validators without touching SAM2 output adaptation, seed selection, or the
well-level segmentation runner.

**Defer beyond Session A:** Session B owns the deterministic fake-predictor end-to-end segmentation
contract. Session C owns the real GPU SAM2 smoke and legacy cleanup.

---

## Stage Logic

Each stage must end with:

- a small functional artifact that can be used by the next stage;
- focused synthetic tests for the new behavior;
- tests under the parallel `tests/data_pipeline/...` tree, not new package-local `src/.../tests/`
  folders;
- a per-file philosophy review after every file is written or edited;
- a commit that contains only that stage;
- an update to `docs/data_pipeline/specs/target/current_state_and_next_steps.md`
  recording what shipped, what remains deferred, and the next concrete stage.

Do not batch multiple stages into one commit. If a stage reveals extra cleanup, leave it as an
explicit follow-up unless it blocks that stage's functional artifact.

---

## Non-Negotiables For Session A

1. Shared identifier API:
   - `build_mask_id`
   - `build_no_mask_id`
   - `build_track_id`
   - `parse_mask_id`
   - `parse_track_id`

2. `parse_mask_id(mask_id)` returns exactly:

   ```python
   (image_id, local_mask_index, is_no_mask)
   ```

   `local_mask_index is None` for no-mask placeholders.

3. Track IDs remain zero-based-compatible:

   ```python
   build_track_id("WELL", 0) == "WELL_track0000"
   ```

   Do not apply one-based embryo indexing inside segmentation.

4. Use the backend path:

   ```text
   segmentation/backends/sam2_video/
   ```

   Do not introduce `segmentation/sam2_video/`.

5. Keep generic frame-mask validation separate from SAM2 prompt validation:
   - `validate_frame_mask_block(df)`
   - `validate_frame_masks(df, frame_inventory)`
   - SAM2 prompt validators are optional in Session A and only if existing code/tests already make
     the migration low-risk.

6. Do not delete `prompt_seeds.py` in Session A. Leave it untouched by default; edit only if imports
   force a tiny compatibility change.

**Doctrine:** identifiers are shared; mask math is segmentation; contracts are generic; SAM2
adaptation waits.

**Pipeline file philosophy applies throughout Session A:**

- no inline ID minting or splitting outside `src/data_pipeline/shared/identifiers/`;
- identifier grammar uses named constants/helpers, not scattered magic strings;
- functions take explicit arguments and do not read haunted globals;
- validators live where the contract lives and fail loud with messages that name the fix;
- tests pin public contracts and failure modes, not implementation details;
- multi-concern files use flow-order banners only when concerns legitimately co-live; split files
  instead of mixing identity, segmentation math, contracts, or SAM2 adaptation.

Every file written in Session A must be checked immediately after editing against
`pipeline_file_philosophy.md`. The file should be well organized, self-documenting, and clear on a
first read: names explain what the code is, docstrings/comments explain why or lifecycle moment, and
the top-to-bottom order follows dependency/flow order. If that review finds a file is hard to scan,
the stage is not done.

---

## Stage A1 - Shared Segmentation Identifiers

**Goal:** one constructor/parser vocabulary for mask IDs, no-mask placeholders, and track IDs.

**Files**

- EDIT `src/data_pipeline/shared/identifiers/constructors.py`:
  - `build_mask_id(image_id, local_mask_index)`
  - `build_no_mask_id(image_id)`
  - `build_track_id(well_id, track_index)`
- EDIT `src/data_pipeline/shared/identifiers/parsers.py`:
  - `parse_mask_id(mask_id)`
  - `parse_track_id(track_id)`
- EDIT `src/data_pipeline/shared/identifiers/__init__.py` to export the shared API.
- UPDATE `src/data_pipeline/shared/identifiers/README.md` if the new mask/track grammar needs
  documentation.
- CREATE `tests/data_pipeline/shared/identifiers/test_identifiers.py` or update the existing
  parallel identifier tests for round trips, malformed strings, no-mask placeholders, and zero-based
  track indices.

Do not create identifier helpers under `segmentation/`. The string grammar is shared and opaque
outside `shared/identifiers`.

Do not mint mask/track IDs with inline f-strings anywhere outside `constructors.py`. Do not parse
them with inline `.split(...)` anywhere outside `parsers.py`.

**Functional artifact:** downstream code can mint and parse canonical IDs without knowing string
format details.

**Verification**

- Synthetic tests prove:
  - `parse_mask_id(build_mask_id(image_id, i)) == (image_id, i, False)`;
  - `parse_mask_id(build_no_mask_id(image_id)) == (image_id, None, True)`;
  - `build_track_id("WELL", 0) == "WELL_track0000"`;
  - `parse_track_id("WELL_track0000") == ("WELL", 0)`;
  - `build_track_id("20250912_B01", 0) == "20250912_B01_track0000"`;
  - `parse_track_id("20250912_B01_track0000") == ("20250912_B01", 0)`;
  - `parse_mask_id("20250912_B01_BF_t0007_m0001") == ("20250912_B01_BF_t0007", 1, False)`;
  - `parse_mask_id("20250912_B01_BF_t0007_mask_none") == ("20250912_B01_BF_t0007", None, True)`;
  - `build_track_id` rejects negative `track_index` values;
  - `parse_track_id` rejects malformed IDs but allows `track0000`.

The underscore-bearing examples are required: parsers must tolerate underscores inside both
`well_id` and `image_id`.

**Commit boundary:** `segmentation identifiers: add shared mask and track ID helpers`.

**End-of-stage status update:** update `current_state_and_next_steps.md` with the commit SHA,
the exact helper names, the zero-based track decision, and Stage A2 as next.

---

## Stage A2 - Mask RLE Utility

**Goal:** add a small, tested RLE utility for binary masks without pulling in frame-mask contract
changes yet.

**Files**

- CREATE `src/data_pipeline/segmentation/masks/mask_rle.py`.
- CREATE `tests/data_pipeline/segmentation/masks/test_mask_rle.py` with synthetic tests covering:
  - empty masks;
  - single-pixel masks;
  - multi-component masks;
  - encode/decode round trip;
  - dtype/shape validation failure paths.

**Functional artifact:** a reusable mask RLE encode/decode module with deterministic behavior on
small synthetic arrays.

**Verification**

- Focused tests pass.
- No frame-mask schema or SAM2 prompt migration is included in this commit.

**Commit boundary:** `segmentation masks: add RLE utility`.

**End-of-stage status update:** update `current_state_and_next_steps.md` with the commit SHA,
the supported RLE behavior, and Stage A3 as next.

---

## Stage A3 - Mask Geometry Utility

**Goal:** add geometric helpers for binary masks, isolated from contracts and SAM2.

**Files**

- CREATE `src/data_pipeline/segmentation/masks/mask_geometry.py`.
- CREATE `tests/data_pipeline/segmentation/masks/test_mask_geometry.py` with synthetic tests covering
  the geometry operations needed by frame-mask validation and later SAM2 adaptation, such as bounding
  box/area/centroid semantics if those are the local conventions.

**Functional artifact:** mask geometry can be computed from synthetic masks without touching frame
mask ingestion or SAM2 output.

**Verification**

- Focused tests pass.
- Geometry helpers reject invalid masks consistently with `mask_rle.py`.

**Commit boundary:** `segmentation masks: add geometry utility`.

**End-of-stage status update:** update `current_state_and_next_steps.md` with the commit SHA,
the exported geometry helpers, and Stage A4 as next.

---

## Stage A4 - Frame-Mask Vocabulary Migration

**Goal:** migrate the frame-mask contract and generic validators to the refined vocabulary. Generic
frame-mask validation must not require prompt tables.

**Files**

- EDIT `frame_masks_contract.py`.
- EDIT `validate_frame_masks.py`.
- EDIT `valid_frame_masks.py`.
- Add or update focused tests in the matching `tests/data_pipeline/...` path for contract columns,
  validation behavior, and no-mask placeholders.

**Vocabulary**

- `prompt_detection_id`: the detection/prompt source identifier.
- `sam2_object_id`: the object identifier as SAM2 understands it.
- `mask_id`: always constructor-minted; includes explicit no-mask placeholders.
- `track_id`: always constructor-minted; zero-based-compatible.

**Required Generic Validation**

- `validate_frame_mask_block(df)`
- `validate_frame_masks(df, frame_inventory)`
- `valid_frame_masks(df)`

Generic frame-mask validation must not require SAM2 prompt context.

The generic validator is the authoritative validator for the frame-mask contract. Keep lifecycle or
strictness differences as explicit options only if needed; do not create a second generic validator
that can drift.

**Optional SAM2 Prompt Validation**

Only if existing code already has SAM2 prompt validators/tests that can be migrated without creating
seed selection or SAM2 output adaptation code, A4 may define stubs or minimal validators under:

```text
segmentation/backends/sam2_video/
```

Optional names:

- `validate_sam2_prompts(...)`
- `validate_frame_masks_against_sam2_prompts(...)`

Otherwise defer SAM2 prompt validators to Session B. A4 must not introduce `seed_selection.py` or
require prompts for generic frame-mask validation.

**Session-A Mask ID Validation Policy**

- Validate `mask_id` parseability.
- Validate `mask_id` uniqueness.
- Validate `track_id` parseability for valid mask rows.
- Validate no-mask placeholder rows exactly:

  ```python
  mask_id == build_no_mask_id(image_id)
  ```

- Require `track_id` to be NA for no-mask placeholder rows.
- Do not require the generic validator to reconstruct deterministic actual-mask
  `local_mask_index` assignment. Session B's adapter owns actual-mask construction policy unless the
  existing implementation makes it trivial and already tested.
- Do not require `track_id == build_track_id(well_id, sam2_object_id)` in generic validation. That
  check belongs in SAM2-specific validation or Session-B adapter tests.
- Error messages must name the violated column and the expected fix, e.g. use constructor-minted
  IDs, make placeholder `track_id` NA, or remove duplicate `mask_id` rows.

**Functional artifact:** a frame-mask table can be validated using constructor-minted `mask_id` /
`track_id` values and explicit no-mask placeholder rows, independent of SAM2 prompt validation.

**Verification**

- Focused tests cover:
  - normal mask rows;
  - no-mask placeholder rows;
  - malformed hand-written IDs rejected by parser/validator;
  - duplicate mask IDs rejected;
  - no-mask placeholder rows must equal `build_no_mask_id(image_id)`;
  - valid mask rows must have parseable `track_id`;
  - no-mask placeholder rows must have NA `track_id`;
  - generic validator works without SAM2 prompt inputs;
  - any SAM2 prompt cross-check, if touched, remains separate.

**Commit boundary:** `frame masks: migrate contract to canonical mask vocabulary`.

**End-of-stage status update:** update `current_state_and_next_steps.md` with the commit SHA,
the final column vocabulary, which validation functions are generic vs SAM2-specific, and Stage A5
as next.

---

## Stage A5 - Legacy Prompt Seed Marker

**Goal:** skip by default. `prompt_seeds.py` is intentionally retained for Session B unless imports
force a tiny compatibility edit.

**Files**

- DEFAULT: leave `prompt_seeds.py` untouched and record that Session B owns migration/retirement.
- ONLY IF IMPORTS FORCE IT: make the smallest compatibility edit needed to keep imports green.

Do not delete `prompt_seeds.py`. Do not add a deprecation docstring just for optics. Do not perform
the Session-B cleanup.

**Functional artifact:** repository readers know `prompt_seeds.py` is intentionally retained for
Session B.

**Verification**

- Import tests and focused segmentation tests still pass.
- Grep confirms no accidental move to `segmentation/sam2_video/`.

**Commit boundary:** skip this commit if the file is intentionally left untouched. If imports force a
tiny compatibility edit, use `segmentation prompts: retain legacy prompt seeds`.

**End-of-stage status update:** update `current_state_and_next_steps.md` with the final Session-A
state, skipped/deferred work, and Session B as next.

---

## Session B - Fake-Predictor End-To-End Contract

Build only after Session A is committed and status-doced.

**Goal:** one well can run end-to-end through the segmentation contract with a fake predictor and
produce valid `frame_masks`, without requiring GPU or real SAM2 inference.

**Scope**

- `seed_selection.py`
- `segment_one_well`
- SAM2 prompt table vocabulary
- `validate_sam2_prompts(...)`
- `validate_frame_masks_against_sam2_prompts(...)`
- `adapt_sam2_output.py`
- fake predictor integration test

**Must not include**

- real SAM2 GPU invocation;
- model/checkpoint path plumbing;
- broad Snakemake production wiring beyond what is needed for the fake-predictor contract;
- deletion of `prompt_seeds.py` unless imports force a tiny compatibility edit;
- `Sam2WellInput` rename unless it is required to keep the fake contract coherent.

**End state**

- fake predictor creates deterministic masks and no-mask placeholders;
- adapter mints `mask_id` / `track_id` using shared constructors;
- actual-mask `local_mask_index` assignment is deterministic and tested by the adapter tests;
- generic `validate_frame_masks(...)` passes;
- SAM2 prompt cross-check passes separately;
- no inline ID minting/parsing appears outside `shared/identifiers`.

Session B should get its own staged plan before coding begins. Keep it deterministic; do not let GPU
availability pull real SAM2 into Session B.

---

## Session C - Real GPU SAM2 Smoke And Cleanup

Build only after Session B's fake-predictor contract is green. The next agent can assume GPU access
when needed for this session and should try to carry the plan through the real SAM2 smoke,
validation, and cleanup end state rather than stopping at scaffolding.

**Goal:** prove the same contracts survive a real, tiny, GPU-backed SAM2 run.

**Scope**

- real SAM2 backend invocation under:

  ```text
  segmentation/backends/sam2_video/
  ```

- tiny frame-capped one-well GPU smoke;
- model/config/checkpoint path handling;
- provenance/logging for SAM2 run inputs;
- real-output validation with generic frame-mask validation and SAM2 prompt cross-validation;
- cleanup/retirement decision for `prompt_seeds.py`;
- `Sam2WellInput` rename/move if still desired;
- final Snakemake/pipeline wiring if Session B intentionally stopped at runner-level integration.

**Must not include**

- changing the shared ID grammar;
- weakening generic validation to accommodate SAM2 quirks;
- replacing the fake predictor test with GPU-only coverage.

**End state**

- real GPU one-well smoke produces `frame_masks`;
- generic frame-mask validation passes;
- SAM2 prompt cross-validation passes;
- no-mask placeholders behave the same way as in fake-predictor tests;
- IDs are constructor-minted and parser-validated;
- old prompt-seed path is retired, quarantined, or explicitly documented as still-live.

Full arc:

```text
Session A = shared primitives + generic contracts
Session B = fake-predictor end-to-end segmentation contract
Session C = real GPU SAM2 smoke + legacy cleanup
```
