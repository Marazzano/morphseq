# Plan: input sufficiency checks for served (model-server) steps

Status: proposed, not started. Read-only investigation only produced this document; no pipeline
source has been changed.

## Context

On 2026-08-31, one well (H09, zero kept GroundingDINO detections — an empty-well sentinel, not a
bug) killed 87 already-succeeded `frame_masks` shards and stalled a 95-well plate at 3% of steps.
Full account: "An empty well (zero kept detections) crashes frame_masks for the WHOLE plate" in
`src/data_pipeline/docs/dev/todos/ORCHESTRATION_TODOS.md`. That specific case is already fixed
(commit `d2548f01`): `has_kept_detections()` now lives beside the contract that mints `is_kept` /
`_det_none` (`object_extraction/detection/frame_detections_contract.py:95`), and both SAM2 call
sites — `tasks.py` (in-process) and `model_servers/adapters/sam2.py:136` (served) — check it before
prompting and, when false, skip the model and write `unprompted_frame_masks(model_inventory)`.

This doc asks the broader question the incident raised: **should served steps get an extra,
narrower layer of input checking that non-served steps don't need**, and if so, where does it live
and how does it actually intercept anything given Snakemake's execution model. This is *not* a
proposal to add defensive coding generally — the user was explicit that the pipeline should stay
non-defensive except at this one high-risk seam.

### The asymmetry, verified

A served client shares a Snakemake **group** with its `service()` rule. Confirmed identical across
all three served steps:

| step | service rule | client rule | socket helper |
|---|---|---|---|
| frame_detections | `service_grounding_dino` (`frame_detections.smk:47`) | `frame_detections_per_well_served` (`:83`) | `service_socket_pattern("gdino")` |
| frame_masks | `service_sam2` (`frame_masks.smk:121`) | `frame_masks_per_well_served` (`:152`) | `service_socket_pattern("sam2")` |
| snip_auxiliary_masks | `service_unet_aux_masks` (`snip_auxiliary_masks.smk:67`) | `build_snip_auxiliary_masks_for_well_served` (`:93`) | `service_socket_pattern("unetaux")` |

In every case `output: socket=service(_..._socket_pattern())` on the service rule, and the client
rule takes `socket=_..._socket_pattern()` as an `input:`. That edge is what forces them into one
group — a client can only run while the server lives, so Snakemake schedules them together.

`profiles/default/config.yaml` sets `keep-going: true`, which is documented there as operating on
**jobs** ("isolated to its own well"). A **group** fails as a unit regardless of `keep-going`
(`ORCHESTRATION_TODOS.md:226-230`). So the asymmetry is real and applies identically to all three
served steps, not just SAM2:

- Bad input to a **non-served** step (`frame_masks_per_well`, the `model`-mode fallback) costs one
  well — `keep-going` isolates it, siblings continue.
- The same bad input to a **served** step costs every well the group's in-flight service is
  currently serving — observed at 87/88 on 2026-08-31, could be worse on a bigger plate.

That asymmetry, not general paranoia, is the entire justification below.

## 1. Served-step inventory

| served step | client `input:` files | adapter | where it would raise on insufficient input |
|---|---|---|---|
| frame_detections (GroundingDINO) | `frame_inventory` (+ `_validated` sentinel) | `model_servers/adapters/grounding_dino.py` | nowhere in the empty case — see §2 |
| frame_masks (SAM2) | `frame_inventory`, `frame_detections` (+ `_validated` sentinel) | `model_servers/adapters/sam2.py` | `validate_sam2_prompts` (`prompt_detections.py:84-87`), called at `sam2.py:149`, if `has_kept_detections` were absent |
| snip_auxiliary_masks (4x UNet) | `snip_inventory` (+ `_validated` sentinel) | `model_servers/adapters/unet_aux_masks.py` | nowhere in the empty case — see §2 |

All three adapters are registered via `model_servers/adapter_base.py:register_adapter` and run
inside `harness.py`'s per-request dispatch (`harness.py:170` `self.adapter.handle(request.payload)`
under `_dispatch_lock`).

## 2. "Insufficient input" conditions that are normal data, per step

The framing assumed all three steps have an H09-shaped failure mode. **That assumption is only
true for SAM2.** Tracing each adapter's call chain down to its validator:

**frame_detections (GroundingDINO).** `handle()` calls `run_frame_detection_df`
(`run_frame_detection.py:109`), which filters to BF projection rows (`_projection_bf_rows`,
`:46-60`) and loops over them, appending detection rows. **An empty loop body is not an error** —
it just appends nothing, producing an empty `frame_detections` DataFrame. The validator explicitly
allows this: `validate_frame_detection_block` (`validate_frame_detections.py:70-75`) checks
required columns, then `if len(df) == 0: return` — a zero-row table is a valid shape by contract.
So "a well with no BF frames in its frame_inventory" is already handled without any special-casing
in the adapter; nothing here can crash the group the way H09 crashed SAM2.

**snip_auxiliary_masks (UNet x4).** `handle()` calls `run_unet_for_snip_inventory`
(`run_unet_snip.py:78`), which filters to `is_valid_snip & processed_snip_path exists`
(`:100-103`) and loops over that filtered set. Same shape as GroundingDINO: an empty filtered set
just means an empty `rows` list, an empty output DataFrame, no exception. `validate_snip_auxiliary_masks`
(`snip_auxiliary_masks_contract.py:51`) is column-presence/dtype/`.any()` checks throughout — every
one of them is vacuously true on an empty DataFrame. `validate_snip_auxiliary_masks_against_snip_inventory`
(`:102`) likewise degrades safely (`set(auxiliary_masks["snip_id"]) - known_snip_ids` is `set() - X = set()`
when empty). So "a well where every snip failed upstream QC (`is_valid_snip=False` for all rows)"
is already handled — same conclusion as GroundingDINO.

**frame_masks (SAM2) — the one real case.** SAM2 is not a per-row map; it needs a non-empty seed
box to call `predictor.add_new_points_or_box` at all. That's structurally different from the other
two, which degrade to "do nothing to zero rows." The already-shipped `has_kept_detections` guard
(`sam2.py:136-147`) is exactly this special case, and it is the *only* one of the three that needs
one.

So, revising the framing's premise: **there is no GroundingDINO- or UNet-side analog of the H09
bug today.** Both of those adapters are naturally total functions over their filtered input,
including the empty case, all the way through their validators. This was verified by reading the
loop bodies and every validator function referenced above, not assumed.

What *would* still be a genuine bug (must fail loud) for each step:
- **frame_detections**: `frame_inventory` missing required columns, or `image_path` files that
  don't exist — `validate_frame_identity_block` / reading a missing file raises, correctly.
- **frame_masks**: `frame_detections` referencing `image_id`s absent from `frame_inventory`
  (`select_segmentation_frame_view` raises on that — `prompt_detections.py:161-163` — and on an
  empty resulting view) — a detection stage that ran but disagrees with the inventory it read.
- **snip_auxiliary_masks**: an off-grid snip image (`assert_on_snip_frame`,
  `run_unet_snip.py:118`) — "affecting every snip in the well identically," deliberately left
  un-caught per that function's own docstring.

None of these should be softened; they're the "genuinely broken" side of the same distinction the
SAM2 fix already drew.

## 3. Where should the check live

The user's candidates: "in the contract code, or alongside the validator as an input validator."
Weighed against `PIPELINE_PHILOSOPHY.md` P6/P7:

- **P6** — a contract lives with the code that **mints** it. `has_kept_detections` already follows
  this: it lives beside `frame_detections_contract.py`'s `is_kept`/`_det_none` vocabulary, i.e.
  with the **producer** (frame_detections), not the **consumer** (SAM2) that needs to ask the
  question. This is deliberate, not incidental — the doc says so directly (`ORCHESTRATION_TODOS.md:196-201`).
- **P7** — one validator per contract, with a **mode flag** rather than forked validators.
  `check_sources` (referenced in the task and used elsewhere in the validator suite, e.g.
  `validate_yx1_acquisition_inventory(check_sources=True)`) is the precedent: same validator,
  different rigor depending on when it's called (write-time vs read-time).

**Does "predicate beside the producer's contract" generalize?** Yes, and the investigation in §2
shows why it already *has* generalized correctly, just not visibly: the reason GroundingDINO and
UNet don't need a `has_valid_input_for(...)` predicate is that their consumers (the served adapter)
never needed to branch on sufficiency in the first place — an empty filtered set flows through
unchanged. The one consumer that *does* need to branch (SAM2 on `is_kept`) already has its
predicate living correctly beside the producer contract that minted the vocabulary it reads.

**Recommendation: no new module, no per-served-step input validator, no mode flag.** The shape the
user is asking about — "a check that everything going into the server is sufficient, living beside
the validator, run before the server is called" — **already exists and is already correctly
placed**, for the one step where the model actually requires non-empty input to run at all. Adding
a parallel `is_sufficient_for(...)` predicate to the frame_detections or snip_auxiliary_masks
contracts today would be exactly the "general defensive coding" the user asked NOT to do: there is
no known insufficiency condition there for it to guard against, so it would be an unjustified
predicate with no caller who needs it (violates P6's "a payload column must earn its place" review
question applied to predicates, not just columns).

The generalizable **rule**, for whoever adds a fourth served step later, is:

> Before wiring a model into a served adapter, check whether the model requires non-empty/non-trivial
> input to run (SAM2 does; a per-row detector or per-row UNet predictor does not). If it does, the
> sufficiency predicate belongs beside the **producer** contract that mints the column the adapter
> would branch on (P6), named for what it answers (`has_kept_detections`, not `is_input_ok`), and
> the adapter's `handle()` calls it before doing the expensive/stateful part of the work — exactly
> the `sam2.py:136` pattern.

This is a documentation/process recommendation (write it into `MODEL_SERVER_WIRING.md` or beside
the adapter README), not a code change, because there is currently nothing left to fix.

## 4. The mechanism question — where CAN this check run

This was the part flagged as unclear, and it's worth being precise because two of the four
candidate insertion points genuinely don't work.

**At DAG-build time — no.** Confirmed by reading the rules: `frame_detections`, `frame_masks`, and
`snip_auxiliary_masks` for a given well are themselves Snakemake outputs of *earlier* rules in the
same DAG (`frame_inventory` comes from materialization; `frame_detections` from the previous
served/non-served step). At DAG-build time, Snakemake resolves wildcards and dependency edges from
declared `input:`/`output:` patterns — it does not read file *contents*. The CSV that would tell you
"H09 has zero kept detections" does not exist on disk until the frame_detections job for H09 has
actually run, which for a `service()` group happens at the same scheduling moment as everything
else in the group. There is no hook in Snakemake's rule-definition phase that can inspect not-yet-materialized
row data.

**In the client process before it opens the socket — this is the only real insertion point, but
only for SAM2's case.** `client.py`'s `main()` (`client.py:136-171`) reads `--payload-json`, which
already contains real, resolved file *paths* (`frame_inventory_csv`, `frame_detections_csv`, etc.)
— by the time the client rule's shell command runs, Snakemake has already confirmed those input
files exist on disk (that's what makes the rule's `input:` edges resolvable at all). So the client
*could* `pd.read_csv` the detections file and check `has_kept_detections` before calling
`call_server(...)`, and skip the socket entirely — this would work mechanically. But it would
duplicate `has_kept_detections`, duplicate `unprompted_frame_masks`, and duplicate
`validate_frame_masks` into the generic, adapter-agnostic `client.py`, which the module's own
docstring says must "never import torch or any model library" and "know nothing about what the
adapter does." Putting model-family-specific skip logic in the client breaks that boundary for no
benefit — the adapter already does the identical check, correctly, one layer down.

**In the adapter, server-side, before `handle()` does real work — this is where it lives today,
and the framing's suspicion is correct that this alone does NOT solve the group-failure problem.**
`harness.py:170` isolates the *server process* correctly regardless of what the adapter does: any
exception from `handle()` is caught, logged, and returned as `Response(ok=False, ...)` without
killing the server or any other in-flight request. That was already true before the SAM2 fix landed
— it is what let the server log 87 `ok` alongside the 1 `FAILED`. The isolation was never the
problem. **The mechanism that actually prevents the blast radius is downstream of the server
entirely**: `client.py:168-171` — `if not response.ok: return 1` — that `1` is what fails the
Snakemake job, and because the job is in the service's group, that failure propagates to the whole
group. So the fix has to make `response.ok` **True** for the "empty well" case, which means the
adapter must not raise for it — which is exactly what `has_kept_detections` does: it branches
*before* calling anything that would raise, writes a valid (if trivial) artifact, and returns
normally.

**Does the client exit 0 having written a valid empty artifact — is that the real mechanism?**
Yes, precisely. Trace it end to end: `sam2.py:136-147` detects the empty-well case, writes
`unprompted_frame_masks(model_inventory)` (a real, contract-valid frame_masks table — "not a
zero-row table," per its own docstring, but one explicit no-mask row per frame) via
`atomic_write_via`, and returns `None` normally. `handle()` returning normally means
`harness.py`'s `except Exception` branch is never entered, so `response = Response(ok=True, ...)`
(`harness.py:172`). The client's `call_server` gets that response, `main()` checks
`response.ok` (True) and returns 0. Snakemake sees exit 0 and a populated output file at the exact
path it expected — indistinguishable from any other successful well. The group survives. **This is
the actual, and only, mechanism**: not a pre-flight check that intercepts anything, but making the
adapter's definition of "success" wide enough to include "correctly did nothing," so the group
never sees a failure to begin with.

## 5. Cost and blast radius of the proposal

Given §3's conclusion, the "proposal" is: **no code changes.** Costed anyway, for the record:

- **Call sites touched**: 0 today. If a future served step needs this pattern, it's 1 predicate
  (beside the relevant producer contract) + 1 call site (in that adapter's `handle()`, before the
  expensive step) + reusing the existing artifact-for-"nothing here" shape if the contract already
  has one (as `no_mask_frame_mask_row` did for frame_masks) or defining one if it doesn't.
- **New module**: none. Extends the existing contract file the producer already owns.
- **What it does NOT cover** — still kills a group after this lands, correctly:
  - Any exception any adapter raises for a condition that is NOT one of the enumerated normal-data
    cases: malformed CSV, missing image files, `select_segmentation_frame_view` raising on a
    detection/inventory mismatch, an off-grid snip in `assert_on_snip_frame`, a CUDA OOM, a corrupt
    checkpoint, a protocol/transport error in `client.py` (`OSError`, `ProtocolError`, `TimeoutError`
    all still return nonzero — `client.py:159-166`).
  - A genuinely bad frame_detections shard (e.g. `is_kept=True` but nonsense bbox coordinates) —
    `validate_sam2_prompts` still runs and still raises for that, deliberately not relaxed.
  - Any server-process-level failure (the model process itself crashing, OOM-killed) — that fails
    the whole group by construction; a resident server dying mid-run is out of scope for this doc
    (see "the blast radius is a SEPARATE, still-open problem" in `ORCHESTRATION_TODOS.md:218-241`,
    option 3, not addressed here).
  - Concurrent normal-data conditions arriving on the same run that nobody has enumerated yet —
    this doc's job was to enumerate the three current served steps exhaustively, not to predict
    future ones.

## What was wrong in the original framing, checked against the code

- **Assumed** GroundingDINO and the UNets have an H09-shaped failure mode analogous to SAM2's.
  **Not true** — both degrade to a valid empty output through their existing filter-then-loop
  structure and existing validators, with no adapter-level guard needed. Verified by reading
  `run_frame_detection_df`, `run_unet_for_snip_inventory`, and every validator function each one
  calls.
- **Implied** the fix might need a new "input validator" module or a mode flag on an existing
  validator (per the P7 `check_sources` precedent). **Not needed** — the existing predicate
  (`has_kept_detections`) already IS the input-sufficiency check the user is describing, already
  lives in the P6-correct place (beside the producer), and there is nothing else today that needs
  one.
- Everything else in the framing — the group-as-failure-unit mechanism, `keep-going` operating on
  jobs not groups, DAG-build-time being impossible, server-side isolation not being the missing
  piece, and "the client exiting 0 with a valid empty artifact" being the actual mechanism — was
  confirmed exactly as stated against `frame_masks.smk`, `frame_detections.smk`,
  `snip_auxiliary_masks.smk`, `profiles/default/config.yaml`, `client.py`, and `harness.py`.

## Test section

No code changes proposed, so no new tests. If the §3 rule is later applied to a fourth served step,
the pattern to test (mirroring the existing SAM2 coverage) is:

- A unit test on the new sufficiency predicate itself, beside its contract's existing tests
  (pattern: wherever `has_kept_detections` is tested today, alongside `frame_detections_contract.py`).
- An adapter-level test asserting `handle()` does NOT raise, and writes the valid "nothing here"
  artifact, when given the insufficient-but-normal input shape (pattern:
  `model_servers/tests/test_sam2_adapter.py`). **Checked: this test file currently has no
  `has_kept_detections` / empty-well case** — grepped for `has_kept_detections`, `empty`, `no_mask`,
  `EMPTY_WELL` and found none. The SAM2 fix (`sam2.py:136-147`) shipped without adapter-level
  regression coverage for the exact incident that motivated it. That gap is real and is the
  single concrete, actionable follow-up this investigation surfaced — even though it is a test-only
  change and this doc's mandate was read-only, it is worth flagging explicitly rather than silently
  noting it in passing.
- No group-level / Snakemake-level test is feasible in CI (services require GPU + real sockets);
  the existing incident write-up in `ORCHESTRATION_TODOS.md`, keyed to job `24155677`, is the
  closest thing to a regression record for the group-failure mechanism itself.
