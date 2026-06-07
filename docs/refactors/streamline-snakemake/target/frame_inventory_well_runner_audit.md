# Audit — frame-inventory + well-runner work (🔍 REVIEW, 2026-06-07)

**Status:** code review of the frame-inventory adapter, the well-runner orchestration layer, and
the front-end Snakefile wiring on branch `mdcolon/20260222_docs_snakemake_remake`.
**Reviewer:** Claude (mdcolon-directed), 2026-06-07.
**Scope reviewed:**
- Commits `a2dd4f46` (well-runner shard helpers), `de74f00f` (frame-inventory scaffold),
  `ca78f010` (frame-inventory adapter rules).
- The uncommitted front-end wiring (Snakefile rename-to-TARGET + registry routing, `tasks.py`
  verb renames, `paths.py`/`well_runner.py` banner updates, `materialize_stitched_images.py`
  well-id filter) committed alongside this audit.
**Specs checked against:** [`front_end_naming_and_flow.md`](front_end_naming_and_flow.md),
[`well_id_throughline_refactor_plan.md`](well_id_throughline_refactor_plan.md),
[`stitched_handoff_contract.md`](stitched_handoff_contract.md).

> **Back-pointers:** this file is referenced from the code it audits so the findings are not lost:
> `orchestration/paths.py`, `orchestration/well_runner.py`,
> `metadata_ingest/frame_inventory/frame_inventory.py`, and
> `pipeline_orchestrator/rules/frame_inventory.smk` each carry an `AUDIT:` pointer here.

---

## Verdict

**Solid, spec-faithful, deliberately staged work.** The per-well/merge split, the no-leakage
identity discipline, and the fail-loud taxonomy all match the locked decisions. The frame-inventory
module is an **honest behavior-preserving adapter** over the legacy `frame_contract.csv` (Scope 2
hasn't run), and it says so everywhere. Verified: 63/63 unit tests pass
(`test_frame_inventory.py`, `test_well_runner.py`, `test_paths.py`); no merged-vs-legacy path
collision; the registry resolves every path to the doc's worked examples.

The findings below are mostly "weaker-than-TARGET because the prerequisite scope hasn't run" —
correct staging, but not always flagged **at the code**, only in the docs. A few are real
simplifications.

---

## What fits the spec well (keep)

- **`well_runner.py`** — the 2a/2b merge split (DAG-planning pure path-resolution vs. runtime
  disk-scan over `.validated` sentinels) is exactly the findings-doc distinction; the no-leakage
  rule (calls `build_well_id`/`validate_well_id`, never an inline f-string or `.split("_")`) is
  obeyed throughout; `output_root` is a parameter everywhere (Scope 3/4 hard rule).
- **`paths.py`** — `_normalize_path_mode` makes `fanout` executable (per_well vs merged must be
  chosen, with a message that says why); the merged template names the experiment and the per-well
  template names the well, so the registry never fabricates a `{well_id}` for the merged file.
- **The frame-inventory module is honest about being a bridge** — every docstring says
  "behavior-preserving adapter over the legacy `frame_contract.csv`." Given Scope 2 (the
  `frame_contract → frame_inventory` rename + regenerate) hasn't run, an adapter is the correct
  interim move.

---

## Findings (priority order)

### 1. 🔴 The frame-inventory rules are a dead DAG branch
`rule all` targets consolidated features, stage predictions, and aux masks. **Nothing requests
`merge_frame_inventory` or the per-well shards**, and segmentation still reads the legacy
`frame_contract.csv` directly (`Snakefile`, `segment_and_track_per_well`). So the four
`frame_inventory.smk` rules parse but never run in a normal invocation — reachable only by naming
their output on the CLI. Defensible *as scaffold* (Scope 5 wires them in when segmentation moves to
the shard), but nothing in the code says "intentionally unwired pending Scope 5."
**Fix:** add a forward-declaration banner to `frame_inventory.smk` (mirroring the `paths.py` one)
so the dead branch doesn't read as a bug. *(Done in the audit commit.)*

### 2. 🟠 `validate_frame_inventory` is far weaker than the contract its name implies
The stitched-handoff contract specifies a **strict, file-level** `validate_frame_inventory_well`:
paths exist, images open, real dims == declared, µm/px > 0, BF contiguous, channels rectangular,
derived ids recomputed from atoms. The implemented `validate_frame_inventory`
(`frame_inventory.py`) checks only **schema columns + nulls + duplicate key** — it is the existing
weak `validate_dataframe_schema` that the contract's Refactor Items explicitly say must be
**promoted**, not reused. Reusing it under the TARGET name risks the name lying (the same sin Win 3
calls out about `echo "ok" > validated`).
**Fix:** either rename to `validate_frame_inventory_schema` (honest interim name) or add a
`# TODO(Scope 2): promote to strict file-level gate per stitched_handoff_contract.md` at the
function. *(TODO added in the audit commit; full promotion deferred to Scope 2.)*

### 3. 🟠 The merged `frame_inventory` is validated twice
`merge_frame_inventory_shards` already calls `_validate_unique_keys` on the concatenated frame,
then `frame_inventory.smk`'s `validate_frame_inventory` rule runs the **same** weak validator over
the merged file again. The merged `.validated` sentinel adds nothing the merge didn't already
guarantee.
**Fix:** drop the merged-level `validate_frame_inventory` rule (the merge self-validates), OR move
all validation into the gate and let the merge be a dumb concat (one-file+sentinel model). The
per-well `validate_frame_inventory_for_well` gate stays — that one is load-bearing.

### 4. 🟡 `merge_frame_inventory_shards` reimplements `concat_well_shards_to_file`
The well-runner just added a generic, suffix-aware `concat_well_shards_to_file` with per-shard
required-column checking — the canonical merge primitive Scope 4 exists to provide. But
`merge_frame_inventory_shards` hand-rolls its own concat + column-drift check + sort. Two
implementations of the same idea, in the same PR.
**Fix:** make `merge_frame_inventory_shards` call `concat_well_shards_to_file(..., required_columns=
REQUIRED_COLUMNS_FRAME_CONTRACT, sort_columns=[...])`. Clear de-duplication, exactly the kind Scope
4 is meant to kill.

### 5. 🟡 The adapter assumes a *global* `well_id` already lives in `frame_contract.csv`
`build_frame_inventory_for_well` filters `frame_df["well_id"] == global_well_id`. The mint sites
were made global (commit `32bf47be`), but Scope 2 (regenerate artifacts) hasn't run — so any
**pre-existing** `frame_contract.csv` on disk still has a local `well_id`, the filter returns
empty, and `if shard.empty: raise` fires with "No frame_contract rows…" — fail-loud (good) but the
message won't say *why* (stale local-id artifact). The schema also still carries both `well_id`
**and** `well_index`, which Scope 2 collapses.
**Fix:** add a hint to the empty-shard error ("…the source may be a pre-Scope-2 artifact whose
`well_id` is still a local label; regenerate it") so the failure is self-explaining.

### 6. 🟡 `time_int` vs `time_index` — the rename half-happened
Docs lock **`time_index`** as canonical (collapse `frame_index` + `time_int`). The adapter sorts on
and keys on **`time_int`** (`frame_inventory.py`, `UNIQUE_KEY_FRAME_CONTRACT`) — consistent with the
*legacy* schema, inconsistent with the *target*. Fine for an adapter, but the `frame_inventory`
product still speaks the old axis name.
**Fix:** add a `# TODO(Scope 2): time_int → time_index` where the column is referenced so the
collapse isn't forgotten.

---

## Already resolved by the front-end wiring (committed alongside this audit)

The uncommitted front-end wiring fixed several drift items an earlier review would have flagged:
- `discover_wells` now reads `scope_metadata_mapped.csv` (not the frame contract) — matches
  Decision 5 of `front_end_naming_and_flow.md` (discovery is metadata-only).
- The checkpoint emits `discovered_wells.txt` and `wells_for_experiment` delegates to
  `run_well_ids_for_experiment` — the well-runner is now wired (the `well_runner.py` banner was
  updated to say so).
- Legacy rules renamed to TARGET (`ingest_plate_metadata`, `ingest_scope_metadata`,
  `map_series_to_wells`, `join_series_mapping_to_scope_metadata`) and routed through `tasks.py`
  verbs (with back-compat aliases) — matches the "one stage, config-dispatch" direction and Win 2.

---

## Does it fit the philosophy of the plan?

**Yes, strongly.** One global `well_id` through-line, no-leakage identity boundary, `output_root`
as a parameter, fail-loud at contract boundaries, "add a step = one registry row" — the well-runner
and registry embody all of it. The frame-inventory adapter is the plan's preferred *style* of
interim step (behavior-preserving, honestly labeled, fail-loud). The deviations are all
"weaker-than-TARGET pending a later scope," which is the correct way to stage this — the one real
slip is #4 (re-duplicating the merge the well-runner just de-duplicated).

## Suggested order of fixes
1. #4 (call `concat_well_shards_to_file`) and #3 (drop double-validate) — remove duplication the
   plan explicitly wants gone.
2. #1, #2, #5, #6 — cheap honesty fixes (banners + TODOs) so the staging is legible at the code.
3. Full strict validator (#2 body) + `time_int → time_index` (#6 body) land in **Scope 2**.
