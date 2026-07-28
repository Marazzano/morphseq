# Tech Debt: `execution: EXECUTION_PER_WELL` Is a Misnomer for `EXPERIMENT`-Fanout Steps

**Status:** known naming gap, mdcolon 2026-07-02. Not a blocker — every `EXPERIMENT`-fanout step
(including the 4 new `*_report` steps) already sets this correctly per the existing convention. The
debt is the NAME reading wrong at the call site, not incorrect behavior.

---

## The gap

`PIPELINE_STEPS`' `execution` field (`orchestration/paths.py`) has exactly two values:

- `EXECUTION_PER_WELL` ("per_well") — one job per well shard
- `EXECUTION_RUN_BATCH` ("run_batch") — one job for the whole run set

`test_every_step_has_required_keys` (`tests/data_pipeline/pipeline_orchestrator/test_paths.py`)
requires every `PIPELINE_STEPS` row to set `execution` to one of these two values — including
`EXPERIMENT`-fanout steps, which have **no per-well concept at all** (one job, one artifact per
experiment). The doc comment above `EXECUTION_PER_WELL`'s definition is explicit that this is
intentional: *"EXPERIMENT-grain steps always use `EXECUTION_PER_WELL`."* So every `EXPERIMENT`-
fanout row in the registry (`ingest_plate_metadata`, `discover_wells`, `map_positions_to_wells`,
and now `death_detection_report`, `physical_embryo_registry_report`, `mask_geometry_report`,
`surface_area_qc_report`, ...) reads `"execution": EXECUTION_PER_WELL` even though it runs as a
single job at experiment grain, never fanned per well.

Read cold, `"execution": EXECUTION_PER_WELL` on an `EXPERIMENT`-fanout step looks like a copy-paste
bug — it says "per well" on a step that has no wells. It is correct by the registry's own
documented rule, but the name actively misleads at the call site, and a future editor (or agent)
who does not read the comment above the constant definition first is likely to "fix" it into
something that then fails `test_run_batch_steps_are_per_well_then_merge` or otherwise diverges from
convention.

---

## Why it was not fixed now

`execution` is a two-value enum with no defined third state (see `orchestration/paths.py`
`EXECUTION_PER_WELL`/`EXECUTION_RUN_BATCH`). Adding a third value (e.g. `EXECUTION_ONCE` or
`EXECUTION_EXPERIMENT`) would mean:

- adding the new constant + updating `_ALLOWED_PATH_MODES`-style validation if `execution` ever
  starts gating behavior beyond documentation (today it is metadata only — "it does not affect
  path construction," per the same comment block)
- updating `test_every_step_has_required_keys`'s `valid_executions` tuple
- auditing every existing `EXPERIMENT`-fanout row (at least 3 pre-existing + the 4 new report rows)
  to decide whether they should all migrate to the new value, which is a naming-only change but
  touches every `EXPERIMENT` step in the registry

This is out of scope for wiring the 4 tier-1 reports into the DAG (see
`docs/refactors/streamline-snakemake/target/specs/viz/report_world.md`); the reports correctly
follow the existing, tested convention rather than inventing a new one mid-task.

---

## The mitigation (in place now)

Every `EXPERIMENT`-fanout row's `execution` line carries an inline comment pointing back to the
rule: `# EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)`. This is
the same mitigation pattern as the `image_id` tech debt entry — not a fix, but a loud enough marker
that a future reader does not mistake documented convention for a bug.

---

## Future refactor path (if `execution` ever starts gating behavior)

Add a third value, e.g. `EXECUTION_ONCE = "once"`, for `EXPERIMENT`-fanout steps specifically:

1. Define `EXECUTION_ONCE` next to `EXECUTION_PER_WELL`/`EXECUTION_RUN_BATCH` in
   `orchestration/paths.py`, with a doc comment explaining it means "one job, experiment grain, no
   per-well concept."
2. Update `test_every_step_has_required_keys`'s `valid_executions` tuple to include it.
3. Migrate every `EXPERIMENT`-fanout row (`ingest_plate_metadata`, `discover_wells`,
   `map_positions_to_wells`, `apply_position_to_well_mapping`, the 4 `*_report` steps, and any
   future `EXPERIMENT` rows) from `EXECUTION_PER_WELL` to `EXECUTION_ONCE`.
4. Confirm `execution_mode(step)` and any caller that branches on `EXECUTION_PER_WELL` vs
   `EXECUTION_RUN_BATCH` (if any exist beyond documentation) still resolve correctly for the new
   value — today `execution` is metadata-only, so this should be a no-op, but verify before
   committing to that assumption.

Worth doing opportunistically the next time someone is already touching the `PIPELINE_STEPS`
registry structure, not worth a dedicated pass on its own.

---

## Where to look when paying this debt

- Enum definition + doc comment: `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py`
  (`EXECUTION_PER_WELL`, `EXECUTION_RUN_BATCH`, lines ~106-120)
- Registry rows affected: every `"fanout": EXPERIMENT` row in `PIPELINE_STEPS`
- Test that enforces the two-value enum today: `tests/data_pipeline/pipeline_orchestrator/test_paths.py`
  (`TestRegistryIntrospection.test_every_step_has_required_keys`,
  `test_run_batch_steps_are_per_well_then_merge`)
