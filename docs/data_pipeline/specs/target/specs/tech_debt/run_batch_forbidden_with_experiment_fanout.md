# Tech Debt: `run_batch` Is Forbidden With `EXPERIMENT` Fanout (may be an over-restriction)

**Status:** open question, mdcolon 2026-07-05. Not a blocker — no step needs the forbidden
combination today. Recorded because the restriction may be an implementation artifact rather than a
real invariant, and we haven't thought it through yet.

---

## The restriction

`orchestration/paths.py` enforces (line ~117): **only `PER_WELL_THEN_MERGE` steps may use
`EXECUTION_RUN_BATCH`.** A `run_batch` + `EXPERIMENT`-fanout step is rejected.

## Why it's questionable

Execution and fanout are meant to be **independent** axes:
- **execution** — how the compute is dispatched (`per_well` = one job per well; `run_batch` = one
  job loads the model once and processes all wells).
- **fanout** — what lands on disk (`EXPERIMENT` = one artifact; `PER_WELL_THEN_MERGE` = per-well
  shards → merge).

All four combinations are *conceptually* coherent — including the forbidden one: a step could load a
model once, run over all wells in a single job, and write **one** experiment-level artifact directly
(no per-well shards, no merge). That is a sensible step; the code just doesn't allow it.

The likely reason for the ban: every batch step we actually have (SAM2, the legacy VAE) writes
per-well shards, so encoding "batch ⟹ shards" kept the rule template uniform. That is a
**scope/simplicity choice, not a conceptual law** — the earlier justification ("a batch step with no
shards has nothing to batch") is wrong: batching is about the *input/job dispatch*, not the *output
shape*.

## Decision

Deferred — left out of `PIPELINE_OVERVIEW.md` deliberately (the overview teaches the two clean axes
and does not mention the restriction). Revisit if a real step ever wants
`run_batch` + `EXPERIMENT` output (one batched job → one experiment artifact). At that point decide
whether to relax the guard or keep the uniform template. Not thinking it through further right now.
