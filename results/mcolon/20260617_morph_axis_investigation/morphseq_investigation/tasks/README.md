# Distribution Engine — parallel implementation tasks

These briefs decompose `docs/HANDOFF_NEXT_distribution_engine.md` +
`docs/PRIMITIVE_ONTOLOGY.md` (the LOCKED spec) into implementable units.

## Read before you touch anything

1. `docs/PRIMITIVE_ONTOLOGY.md` — THE locked spec. Everything defers to it.
2. `docs/DISTRIBUTION_COMPARISON_MOTIVATION.md` — the *why*.
3. `docs/HANDOFF_NEXT_distribution_engine.md` — the map + the four steps.
4. Your own task file below.

## Dependency graph (READ THIS — it dictates who can start when)

```
        TASK_0_foundation      ← BLOCKING. Serial. Must be committed + reviewed
        (the four nouns +        BEFORE any other task starts. Freezes the object
         all make_*_id +         contract AND the grid_id hash so nobody diverges.
         grid_id hash +
         invariant guards)
             │
     ┌───────┼────────┬──────────┐         these four run IN PARALLEL,
     ▼       ▼        ▼          ▼          each imports the frozen objects
  TASK_A   TASK_B   TASK_C    TASK_D
  grid     labelers compare   plotting
     └───────┴────────┴──────────┘
                 │
                 ▼
           TASK_E_b9d2_example   ← integration. Serial. Runs LAST. Proves it composes.
```

- **TASK_0 is non-negotiably first and serial.** Do not start A/B/C/D until it is
  committed on `main` (or a shared branch you all pull). If you start early you
  will each invent a different object shape and merge will be hell.
- A/B/C/D are parallel. B is the biggest (~40% — it re-expresses live peak code).
  C and D can develop against hand-built fixture objects while B is in flight.
- TASK_E is the acceptance target; it wires everything.

## Commit discipline (REQUIRED — this applies to you, the agent)

You **must** commit at the checkpoints marked `⟢ COMMIT` in your task file, not
just at the end. This keeps the parallel tracks reviewable and rebased cleanly.

Rules:
- Branch per task: `git checkout -b dist-engine/task-<X>-<shortname>` off the
  latest `main` that contains TASK_0.
- Commit message format: `dist-engine(task-<X>): <what landed at this checkpoint>`
- End every commit body with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Do not squash away the checkpoint commits.** They are the audit trail.
- Run the module's tests (`PYTHONPATH=src:$PYTHONPATH conda run -n
  segmentation_grounded_sam --no-capture-output python -m pytest <your test file>`)
  green BEFORE each `⟢ COMMIT`. A checkpoint commit with red tests is a defect.
- Open a PR when your task's final checkpoint is committed; do not merge into
  other tasks' branches.

## Where the code lands

New package: `morphseq_investigation/engine/` (create it in TASK_0). Suggested layout:
```
engine/
  objects.py        # TASK_0: Distribution, Grid, DensityGrid, SampleSet, LabelGroup, label_groups
  identifiers.py    # TASK_0: make_distribution_id / make_sample_set_id / make_grid_id
  invariants.py     # TASK_0: coverage + assignment-consistency + ordered-feature guards
  grid.py           # TASK_A: build_grid + construction methods
  labelers.py       # TASK_B: peak_finding + genotype labelers
  compare.py        # TASK_C: compare_label_groups + correspondence policies + agreement
  plotting.py       # TASK_D: faceting-engine IR emitters (KDE strips, HDR overlays)
  helpers.py        # (front door) label_dataframe — build LAST, on the dumb objects
tests/engine/       # mirror the above, one test module per source module
```
Confirm this layout against the ontology before committing TASK_0 — the handoff
says re-derive, don't trust a stale ordering.
