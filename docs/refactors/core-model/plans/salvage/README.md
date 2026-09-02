# Salvage — Phase 1

**This is not live code. It is not on `sys.path` and nothing imports it.**

## `pipeline_contracts.py`

348 lines, recovered 2026-08-27 from the untracked working tree of the `morphseq-phase1a` worktree
before that worktree was removed. It is the **only surviving artifact** of the Phase 1 attempt that
a status report claimed was complete and passing 58/58 tests. It was never committed, never
reviewed, and never run — no test imports it and no entry point references it.

It was written into `src/core/data/`, which was in `.gitignore` at the time, which is why git never
saw it. That ignore rule was removed in `b76528eb`.

Kept here deliberately rather than in `src/core/data/`: dropping unreviewed, untested code into the
live package is how it acquires false authority. Read it as a starting point for Phase 1, verify it
against `../../contracts/MANIFEST_SCHEMA.md`, and move it into `src/core/data/` only with tests.

Context: `../../reports/GROUND_TRUTH_2026-08-27.md`, Claims 9–10.
