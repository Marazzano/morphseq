# Archive — core-model refactor

**Nothing in this folder describes current state.** Every file here is retained for provenance
only. Each carries a banner saying what it was accurate against and what refuted or superseded it.

Do not cite an archived document as evidence for a claim about code. Verify against the code, or
against `../reports/GROUND_TRUTH_2026-08-27.md`, which carries the command and output for each
claim it settles.

| File | Status |
|---|---|
| `CORE_REFACTOR_PHASE0_AUDIT.md` | Accurate at `6f6e0f3f^`. All findings closed by `6f6e0f3f`. |
| `NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md` | Accurate at `6f6e0f3f^`. One finding still open (`contrastive_transform`). |
| `TRAINING_READINESS_REMAINDER.md` | **Fabricated implementation state.** Four cited commits do not exist. |
| `REVIEW_2026-08-26.md` | Pre-merge thread review; superseded by the ground-truth audit. |

## Why this folder exists

On 2026-08-24 a status report asserted that Phase 1 was implemented and passing 58/58 tests across
four named commits. None of those commits existed. The claim was inherited by `PLAN.md`, dispatched
to agents, and went unchecked for a week — while a generated `STATUS.md` had been printing
`commit unavailable` against all four SHAs the entire time.

The rule that follows from it: **no hand-written document asserts implementation state.** Only
`STATUS.md` does, and it is generated.
