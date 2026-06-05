# Archived pre-`target/` refactor docs (2026-06-05)

These docs were the working refactor spec **before `target/` became the primary
target.** They were archived (not deleted) on 2026-06-05 because they contain
**stale terminology** but **still-useful reference material.**

## Why archived
Terminology drift vs. the wired Snakefile (verified 2026-06-05):
- they say **`frame_manifest.csv` / `stitched_image_index.csv`**
- the real Snakefile uses **`frame_contract.csv` / `stitched_inventory.csv`**
- they predate downstream stages being wired into the Snakefile (now they are)

**For current truth, use `target/` (authoritative) and
`../ORIENTATION_for_final_review.md`.** Treat everything here as intent/history.

## What's still useful here (the reason we kept them)
| File | Still-useful content | Caveat |
|---|---|---|
| `data_output_structure.md` | **dataset output tree** + contract files / required columns | filenames use the old `frame_manifest`/`stitched_image_index` names |
| `processing_files_pipeline_structure_and_plan.md` | **`src/` script/module structure**, contract definitions, debugging flow | architecture spec uses old contract names |
| `snakemake_rules_data_flow.md` | rule-by-rule I/O intent | rule names/contracts drifted |
| `DATA_INGESTION_AND_TESTING_STRATEGY.md` | symlink strategy, test-dataset + stepwise validation guidance | `-m data_pipeline...` invocation pattern still valid |
| `phase_3_implemtation_plan.txt` | historical phase-3 planning | superseded by the Scopes in `target/` |
| `data_ouput_strcutre.md` | (253-byte typo'd stub of the above) | redundant; kept for completeness |

When pulling structure info from these, **cross-check names against the live
Snakefile** (`src/data_pipeline/pipeline_orchestrator/Snakefile`) and `target/`.
