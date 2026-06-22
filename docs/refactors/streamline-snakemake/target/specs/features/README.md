# Feature World - computed feature planning

**Status:** planning index, 2026-06-22. This folder owns computed-feature targets plus
feature-derived QC targets. QC lives here when it consumes feature tables and emits flag tables over
the same `snip_id` universe.

**Read first:** `targets/feature_world.md`.

---

## Docs

| Doc | The one question it answers |
|---|---|
| `targets/feature_world.md` | Which computed features and feature-derived QC targets can start, what do they depend on, and what proves each one is ready? |
| `targets/legacy_embeddings.md` | Full spec for the legacy VAE embeddings step: output location, config knobs, 3.9 env boundary, batch-execution constraint, and done-when criteria. |

---

## Boundary

Feature world starts after object extraction has produced validated object/mask products. It does not
own detection, segmentation, tracking, auxiliary-mask production, or analysis-ready joins.

The feature implementation pattern still follows the shared per-well doctrine: one registry row, one
product-local folder, one thin task/entrypoint, one templated rule, and tests under the parallel
`tests/data_pipeline/...` tree. Product folders live directly under `feature_extraction/` or
`quality_control/`; do not add a redundant `stages/` folder.

The boundary between features and QC is:

- features compute measured or predicted values;
- QC consumes feature tables and emits boolean/informational flags plus QC-specific annotations.
