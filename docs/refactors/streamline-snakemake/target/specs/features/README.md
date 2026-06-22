# Feature World - computed feature planning

**Status:** planning index, 2026-06-22. This folder owns computed-feature targets only. QC stages
use the same per-well stage pattern but should get a sibling QC world instead of being folded into
this feature section.

**Read first:** `targets/feature_world.md`.

---

## Docs

| Doc | The one question it answers |
|---|---|
| `targets/feature_world.md` | Which computed features can start, what do they depend on, and what proves each one is ready? |

---

## Boundary

Feature world starts after object extraction has produced validated object/mask products. It does not
own detection, segmentation, tracking, auxiliary-mask production, QC flags, or analysis-ready joins.

The feature implementation pattern still follows the shared per-well doctrine: one registry row, one
compute function, one thin task/entrypoint, one templated rule, and tests under the parallel
`tests/data_pipeline/...` tree.
