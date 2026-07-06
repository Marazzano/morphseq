# Tech Debt: Acquisition Bundles Two Distinct Phases (Metadata Ingest + Materialization)

**Status:** known design smell, mdcolon 2026-07-04. Recorded as an honest tech-debt entry — not a
blocker, and the boundary is unclear enough that no move is proposed yet. Named so the idea is not
lost.

---

## The gap

The `acquisition` stage actually spans **two fundamentally different phases**:

1. **Metadata ingestion** — plate + scope metadata ingest, culminating in **well discovery**
   (`discover_wells`). This is where "which wells exist" is answered. It is the fan point.
2. **Materialization** — after wells are discovered, materialize each well's images. This is still
   **per-scope** (stitching mechanics differ per microscope) and carries its own nuances.

In retrospect these two could plausibly have been their own stages: the first is CPU-light metadata
work that ends at a checkpoint; the second is the per-scope pixel work that closes the scope world.
Today they share one stage folder (`acquisition/`) and one `stage` string in `PIPELINE_STEPS`.

## Why it's deferred

- It is **not** causing incorrect behavior — the steps, paths, and DAG are all correct as-is.
- The **right boundary is unclear.** Well discovery is the natural seam, but materialization also
  writes the `frame_inventory` product that downstream stages consume, so a clean split would need
  to decide where that product's stage lives.
- A split is **wide, mechanical churn** (a new top-level output folder + `stage` strings + every
  affected `PIPELINE_STEPS` row) for a mostly-cosmetic gain right now.

## If/when revisited

Decide the seam first (well discovery vs materialization output), then follow the same
stage-reconciliation pattern used elsewhere: land the `PIPELINE_STEPS` stage strings and the source
folder layout in one commit so they cannot drift. See
[`source_tree_stage_alignment.md`](../source_tree_stage_alignment.md) for the source↔output stage
alignment doctrine this would extend.
