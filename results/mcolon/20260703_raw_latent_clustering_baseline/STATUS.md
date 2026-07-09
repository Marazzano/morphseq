# STATUS — 2026-07-04

**PAUSED at the gate. Not a bug — a batch-effect problem. Do not run steps 3–6 yet.**

## Where things are

- Scripts `0_load` → `1_cluster` → `2_triptych` all run clean.
- Gate HTML `figures/triptych.html` renders fine (self-contained Plotly, 3 panels:
  raw z_mu_b UMAP init / raw z_mu_b condensed / PBX baseline condensed).
- Baseline path + condensation params are **verified correct** (see README "Data";
  pinned by the final principal-tree run's `metadata.json`). Earlier wrong-baseline
  (`combined_shrunk`, 235 emb) and k-means/Leiden framing are fixed/removed.

## The finding (why we paused)

Raw z_mu_b condensation fragments into disjointed vertical **"columns"** over time
instead of coherent cross-time phenotype groups. Cause = **experimental batch effects**
in the raw latent. The margin/classification baseline never showed this because the
**classifier regresses batch out** — so the old "coherent" groupings were partly a
product of that correction, not pure phenotype.

Full reasoning: README `## Findings`. Memory: `project_raw_latent_batch_effects.md`.

## Next (unstarted decision)

Pick a **confound-aware** batch-correction strategy for z_mu_b before re-running the
null/discrepancy steps. Batch and phenotype may be partially confounded across the 3
PBX experiments — naive per-experiment centering risks removing real phenotype signal.
Guard: any correction must not flatten known-good PBX structure.
