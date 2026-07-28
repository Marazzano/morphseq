# Batch-Entry Seam Investigation — Summary & Next Steps

## The problem

Raw z_mu_b trajectory condensation shows a "cliff"/seam at time bins where a new
experiment first enters the timeline: **48 hpf** (`20251207` enters) and, weaker,
**72 hpf** (`20260306` enters). At the seam, the entering batch's bundle lands off
the manifold of the batch already present. Measured by `ridge_score`
(`seam_bridge_init.py`): fraction of observations lacking a neighbor within local
ridge spacing at the adjacent time bin.

## What we established (in order)

1. **Root cause is the per-bin UMAP + Procrustes initializer.** Each time bin gets
   an independent UMAP (arbitrary frame), stitched to `t-1` via Procrustes using
   **only same-embryo anchors**. A batch entering at bin `t` has no `t-1` rows →
   zero anchors → its bundle is placed by a rotation solved purely from the *other*
   batch. It's a lack-of-bridge problem, not a feature batch effect.
   (`init_embedding.py` analysis; `RELATIVE_GEOMETRY_AUDIT.md`.)

2. **The solver alone cannot close it.** Condensation closes every *ordinary*
   temporal gap (~60% → ~2-4%) but not the batch-entry bins. Turning off fidelity
   (`25_condense_no_fidelity.py`) left t7 unchanged (0.60→0.61): not an anchoring
   artifact. Attraction/coherence need temporal co-travel, which an entering batch
   lacks.

3. **Init-side fixes don't hold.** Global UMAP (`26_`) helped x0 but smeared the
   whole manifold after condensation. Bidirectional/future-anchor alignment moved
   the strand but the ridge barely changed (the entering strand is internally
   consistent, just collectively offset — aligning it to its own future is a no-op).

4. **The margin (classifier) space does NOT have this problem** — it mixes
   experiments (cross-batch 10nn overlap 0.37-0.61) because the classifier
   projection downweights batch-orthogonal variation. Residual seam even there is
   ~0.34 at entries (a realistic floor). (`27_margin_experiment_mixing.py`.)

5. **Raw features ARE mixable where batches co-occur** (0.33 cross-batch mixing at
   72 hpf for `20260304`↔`20260306`). PCA-3 (`28_`) = 92.7% var; PC1 = time
   (corr 0.72). The apparent "batch separation" is the batch–time confound (batches
   sample different windows), NOT a removable batch axis (|corr(PC,1207)|≈0).

6. **Dropping `20251207` removes the 48hpf seam but the 72hpf entry still seams**
   (`29_`) — even though those batches DO overlap in features. So the seam is
   structural to time-binning + entry-stitching, not specific to one bad batch.

## The rescue force we built (`30_temporal_coalesce_force.py`)

A support-conditioned "temporal zipper": for each point lacking cross-time support
(severity `z = d_nearest / s_local`, logistic-gated), pull toward the adjacent-bin
neighbor centroid; strand-propagate the direction along the same embryo so the
segment translates coherently (elasticity doesn't fight). Plus optional coherence
suppression, repulsion damping, and annealed noise.

### Diagnostic machinery (reusable)
- `31_zipper_force_balance.py` — momentum-decomposed force ledger: projects each
  force's actual update onto the merge direction. Proved the zipper works but the
  base stack opposes.
- `32_support_statistic_separation.py` — `d/s_local` separates seam (q90=7.5) from
  healthy (q90=0.66); peer-relative z-score rejected (self-reference trap).
- `33_per_force_decomposition.py` — decomposes EVERY force at the seam. **Key
  finding: REPULSION is the dominant merger-opposing force** (grows monotonically,
  larger than the zipper at the contact zone). Elasticity opposes transiently then
  fades once the strand moves coherently.

## The result: a theoretical maximum

Eight mechanisms (solve-time force, pre-merge, anneal, coherence-suppression,
strand-propagation, repulsion-damping, noise, and combinations) all plateau at
**t7 ridge ≈ 0.56-0.61**. Interleaving `@t7` goes 0.19 (baseline) → 0.23 (best),
vs ~0.5 for a truly merged region. Two independent proofs this is a ceiling:
- **Repulsion** enforces a max local density: the two dense groups jam like
  tectonic plates and cannot interdigitate under zero-temperature gradient descent.
- **Feature floor**: 75% of 48hpf entrants have NO 44hpf feature-neighbor at strict
  scale (26% at 2×). `20251207` never existed at 44 hpf; nothing there is its
  biological predecessor. Forcing the seam to 0 would fabricate a connection the
  data lacks (the Harmony mistake).

**The forces already push the embedding to this floor.** The correct behavior is
that the system merges every *supported* bin to ~0 and declines to invent the
unsupported connection.

## Tooling shipped
- `viz/condensed_time_slice_viewer.py`: `time_slice_html(..., views=[...])` —
  multi-view HTML with a dropdown (experiment / genotype / continuous z_mu_b dims),
  true per-timepoint coloring on a global colorbar, faint trajectory lines
  (`trajectory_trace_alpha`). `make_multiview.py` wraps it (dims 33/71/85).
- Run outputs are config-tagged (`<tag>__by_experiment.html`, `__ridge.txt`,
  `__positions.npz`) so runs don't overwrite each other.

## NEXT GOAL: find the MVP

We implemented *many* mechanisms to characterize the problem. Most were diagnostic,
not production. The next step is to **distill the MVP**: the single, minimal
intervention that captures the real, supportable behavior — "less is more."

Open questions for the MVP:
- Is the seam worth "fixing" at all, or is the honest deliverable the *diagnosis*
  (this entry has no predecessor) + the ridge metric as an honest QC signal?
- If a fix is wanted: which ONE constraint to relax (repulsion is the measured
  blocker) to approach the ceiling, rather than stacking forces?
- Should the production representation just be the margin space (which mixes
  experiments correctly by construction) rather than raw z_mu_b + a rescue force?

The constraint-accounting experiment (quantify each base force's cost vs the
theoretical max, relax the cheapest) is the natural next measurement.
