# Unsupervised raw-latent clustering baseline (IDEA — not committed)

Status: **idea / scratch.** Not scheduled. Captures a design we talked through so it
isn't lost. Nothing here is built yet.

## Why this exists

Current unsupervised structure (trajectory condensation) is fed **classifier
margins / multiclass probabilities**, which are clipped at class boundaries — a
distorted space (see `src/analyze/trajectory_condensation/README.md` "Legacy input
path (deprecated)"). Every result we currently trust — TFAP2, and the PBX
condensation — shares that one distortion, so their agreement was never
independent confirmation. TFAP2 came out as "mixture," and we **cannot currently
distinguish soft-mixing (incomplete penetrance) from a non-linear / >2D phenotype**
— both look the same after margin projection.

Goal: a genuinely label-free baseline that clusters the **raw embedding** directly,
so we can eventually adjudicate mixing-vs-non-linear on TFAP2. But TFAP2 is the
*hard test we withhold*. **PBX is the positive control** (phenotypes ~one
direction, known-good) and must pass first.

**What we are actually testing:** does the raw latent + its cross-bin connectedness
extract *more* usable signal than the linearity-constrained classifier? The
classifier's linearity is a real constraint that may discard relevant structure;
our bet is that clustering the raw latent recovers it. This is an
improvement-or-not question, decided empirically. We are NOT pre-litigating how much
of the latent variance is nuisance (batch/position/stage) — that only matters if the
clustering *fails*, and if it fails, the earliest-bin leakage tripwire (see Metric)
tells us to go diagnose it then. Don't front-load that worry.
(There was a past observation that the linear classifier did *better* by discarding
irrelevant information; we are explicitly bracketing that for now and just seeing
whether the raw-latent approach improves things or not.)

## Design (agreed)

- **Data / space:** raw `z_mu_b` embeddings, NOT margins.
  Source: `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_{exp_id}.csv`
  (this is the same source the classifier read pre-margin — clean apples-to-apples).
- **Cluster unit:** one point per `(embryo, time_bin)` — the same aggregation the
  classifier does. Use `bin_embryos_by_time` in `src/analyze/utils/binning.py`.
- **Bin width = 4 hpf.** Chosen so it's a **single-variable comparison** against the
  existing PBX condensation (already bin4): any difference is attributable to
  *space* (raw-latent vs margin), not bin width. (bin=2 barely changes PBX since
  it's ~one direction, so not worth the rewrite.)
- **The clustering is cross-bin by construction — this is the point, not a side
  effect.** Both baselines build their structure over the points *pooled across all
  time bins*, so a point's cluster assignment at bin *t* depends on connectivity that
  spans other bins. This is the same broad-length information the classifier
  exploits, and it's deliberately what we want: the question is whether the raw
  latent + its connectedness gets *more* out of that cross-bin information than the
  linearity-constrained classifier does. (It's also what gives the embryo-ID null
  its teeth — see Null below.)
- **Keep two stages distinct — do not fuse them:**
  1. *Clustering* happens **cross-bin** (graph built over points pooled across all
     bins; assignments depend on structure spanning bins).
  2. *Comparison / readout* is a **post-evaluation** that happens **per–time-bin**.
     The clustering already handled the cross-bin work; here we only compare its
     outputs. For a given bin, build the **embryo-level connectedness graph** (which
     embryos are linked/co-clustered) and compare **graph vs graph**:
     - **across the three methods** — raw-distance, Leiden, and the margin/classifier
       baseline — each fed a *different input embedding* to the clustering machinery;
       recapitulation = the three graphs **agree**.
     - **against the null** — all three must be **decisively above** the ID-shuffled
       null.
     Aggregate over bins into a curve. The comparison unit is the **distribution of
     connectedness at the embryo level** (align/compare graphs), NOT individual point
     assignments.
  "Compare within a bin" is the evaluation lens; it does **not** mean the clustering
  was done within a bin. Conflating the two wrongly makes the null look toothless.
- **Recapitulation criterion, stated plainly:** the three method-graphs agree with
  each other *and* all sit well above the null, per bin. That agreement — across
  different input embeddings, not just raw-vs-margin — is what tells us we've
  recovered the same clustering modes without the linearity constraint.
- **Two raw-latent baselines — they carry DIFFERENT information, run both:**
  (these are two of the three methods compared in the readout above; the third is the
  margin/classifier baseline, fed a different input embedding.)
  - *raw-distance grouping* in `z_mu_b` = **global** geometry (are clouds far apart;
    blind to shape).
  - *Leiden on `z_mu_b` kNN graph* = **local** connectivity (follows a curved
    manifold; scale-free).
  - On PBX they should *agree* (no curve to trip on) — agreement is a free
    consistency check on the "one axis" assumption. On TFAP2 later, their
    **disagreement is the mixing-vs-non-linear signal** (global sees one blob,
    local threads/splits the curve). Read raw-distance as primary evidence.
  - (No HDBSCAN — explicitly dropped.)

## Null (the important part — get the permutation right)

- **Null = shuffle EMBRYO IDs, not genotype labels.**
  Shuffling labels is toothless: an unsupervised method never sees labels, so it
  recovers the same clusters and tests nothing. Instead scramble which
  `(embryo, bin)` points get stitched into one trajectory — this breaks real
  temporal/identity coherence while keeping the marginal point cloud identical.
  If real embryo-linked clustering differs from the ID-shuffled version, the
  structure lives in the trajectories, not the point distribution.
- **Why this null has teeth (and isn't circular):** because the clustering is
  cross-bin (see Design), a point's cluster assignment depends on which other points
  are linked to it as the same embryo over time. Scrambling embryo IDs therefore
  *changes the clustering* even though the pooled marginal cloud is byte-identical.
  The marginals being identical is exactly what makes it a clean null — it isolates
  the trajectory/identity coherence as the only thing that differs. (This null would
  be toothless for a purely within-bin method, where the identical cloud forces
  identical clusters; it is not toothless here.)
- **No synthetic/toy data.** PBX is a real known-good case; that's the whole point
  of using it as the control.

## Metric & the key improvement: signal-over-time

- Turn each clustering into a **graph** (cluster adjacency). Compare
  **real-graph vs null-graph per time bin** — a graph-structure discrepancy.
- **Render discrepancy as a heatmap over time.** Existing heatmap utility is a fine
  first try; if it doesn't fit the per-cell scalar cleanly, just render directly.
  Not a blocker — quick check at build time, no need to decide now.
- The discrepancy-over-time curve **is a signal-emergence profile**, measured
  unsupervised:
  - **Early bins:** embryos undiverged → real ≈ null → small discrepancy →
    correctly reports "no signal yet."
  - **Late bins:** strong structure → real ≫ null → large discrepancy.
- **State this as a falsifiable prediction up front:** real ≈ null early, real ≫
  null late. Large discrepancy at the *earliest* bins would mean leakage or the
  embedding encoding batch/position rather than phenotype — so the over-time shape
  self-validates the pipeline.
- This is the independent, label-free analogue of the existing AUROC
  emergence-explorer.

## Pass criterion (define BEFORE looking)

- Discrepancy is **near-zero early and large late** (the emergence shape), AND real
  PBX structure is **visibly distinct from the null** at the bins where structure
  exists.
- **The PBX bar is: recapitulate the known-good PBX clustering, OR do better,
  *without* the linearity constraint.** PBX being "easy" (~one direction) is the
  point of the control, not a weakness: if the unconstrained raw-latent method can't
  even reproduce the known-good easy case, it's dead; if it does, the machinery is
  validated before we spend TFAP2 on it. We do NOT need a curved/known-non-linear
  case to calibrate the global-vs-local disagreement readout in order to pass PBX —
  that calibration can wait for a case that actually has curvature.
- Bar is **"non-degenerate," not "beats the margin/phenotype-direction method."**
  We want to *replace* the distorted margin space, so agreement with old outputs is
  nice-to-have, not the target. **Disagreement with the margin case is a signal,
  not a rejection.**
- Don't let PBX become another force-sweep rabbit hole (see the big sweep tree in
  `20260329`). Non-degenerate + right emergence shape = done.
- **Pass PBX → then TFAP2** (the withheld hard case), same machinery.

## Concrete dependencies / references

- Binning: `src/analyze/utils/binning.py::bin_embryos_by_time`
  (`time_col="predicted_stage_hpf"`, `bin_width=4.0`).
- Raw latents: `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_{exp_id}.csv`, `z_mu_b` cols.
- Existing PBX condensation to compare against (both margin-space — "raw" vs
  "shrunk" there means raw-margin vs centered-margin, NOT raw-latent):
  - `results/mcolon/20260329_pbx_crispant_analysis_cont/results/positioning/pairwise/*/condensed_positions.npz`
  - `results/mcolon/20260407_pbx_analysis_cont/results/positioning/trajectory/combined_shrunk_condensation_{4,5}class_bin4_perm500_*`
  - principal tree: `results/mcolon/20260407_pbx_analysis_cont/results/principal_tree/`
- Condensation input builders (the DEPRECATED margin path both PBX & TFAP2 use):
  `05_pbx_condensation.py` → `schema.from_multiclass_csv` / `from_pairwise_margin_csv`;
  TFAP2 `results/mcolon/20260413_tfap2_followup/scripts/03_run_condensation.py`
  pivots `values="class_signed_margin"`.
- Phenotype-direction module (projection vectors that fed the "shrunk" runs — NOT
  clustering outputs): `results/mcolon/20260409_pbx_additivity/phenotype_direction/`
  (`projection.py`, `vectors.py`, `centering.py`).
- Condensation package (for the margin-space comparison arm):
  `src/analyze/trajectory_condensation/` — `tc.run_condensation`, `tc.load_run`,
  `bin_embryos_by_time` unit matches its `(embryo, time)` rows.

## Rough build order (when/if committed)

1. Pull PBX `z_mu_b`, bin4 via `bin_embryos_by_time`, restrict to supported window.
2. raw-distance grouping + Leiden on the binned points.
3. embryo-ID-shuffle null; recluster.
4. cluster→graph per bin; real-vs-null graph discrepancy per bin.
5. discrepancy-over-time heatmap (try existing utility; render directly if it doesn't fit).
6. lay next to existing PBX condensation; check pass criterion.
7. if pass → repeat on TFAP2, add the raw-distance/Leiden disagreement readout.
