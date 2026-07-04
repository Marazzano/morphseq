# WT-Calibrated Phenotype Geometry

A measurement framework for decomposing within-group phenotype geometry into
**distribution shift → support geometry → density geometry → component structure**,
each calibrated against a matched-N wildtype null, each with explicit per-statistic
uncertainty.

The **framework** (the decision tree + its axioms) is the stable contribution; the
specific statistics are replaceable implementations of each stage's *job*.

## Axioms

1. **Wildtype defines the reference geometry.** WT is the coordinate system, not a
   comparison dataset. Every statistic asks "is this unusual relative to what WT
   produces at the same stage and sample size?" No absolute thresholds.
2. **Confidence is orthogonal to the call.** Whether a distribution is continuous
   or discrete is one question; how much to trust it is another. Confidence is
   computed **per statistic** (N-adequacy differs by metric) and never changes a
   call.
3. **Reference and null are distinct WT roles.** WT-as-reference defines what normal
   geometry *is* (its own shape); WT-as-null is the finite-sampling variability a
   matched-N draw produces. Both are needed.

## Decision tree (each stage conditions the next)

```
Stage 1  distribution_shift   different from WT?
           No  -> "wildtype-like", stop.
           Yes -> Stage 2.
Stage 2  support_geometry     WHERE does probability exist? connected / broken?
           connected -> Stage 3.
           discrete  -> Stage 4.
Stage 3  density_geometry     HOW is probability distributed inside the continuum?
Stage 4  component_geometry   how many stable pieces?
(+)      confidence           per-statistic, attached throughout.
```

The conditioning is why it's a tree, not a checklist: skewness of a mixture is not
the same biological object as skewness of a connected continuum, so density
descriptors only run once support is established.

## Modules

| file | stage / role | key API |
|------|--------------|---------|
| `phenotype_geometry.py` | orchestrator (owns conditioning) | `run_phenotype_geometry(group, wt)` |
| `distribution_shift.py` | Stage 1 — differ from WT | `compute_distribution_shift`; pluggable metrics (Wasserstein, JS) |
| `support_geometry.py` | Stage 2 — where is probability | `compute_support_geometry`; valley_depth, MST, Fiedler |
| `density_geometry.py` | Stage 3 — how distributed | `compute_density_geometry`; variance, skew, kurtosis, tail, entropy |
| `component_geometry.py` | Stage 4 — how many pieces | `compute_component_geometry`; GMM-BIC + HDBSCAN |
| `confidence.py` | cross-cutting uncertainty | `score_confidence` (continuous [0,1] → tier) |
| `synthetic_scenarios.py` | ground-truth generators | 12 scenario families, N=5/8/12/20 |
| `validate_framework.py` | **Phase A gate** | asserts each stage's behavior on synthetics |
| `run_real_anchors.py` | real cep290/b9d2 application | full tree on hand + embedding axes; per-bin plus full-reference audit |
| `connectedness.py` | deprecated shim → `support_geometry` | back-compat re-exports |

## Statistics and their jobs

- **valley_depth** — KDE super-level density separation. Detects density-visible
  gaps (an empty interior between filled regions). The required evidence for a
  broken-support call.
- **mst_max_edge** — largest MST edge / median edge. Detects unsupported jumps;
  works at very small N.
- **fiedler** — algebraic connectivity of a kNN graph (inverted to
  "disconnectedness"). Global connectivity; fires on any thin manifold, so it
  corroborates rather than solely triggers.

A broken-support call requires **density-visible separation** (valley_depth) plus
graph corroboration — this is what keeps sparse/curved-but-connected manifolds
(crescent, spiral, outliers) from being mislabeled discrete. Disagreement between
statistics is surfaced (`SupportGeometryBundle.disagreements`), not averaged away —
it is biologically informative until proven otherwise.

## Validation-first

`validate_framework.py` is the gate: the framework is exercised on synthetic data
with known ground truth (continuous / discrete / variance-only / hole / crescent /
spiral / outliers / small-middle at multiple N) and asserts each stage behaves per
its theoretical interpretation **before** any real data is touched. Run it first:

```
conda run -n segmentation_grounded_sam --no-capture-output python validate_framework.py
```

Then the real anchors:

```
conda run -n segmentation_grounded_sam --no-capture-output python run_real_anchors.py
```

`run_real_anchors.py` reads the cleaned reference tables produced by
`results/mcolon/20260607_sci_cilia_gene14_imaging_qc/0_load_and_clean_datasets.py`.
For the biological anchors, the mutant group is restricted to the labeled
homozygous phenotype classes (`b9d2`: CE/HTA, `cep290`: High_to_Low/Low_to_High);
the output records `group_label_counts` so low-support cases are visible.

## Known hard cases (documented, not hidden)

The 2-D-projection support test has understood blind spots, surfaced by the
synthetic sweep and kept as **report-only** (not gate failures) until the
embedding / higher-dimensional Stage-2 extension lands:

- **weak separation** — two blobs whose gap is smaller than their width overlap
  after shape-normalization; correctly read as connected. Also invisible to the
  Stage-1 location-shift test when centered on WT (a symmetric split).
- **evenly-spaced trimodal** — a filled middle mode leaves no single deep density
  valley, so `valley_depth` reads connected even though `fiedler` detects the
  disconnection. Fiedler flags it; the density-gap primary does not.
- **annulus / hole** — a ring is connected in 2-D (you can go around the hole), so
  "broken support" is genuinely ambiguous at this dimensionality.

The biologically-relevant anchor case — a **two-fate split** (`two_discrete`) — is
asserted and passes cleanly. These hard cases are exactly the spec's "a 1-D axis
can hide a mode" caveat and motivate the embedding-space extension.

## Outputs

- `tables/phenotype_geometry_summary.csv` — one row per (gene × timebin × axis):
  changed_from_wt, support_call, per-statistic p-values, confidence,
  `estimated_transition_hpf`, and the labeled homozygous `group_label_counts`.
- `tables/phenotype_geometry_full_reference_summary.csv` — same anchor test after
  pooling the entire labeled homozygous reference across stages. This is the audit
  table for checking whether per-bin support was hiding a split.
- `tables/bootstrap_nulls.npz` — the full WT bootstrap null for every statistic
  (storage is cheap; keeps re-thresholding / re-combining free).
- `plots/framework_validation.png` — synthetic validation summary.
- `plots/phenotype_geometry_trajectory.png` — support-brokenness over developmental
  time, hand vs. embedding axis, with the estimated variance→mode transition marked.

See `docs/todos_scratch/morph_axis_discreteness_spec.md` for the full design.
