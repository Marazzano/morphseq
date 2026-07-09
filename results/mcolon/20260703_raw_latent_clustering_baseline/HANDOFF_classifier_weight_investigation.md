# Handoff: classifier-weight vs raw-latent-difference investigation

## Origin question

Scripts 8/9 showed the classifier-margin space substantially reorganizes the
raw 80-dim `z_mu_b` neighbor structure, despite margin being a heavy
compression (~10 scalar pairwise axes) of the raw manifold. The question:
is classification finding a genuine *subset* of what's different in the raw
latent, or is raw latent space "leaving stuff on the table" that margin space
never looks at?

## What was built (scripts 10-12)

- **`10_classifier_weight_heatmap.py`** — fits `extract_classifier_directions`
  (logistic-regression unit coefficients, one per raw `z_mu_b` dim) for each
  crispant vs `inj_ctrl`, at every 4hpf time bin. Heatmap of `|coefficient|`
  across all 80 dims. Also a pooled view snapped to key stage checkpoints.
- **`11_batch_reproducibility_check.py`** — at the bin center nearest 72hpf,
  refits per-experiment (3 independent batches) vs pooled, for all 3 crispant
  comparisons AND a negative control (`wik_ab` vs `inj_ctrl`, no CRISPR).
  Checks whether "hot" dims are reproducible across independent experiments
  or an artifact of pooling.
- **`12_hot_dims_raw_vs_coef.py`** — companion plot: raw values of the top-2
  classifier-hot dims, boxplot/strip by genotype colored by experiment,
  alongside a bar chart of `|coef|` per-experiment vs pooled for both the
  negative control and the strongest crispant comparison.

All three use **`bin_width=4.0`** via `extract_classifier_directions`, matching
script 10 exactly. `extract_classifier_directions` reports
`time_bin_center = raw_time_bin + bin_width/2` — a raw `time_bin` of 68 reports
back as bin center 70.0. Scripts 11/12 derive the nearest-to-72 bin from the
fit itself rather than hardcoding, to avoid silently drifting off script 10's
actual bins (an earlier draft snapped to raw `time_bin==72` with re-binning
disabled, which is a DIFFERENT bin than script 10 uses, and produced a
different, wrong top-dim ranking — z_mu_b_69 instead of the real z_mu_b_71).

## Finding

Two raw `z_mu_b` dimensions (`z_mu_b_33`, `z_mu_b_71`) dominate the classifier's
`|coefficient|` for every crispant-vs-`inj_ctrl` comparison, at essentially
every developmental time bin (script 10 heatmap) — this is NOT a spread-thin
importance profile across all 80 dims, it's concentrated on 2. That directly
supports the original hypothesis: classification finds a small, sparse subset
of the raw latent's degrees of freedom, not a diffuse combination of all of
them.

Both dims are large and reproducible across all 3 independent experiments
(script 11), including in a `wik_ab`-vs-`inj_ctrl` negative control that has
no CRISPR difference at all — so coefficient magnitude ALONE does not
distinguish "real CRISPR-phenotype axis" from "some other structure shared by
every injected group."

The raw-value plot (script 12) breaks that tie:
- **`z_mu_b_71`** shows a real, visible, near-monotonic gradient: WT/inj_ctrl
  ~2.7-2.8, dropping through pbx1b -> pbx4 -> double crispant (~2.0 -> 1.9 ->
  1.5). A dose-like structure consistent with genuine CRISPR-driven biology.
- **`z_mu_b_33`** shows almost NO visible separation between any group by eye
  — every genotype's median sits in the same ~0.7-0.9 band. Its large,
  reproducible classifier coefficient is not explained by its own marginal
  distribution.

## What this says about the raw latent's structure

A dimension can carry no marginal (univariate) signal and still get a large,
reproducible logistic-regression coefficient if it participates in a
multivariate combination with other dims — e.g. as a suppressor variable that
cancels shared nuisance variance out of the truly-informative dims, or as
half of a correlated pair whose *combination* isolates a direction neither dim
reveals alone. `z_mu_b_33`'s behavior (big coefficient, flat raw distribution,
reproducible across every comparison including the negative control) is the
signature of exactly this: the classifier is not reading `z_mu_b_33` as
"different by genotype," it's using it in combination with other dims
(plausibly including `z_mu_b_71`) to sharpen a decision boundary.

This means the raw z_mu_b manifold's structure is NOT simply "a few dims
matter, the rest are noise" — some of its most classifier-important dims only
make sense jointly, not individually. That's a genuine piece of latent
structure worth understanding on its own, independent of the original
margin-space-reorganization question.

## Next steps (not yet done)

- Correlate `z_mu_b_33` against `z_mu_b_71` and the other next-ranked dims
  (e.g. 40/48/53 at bin 70) to check for the suppressor/correlated-pair
  pattern directly, rather than inferring it from coefficient behavior alone.
- Check `z_mu_b_33` against known structural covariates (`total_length_um`,
  batch/experiment_id, imaging condition) to see if it's tracking something
  interpretable.
- Extend the reproducibility check (script 11/12 logic) across more time bins
  to see whether `z_mu_b_71`'s dose-gradient holds throughout development or
  is stage-specific.
- `20260306` (one of the 3 experiments) has zero WT/inj_ctrl samples at the
  bin used in scripts 11/12 — worth checking whether a different bin gives
  full 3-way experiment coverage for a cleaner reproducibility read.
