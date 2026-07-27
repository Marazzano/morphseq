# Temporally Smooth Batch-Correction Field

## Purpose

This is a temporally regularized registration layer for the 2D initializer. It
is not a modification to trajectory condensation. Developmental time is
observed biological metadata rather than an optimizable coordinate; the
correction field therefore operates only in initializer XY space.

```text
p_i = (x_i, y_i),   batch = b_i,   time = t_i
p_i' = p_i + u_{b_i}(t_i)
t_i' = t_i
```

The field is estimated where matched-stage experimental overlap supplies
evidence, then regularized to vary smoothly through time. It must not invent
independent corrections for every time slice.

**Model assumption.** The dominant technical batch effect is a smoothly
varying, experiment-specific translation in initializer XY space that is shared
across genotypes.

## V0 — translation field MVP

V0 parameterizes the correction field as a batch-specific, time-varying
translation field:

```text
u_b(t) = [Δx_b(t), Δy_b(t)]
```

### Estimation

- Use only overlapping time bins containing at least two experiments.
- Estimate the translation field using only control populations, under the
  assumption that the technical batch displacement is shared across all
  genotypes within an experiment.
- Fit one strongly regularized smooth function per experiment and XY dimension
  (for example, a low-degree spline).
- Choose a reference experiment with correction fixed to zero.
- Apply the resulting field uniformly to every genotype within that experiment.

### Hard constraints

- Time is immutable: `Δt = 0`.
- The correction is translation-only; it cannot rotate, rescale, shear, or
  independently rearrange phenotype neighborhoods.
- Bins with no experimental overlap contribute no direct evidence for the
  correction field. Their correction is inferred only through the temporal
  smoothness prior and must be flagged as interpolated or extrapolated.

### V0 acceptance criteria

V0 is a diagnostic MVP, not a production correction, until it demonstrates all
of the following:

1. Cross-experiment neighborhood connectivity improves in overlap bins,
   especially 48–76 hpf.
2. The visible 48 hpf seam decreases in the corrected initializer and after
   unchanged condensation.
3. Within-experiment crispant-versus-control separation is preserved.
4. Control-derived translation fields are small, smooth, and broadly consistent
   with those independently estimated from each crispant group. Large
   discrepancies suggest the displacement is not purely technical.
5. Results remain stable under reasonable spline smoothness choices and
   leaving out one overlap bin.

Failure of V0 is informative: it says a time-varying translation is not enough;
it does not justify increasing correction strength blindly.

## Quick iterative prototype

`24_iterative_correction_field.py` implements the smallest Harmony-inspired
test on the existing 2D initializer:

```text
fix 20251207_pbx as the bridge/reference experiment
initialize every u_b(t) = 0
repeat:
    softly match non-reference controls to reference controls in each overlap bin
    robustly average match displacement into one residual translation per bin
    smooth each experiment's residual translations over time
    u_b(t) <- u_b(t) + 0.5 * residual_b(t)
    reapply the accumulated field and rematch
```

The reference experiment overlaps `20260304` at 48--72 hpf and `20260306` at
72--76 hpf. Before the first overlap the MVP keeps the field at zero, so the
incoming 20260304-only coordinate frame remains untouched through 44 hpf and
the learned 48 hpf update can repair the 44-to-48 seam. After the last overlap
it uses constant endpoint extrapolation to preserve continuity. Unsupported
times are flagged and cannot acquire an independently varying correction.

The prototype writes `corrected_initializer.npz`, experiment/genotype HTML
viewers, the fitted field, overlap diagnostics, and convergence history under
`figures/iterative_correction_field_v0/`. It does not run condensation; the
initializer seam is the first and cheapest gate.

## V1 — only after V0 is validated

V1 may generalize the same abstraction while preserving the V0 constraints and
diagnostics. The allowed escalation order is:

```text
V0: p' = p + u_b(t)                         # smooth translation
V1a: p' = A_b(t) p + u_b(t)                 # smoothly regularized affine field
V1b: p' = p + u_b(x, y, t)                  # spatially varying field
```

`A_b(t)` must be strongly regularized toward the identity. A spatially varying
field requires stronger geometry-preservation checks because it can alter local
phenotype relationships.

The joint graph initializer is a separate future infrastructure track, not an
automatic V1 replacement:

```text
within-time phenotype-neighbor edges
+ same-embryo adjacent-time edges
+ matched-time cross-experiment control anchors
→ one shared 2D initializer
```

It is warranted if V0 cannot repair the seam without harming biology, or if
the existing per-bin UMAP plus Procrustes chain is shown to recreate the seam
after a successful V0 correction.

## Non-goals

- Do not move observations across time.
- Do not treat distant developmental stages as batch-matching pairs.
- Do not use genotype labels to fit the correction field.
- Do not overwrite existing raw, per-bin-Harmony, or global-Harmony outputs.
