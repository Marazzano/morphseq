# Positive-response validation

This isolated exploratory analysis asks whether positive gold motile-cilia
score changes after foxj1a perturbation are larger than ordinary
embryo-to-embryo sampling noise.

The analysis is intentionally separate from the main foxj1a inspection. It
does not modify the parent report or its figures.

## What it checks

1. Ranks every positive score change with its model p-value, existing q-value,
   gold-module-only BH q-value, confidence interval, standardized effect, and
   embryo counts.
2. Shows every embryo for the six leading positive candidates.
3. Removes each embryo once to identify results driven by a single embryo.
4. Divides control embryos into all possible balanced pseudo-arms to estimate a
   cell-type-specific sampling-noise floor.

The control-only null does **not** measure foxj1a/control plate confounding.
Passing it is useful evidence against ordinary sampling noise, but is not proof
of biological adaptation.

## Main files

- `positive_response_validation.Rmd` — readable, executable analysis.
- `report.html` — rendered report.
- `data/positive_candidate_summary.tsv` — one row per positive cell type.
- `figures/01_top_candidate_embryo_values.png` — raw leading candidates.
- `figures/02_noise_floor_and_stability.png` — sampling-null and outlier checks.

Component-gene coherence, leave-one-gene-out scores, and compensatory regulator
expression are not claimed here because the compact artifacts do not contain
individual genes for these positive candidates. That extraction should be done
only after selecting candidates worth the large-CDS read.

## Preliminary result

- 35 of 92 tested cell types have a positive point estimate.
- Only `mesenchymal cell, pharyngeal arch` has a nominal 95% model interval
  above zero (`difference = 0.000351`, `p = 0.0417`).
- No positive result passes the model's full q-value or a gold-module-only BH
  correction; the best gold-module-only q-value is `0.403`.
- No positive result exceeds its cell-type-specific 95% control-only sampling
  threshold. Pharyngeal-arch mesenchyme is closest (`empirical p = 0.0648`;
  observed effect is `0.985` times the 95% noise threshold).
- Removing any one embryo leaves the direction positive for 22 of the 35
  candidates, including all six leading candidates. Their direction is not
  obviously driven by one embryo, but their magnitude remains compatible with
  sampling noise.

The current evidence therefore supports **candidate positive responses**, not
adaptation. A coherent increase across component genes would be the next useful
test for a deliberately selected candidate, but would require one scoped read
of the large CDS. Independent replication is still needed because the
foxj1a/control comparison is confounded with hash plate.
