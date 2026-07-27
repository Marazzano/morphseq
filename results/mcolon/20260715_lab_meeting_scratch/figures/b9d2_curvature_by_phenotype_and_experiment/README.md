# B9D2 curvature faceted by phenotype and colored by experiment

`curvature_faceted_by_phenotype_colored_by_experiment.png` uses the native
`plot_feature_over_time` faceting API and Build06 trajectories.

`total_length_faceted_by_phenotype_colored_by_experiment.png` uses the identical
cohort, cross-bin phenotype labels, facets, and experiment colors for
`total_length_um`.

- facet rows: `CE`, then `HTA`
- facet columns: every B9D2 pair found in the five Build06 experiments, including
  pairs represented in only one experiment (`b9d2_pair_1`, `_2`, `_4`, `_5`,
  `_6`, `_7`, and `_8` in the current data)
- color/group: Build06 experiment
- Build06 experiments: `20251104`, `20251119`, `20251121`, `20251125`, `20260206`
- cohort: QC-passing homozygous embryos
- phenotype rows: CE/HTA predictions from the saved
  `b9d2_homozygous_phenotype.pkl` per-bin label-transfer model; per-bin
  probabilities are pooled across supported time bins into one cross-bin label
  per embryo
- features: `baseline_deviation_normalized` and `total_length_um` in separate PNGs
- pale lines: individual embryo trajectories
- dotted lines: experiment-level median trends in 3 hpf bins
- uncertainty bands: omitted

The row and column ordering are fixed with `FacetSpec`. The raw Build06 `phenotype`
column is not used because it repeats genotype rather than containing CE/HTA calls.
