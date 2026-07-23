# B9D2 phenotype trajectories by pair

`phenotype_trajectories_by_pair.png` recreates the historical 2×6 trajectory grid,
but groups and colors embryos by phenotype rather than genotype.

- columns: B9D2 pairs 2, 4, 5, 6, 7, and 8
- rows: normalized curvature and total body length
- pale lines: individual QC-passing embryos; wild-type trajectories use extra-low
  opacity so they sit behind the mutant trajectories
- thick lines: median phenotype trends in 3 hpf bins, lightly smoothed
- colors: canonical `CE` green, `HTA` orange, and `Not Penetrant` gray
- mapping: `BA_rescue` is pooled into `HTA`; unlabeled/wildtype embryos remain
  `Not Penetrant`

The source is the cleaned B9D2 reference table from the SCI cilia QC analysis.
