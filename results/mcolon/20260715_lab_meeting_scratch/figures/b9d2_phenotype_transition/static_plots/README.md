# B9D2 static phenotype-transition plots

`01_summary_error_band__CE__HTA_overlay.png` is the B9D2 counterpart to the NWDB
CEP290 `High_to_Low` + `Low_to_High` overlay.

`02_summary_error_band__CE__HTA_overlay__total_length_um.png` uses the identical
cohort and rendering contract for embryo length, which captures the axial-shortening
component of the B9D2 phenotype.

`03_individual_trajectories__CE__HTA_overlay.png` and
`04_individual_trajectories__CE__HTA_overlay__total_length_um.png` show the raw
per-embryo trajectories plus the dashed phenotype trend, with no uncertainty band.

Rendering contract:

- features: normalized body-axis curvature and total length in µm (separate PNGs)
- cohort: QC-passing B9D2 homozygous reference embryos
- phenotype mapping: `CE` stays `CE`; `BA_rescue` is pooled into `HTA`;
  non-penetrant embryos are not included in this two-curve summary
- time range: 24–120 hpf
- summary: median in 3 hpf bins
- uncertainty: interquartile range
- trend: dashed Gaussian-smoothed line (`sigma=1.5` bins)
- colors: canonical B9D2 green/orange palette from `analyze.viz.styling`
- legend: omitted to match the source presentation still

The generation script and full input provenance are documented in the run-level
`README.md`.
