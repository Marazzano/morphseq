# CEP290 static phenotype-transition plots

These are regenerated from the cleaned SCI cilia QC reference table using the
same rendering contract as the B9D2 summaries.

- `01_summary_error_band__High_to_Low__Low_to_High_overlay.png`: normalized
  body-axis curvature
- `02_summary_error_band__High_to_Low__Low_to_High_overlay__total_length_um.png`:
  total embryo length in µm
- `03_individual_trajectories__High_to_Low__Low_to_High_overlay.png`: raw
  per-embryo curvature trajectories
- `04_individual_trajectories__High_to_Low__Low_to_High_overlay__total_length_um.png`:
  raw per-embryo length trajectories

Both plots contain QC-passing homozygous `High_to_Low` and `Low_to_High` embryos,
span 24–120 hpf, show medians in 3 hpf bins with IQR bands, and use dashed trends
smoothed with `sigma=1.5` bins. Colors come from the canonical package palette.
The individual variants retain the dashed summary trend and omit only the uncertainty band.
