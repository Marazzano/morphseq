# 20260805 — Hotfish temperature vs stage

Refined reproduction of the key staging figures from `../20260528`.

## Files
- `hotfish_stage_utils.py` — portable staging/plotting helpers (temperature colormap,
  timepoint markers, bootstrap SE, cohort summaries, 19C toggle).
- `hotfish_stage_figures.ipynb` — the analysis notebook: (i) stage vs temperature,
  (ii) transcriptional vs morphological stage, (iii) stage variability vs temperature.

## Running it (on the laptop)
The notebook reads a data cache. Point it at your local cache one of two ways:

1. Set the env var before launching Jupyter:
   `export MORPHSEQ_DATA_ROOT="/Users/nick/Projects/data/morphseq/results/20260528"`
2. Or edit `_default_data_root()` in `hotfish_stage_utils.py`.

If the pre-built `joint_141_morph_seq.csv` is in that dir, both stages load directly.
Otherwise the loader assembles morph stage from `hf_pca_morph_df.csv` and looks for a
seq-staging file (`seq_to_morph_pca_pd.csv` / `time_predictions.csv`); if none is found,
the transcriptional (`pseudostage`) panels are skipped automatically.

## Stage provenance (audit)
- **Morphological stage** = `mdl_stage_hpf` — upstream sklearn Poly→Linear model
  (`morph_stage_model.joblib`) on morph-VAE PCA coords. Precomputed, not re-fit here.
- **Transcriptional stage** = `pseudostage` — upstream Hooke regression
  (`bead_expt_linear` → `time_predictions.csv`), merged via `morphseq_metadata.csv`.
  Precomputed, not re-fit here.
- Arrhenius reference: `6 + (timepoint-6)*(0.055*temperature - 0.57)`.

## 19C toggle
`INCLUDE_19C` at the top of the notebook (default `False`). When `True`, the 19C cohort
is kept everywhere and figures write to `figures/with_19C/` instead of `figures/no_19C/`.

## Figure output
`figures/no_19C/` or `figures/with_19C/` (next to the notebook), each figure as `.png` + `.pdf`.
