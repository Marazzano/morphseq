# Focus QC fine-tuning plan

Date: 2026-06-30

Original goal: determine whether the pre-fine-tuning focus signal really needs local
normalization/context, and whether the pixel-level Sobel threshold `grad > 0.04` should be retuned
before pipeline wiring.

## Final decision after review

Use the fine-tuned focus gate:

```text
normalization_mode = local_context
interior_gradient_threshold = 0.02
focus_flag = interior_strong_edge_frac < 0.50
```

This is a deliberate low-information exclusion compromise. The review showed that this gate catches
the whole-embryo ghost/structureless anchors and removes a small low-information tail (~2.1% overall
in the 7,500-row sample; concentrated in 20251125). Some bright/dorsal embryos with genuinely little
internal structure are removed. That is acceptable: for the first pipeline gate, we prefer to remove
low-information snips rather than preserve every borderline bright/dorsal case.

## Inputs

- `tables/interior_structure_cross_experiment_metrics.csv`
  - 7,500-row cross-experiment population used by the latest focus handoff
  - includes `image_path`, `mask_path`, `top_spread_p99_p90`, current
    `interior_strong_edge_frac`, and current `interior_std`
- `tables/anchor_interior_structure_metrics.csv`
  - curated focus/blur/dorsal/saturation anchors

## Sweep

For each image/mask row, recompute eroded-interior Sobel edge fractions under these normalization
modes:

- `local_context`: robust 1st-99th percentile rescale over mask plus a dilated local context band
  (the current research-script behavior)
- `mask_only`: robust 1st-99th percentile rescale using only embryo-mask pixels
- `raw_255`: no percentile normalization; use raw grayscale divided by 255
- `crop_global`: robust 1st-99th percentile rescale over the whole local crop

For each normalization mode, sweep:

- Sobel strong-edge threshold: `0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10`
- embryo-level focus gate: report summaries at `interior_strong_edge_frac < 0.20` and `< 0.25`

The crop/mask geometry stays fixed:

- crop around mask with 32 px pad
- erode mask by 12 px, fallback to 6 px, then full mask if interior has fewer than 200 pixels
- Gaussian smooth normalized image with `sigma=1.0`
- Sobel edge magnitude from `skimage.filters.sobel`

## Outputs

All outputs land under `fine_tuning/`.

- `tables/focus_sobel_threshold_sweep_metrics.csv`
  - per-snip metric values for every normalization mode and Sobel threshold
- `tables/focus_sobel_threshold_sweep_summary.csv`
  - per-experiment flag fractions for each mode/threshold/gate
- `tables/focus_sobel_threshold_anchor_summary.csv`
  - anchor hit table for each mode/threshold/gate
- `figures/interior_strong_edge_frac_histograms_by_mode.png`
  - population histograms for `interior_strong_edge_frac`
- `figures/focus_vs_top_spread_threshold_grid.png`
  - focus-vs-bright-tail spread panels across normalization modes and Sobel thresholds
- `figures/focus_flag_fraction_heatmap.png`
  - flag-rate heatmap by normalization mode and Sobel threshold

## Review questions

1. Does `raw_255` separate ghosts/blur anchors similarly to `local_context`?
2. Does `mask_only` inflate edge fractions for flat bright or saturated embryos?
3. Is the initial `grad > 0.04` setting stable, or is the anchor/population behavior better at a
   different point in the swept range?
4. Does the initial `0.25` embryo-level gate remain reasonable under the chosen normalization mode?
5. If raw crops are used instead of projection-image transient crops, rerun this same script on that
   raw-crop evidence and compare the same outputs before wiring pipeline thresholds.
