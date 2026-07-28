# Legacy Keyence focus-stack comparison

This diagnostic routes only the Keyence focus projection through the legacy
Build01 source:

- `src.build.data_classes.MultiTileZStackDataset` for the legacy joint-tile
  percentile sampling/rescaling and float16 conversion.
- `src.build.export_utils.LoG_focus_stacker` for the legacy LoG selection.
- The current experiment stitch map is reused so alignment is held constant.
- Production pipeline images and inventories are not modified.

## Result

For `20260702_hotchem_30hpf_plate01` at `t0000`, the current and legacy
projections are substantially different:

| well | current mean | legacy mean | mean absolute difference (0-255) |
|---|---:|---:|---:|
| A01 | 189.29 | 74.77 | 134.69 |
| A02 | 195.04 | 70.26 | 143.14 |
| B01 | 177.07 | 95.84 | 122.76 |

The legacy A02 projection does not reproduce the current bright-background,
white-patch rendering. The old path retains the legacy dark-field polarity and
selects visibly different detail around the embryo and debris. This supports a
regression in the new focus materialization path rather than saturation already
present in the source Z planes.

The implementation difference most likely responsible is contrast handling:

- Legacy Keyence Build01 derives one shared 0.1--99.9 percentile range across
  all tiles, rescales once to `[0, 1]`, converts to float16, and runs LoG.
- The current Keyence materializer calls `im_rescale` separately for each tile,
  then passes that result to `materialize_ff_projection`, which calls
  `im_rescale` a second time before LoG.

See `outputs/comparison_metrics.csv`, the `__legacy_build01.png` projections,
and the `__absdiff.png` images for the direct comparisons.

## Improved shared-bound experiment

`generate_improved_comparison.py` adds two deterministic variants for A02 t0000:

- `shared_clean`: exact shared 0.1--99.9% uint16 histogram bounds, one clipped
  affine transform for LoG scoring, float32 processing, raw-pixel gathering,
  and one shared display transform.
- `improved_unclipped`: the same route, except LoG scoring uses the unclipped
  affine tensor; clipping is reserved for the final display transform.

The exact shared bounds were `lo=5654`, `hi=56629`. The two variants selected
different Z planes at only 7,613 of 2,073,600 tile pixels (0.3671%). Both restore
the legacy appearance without the current pipeline's washed-out rendering.

Against the legacy stitched image:

| variant | mean absolute pixel difference | SSIM | embryo-crop Laplacian variance |
|---|---:|---:|---:|
| legacy | 0.000 | 1.0000 | 3126.92 |
| shared clean | 2.301 | 0.8758 | 3138.38 |
| improved unclipped | 2.420 | 0.8694 | 3134.76 |

The embryo montage is
`outputs/20260702_hotchem_30hpf_plate01_A02_BF_t0000__embryo_comparison_montage.png`.

## Reproduce

```bash
env PYTHONPATH=/net/trapnell/vol1/home/mdcolon/proj/morphseq \
  /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260712_legacy_focus_stack_comparison/compare_legacy_keyence_focus.py \
  --experiment 20260702_hotchem_30hpf_plate01 \
  --experiment-root pipeline/output/acquisition/20260702_hotchem_30hpf_plate01 \
  --wells A01 A02 B01 \
  --output-dir results/mcolon/20260712_legacy_focus_stack_comparison/outputs \
  --device cpu
```
