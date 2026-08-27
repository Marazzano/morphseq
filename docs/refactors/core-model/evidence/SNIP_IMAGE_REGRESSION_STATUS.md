# Snip image regression and compatibility status

**Status:** Major configuration regressions fixed in code; stored corpus not regenerated; exact
legacy/checkpoint parity not yet accepted

**Last reviewed:** 2026-08-24

**Priority:** Current model-input blocker

## Purpose

This document is the durable account of why snips produced by the replacement data pipeline
differed from legacy production snips, what those differences did to the morphology checkpoint,
which causes have been fixed, and what must still be demonstrated before the new snips are treated
as checkpoint-compatible.

The final root-cause analysis supersedes the earlier handoff's statement that the saturation cause
was unknown:

- [Final root-cause report](../../../../results/nlammers/20260729_morphseq_integration/root_analysis/IMAGE_QUALITY_ROOT_CAUSE_REPORT.md)
- [Earlier investigation handoff](../../../../results/nlammers/20260729_morphseq_integration/PIPELINE_IMAGE_REGRESSION.md)

## Executive conclusion

The conspicuous legacy/current difference was caused by two independent preprocessing regressions:

1. The canonical snip scale changed from **6.5 to 7.8 micrometers per pixel**.
2. The Gaussian mask-blend radius changed from **75 to 20 micrometers**.

The scale change made embryos smaller on the unchanged 576 x 256 canvas. The blend change narrowed
the transition into the synthetic background and retained many more bright CLAHE pixels near the
embryo silhouette, producing the clipped or saturated appearance.

Both defaults were restored and centralized in code on 2026-08-02 (`37aeb639`). They are currently:

```python
DEFAULT_TARGET_PIXEL_SIZE_UM = 6.5
DEFAULT_BLEND_RADIUS_UM = 75.0
```

That resolves the two diagnosed configuration defects for future runs. It does **not** establish
that the problem is closed:

- The audited production corpus predates the correction and was not regenerated.
- A post-fix, real-data legacy-parity acceptance run was not documented.
- Mask and orientation inputs still differ from legacy production.
- A later rendering rewrite on `origin/main` introduced small, measured raster differences and
  retained a known 180-degree orientation-flip tail.

The practical status is therefore:

> **Fixed in code; unresolved in stored data; exact checkpoint compatibility unproven.**

## Evidence base

The root-cause analysis paired 93 wells from `20250612_30hpf_ctrl_atf6`. Legacy and replacement
pipeline snips came from the same experiment and were re-encoded through the same
`20241107_ds_sweep01_optimum` checkpoint. Controlled rerenders varied scale and blend radius while
holding the source image, mask, crop, CLAHE, background, and encoding fixed.

This was a production A/B plus counterfactual rerender, not a comparison against a synthetic
reconstruction.

## Substantive legacy/current differences

| Contract element | Legacy production | Replacement pipeline at diagnosis | Consequence | Current status |
|---|---:|---:|---|---|
| Snip scale | 6.5 um/px | 7.8 um/px | Embryo 16.7% smaller in each linear dimension; about 30% fewer foreground pixels | Restored to 6.5 in code |
| Gaussian blend radius | 75 um | 20 um | Much narrower taper; bright boundary pixels retained instead of attenuated | Restored to 75 in code |
| Embryo mask | Lower-resolution UNet/JPEG-era mask | Full-frame SAM2 mask | Different silhouette, crop support, and downstream auxiliary-mask input | Intentional drift; parity effect open |
| Pose/orientation | Yolk-guided where yolk was available | Mass-distribution fallback because yolk masks are not wired | Possible 180-degree pose differences | Tracked, not resolved |
| Raster renderer on later `origin/main` | `skimage` resize, rotate, then crop | OpenCV area resize plus direct affine render | Small interpolation and placement differences | Measured, not byte-identical |

### Scale regression

On a fixed-size canvas, changing from 6.5 to 7.8 um/px predicts an embryo foreground-area ratio of:

```text
(6.5 / 7.8)^2 = 0.694
```

The measured legacy/current ratio was **0.700**. Foreground fill fell from 0.1287 to 0.0901.

The scale change also occurred before the model's 2x resize from 576 x 256 to 288 x 128. It
therefore discarded information before an already lossy model-input reduction and created a direct
train/serve mismatch for a checkpoint trained on 6.5 um/px snips.

### Blend-radius regression

The Gaussian width is:

```text
blur_sigma_px = blend_radius_um / snip_pixel_size_um
```

The effective taper changed from approximately 11.54 pixels in legacy production to 2.56 pixels in
the diagnosed replacement run:

```text
legacy: 75 / 6.5 = 11.54 px
new:    20 / 7.8 =  2.56 px
```

For the paired foreground pixels:

| Output | Mean | p95 | Fraction at or above 250 |
|---|---:|---:|---:|
| Legacy production | 122.67 | 226 | 0.003485 |
| Replacement production | 139.96 | 238 | 0.012650 |

The replacement snips therefore contained **3.63 times** the legacy saturated-pixel fraction and
were more saturated in 89 of 93 wells.

Holding every other input fixed and varying only the blend radius gave:

| Blend radius | Mean | p95 | Fraction at or above 250 |
|---:|---:|---:|---:|
| 20 um | 139.64 | 238 | 0.012615 |
| 50 um | 128.04 | 232 | 0.007105 |
| 60 um | 124.53 | 229 | 0.004989 |
| 75 um | 119.42 | 223 | 0.001875 |

Reverse-fitting a stored legacy final image against its stored pre-blend CLAHE image and mask
recovered a radius of 75.1 um. The equivalent current reverse fit recovered approximately 20 um.

The 20 um value was introduced to reduce contamination from neighboring embryos. That concern is
legitimate, but the preprocessing distribution was changed without versioning the image contract,
retraining the checkpoint, or validating the unchanged checkpoint. The preferred future repair is
to prevent non-target source pixels from entering the halo while preserving a checkpoint-compatible
taper, then validate the resulting distribution.

## Hypotheses that were excluded

The paired experiment and controlled rerenders ruled out the following as primary causes of the
legacy/current size and saturation regression:

- Keyence stitched/focus-stack source images or flat-field correction.
- Image polarity or a new inversion.
- CLAHE implementation or tested scikit-image version differences.
- JPEG versus PNG encoding.
- Per-image dtype/range stretching.
- Synthetic-background statistics.
- Scale as the main cause of saturation: changing scale at a fixed 20 um blend radius had little
  effect on the saturated fraction.

Matched old/current raw crops correlated at 0.992. Under identical downstream geometry, the current
source was slightly **less** saturated. The old Keyence stitching/seam defect was addressed in July,
and the remaining tile-gain warning should not be treated as the cause of this snip regression.

This does not prove that tile-wide gain can never clip a Keyence image. It establishes the narrower
claim needed here: it did not explain the paired legacy/current snip defect.

## Checkpoint impact

The same legacy checkpoint was used for both image generations. Positive controls showed that
re-encoding each image set reproduced its corresponding stored latent table, so the latent
difference was an image-input difference rather than an encoding-pipeline bookkeeping error.

Across the 93 paired wells:

- Median direct per-dimension correlation: **0.687**.
- Standardized RMSE: **1.009 legacy standard deviations**.
- Pairwise latent-distance correlation: **0.660**.
- Nearest-reference self-match: **18.3%**, versus about 1.1% by chance.
- Temperature R-squared: **0.582 legacy versus 0.353 replacement**.

In the controlled 2 x 2 scale/radius experiment, the scale change produced a standardized RMS
latent perturbation of 0.721 and the radius change 0.568, with a material interaction of 0.419.
Scale was the larger direct representation perturbation; radius was the direct visual-saturation
cause.

## Resolution status

### 1. Code defaults: resolved

The compatibility defaults live in
[`defaults.py`](../../../../src/data_pipeline/object_extraction/snip_processing/defaults.py) and are
used by the entrypoint, orchestration rule, task CLI, augmentation function, and alternate pipeline
wrapper. The live orchestration config also specifies 6.5 um/px and 75 um explicitly.

Tests pin the shared default values and demonstrate that the 75 um taper attenuates boundary
highlights more strongly than the rejected 20 um setting.

### 2. Stored production corpus: unresolved

The merged inventory for the paired GENE7 experiment was written on 2026-07-27, before the
2026-08-02 correction. It therefore refers to the diagnosed pre-fix snips.

The 2026-08-19 reconnaissance found that all 133 readable merged inventories lack both
`source_micrometers_per_pixel` and `snip_micrometers_per_pixel`; see
[`reports/PIPELINE_RECON.md`](../reports/PIPELINE_RECON.md). The audited corpus contains 699,505 snips,
but its inventories do not prove that any row was rendered under the corrected scale contract.

Existing snips must not be assumed checkpoint-compatible merely because the current code defaults
are correct.

### 3. Exact legacy parity: unresolved

Restoring 6.5/75 did not recreate legacy embeddings exactly in the controlled analysis. Remaining
differences include the mask source, orientation evidence, and source/crop assets. A 6.5/20 arm was
numerically closest to legacy under several metrics only because its harder blend accidentally
compensated for other drift; that is not evidence that 20 um is correct.

Exact byte equality is not required if the new image contract is deliberately accepted, but model
compatibility must be established with a real-data image and embedding gate.

### 4. Later rendering rewrite: branch-dependent and not fully closed

`origin/main` contains the snip-product/geometry rewrite merged as PR 31 (`5976f8d2`). The current
`core-model-refactor` branch contains the 6.5/75 correction but is not descended from that merge.

During development of the rewrite, an initial geometry implementation produced a median 7.8-pixel
centroid shift from the prior renderer. Correcting the placement reference reduced this to 0.72
pixels, with median SSIM approximately 0.995 and median embedding displacement 2.28%. The remaining
difference was attributed mostly to the OpenCV resampling kernel and direct rendering path.

That validation also found an approximately 2.5% sample tail of 180-degree flips. The
mass-distribution `vert_ratio` orientation rule could cross its 0.5 threshold depending on whether
it saw the source mask or a resampled mask. This survivor was documented but not fixed.

Any acceptance run must name the branch/commit being accepted. A pass on `core-model-refactor` does
not automatically accept the renderer currently on `origin/main`.

## SeaHUB segmentation-quality note

The early SeaHUB workflow called GroundingDINO box detection "segmentation" and principally gated
FOVs by whether eight boxes remained after NMS. In the GENE6/Pbx validation, the FOV reached the
expected count, but manual review found **two bona fide bad embryo crops** that the count-only QC did
not detect. This is recorded in the
[`SeaHUB README`](../../../../results/nlammers/20260723_seahub/README.md).

The later production design improved the contract by making precomputed box-prompted SAM2 masks
authoritative, selecting the prompt-associated connected component, filling holes, requiring
complete mask coverage, and retaining only the largest connected component as a final snip-stage
defense.

No full-corpus quantitative or visual segmentation acceptance report was found. The defensible
status is therefore:

> The known handoff mechanisms were improved, but empirical SeaHUB segmentation quality was not
> comprehensively signed off.

This is secondary to the general snip regression, but it should remain tracked.

## Individual z-slice snips: related identity gap

Full-frame Keyence z-plane materialization exists. Per-embryo masked z-plane snips do not. They
remain a design item in
[`PLAN_zslice_stitch_resolution.md`](../../../../src/data_pipeline/docs/PLAN_zslice_stitch_resolution.md).

Before adding them, the artifact contract needs an explicit product and z axis. The current
`snip_id` grammar has no focus/z component, so a focus-projection snip and individual z-plane snips
could collide or join incorrectly. At minimum, the inventory must carry product type and z
index/position explicitly. This is tracked here for continuity but is not the current image-quality
priority.

## Minimum acceptance exercise

Do not start with a bulk corpus rebuild. Regenerate one representative GENE7 plate under a fresh,
immutable output root using the exact code revision intended for production.

### Image gate

Pair regenerated and legacy snips by well/embryo and report:

1. Foreground area and linear extent.
2. Mean, p95, and fraction of pixels at or above 250.
3. Radial/taper profile across the embryo boundary.
4. Orientation disagreement and explicit 180-degree flips.
5. Neighboring-embryo/debris leakage into the 75 um halo.
6. Representative contact sheets, including the largest outliers rather than only medians.

### Model gate

Encode both sets through `20241107_ds_sweep01_optimum` and report:

1. Per-dimension correlation and standardized RMSE.
2. Pairwise-distance correlation and nearest-reference self-match.
3. Preservation of developmental and temperature signal.
4. Auxiliary-mask/QC changes caused by the regenerated snips.

### Provenance gate

Record with every candidate run:

- Git commit and branch.
- Target micrometers per pixel.
- Blend radius.
- Mask source/version.
- Orientation policy and evidence source.
- CLAHE settings.
- Background model and noise scaling.
- Frame shape, resampling kernel, and file encoding.

Only after those gates pass should the wider corpus be regenerated. For the 6.5/75 correction,
upstream image materialization and native-resolution segmentation do not need to rerun. The
required downstream chain is snip processing, snip auxiliary masks/QC, fraction-alive/viability,
legacy embeddings, and consumers of those products.

## Decision summary

- **Use 6.5 um/px and 75 um as the compatibility baseline.**
- **Do not use the pre-August stored replacement snips as checkpoint-compatible inputs.**
- **Do not attribute the regression to Keyence stitching.**
- **Treat exact legacy parity as an acceptance question, not as an implication of restoring two
  defaults.**
- **Track orientation/yolk-mask drift, but do not let it displace the immediate representative
  rerender and embedding test.**
- **Keep the SeaHUB count-only-QC miss visible until a full-corpus segmentation review exists.**
