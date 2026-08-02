# Snip image-quality regression: root-cause report

Date: 2026-07-30

Scope: legacy versus current snips for `20250612_30hpf_ctrl_atf6`, with production
provenance, controlled rerenders, and re-encoding through
`20241107_ds_sweep01_optimum`.

## Executive conclusion

The apparent degradation is two independent preprocessing regressions:

1. **Scale regression:** the canonical snip scale changed from **6.5 to
   7.8 µm/px**. On the unchanged 576×256 canvas this makes an embryo 16.7% smaller
   in each linear dimension and predicts an area ratio of
   `(6.5 / 7.8)^2 = 0.694`. The observed legacy/current foreground-area ratio is
   0.700.
2. **Saturation regression:** the Gaussian mask-blend radius changed from **75 to
   20 µm**. This is the direct cause of the brighter, clipped-looking embryo.
   Because blur sigma is `blend_radius_um / pixel_size_um`, the effective taper
   narrowed from 11.54 px in legacy production to 2.56 px currently. Bright
   CLAHE pixels near the silhouette are therefore retained instead of being
   attenuated into the dark synthetic background.

The source image, polarity, CLAHE implementation and library version, JPEG/PNG
encoding, flat-field data, interpolation, dtype/range handling, and background
statistics do not explain the regression.

## Quantitative evidence

Ninety-three legacy/current embryo wells pair exactly.

For fixed foreground pixels (`>30`):

| Output | Mean | p95 | Fraction >=250 |
|---|---:|---:|---:|
| Legacy production | 122.67 | 226 | 0.003485 |
| Current production | 139.96 | 238 | 0.012650 |

Current production therefore has **3.63×** the saturated-pixel fraction
(paired-bootstrap 95% CI 3.19–4.17×), and is more saturated in 89/93 wells.

Holding the current source, mask, crop, CLAHE, background, and encoding fixed:

| Blend radius | Mean | p95 | Fraction >=250 |
|---|---:|---:|---:|
| 20 µm | 139.64 | 238 | 0.012615 |
| 50 µm | 128.04 | 232 | 0.007105 |
| 60 µm | 124.53 | 229 | 0.004989 |
| 75 µm | 119.42 | 223 | 0.001875 |

The diagnostic 7.8/20 rerender matches all 93 current production snips
pixel-for-pixel. Reverse-fitting a stored legacy final image against its stored
legacy CLAHE intermediate and mask yields sigma 11.55 px, or 75.1 µm at
6.5 µm/px; the residual is JPEG noise. The corresponding current reverse fit is
approximately 20 µm.

Changing scale alone does not explain saturation: at radius 20, the fixed-
foreground saturation is 0.012615 at 6.5 µm/px and 0.012650 at 7.8 µm/px.

## Provenance

- The exact legacy runner produced the inspected files on 2025-07-07. It called
  `extract_embryo_snips` without overriding its defaults. Contemporaneous
  commits `cbcdd338fa1d` and `909a80b8e68f` define `outscale=6.5` and
  `dl_rad_um=75`.
- The new pipeline config introduced `target_pixel_size_um: 7.8` in commit
  `93ffd7904b51` on 2026-03-02. No supporting calibration or compatibility
  rationale was found for using 7.8 as the canonical snip scale.
- Commit `1c1d7a9f6a00` introduced `blend_radius_um: 20.0` on 2026-07-06, with
  the explicit rationale that a smaller radius avoids carrying close neighbors
  into a snip. Commit `72931d2ad0da` established it in the replacement direct
  entrypoint.

The neighbor concern is legitimate, but the change was made without versioning
the preprocessing contract or retraining/revalidating the unchanged legacy
checkpoint.

## Hypotheses excluded

- **CLAHE:** legacy stored and reconstructed CLAHE distributions match. The same
  raw crops processed by scikit-image 0.19.3, 0.20.0, 0.22.0, and 0.25.2 were
  byte-identical.
- **Source/flat-field images:** matched raw crops correlate at 0.992. Swapping
  old versus current source through identical current geometry makes the current
  source slightly less saturated.
- **Polarity:** old and current sources have positive correlation; inversion
  produces negative correlation. Neither production chain performs a new
  inversion.
- **JPEG versus PNG:** converting current PNGs to legacy-like JPEG slightly
  increases, rather than removes, the high-intensity tail.
- **Background noise:** current on-disk production uses the legacy-like 0.1×
  background scaling. The alternate `ops.py` path was not the producer of the
  inspected current PNGs, and dark background values cannot create pixels >=250
  inside the embryo.
- **Scale as the saturation mechanism:** fixed-radius counterfactuals give
  essentially unchanged saturation at 6.5 and 7.8 µm/px.

## Checkpoint impact

The same `20241107_ds_sweep01_optimum` checkpoint is serving both generations.
Positive controls verify the comparison:

- re-encoded legacy PNGs reproduce the stored legacy latent table with median
  per-dimension correlation 0.9999997;
- re-encoded current PNGs reproduce `analysis_ready` with maximum absolute error
  1.1e-6;
- the 7.8/20 counterfactual equals the current stored latent vectors exactly.

Across the 93 wells, current versus legacy 100-dimensional embeddings have:

- median direct per-dimension correlation 0.687;
- standardized RMSE 1.009 legacy standard deviations;
- pairwise-distance correlation 0.660;
- nearest-reference self-match 18.3% (chance is approximately 1.1%).

For biological dimensions 20–99, pairwise-distance correlation is 0.646 and
standardized RMSE is 0.910. Temperature R² falls from 0.582 in legacy embeddings
to 0.353 currently.

In the controlled 2×2 rerender, the scale change produces a standardized RMS
latent perturbation of 0.721 and the radius change 0.568, with a material
interaction (0.419). Scale is the larger direct model perturbation; radius is
the direct image-saturation cause.

Important caveat: current-source/current-mask 6.5/75 renders do not recreate the
legacy embeddings exactly. Current segmentation, rotation, and source assets
also differ. In this factorial, 6.5/20 is numerically closest to the stored
legacy latent vectors for several metrics because the harder blend partially
compensates for those other differences. That is accidental compensation, not
evidence that 20 µm matches the training image distribution. Restoring two
configuration values must therefore be followed by an embedding acceptance
test; it cannot be assumed to restore full legacy model behavior.

## Blast radius

Affected products are the canonical snips and anything consuming their pixels:
snip auxiliary masks and QC, fraction-alive/viability outputs, latent
embeddings, and downstream `analysis_ready` joins or biological analyses using
those outputs.

The 6.5→7.8 snip-grid change does not alter upstream native-resolution
segmentation geometry, curvature, or surface-area measurements for these
Keyence experiments. Those do not need to be recomputed solely because of this
snip regression.

Secondary drift remains in mask generation and pose: legacy used low-resolution
UNet/JPEG masks plus yolk-guided orientation; current uses full-frame SAM2 masks
and mass-distribution orientation. Controlled same-input tests show that these
are not the saturation cause, but they matter for exact legacy embedding
parity.

## Recommended response

1. **For visual/training-image parity, restore the canonical image settings to
   6.5 µm/px and 75 µm.** Rerender a representative plate before a bulk rebuild.
2. **Treat embedding parity as a separate acceptance gate.** Compare regenerated
   vectors, pairwise geometry, developmental/temperature signal, and auxiliary
   QC against legacy references. Exact continuity may require recreating legacy
   mask/orientation preprocessing, not just changing two knobs.
3. **Resolve the neighbor conflict explicitly.** If a 75 µm outward blur carries
   adjacent embryos, replace out-of-mask source pixels with synthetic
   background before applying a legacy-width taper. Validate or retrain the VAE
   before adopting this redesigned distribution.
4. **Version the preprocessing contract with each checkpoint:** scale, blend
   radius, mask/pose method, CLAHE settings, background model, frame shape, and
   file encoding. Record these in every snip inventory.
5. **Add golden-image regression tests:** foreground area, p95 and saturation,
   taper profile, exact stored fixtures, and checkpoint embedding drift.
6. After acceptance, rerun snip processing and snip-dependent downstream
   products. Upstream image materialization, detection/segmentation, and native
   geometry need not be rerun unless the chosen compatibility plan changes
   masks or orientation.

No production pipeline source or configuration was changed during this
investigation.
