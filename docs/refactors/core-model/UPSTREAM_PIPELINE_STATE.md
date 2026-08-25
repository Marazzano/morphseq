# Upstream pipeline state — what the core-model refactor is building on

Snapshot: 2026-08-24. Assembled from the pipeline's own written record, not from a fresh run.
Canonical tracker: [`docs/data_pipeline/PLANNED_REVISIONS.md`](../../data_pipeline/PLANNED_REVISIONS.md).

This refactor consumes pipeline snips as its training corpus. Three upstream issues bear on that
directly; one of them is not cosmetic.

---

## 1. The snip corpus has a known image-quality regression — 🟡

**Read first:** `results/nlammers/20260729_morphseq_integration/root_analysis/IMAGE_QUALITY_ROOT_CAUSE_REPORT.md`.
Tracked as PLANNED_REVISIONS §5.

Two preprocessing values drifted away from the settings the current checkpoint was trained under:

| | legacy | regressed | now |
|---|---|---|---|
| snip scale | 6.5 µm/px | 7.8 | **6.5 (restored)** |
| blend radius | 75 µm | 20 | **75 (restored)** |

The radius change is the direct cause of the bright/clipped look — blur sigma is
`blend_radius_um / pixel_size_um`, so the taper collapsed 11.54 px → 2.56 px and the saturated-pixel
fraction rose **3.63×** (more saturated in 89/93 wells). The scale change shrank embryos 16.7% per
linear dimension on the fixed 576×256 canvas.

**Why this matters here, concretely.** Snips written between 2026-03-02 (`93ffd7904b51`) and the
restoration are drawn from a different image distribution than both legacy snips and snips written
now. Any training package assembled across that window mixes two distributions. Measured effect on
embeddings from the *same* checkpoint: median per-dimension correlation 0.687, pairwise-distance
correlation 0.660, temperature R² 0.582 → 0.353.

**Two things are still open and both gate this refactor:**

- **No embedding acceptance test has been run.** Restoring the two knobs is explicitly *not* proven
  to restore model behaviour — segmentation (SAM2 vs low-res UNet), pose (mass-distribution vs
  yolk-guided), and source assets also drifted. The report warns that 6.5/20 scored closest to legacy
  on several metrics by *accidental compensation*, so metric proximity is not evidence here.
- **The neighbour conflict that motivated the 20 µm radius is unresolved.** A 75 µm outward blur can
  carry adjacent embryos into a snip. The proposed fix — synthetic-background fill outside the mask
  before a legacy-width taper — changes the image distribution again and would need validation or a
  retrain.

**Action for this refactor:** date-stamp the snip corpus and confirm which side of the regression each
experiment's snips were written on before treating the package as homogeneous.

### A gap in our own Phase 0 recon

[`reports/recon_tables/pixel_scale_profiles.csv`](reports/recon_tables/pixel_scale_profiles.csv) is
**1 byte — empty**. Pixel scale is precisely the axis of half this regression, so the recon currently
has no coverage of the thing most likely to be heterogeneous across the corpus. Worth re-running.

By contrast [`reports/recon_tables/intensity_by_experiment.csv`](reports/recon_tables/intensity_by_experiment.csv)
already carries `saturated_255_fraction_{p05,median,p95}` per experiment — that column is a direct
readout of the regression and can be used to date-stamp the corpus without re-rendering anything.

---

## 2. SeaHub data is built but unverified — 🟡

Tracked as PLANNED_REVISIONS §6. Docs in [`../seahub/`](../seahub/); runbook at
`results/nlammers/20260723_seahub/HANDOFF.md`.

The newest pipeline commit (`409d9c83`) is *"Patch to resolve issues with SeaHUB embryo image
extraction. **Still needs testing**"* — the full corpus run has not been verified end to end.

For this refactor the load-bearing caveat is that **µm/px is a placeholder (7.8) for all SeaHub rows**.
Physical scale is uncalibrated there, so any physical-size feature must exclude
`source_scope == 'seahub'`. Focus and motion QC also false-flag ~all SeaHub rows (single-z, annotate-only),
and a SeaHub "plate" is synthetic — it mixes perturbations and stages, so plate is never a shared condition.

---

## 3. Z-slice snips were designed but never built — 🔴

P3 in [`docs/data_pipeline/PLAN_zslice_stitch_resolution.md`](../../data_pipeline/PLAN_zslice_stitch_resolution.md).

If this refactor wants z-slices as training input in addition to full-frame, the machinery does not
exist yet. The design is scoped: extend snip processing to load z-planes per `image_id` via
`load_z_stack_images_from_image_id` and apply the same crop/rotate/mask — one mask and one rotation
per stack. Storage was the blocker, and the 6.5 µm/px + jpg decision was made partly to keep this
viable (source resolution must be ≥ snip target, which 6.5 now satisfies).

Related and still open: Keyence `z_stack` materialization raises `NotImplementedError`, currently
worked around by dropping `focus_flag` and `motion_blur_flag` from `snip_qc` (§2).

---

## 4. Throughput — mostly resolved 🟢

Not a correctness concern, but relevant if this refactor triggers bulk re-runs. Resident model servers
landed and removed the per-well model-reload cost: `snip_auxiliary_masks` 2.88×, `frame_detections` ~3×.
Both are behind config toggles defaulting **off** pending an end-to-end run. `frame_masks` was measured
as not worth serving on long time series (1.02×) and is enabled only by SeaHub overlays. GPU rules now
declare `resources: gpu=1`, but that is inert without `--resources gpu=1` on the CLI — it is baked into
the SGE scripts, not into a profile.

---

## Bottom line

The one item that should change what this refactor does is **§1**. The corpus is not known to be
homogeneous, the acceptance test that would establish it has not been run, and our own pixel-scale
recon table is empty. Everything else is either annotation-only (§2), not-yet-built (§3), or
performance (§4).
