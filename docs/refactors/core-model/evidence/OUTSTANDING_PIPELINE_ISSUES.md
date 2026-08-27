# Outstanding data-pipeline issues

**Snapshot:** 2026-08-24

**Scope:** A consolidated account of the outstanding `src/data_pipeline` issues discussed across
recent investigations. The current priority is compatibility between legacy and replacement-pipeline
snips. SeaHUB-specific concerns, individual z-slice export, and lower-priority pipeline risks are
included so they are not lost.

This document distinguishes four states:

- **Resolved in code:** the implementation was corrected.
- **Open in stored data:** existing outputs still reflect the old behavior or cannot prove otherwise.
- **Open validation:** code exists, but the relevant real-data acceptance exercise has not occurred.
- **Tracked:** a real concern that is not currently a priority.

## Executive issue register

| Priority | Issue | Current status |
|---|---|---|
| P0 | Replacement snips differed materially from legacy snips | Two main configuration causes fixed in code; stored corpus and checkpoint compatibility remain unresolved |
| P0 | Corpus provenance cannot identify the rendering contract | Open: merged inventories do not record source or snip pixel scale |
| P1 | Pipeline-wide visual-quality acceptance is weak | Open: production tests establish materialization, not legacy or visual parity |
| P1 | SeaHUB segmentation/crop quality | Improved architecture, but no full-corpus signoff; early count-only QC missed real bad crops |
| P2 | The restored 75 µm blend can admit neighboring material | Open design conflict; must be tested without silently changing checkpoint inputs again |
| P2 | Orientation remains heuristic without yolk masks | Open and tracked; a small 180-degree flip tail has been measured |
| P2 | Per-embryo individual z-slice snips | Designed but not implemented; product identity must be extended first |
| P3 | SeaHUB calibration, routing, and single-z QC semantics | Known and documented; not a current priority |
| Resolved/stale | Keyence stitching as the cause of the snip regression | Ruled out; the July stitching/seam work addressed the older defect |
| Resolved/stale | Full-frame Keyence z-stack materialization | Implemented; some planning text and runtime comments still describe the former limitation |

## 1. Legacy versus replacement-pipeline snips — current priority

The decisive investigation compared 93 matched wells from `20250612_30hpf_ctrl_atf6`. It found two
independent preprocessing regressions.

| Difference | Legacy | Replacement at diagnosis | Effect |
|---|---:|---:|---|
| Snip scale | 6.5 µm/px | 7.8 µm/px | Embryos became 16.7% smaller in each linear dimension; measured foreground-area ratio was 0.700 |
| Gaussian blend radius | 75 µm | 20 µm | The edge taper narrowed sharply and retained bright CLAHE pixels near the silhouette |
| Mask/pose inputs | Lower-resolution UNet/JPEG masks; yolk-guided orientation where available | Full-frame SAM2 masks; mass-distribution orientation | Additional silhouette and orientation drift, but not the cause of saturation |

The visual regression was substantive:

- Mean foreground intensity rose from 122.7 to 140.0.
- Foreground p95 rose from 226 to 238.
- The fraction of pixels at or above 250 rose 3.63-fold.
- Replacement snips were more saturated in 89 of 93 wells.
- Median embedding-dimension correlation was 0.687 and pairwise-distance correlation was 0.660.
- Temperature signal fell from an R-squared of 0.582 to 0.353.

The blend-radius change caused the clipped appearance. Its effective Gaussian width changed from
`75 / 6.5 = 11.54` pixels to `20 / 7.8 = 2.56` pixels. Controlled rerenders and reverse-fitting of
stored images recovered approximately 75 µm for legacy outputs and 20 µm for replacement outputs.

### What has been fixed

Commit `37aeb639` restored and centralized the compatibility defaults:

```python
DEFAULT_TARGET_PIXEL_SIZE_UM = 6.5
DEFAULT_BLEND_RADIUS_UM = 75.0
```

The entrypoint, orchestration rule, task CLI, augmentation function, and alternate wrapper now use
the shared defaults. This resolves the two diagnosed configuration defects for future runs.

### What remains unresolved

1. **The stored production corpus was not regenerated.** The paired GENE7 inventory predates the
   2026-08-02 correction. Current code defaults do not retroactively repair its snips.
2. **The corpus cannot prove its rendering scale.** All 133 readable merged inventories in the
   Phase 0 reconnaissance lack both `source_micrometers_per_pixel` and
   `snip_micrometers_per_pixel`.
3. **Checkpoint compatibility has not been accepted.** Restoring 6.5/75 did not reproduce legacy
   embeddings exactly because masks, orientation evidence, and source/crop assets also changed.
4. **The current branch is not the whole production story.** `origin/main` contains the later
   PR 31 renderer rewrite (`5976f8d2`), while `core-model-refactor` contains the 6.5/75 correction
   but is not descended from that merge. Any acceptance result must name the tested commit.
5. **A small pose-error tail survives.** Validation of the later renderer found roughly 2.5% of
   samples could flip by 180 degrees when resampling moved the mass-distribution `vert_ratio`
   across its 0.5 threshold.
6. **The neighbor-contamination problem that motivated 20 µm remains real.** A 75 µm outward blur
   can include a nearby embryo or debris. Replacing non-target pixels before blending is promising,
   but it would again change the model-input distribution and therefore requires an A/B gate.

### What was not responsible

The paired analysis ruled out Keyence stitching/flat-field correction, polarity, CLAHE version,
JPEG-versus-PNG encoding, dtype/range stretching, and background statistics as primary causes of
this regression. Matched raw crops correlated at 0.992, and the replacement source was slightly
less saturated under identical geometry.

The older Keyence tile-stitching/seam issue was addressed in July. It remains reasonable to track
tile-gain behavior as a general image risk, but it should not be presented as a live cause of the
legacy/current snip discrepancy.

### Required closure exercise

Regenerate one representative GENE7 plate under a fresh output root using the intended production
commit. Pair it with legacy outputs and measure:

- Foreground area, extent, intensity p95, saturation fraction, and boundary taper.
- Orientation disagreement, including explicit 180-degree flips.
- Neighboring-embryo or debris leakage into the blend halo.
- Embedding correlation, standardized RMSE, pairwise geometry, self-match, and preservation of
  developmental and temperature signal.
- Auxiliary-mask and QC changes.

Only after this passes should the wider snip corpus and dependent products be regenerated.

## 2. Pipeline-wide snip quality, QC, and provenance

The parity regression is the immediate problem, but it exposed a broader weakness: the pipeline
does not have a real-data golden-image suite for orientation, clipping, boundary preservation,
neighbor leakage, and cross-microscope consistency. `is_valid_snip` primarily establishes that a
snip was successfully materialized; it is not a visual-quality verdict.

The Phase 0 corpus audit found:

- 699,505 inventory rows across 133 readable inventories.
- 28 inventory-bearing experiments, representing 167,603 snips, have no QC artifact.
- Of 531,902 QC-evaluable rows, 185,204 pass `use_snip` and 346,698 fail.
- Only 176,466 rows are definite survivors of the combined metric-training gate.

These numbers do not imply that every excluded snip is visually corrupt. They do show that the
corpus cannot be treated as uniformly analysis-ready and that missing QC must remain unresolved,
not silently converted to failure.

Additional tracked issues:

- Yolk masks are not wired into current snip processing, so orientation normally uses a
  mass-distribution heuristic.
- Rotation/crop and preprocessing provenance are insufficient for auditing an individual failure.
- The preprocessing contract should record scale, blend radius, mask source, orientation policy,
  CLAHE settings, background model, frame shape, resampling method, and encoding.
- The Phase 0 `pixel_scale_profiles.csv` is empty because the relevant scale fields are absent.
  The existing per-experiment saturation table can help identify likely rendering eras, but it
  cannot replace explicit provenance.

## 3. SeaHUB-specific issues

SeaHUB has a distinct front end: detector boxes are reconciled into embryo crops, padded into
single-frame grayscale inputs, and handed to the shared back half. The main unresolved concern from
past discussions is segmentation/crop quality.

### Segmentation quality

In the early three-experiment review, GroundingDINO retained the expected eight detections after
NMS, but manual review found two bona fide bad embryo crops in one FOV. Count-only FOV QC therefore
missed genuine failures.

The later production contract improved the handoff by using cleaned box-prompted SAM2 masks as
authoritative inputs, retaining the prompt-associated connected component, filling holes, checking
crop coverage, and applying a largest-component defense downstream. These are meaningful safeguards,
but no full-corpus quantitative or visual segmentation acceptance report was found. Commit
`409d9c83` also describes the extraction patch as still needing testing.

Current status: **the architecture was improved, but SeaHUB segmentation quality has not received a
documented full-corpus signoff.**

### Known lower-priority SeaHUB caveats

- Scale calibration and fallback behavior remain important for physical-size features, but are not
  the current priority.
- Bundle construction must remain one invocation over the full manifest because canvas dimensions
  are corpus-wide per call.
- Focus and motion QC are not biologically interpretable in the ordinary way for arbitrary
  `single_z` images and should remain diagnostic rather than exclusionary.
- Synthetic SeaHUB plates mix stages and perturbations; plate identity is not a shared biological
  condition.
- The general 6.5/75 snip compatibility and neighbor-halo questions also apply once SeaHUB enters
  the shared snip-processing back half.

## 4. Individual z-slice snips

Two distinct capabilities have repeatedly been conflated:

1. **Full-frame z-plane materialization:** implemented for both YX1 and Keyence. Keyence support
   landed in July and current code writes z-stack rows and files. Several older configs and the
   living revision tracker still contain stale `NotImplementedError` language.
2. **Per-embryo masked z-plane snips:** not implemented. The plan proposes loading all z planes for
   an `image_id` and applying the stack's crop, rotation, and mask to each plane.

Before implementing the second capability, the artifact identity contract needs explicit product
and z axes. The current `snip_id` grammar has no focus/z component, so an individual z-plane snip
could collide or join incorrectly with the focus-projection snip. At minimum, inventories must carry
the product type and z index/position explicitly.

This is a real design gap but is secondary to establishing ordinary snip compatibility.

## 5. Other pipeline mechanics worth retaining

These are not current image-quality blockers, but remain relevant if a bulk regeneration is needed:

- Resident model servers removed much of the repeated model-load cost for auxiliary masks and
  detections, but their toggles default off pending end-to-end validation.
- GPU rules declare `gpu=1`, but scheduling enforcement still requires a CLI/profile resource
  budget. The current SGE scripts provide it.
- The snip-inventory and frame-detection merge/validation races described in older notes have been
  fixed and should not remain on the active-issue list.

## Recommended order of work

1. Run the representative GENE7 legacy/current rerender and checkpoint acceptance exercise.
2. Establish which existing experiments were rendered under which scale/blend contract; do not
   assume the stored corpus was fixed.
3. Add the missing preprocessing provenance and a small real-data golden-image regression suite.
4. Perform a bounded SeaHUB segmentation/crop review using the authoritative masks.
5. Scope the z-slice snip identity and storage contract before implementing export.

## Primary evidence

- [Final snip root-cause report](../../../../results/nlammers/20260729_morphseq_integration/root_analysis/IMAGE_QUALITY_ROOT_CAUSE_REPORT.md)
- [Earlier snip investigation handoff](../../../../results/nlammers/20260729_morphseq_integration/PIPELINE_IMAGE_REGRESSION.md)
- [SeaHUB validation README](../../../../results/nlammers/20260723_seahub/README.md)
- [SeaHUB handoff](../../../../results/nlammers/20260723_seahub/HANDOFF.md)
- [SeaHUB implementation contract](../../seahub/SEAHUB_IMPLEMENTATION_CONTRACT.md)
- [Z-slice plan](../../../data_pipeline/PLAN_zslice_stitch_resolution.md)
- [Phase 0 pipeline reconnaissance](../reports/PIPELINE_RECON.md)
- [Detailed snip-regression status](SNIP_IMAGE_REGRESSION_STATUS.md)
