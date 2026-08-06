# Snip-image regression: pixel scaling and saturation

Investigated 2026-07-29/30. Handoff document for a second agent.

Two independent defects in the current pipeline's snip images, found by A/B-ing against legacy
snips that exist on disk for the **same experiments**. Issue 1 is solved and has a one-line fix
pending a decision. Issue 2 is localized but **not** solved.

Both are confined to the snip raster and everything computed on it — critically including the VAE
embeddings. Physical measurements and QC are unaffected (§3).

---

## 0. TL;DR

| | issue 1 — scale | issue 2 — saturation |
|---|---|---|
| Symptom | embryos ~30% smaller in area | ~4x more blown-out pixels, head region worst |
| Status | **SOLVED** | **OPEN** |
| Cause | `target_pixel_size_um` 6.5 -> 7.8 | unknown; ruled out FF, scale, CLAHE code |
| Fix | change one config value, re-run 4 steps | unknown |
| Confidence | very high (3 independent confirmations) | high that it's real; cause unidentified |

---

## 1. The evidence base — a real A/B, not a reconstruction

Legacy snips exist on disk for all six 20250612 (GENE7) plates, produced by the previous pipeline
generation from the *same* raw images:

- `data/morphseq/training_data/bf_embryo_snips/{experiment_id}/{experiment_id}_{well}_e00_t0000.jpg`
  — final snips (CLAHE + background blend), 256x576 grayscale
- `data/morphseq/training_data/bf_embryo_snips_uncropped/{experiment_id}/...`
  — **CLAHE'd but NOT background-blended.** This is the key intermediate for issue 2.
- `data/morphseq/metadata/embryo_metadata_files/{experiment_id}_embryo_metadata.csv`
  — carries `snip_um_per_pixel`, authoritative for the legacy scale

Current pipeline equivalents:

- `pipeline/output/object_extraction/{exp}/snips/per_well/{well_id}/snips/{physical_embryo_id}/{snip_id}.png`
- `pipeline/output/object_extraction/{exp}/snips/per_well/{well_id}/{well_id}_snip_inventory.csv`
  — has `processed_snip_path`, `embryo_mask_snip_path`, `image_path`, `mask_id`

Both canvases are **256x576**, so all differences are content, not geometry.

95 paired embryos were compared for `20250612_30hpf_ctrl_atf6`. Measurement method matters: both
production sets have near-black backgrounds, so a fixed threshold (`>30`) is a fair foreground
definition for both. **Do not use Otsu** — it latches onto injected background noise in re-renders
and produces meaningless numbers. That mistake cost an hour and briefly produced a false
"hypothesis refuted" result.

---

## 2. Issue 1 — pixel scaling. SOLVED.

### Cause

`target_pixel_size_um` is **7.8** in the live config. The legacy pipeline used **6.5**
(confirmed from `snip_um_per_pixel: 6.5` in the legacy per-experiment metadata for this same
experiment). 7.8 / 6.5 = 1.20, a 20% coarser raster.

### Confirmed three independent ways

| method | area ratio | linear ratio |
|---|---|---|
| Predicted from 6.5 -> 7.8 | 0.694 | 0.833 |
| Old vs new **production** snips, 95 paired embryos | **0.700** | 0.837 |
| Controlled re-render: same embryo, same code, only the scale differs | **0.708** | 0.841 |

Foreground fill of the canvas: legacy 0.1287 -> current 0.0901.

### Why the mechanism is clean

`crop_to_embryo_bounds` in
`src/data_pipeline/object_extraction/snip_processing/extraction.py` takes a **fixed-size window
centered on the mask centroid**. It does *not* scale the embryo to fit the canvas. Both pipeline
generations work this way. Therefore apparent embryo size is a pure function of
`target_pixel_size_um`, with no confounds.

### Where the number lives — note the inconsistency

| location | value |
|---|---|
| `pipeline_orchestrator/config.yaml:180` | **7.8** (the live value) |
| `pipeline_orchestrator/rules/snip_processing.smk:87` | 7.8 (fallback) |
| `pipeline_orchestrator/tasks.py:1479` | 7.8 (argparse default) |
| `snip_processing/entrypoints/run_snip_processing.py:99` | 7.8 (kwarg default) |
| `snip_processing/pipelines/snip_processing.py:45` | **6.5** (fallback — the odd one out) |

Four places say 7.8, one says 6.5. The 6.5 matches the legacy pipeline exactly, which reads as the
vestigial original with the default having been moved to 7.8 everywhere else and one site missed.

### Where 7.8 probably came from

`src/data_pipeline/acquisition/seahub/integration.py:64` —
`SeaHubIntegrationConfig.micrometers_per_pixel: float = 7.8`, with
`calibration_status: str = "placeholder"` and, at line 605, the pipeline's own warning:

> `"calibration_issue": "SeaHub pixel calibration is unverified; revisit 7.8 um/px."`

7.8 appears nowhere else in the tree. The most plausible history is that a SeaHub placeholder
calibration was promoted into the global snip config.

### Corroborating detail

`frame_inventory` carries two scales per Keyence experiment:

- `BF/projection/focus_stack` — 1440x3420 at raw **1.887 um/px** (this is what snips read)
- `BF/z_stack` — 418x993, downsampled to exactly **6.50057 um/px**

So 6.5 survives elsewhere in the pipeline as a deliberate resolution choice. That further isolates
7.8 in `snip_processing` as anomalous.

### Consequence

The embryo carries ~30% fewer pixels *before* the 2x reduction to the 288x128 model input
(`config.yaml` documents the model input as `[576, 256]` downsampled 2x). Information is destroyed
ahead of an already lossy downsample.

---

## 3. Blast radius — what the scale does and does not touch

`target_pixel_size_um` is consumed in exactly one place: `snip_processing.smk:87` ->
`process_snip_row`. Nothing upstream reads it.

**NOT affected** — computed at full-frame native resolution (1.887 um/px for Keyence):

| product | why |
|---|---|
| `mask_geometry` (`area_um2`, `length_um`, `width_um`, `perimeter_um`, centroids, bbox) | decodes `frame_masks` RLE at full frame; takes `image_micrometers_per_pixel` from `frame_inventory` |
| `curvature_metrics` (`total_length_um`, `mean_curvature_per_um`, `baseline_deviation_*`) | same RLE-decode path |
| `pose_kinematics` | derived from full-frame centroids |
| `stage_predictions` | no pixels involved |
| **`surface_area_qc`** | consumes `area_um2` -> the separate QC audit is uncontaminated |

**AFFECTED** — lives on the snip grid:

| product | note |
|---|---|
| the snip PNGs | the model input itself |
| **`legacy_embeddings`** | encodes `processed_snip_path` directly. This is the one that matters. |
| `snip_auxiliary_masks` (UNet: focus / bubble / via / yolk) | runs on the snip crop at snip resolution |
| `fraction_alive` | ratio of via ∩ embryo on the snip grid — scale-invariant in principle, but the UNet was trained at the legacy scale, so prediction quality at 7.8 is an open question |
| `focus_qc`, `motion_blur_qc` | fed from snip-grid aux masks |
| `viability_dead_flag` | via death detection, which consumes `fraction_alive` |

**Fix blast radius:** `snip_processing -> snip_auxiliary_masks -> fraction_alive ->
latent_embeddings`. Segmentation, `frame_masks`, `mask_geometry`, `curvature_metrics`, and
`stage_predictions` do **not** need to re-run. Much cheaper than a full rebuild.

### Separate, worse, and upstream: SeaHub calibration

For `20260724_seahub_*` experiments, `frame_inventory.image_micrometers_per_pixel = 7.8` is the
**native** calibration — there is no `raw_micrometers_per_pixel` at all. Since `mask_geometry`
reads that field directly, **every physical measurement for SeaHub experiments, and therefore
`surface_area_qc`, rests on the placeholder the code itself flags as unverified.** This is a
different and arguably more consequential problem than the snip scale, and it lands on the SeaHub
back-half work rather than on 20250612. Not investigated further.

---

## 4. Issue 2 — saturation. OPEN.

### Symptom, quantified

Within the embryo foreground, legacy -> current:

| metric | legacy | current | change |
|---|---|---|---|
| fraction of pixels >= 250 | 0.0030 | 0.0117 | **3.94x** |
| p95 intensity | 224 | 238 | +14 |
| mean intensity | 123 | 141 | +18 |

Visually worst in the head/yolk region — the densest structure.

### Ruled out

**1. Not the FF / focus-stack images.** Direct comparison for the same well:

| image | shape | mean | p99 | fraction >= 250 |
|---|---|---|---|---|
| legacy `stitched_FF_images/A01_t0000_stitch.jpg` | 3420x1440 | 93.8 | 249 | 0.00929 |
| current `projection/focus_stack/..._BF_t0000.png` | 3420x1440 | 91.4 | 242 | **0.00424** |

The current FF is *less* saturated. So neither pre-FF downsampling nor FF-stage normalization is
responsible — this eliminates both of the hypotheses in the original framing of the question.

**2. Not primarily the scale change.** Controlled re-render, same embryo and code, only
`target_pixel_size_um` varying: saturation 0.0116 (6.5) -> 0.0145 (7.8) = **1.25x**, p95 flat at
~238-239. That accounts for a small fraction of the observed 3.94x.

**3. Not a CLAHE code change.** Both generations call
`skimage.exposure.equalize_adapthist(image) * 255` with **all defaults** (kernel_size =
`image.shape // 8` = 72x32 px, clip_limit 0.01, nbins 256):

- legacy: `src/build/build03A_process_embryos_main_par.py:235`
- current: `src/data_pipeline/object_extraction/snip_processing/augmentation.py:27`

Same CLAHE-then-Gaussian-tapered-blend order in both.

**4. Not the per-image intensity stretch.** `extract_embryo_crop` contains
`skimage.exposure.rescale_intensity(image, in_range='image', out_range=(0,255))`, but it is guarded
by `if image.dtype != np.uint8` and the focus-stack PNGs are uint8. It never fires. (Note there is
also a dead sibling function `extract_and_rescale` in the same file with no callers — do not confuse
them.)

### The decisive clue

**The current code at the LEGACY scale already produces legacy-unlike saturation.** Re-render at
6.5 um/px gives 0.0116, which matches current production (0.0117) rather than legacy production
(0.0030). So the residual ~3x lives inside the current snip chain and is independent of scale.

### Nick's lead — untested, and the most promising one

**The snips are inverted.** Bright "blown out" regions in the snip correspond to *dark* regions in
the original image. That reframes the problem: this is plausibly a **background-subtraction or
background-estimate** effect rather than highlight clipping.

Supporting circumstantial evidence: `estimate_background_stats_full_frame` in
`snip_processing/ops.py` defines background as *source stitched image pixels where embryo_mask == 0,
computed in full-frame space before resize/rotate/crop*. Its docstring says it was
"ADAPTED FROM `build03A_process_images.py::estimate_image_background()`" and notes the original was
"broken" and returned "hardcoded fallback values". **An adaptation that fixed a broken function is
exactly the kind of change that would shift background statistics between generations** — and
`background_mean` / `background_std` feed `blend_with_background_noise`, which composites the final
snip. Worth checking whether the legacy pipeline effectively ran with those hardcoded fallbacks
while the current one computes real statistics.

Caveat on interpretation: background blending affects pixels *outside* and at the *edge* of the
mask via the Gaussian taper. It should not directly brighten the interior. So if the interior is
saturating, either the taper reaches further than expected, or the effect is upstream of the blend.
The pre-CLAHE comparison below distinguishes these.

---

## 5. Next steps — ordered, tied to specific files

### Step 1. Isolate where saturation enters: pre-CLAHE vs post-CLAHE

This is the single highest-value experiment and it needs no new code.

`process_single_snip` (`snip_processing/process_snips.py`) already accepts
`save_raw_crops: bool` + `raw_crops_dir`, and writes the **pre-CLAHE, pre-blend** crop as a TIF at
Step 3/4 boundary. `augment_snip` also **returns** `clahe_only` (the CLAHE'd, un-blended image) as
its second value, which `process_snip_row` currently discards.

Do this:

1. Call `process_single_snip` directly for ~10 wells of `20250612_30hpf_ctrl_atf6` with
   `save_raw_crops=True`, capturing raw crop, `clahe_only`, and final.
2. Compare three histograms against the legacy counterparts:
   - **pre-CLAHE crop** vs — no legacy equivalent on disk, but the legacy source is the same FF
     image, so compare against a legacy-scale re-render of the same crop
   - **`clahe_only`** vs `bf_embryo_snips_uncropped/` (legacy CLAHE'd, un-blended) — **this is the
     direct comparison that localizes the defect**
   - **final** vs `bf_embryo_snips/`

Interpretation:
- pre-CLAHE crops match, `clahe_only` diverges -> the fault is in the CLAHE invocation (input
  dtype/range, or the effective kernel given the changed embryo-to-canvas ratio)
- pre-CLAHE crops already diverge -> the fault is in Step 1-3 (`extract_embryo_crop` /
  `apply_rotation_to_snip` / `crop_to_embryo_bounds`)
- `clahe_only` matches but final diverges -> the fault is in `blend_with_background_noise`, which
  points straight at the background-stats change and confirms Nick's lead

Reference implementation to copy: the re-render harness in §6 below.

### Step 2. Test the background-stats hypothesis directly

Files: `snip_processing/ops.py::estimate_background_stats_full_frame`,
`snip_processing/augmentation.py::blend_with_background_noise`, and the legacy
`src/build/build03A_process_images.py::estimate_image_background`.

1. Read the legacy `estimate_image_background` and determine what it actually returned in practice
   (the current docstring claims it returned hardcoded fallbacks).
2. Print `background_mean` / `background_std` as the current pipeline computes them for these
   experiments. `process_snip_row` records both on every manifest row
   (`background_mean`, `background_std`, `background_definition`) — find the snip_processing
   manifest and read them rather than recomputing.
3. Re-render with the legacy values substituted and see whether saturation drops to ~0.003.

### Step 3. Rule out the inversion/polarity path

Confirm where inversion happens and whether it is symmetric between generations. `SeaHubIntegrationConfig`
has `flip_polarity: bool = True`, so polarity handling exists in the codebase and is configurable.
Establish: are Keyence snips inverted, at which step, and identically in both generations? If the
current pipeline inverts at a different point relative to CLAHE or the blend, that alone could
produce the asymmetry.

### Step 4. Decide and apply the scale fix

Decision for Nick, not for the agent: set `config.yaml:180 target_pixel_size_um` to 6.5 (restoring
parity with the checkpoint's training distribution) or leave 7.8 and retrain.

Strong argument for 6.5: the VAE checkpoint in use is `20241107_ds_sweep01_optimum` — the **same
checkpoint** that produced the legacy latents, per `embedding_model_name` in `analysis_ready`. It was
trained on 6.5 um/px snips. Running it on 7.8 um/px inputs is a train/serve distribution mismatch.

If changed, also reconcile `snip_processing.py:45` (already 6.5), `snip_processing.smk:87`,
`tasks.py:1479`, and `run_snip_processing.py:99` so all five sites agree, and consider whether the
value belongs in one place rather than five.

Then re-run only: `snip_processing -> snip_auxiliary_masks -> fraction_alive -> latent_embeddings`.

### Step 5. Quantify the model-response impact (Nick's "track 2")

Once both defects are understood, measure what actually changed in embedding space rather than in
pixel space:

- 93 wells of `20250612_30hpf_ctrl_atf6` have both legacy and current latents.
- Legacy latents: `data/morphseq/legacy/20241107_ds_sweep01_optimum/morph_latents_{exp}.csv`
  (`z_mu_n_00–19` + `z_mu_b_00–79`, plus sigmas).
- Current latents: the `z_mu_00–99` block in `analysis_ready`.
- **Same checkpoint**, so dimensions should correspond — but the flat <-> `n`/`b` ordering is
  **unverified**. Recover the permutation by correlating across the 93 shared wells before drawing
  any conclusion. Expect strong-but-imperfect correlation because the images genuinely differ.
- Then report: per-dimension correlation, shift in PCA/UMAP space, and whether biologically
  meaningful structure (temperature, genotype) is preserved or degraded.

Join on `well_id`. Legacy ids are `..._A01_e00_t0000`; current are `..._A01_e01_BF_t0000` — the
embryo index is off by one and there is no channel token in legacy. See
`src/morphseq_integration/HANDOFF.md` §2a for the full compatibility list.

---

## 6. Reproducing the measurements

Environment — only `points-ml` has pandas + pyarrow + matplotlib + skimage together:

```bash
cd results/nlammers/20260729_morphseq_integration
PYTHONPATH=/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src:. \
  /net/trapnell/vol1/home/nlammers/micromamba/envs/points-ml/bin/python <script>
```

Helpers already written and working in `qc_diagnostic_utils.py` (same directory):
`load_analysis_ready`, `load_snip_inventory`, `load_frame_masks`, `_resolve` (handles the
absolute-vs-output-root-relative path mix in the inventories), `mask_outline`, `build_pair_image`.

Re-render harness — the pattern that worked:

```python
from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle
from data_pipeline.object_extraction.snip_processing.process_snips import process_single_snip

# full-frame mask must be written to disk as PNG; process_single_snip takes paths, not arrays
mask = decode_binary_mask_rle(json.loads(frame_masks_row.mask_rle))
skio.imsave(mask_png, (mask * 255).astype(np.uint8), check_contrast=False)

process_single_snip(
    snip_id=sid,
    image_path=<focus_stack png>,          # from snip_inventory.image_path
    mask_path=mask_png,
    yolk_mask_path=None,
    output_shape=(576, 256),
    pixel_size_um=1.887209,                # frame_inventory.image_micrometers_per_pixel
    target_pixel_size_um=6.5,              # or 7.8
    background_mean=0.0, background_std=0.0,   # force black bg so a fixed threshold is valid
    blend_radius_um=20.0,
    save_raw_crops=True, raw_crops_dir=<dir>,  # <-- for Step 1
    processed_dir=<dir>,
)
```

Two traps that cost real time:

1. **Never measure fill/saturation with Otsu.** Use a fixed threshold (`>30`) and force
   `background_std=0.0` in re-renders. Otsu on a noisy synthetic background yields ~0.45 fill for
   everything and produced a spurious "hypothesis refuted".
2. **`yolk_mask_path=None` changes rotation** (`rotation_source` falls back to `embryo_only`), so
   re-renders are ~0.71x the fill of production even at matched scale. *Relative* comparisons between
   two re-renders are valid; absolute comparisons against production are not. Supply the real yolk
   mask from `snip_auxiliary_masks` if absolute parity is needed.

Existing figure: `figures/_sleuth_old_vs_new_snips.png` — 4 rows (legacy production, current
production, re-render @6.5, re-render @7.8) x 6 wells. The visual match of row 1 to row 3 and row 2
to row 4 is what confirmed issue 1.

---

## 7. Corrections to be aware of

During this investigation I twice stated a conclusion that later proved wrong. Both are corrected
above, but flagged here so the next agent does not resurrect them from older notes:

1. I initially claimed the scale hypothesis was **refuted** by a controlled re-render. That was an
   Otsu artifact. The scale hypothesis is **confirmed**.
2. I initially framed saturation as caused by the scale change (via CLAHE seeing a different
   embryo-to-canvas ratio). The controlled test shows scale contributes only 1.25x of ~3.94x. The
   cause is **unidentified**.

Nothing in the pipeline has been changed. All work so far is read-only analysis plus new files under
`results/nlammers/20260729_morphseq_integration/` and `src/morphseq_integration/`.
