# MorphSeq / Latent Morphology Model — Consolidated Decision Record

**Updated:** 2026-08-18 · **Supersedes:** the recovered "MorphSeq Refactor — Decision Synopsis"
**Sources:** Phase 0 audit (file:line cited) · `NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md` (2026-08-13) · session decisions
**Tags:** [NICK] approved · [CLAUDE] proposed, unratified · [AUDIT] externally verified

> Written to the project so no future session starts from zero again. If a chat contains a decision
> not recorded here, it is invisible to future sessions — put it here.

---

## 1. Verified facts (Phase 0 audit, with citations)

- **Metric path is MIS-WIRED, not missing.** `src.core.data` NTXent machinery works. Two root causes:
  (a) `model_configs.py:8` imports `NTXentDataConfig` from the empty `src/data` shim;
  (b) metric hydra config sets `dataconfig.target: "BasicDataset"` → `metricVAE.forward` takes the
  vanilla branch → loss's `unbind(dim=1)` fails. **Both must be fixed; either alone leaves it dead.**
- `initialize_model` does **not** throw on missing `metric_array` — `hasattr`-guarded (`run_utils.py:339-340`).
- Loader reads a legacy table (`dataset_utils.py:29-38`) and re-derives `embryo_id` via `snip_id[:-6]`.
- `_pixel_scale` is a literal `128*288`, independent of `input_dim` (`loss_functions.py:185-187`).
- Metadata dies at the snip→ImageFolder boundary (`dataset_classes.py:84-154`).
- Interface is `load_encoder` + `arch_spec.json`. Pipeline `latent_embeddings` uses a third path and
  emits flat `z_mu_*`, not `z_mu_b_*`.
- `metric_array`: `(P,P)`, `+1`/`0`/`−1`, built offline in `build04`, **no serialization consumers**.
- Optical covariates present: `micrometers_per_pixel`, `objective_magnification`, `microscope_id`,
  `z_position`. `numerical_aperture` absent.
- **Signed δ must be BUILT** — only a per-pixel LoG-argmax map exists.
- FF vs slice derivable from `image_product_type`/`projection_method`; no typed boolean.
- Hygiene: `tv_weight` computed but never summed; `accumulate_grad_batches` never passed to Trainer
  and ignored under manual optimization; `logvar` clamp in `VAE` missing in `metricVAE`;
  `src/vae` dead for `src/core` but **live for `src/analyze`**.

## 2. Pipeline contract (new integration audit)

- Canonical snip `(H, W) = (576, 256)`, 8-bit non-interlaced grayscale PNG.
- Identity spine: `experiment_id → well_id → physical_embryo_id → embryo_id → snip_id`, with
  `physical_embryo_id = {well_id}_e{n}`, `embryo_id = {pid}_{channel_id}`, `snip_id = {embryo_id}_t{t}`.
  **No focus axis in the grammar.**
- Sources: `snip_inventory` (paths/identity), `stage_predictions`, `snip_qc`, `plate_metadata`.
- `is_valid_snip` (materialised) ≠ `use_snip` (QC verdict).
- Plate guarantees only `genotype`, `start_age_hpf`, `temperature`, `medium`. **`short_pert_name` not guaranteed.**
- Masks are colocated with their snips (naming convention to be discovered).

## 3. Decisions — approved

| # | Decision | Note |
|---|---|---|
| D1 | `analysis_ready` is **not** the training boundary | manifest built from the four sources above [NICK] |
| D2 | Replace `metric_array` with a **rule-based relation function**, class×class | curation revision pending [NICK] |
| D3 | Carry **optical covariates** through the manifest | non-model-facing in phase one [NICK] |
| D4 | Carry **`image_product_type`**; phase one filters **FF-only** via an explicit list | no ID-format change [NICK] |
| D5 | Fix **both** metric mis-wiring causes | [NICK] |
| D6 | Keep **288×128** default; image size becomes a **config parameter** | [NICK] |
| D7 | Derive `_pixel_scale` from `input_dim` | forced by D6 [NICK] |
| D8 | **On-the-fly** downsampling; no exported training set | cache only if measurement demands it [NICK] |
| D9 | Delete the **margin term** from metric loss (confirmed inert) | [NICK] |
| D10 | Delete `accumulate_grad_batches` (dead config) | [NICK] |
| D11 | Splits **group-disjoint at `physical_embryo_id`**, persisted by ID | [AUDIT] |
| D12 | Decoder δ-pathway: **thin, injected late** | [NICK] |
| D13 | Predictive z-slice reconstruction: **excluded** | [NICK] |
| D14 | Microscope/optical conditioning: **add explicitly** | [NICK] |
| D15 | Metric path made functional now; **metric policy revisited separately** | [NICK] |

**Augmentations:** wanted — embryo rescaling ~50%± FOV-constrained; brightness/contrast;
reflections/rotations FOV-constrained (believed implemented). Dirt-speck texture **tabled**.
Mild segmentation-error mimicry **included but discuss-first**.

**Corrected this session:** deriving `_pixel_scale` from `input_dim` makes λ transfer *better*
across geometries, not worse. The earlier synopsis recorded the opposite; it was wrong. At 288×128
the derived value equals the literal, so no existing run is invalidated.

**Struck:** the 7.8 µm/px figure. Do not propagate. Mixed-scale detection across the cohort is still
a required check.

## 4. Open — needs Nick

1. Metric relation policy + curation revision (blocks anything consuming relation semantics).
2. ImageFolder boundary / where view-generation sits and how the mask reaches it. *Discuss-first.*
3. Mask-jitter design (magnitudes, asymmetry, boundary-only erosion, desync from `area_um2`). *Discuss-first.*
4. "FF is done" convergence criterion for staged FF training (staged-vs-joint itself is pinned).
5. Whether to adopt SupCon `L_out` and accept the λ_metric/τ retune. [CLAUDE, unratified]
6. Run-artifact layout / experiment tracker convention.
7. Phase-one experiment ID list.
8. Checkpoint compatibility requirement.

## 5. Unverified — repo/data checks

- `contrastive_transform` per-view augmentation independence; whether FOV-constrained ±50% rescaling exists.
- Mask file naming convention and coverage.
- `image_product_type` / `projection_method` value sets; whether `snip_id` stays unique if both appear.
- Actual µm/px consistency across the cohort.
- Read+decode throughput off the real mount (decides D8's cache question).
- Whether `src.data_pipeline` path helpers import cleanly in the training env.

## 6. Deferred

Temporal↔atemporal exchange rate · latent distillation · free bits · eval-suite design ·
tree/hyperbolic prior · dirt-speck augmentation (only via real-background harvesting) ·
pipeline embedding interface · native 576×256 training.
