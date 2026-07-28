# MorphSeq `src/core` Refactor — Phase 0 Audit

Read-only audit. No edits made to any audited code. Every factual claim is tagged
`[VERIFIED file:line]`, `[INFERRED]`, or `[DOC-ONLY]`.

**Scope:** `src/core` (model code being refactored) and `src/data_pipeline` (new Snakemake
pipeline). `src/vae` deprecated / out of scope (confirmed in Part 3.2).

---

## PART 1 — System architecture & machinery (PRESERVE)

### O1 — Loss composition

**1. Every loss term** (all in `src/core/losses/loss_functions.py`, assembled in the two `forward` methods):

| Term | Config key/weight | Computed | Summed into total |
|---|---|---|---|
| Pixel recon (L1/L2/bce) | `reconstruction_loss` | `recon_module` `[VERIFIED loss_functions.py:34-55]` | yes, scaled by `_pixel_scale()` |
| KLD | `kld_weight`, `schedule_kld` | `_compute_vae_terms` `[VERIFIED loss_functions.py:210]` | yes |
| LPIPS perceptual | `pips_weight`, `pips_flag` | `calc_pips_loss` `[VERIFIED loss_functions.py:20]` | yes |
| GAN (generator) | `gan_weight`, `use_gan`, `gan_net` | `[VERIFIED loss_functions.py:219-224]` | yes |
| Metric (NT-Xent) | `metric_weight`, `schedule_metric` | `_nt_xent_loss_euclidean` `[VERIFIED loss_functions.py:325]` | yes (NTXent only) |
| TV | `tv_weight` | `calc_tv_loss` `[VERIFIED loss_functions.py:14]` | **NEVER — see trap 4** |

**2. Total-loss assembly.**
- Plain VAE: `pixel_loss_w + kld_weight*kld + pips_weight*pips + gan_weight*gan` `[VERIFIED loss_functions.py:249-254]`.
- MetricVAE: same four terms **plus** `metric_weight*metric_loss` `[VERIFIED loss_functions.py:303-309]`.
  Weights are mutated live each step by the Lightning ramp schedulers (see O2).
- `_pixel_scale()` rescales the per-pixel mean by `(128*288)/100` (L2) or `(128*288)/10/100` (L1)
  `[VERIFIED loss_functions.py:183-187]`.

**3. Metric loss form** — `_nt_xent_loss_euclidean` / `_nt_xent_loss_multiclass`
(`loss_functions.py:325-379`):
- **Distance**: squared Euclidean `cdist(...).pow(2)` over `biological_indices` only `[VERIFIED :328-339]`.
- **Normalizer**: `sigma = N = latent_dim_bio/2`, then
  `dist_normed = (-(dist/sigma)**0.5 + margin)/temperature` `[VERIFIED :340-342]`. The effective
  distance is `sqrt(dist²/(D_bio/2))` = the claimed `sqrt(D_bio/2)`-style normalizer. **CONFIRMED.**
- **Kernel**: logit is **affine in distance** — `(-distance + margin)/temperature`. No exponent /
  `beta`. **CONFIRMED.**
- **Temperature `τ`**: `cfg.temperature`, divides the whole affine logit `[VERIFIED :342]`. **CONFIRMED.**
- **Sum-over-positives inside the log** (`L_in`): numerator `logsumexp` over positives, denominator
  `logsumexp` over positives+negatives `[VERIFIED :371-379]`. **CONFIRMED** (`L_in` form).
- **Margin `m`**: `cfg.margin`, added to **every** logit uniformly `[VERIFIED :342]`. Because it shifts
  numerator and denominator logits identically, it **cancels** in `numerator − denominator` →
  **inert/shared across all logits**. **CONFIRMED** (margin is currently a no-op given the affine form).

**4. TRAP — `tv_weight` unsummed. CONFIRMED.** `calc_tv_loss` is defined `[VERIFIED loss_functions.py:14]`
and `self.tv_weight` is stored `[VERIFIED loss_functions.py:140]`, but `calc_tv_loss` is **never called**
and `tv_weight` **never enters** either total-loss expression. Config default `tv_weight: float = 0`
`[VERIFIED loss_configs.py:47]`.

**5. TRAP — `logvar` clamp parity. CONFIRMED.** `VAE.forward` clamps:
`log_covariance.clamp(min=-10.0, max=5.0)` `[VERIFIED legacy_models.py:38]`. `metricVAE.forward` does
**not** clamp in either branch — `mu, logvar = encoder_output.embedding, encoder_output.log_covariance`
`[VERIFIED legacy_models.py:90]` and NT-Xent branch `logvar = enc.log_covariance`
`[VERIFIED legacy_models.py:106]`. The metric model can emit unclamped logvar.

### O2 — Lightning wrappers & optimization

**1. LightningModules**: `LitModel` (live VAE/metricVAE wrapper) and `LitAutoencoderKL` (separate
LDM-only wrapper, not on the metric path) `[VERIFIED pl_wrappers.py:19, :382]`.

**2. Manual optimization. CONFIRMED.** `self.automatic_optimization = False`
`[VERIFIED pl_wrappers.py:44]`. `training_step` fetches `self.optimizers()`, runs `manual_backward`,
optional `clip_grad_norm_`, then `opt_G.step(); opt_G.zero_grad()`; the discriminator is a separate
manual step with `opt_D` `[VERIFIED pl_wrappers.py:108-204]`.

**3. Optimizers/schedulers/clipping/logging.**
- `configure_optimizers`: Adam with 3 param groups (encoder-base `lr_encoder`, decoder `lr_decoder`,
  heads `lr_head`), plus a second Adam for `D` when `use_gan` `[VERIFIED pl_wrappers.py:292-316]`.
- **No LR scheduler** — weight ramps are applied by mutating `loss_fn.*_weight` via cosine ramps keyed
  on `current_epoch` `[VERIFIED pl_wrappers.py:246-284]`, not by a torch scheduler.
- Gradient clipping is manual (`grad_clip_norm`, default 0 = off) `[VERIFIED pl_wrappers.py:149-152,
  train_config.py:56]`.
- Logs: per-term losses, weights, grad norms, and (val) LPIPS/SSIM + latent anisotropy/condition number
  `[VERIFIED pl_wrappers.py:60-105, :206-234]`.

**4. TRAP — `accumulate_grad_batches` inert. CONFIRMED.** Set as
`LitTrainConfig.accumulate_grad_batches: int = 2` `[VERIFIED train_config.py:40]` (and `1` in the metric
hydra config `[VERIFIED metric_vae_timm.yaml:50]`), but it is **never passed to `pl.Trainer`** — the
Trainer kwargs omit it `[VERIFIED run_utils.py:216-224]` — and under manual optimization Lightning
ignores it entirely; `training_step` steps the optimizer every batch with no hand-rolled accumulation
`[VERIFIED pl_wrappers.py:157-158]`. It is dead config.

### O3 — Model config system & instantiation

**1. Config structure & YAML stacking.** `metricVAEConfig`/`VAEConfig` are pydantic dataclasses
composing `ddconfig` (arch) + `lossconfig` + `dataconfig` + `trainconfig`
`[VERIFIED model_configs.py:40-52, :87-93]`. Layering (highest wins): **Hydra defaults list** (`_self_`
then `model: metric_vae_timm`) `[VERIFIED base_cluster_metric.yaml:1-3]` → per-model YAML
(`config_target` + `ddconfig`/`lossconfig`/… blocks) `[VERIFIED metric_vae_timm.yaml]` → `from_cfg`
deep-merges the user dict on top of `asdict(cls())` defaults via `deep_merge` (recursive, b-wins)
`[VERIFIED model_configs.py:99-133, model_utils.py:5-18]`. `prune_empty` drops None/`""` leaves so they
don't clobber defaults `[VERIFIED model_utils.py:21-35]`.

**2. Parameterizable vs hard-coded.** Arch (`latent_dim`, `input_dim`, backbone `name`,
`frac_nuisance_latents`, conv geom), all loss weights/schedules, `temperature`/`margin`/`time_window`/
`self_target_prob`, train LRs/epochs are config `[VERIFIED arch_configs.py, loss_configs.py,
train_config.py]`. Hard-coded: image geometry `128×288` (trap 4).

**3. Instantiation from config.** `initialize_model` → `get_obj_from_str(config_target).from_cfg(...)`
→ `build_from_config` dispatches encoder/decoder by `ddconfig.name`/`is_timm_arch`
`[VERIFIED run_utils.py:325-338, factories.py:12-44]`.
- **AutoModel-style loader**: CONFIRMED present but **named `load_encoder` / `arch_spec.py`**, not
  `load_from_folder`. `save_arch_spec` writes `arch_spec.json` at train time; `load_encoder(run_path)`
  rebuilds the model from that spec + checkpoint with **no Hydra/loss/data imports**
  `[VERIFIED arch_spec.py:119-349]`. A heavier `load_trained_model` also exists (needs the Hydra config)
  `[VERIFIED run_utils.py:398-479]`.
  - `[DOC-ONLY]` the literal `AutoModel.load_from_folder` appears only as a **comment** in
    `perceptual_custom.py:5` and in the deprecated `src/vae` stack. The live `src/core` equivalent is
    `load_encoder`.
- **Encode/decode contract** exposed by the rebuilt model: `model(x: (B,C,H,W)) → ModelOutput` with
  `.mu (B,D)`, `.logvar (B,D)`, `.recon_x (B,C,H,W)`, `.z` `[VERIFIED arch_spec.py:283-298,
  legacy_models.py:42-49]`.

**4. TRAP — hard-coded `128×288` geometry. CONFIRMED, multiple live sites:**
- `_pixel_scale()`: literal `(128*288)` `[VERIFIED loss_functions.py:185-187]`.
- `LegacyArchitecture.input_dim = (1, 288, 128)` and `TimmArchitecture.input_dim = (1, 288, 128)`
  `[VERIFIED arch_configs.py:18, :47]`; `ArchitectureAELDM.resolution = [288,128]` `[VERIFIED :62]`.
- `ArchiveSpec.input_dim` default `[1, 288, 128]` `[VERIFIED arch_spec.py:77]`.
- `input_dim` IS config-derived at instantiation, but `_pixel_scale` is a literal constant independent
  of `input_dim` — that one is the true hard-code that mis-scales if geometry changes.

### O4 — Parameter-sweep machinery

**1/2. Definition & execution.** Sweeps are **per-run shell scripts**, not a Hydra multirun sweeper.
Each `sweepNN_files/*.sh` invokes `training_cluster.py` (which prepends repo root to `sys.path` and runs
`@hydra.main(config_name="base_cluster_metric")`) with **CLI overrides** for the varied keys
`[VERIFIED training_cluster.py:32-48; ls src/core/run/sweep01_files … sweep11]`. The base config has
**no `hydra/sweeper`/`launcher` block** `[VERIFIED base.yaml, base_cluster_metric.yaml — grep negative]`,
so "a sweep" = a directory of hand-written scripts (e.g. `sweep010_patch4_pips_lo/med/hi.sh` vary
`pips_weight`) `[INFERRED from filenames + CLI-override pattern]`.

**3. Output naming.** Hydra `run.dir` = `.../training_outputs/${job.name}_${now}`
`[VERIFIED base_cluster_metric.yaml:9]`; W&B `project`/`group`/`run_name` derive from
job name+timestamp+num `[VERIFIED :16-23]`. Checkpoints land in `<run>/checkpoints/` (`last.ckpt`,
`epoch{NN}.ckpt`, plus `special/` epochs from `save_epochs`) `[VERIFIED run_utils.py:187-201]`.
`arch_spec.json` + the `.hydra/config.yaml` associate a checkpoint back to its config
`[VERIFIED run_utils.py:162-166, arch_spec.py]`. **Preserve for reproducibility**: the `.hydra/` dump,
`arch_spec.json`, `split_indices.pkl`, and the run-name template in `parse_model_paths`/`train_vae`
`[VERIFIED run_utils.py:159, :286-305]`.

---

## PART 2 — Data → model contract

### O5 — Pipeline output surfaces vs loader expectations

**1. Frame inventory** (`src/data_pipeline`): produced per-well by materialization; contract
`frame_inventory_contract.py:44-59`. Required atoms: `experiment_id, well_index, channel_id,
time_index, z_index, image_product_type, projection_method, elapsed_time_s, acquisition_time_s,
image_path, image_micrometers_per_pixel, image_width_px, image_height_px` + write-policy cols; derived
`well_id, image_id` `[VERIFIED]`. **`src/core` never consumes the frame inventory** — the model side
starts at snips/embeddings, so there is no direct loader counterpart `[INFERRED — no frame_inventory
reader in src/core]`.

**2. Analysis-ready table** (labels/metadata): per-`snip_id`, assembled by importing every mint-site
contract `[VERIFIED analysis_ready/contract.py:44-118]`. Column families: identity spine
(`experiment_id→well_id→physical_embryo_id→embryo_id→snip_id` + frame provenance), feature payloads
(curvature, stage `predicted_stage_hpf`, mask geometry `area_um2`…, pose, `fraction_alive`,
`embedding_model_name`), QC verdict (`use_snip`, `qc_fail_reasons`), broadcast plate metadata
`[VERIFIED]`. **Loader side**: `make_seq_key` reads a completely different, **legacy** table
`metadata/embryo_metadata_df_train.csv` expecting `snip_id, experiment_id, experiment_date,
inferred_stage_hpf|predicted_stage_hpf, short_pert_name` `[VERIFIED dataset_utils.py:29-38]`. Different
schemas — see divergence list.

**3. `latent_embeddings` step.** Located at `src/data_pipeline/feature_extraction/legacy_embeddings/`
`[VERIFIED encode.py, contract.py]`.
- **Does NOT use `AutoModel.load_from_folder`.** Loads via `load_legacy_vae_encoder(model_dir)`, which
  reconstructs a standalone conv encoder from `encoder_reformatted.pt` + `model_config.json` (production
  `SeqVAEConfig`, `latent_dim=100`, `input_dim=[1,288,128]`) `[VERIFIED
  legacy_vae_inference_loader.py:51-198]`. Encode interface = `EncoderProtocol.encode_batch(x:[B,C,H,W])
  → {"mu","logvar"}` `[VERIFIED encode.py:32-44]`.
- **Emits**: `snip_id`, `z_mu_00…` (zero-padded), optional `z_sigma_00…` (= **logvar**, not std),
  `embedding_model_name` `[VERIFIED encode.py:91-100, contract.py:21-22]`.
- **Does NOT emit `z_mu_b_*`.** `[DOC-ONLY]` refuted for the pipeline path — `z_mu_b_*`/`z_mu_n_*` exist
  only in the legacy `assess_vae_results` disentangled path; the pipeline writes a **flat** `z_mu_*`
  block `[VERIFIED analysis_ready/contract.py:61-66 comment + select_latent_columns]`.

**4. Divergence list** → consolidated below (Core↔pipeline divergence map).

### O6 — Snip generation

**1. Producer & geometry.** `src/data_pipeline/object_extraction/snip_processing/`
(`process_snips.py`/`extraction.py`/`ops.py`). Geometry is **config-derived**: `target_pixel_size_um`,
`output_height_px`, `output_width_px`; scaling = `pixel_size_um / target_pixel_size_um`
`[VERIFIED extraction.py:50, ops.py:229-231]`. New-entrypoint defaults: `target_pixel_size_um=7.8`,
`output_height_px=576` `[VERIFIED run_snip_processing.py:97-98]`; the pipeline runner default is `6.5`
`[VERIFIED pipelines/snip_processing.py:45]`. Grayscale, 1 channel `[INFERRED from legacy
input_dim=(1,…) + snip contract]`.

**2. Per-snip metadata that survives.** The snip manifest carries `snip_id, embryo_id, well_id,
image_id, time_int, image_micrometers_per_pixel, target_pixel_size_um, output_height_px,
output_width_px, rotation_angle_*, background_mean/std, …` `[VERIFIED snip_processing/contract.py:9-58]`.
But the **model input is just the image tensor** — `NTXentDataset`/`BasicDataset` load pixels via
`ImageFolder` and attach only `snip_id`-derived vectors from the seq_key; **none of the snip manifest's
optical/geometry columns reach the model** `[VERIFIED dataset_classes.py:84-154]`. They are dropped at
the snip→ImageFolder boundary.

**3. Reconcile geometry.** Pipeline snips are `576`-tall at `7.8µm/px` (new) vs the model's hardcoded
`288×128`. **Divergence** — a pipeline snip is not shape-compatible with the model input without a
resize; `load_encoder`/eval transforms resize to `input_dim[1:]` `[VERIFIED run_utils.py:466-467]`, but
the training loader (`ImageFolder`+`contrastive_transform`) relies on the transform's `target_size`
`[VERIFIED dataset_configs.py:143]`.

### O7 — Data loader, sampler, `make_metadata`

**1. Batch assembly / positive-pair guarantee.** `LitModel.train_dataloader` uses a plain
`SubsetRandomSampler` over `train_indices` `[VERIFIED pl_wrappers.py:319-332]`. **There is no
batch-level positive-pair sampler** — instead, each `NTXentDataset.__getitem__` returns a
**self-contained pair** `(X, Y)` where `Y` is chosen at item time from same-embryo/same-class + age-
window candidates `[VERIFIED dataset_classes.py:107-146]`. Positives are guaranteed *per item*, not
*per batch* (the batch-`pair_matrix` in the loss just pairs view-0 with view-1 of the same item)
`[VERIFIED loss_functions.py:333-334]`.

**2. `make_metadata`.**
- Live `NTXentDataConfig.make_metadata` is **NOT a no-op** — it runs `split_train_test`, loads
  `metric_key.csv`, builds `seq_key_dict` (`pert_id_vec/e_id_vec/age_hpf_vec`) and the sorted
  `metric_array`, and the train/eval/test bool vectors `[VERIFIED dataset_configs.py:211-245]`. Consumed
  by `NTXentDataset` + plumbed to the loss.
- **The no-op shim `make_metadata` exists in a DIFFERENT module**: `src/data/dataset_configs.py`
  `BaseDataConfig.make_metadata → return None` `[VERIFIED src/data/dataset_configs.py:88-90]`, and its
  `NTXentDataConfig` is an empty `pass` placeholder `[VERIFIED :112-115]`. So `[DOC-ONLY] "make_metadata
  is a no-op shim"` is **true only of `src/data`, false of the live `src.core.data`** — see Part 3.1.

**3. Paired-forward vs single-image config. CONFIRMED mismatch.** `metricVAE.forward` in the NT-Xent
branch expects `x` of shape `(B, 2, C, H, W)` and does `x.unbind(dim=1)` `[VERIFIED legacy_models.py:85,
:98]`; the loss also does `x.unbind(dim=1)` `[VERIFIED loss_functions.py:291]`. But the **wired metric
hydra config sets `dataconfig.target: "BasicDataset"`** — a single-image `ImageFolder` dataset yielding
`(C,H,W)` `[VERIFIED metric_vae_timm.yaml:42; dataset_classes.py:22-31]`. A `BasicDataset` batch is
`(B,C,H,W)`, so `metricVAE.forward` would take the `len(x.shape)!=5 → vanilla` branch
`[VERIFIED legacy_models.py:85-92]` and the metric loss's `x.unbind(dim=1)` would fail. This is the
reported non-functional metric path.

**4. `initialize_model` AttributeError on missing `metric_array`. REFUTED.** It does **not** raise — it
guards with `if hasattr(model_config.lossconfig, "metric_array")` before assigning
`[VERIFIED run_utils.py:339-340]`. `[DOC-ONLY]` the claimed raise does not exist here. (A real failure
would instead surface later when the loss indexes an empty `metric_array`, only if
`self_target_prob < 1.0` `[VERIFIED loss_functions.py:354-358]`.)

### O8 — Metric pair determination & `metric_array` removal-safety map

**1. Pair determination path.** Genotype/class relations are decided **once, offline** in `build04` by
phenotype/control/crispant/background rules, written to `perturbation_metric_key.csv`
`[VERIFIED build04_perform_embryo_qc.py:1615-1657]`. At train time `make_metadata` loads
`metric_key.csv`, reorders it to `perturbation_id` order → `self.metric_array`
`[VERIFIED dataset_configs.py:216-237]`. Item-time sampling (`__getitem__`) reads
`metric_array[pert_id, :]==1` to find positive perturbation classes, intersects with age-window +
train/eval bool → picks `Y` `[VERIFIED dataset_classes.py:113-135]`. Loss-time,
`metric_array[pert_vec][:,pert_vec]` builds the batch target matrix `[VERIFIED loss_functions.py:356-366]`.

**2. Every WRITE.**
- **Origin**: `build04_perform_embryo_qc.py:1628-1657` → `perturbation_metric_key.csv` `[VERIFIED]`.
- **In-memory (live)**: `NTXentDataConfig.make_metadata` → `self.metric_array`
  `[VERIFIED dataset_configs.py:231-237]`; plumbed by `run_utils.py:340` into `lossconfig.metric_array`
  `[VERIFIED]`.
- **In-memory (legacy)**: `src/core/functions/dataset_utils.py` samplers read
  `self.model_config.metric_array` (never write) `[VERIFIED :254, :364]`; `src/vae/**` and
  `src/legacy/vae/**` configs build their own `metric_array` `[VERIFIED seq_vae_config.py:146-152,
  morph_iaf_vae_config.py:172-177, + legacy copies]`.

**3. Every READ.**
- Loss: `loss_functions.py:356-359` `[VERIFIED]`.
- Sampler (live): `dataset_classes.py:114-115` `[VERIFIED]`.
- Config plumbing: `run_utils.py:339-340` `[VERIFIED]`.
- Loss config field: `loss_configs.py:126` (`MetricLoss.metric_array`) `[VERIFIED]`; data config field:
  `dataset_configs.py:161` `[VERIFIED]`.
- Legacy samplers: `functions/dataset_utils.py:255, 365-366` `[VERIFIED]`.
- **Logging/eval/checkpoint/serialization**: **none live.** The only checkpoint-serialization hook is
  fully **commented out** in `callbacks.py:15-29` (a disabled `metric_array.npy` save) `[VERIFIED]`.
  `arch_spec.json` does **not** persist `metric_array` `[VERIFIED arch_spec.py:42-90]`. Removal has **no
  serialization consumers to migrate**.

**4. Shape/dtype/semantics.** `(P, P)` square over unique perturbation classes; built `int16`, cast
`int8` at loss time `[VERIFIED build04:1628, loss_functions.py:356]`. Values: **`+1` = positive
reference** (same class + diagonal), **`0` = negative** (default), **`−1` = neutral/excluded**
(unspecified/uncertain relations) `[VERIFIED build04:1630-1653; consumed as `==1`/`==-1`/else at
loss_functions.py:359-366 and dataset_classes.py:115]`.

**5. Silent-break surface if removed** (the removal blocker):
- `loss_functions.py:354-369` — the entire `self_target_prob < 1.0` cross-class branch depends on it;
  without it only self/same-embryo positives remain `[VERIFIED]`.
- `dataset_classes.py:113-127` — `other_option_array` (cross-embryo positives) collapses to empty
  `[VERIFIED]`.
- `run_utils.py:339-340` plumbing must be updated in lockstep with the loss/data config field removal
  `[VERIFIED]`.
- `functions/dataset_utils.py` legacy samplers (if still reachable) would break `[VERIFIED :254, :364]`
  — but confirm they're dead (only `set_inputs_to_device` is imported from `src.functions.dataset_utils`
  by `src/core` `[VERIFIED core_utils_segmentation.py:7]`; the `*DatasetCached` classes have no live
  importer `[INFERRED]`).
- Upstream `build04` producer + `perturbation_metric_key.csv` / `metric_key.csv` on disk become orphaned
  `[VERIFIED build04:1657]`.

### O9 — Conditioning-metadata availability

**1. Optical/acquisition covariates.** **Present** in scope metadata: `micrometers_per_pixel`,
`objective_magnification`, `microscope_id`, `image_width/height_px`, `z_position`
`[VERIFIED scope_metadata_contract.py:14-39]`; per-frame `image_micrometers_per_pixel` in the frame
inventory `[VERIFIED frame_inventory_contract.py:55]`; per-snip `image_micrometers_per_pixel` +
`target_pixel_size_um` `[VERIFIED snip contract:30-31]`. **`numerical_aperture` is ABSENT** everywhere
`[VERIFIED — grep negative across src/data_pipeline]`. Instrument identity is a string
(`microscope_id`/`scope_name`) **plus** magnification + pixel-size, i.e. richer than a bare scope-name
string.

**2. Focal offset δ.** **No per-frame best-focus plane and no signed δ exist.** `log_focus.py` computes
a **per-pixel** LoG-sharpness argmax-z (`idx`) used for focus-stacking, stored as the per-pixel
`focus_index_map` `.npz` provenance `[VERIFIED log_focus.py:76-77; frame_inventory_contract.py:84-86]`.
There is no scalar frame-level best-focus index and no `slice − best_focus` signed offset. A signed δ /
frame-level best-focus detector (variance-of-Laplacian / Brenner) would be a **new preprocessing
dependency** `[VERIFIED — grep negative for best_focus/brenner/frame-level δ]`.

**3. Acquisition mode (FF vs slice).** Encoded structurally via
`image_product_type ∈ {projection, z_stack}` + `projection_method` (e.g. `focus_stack`, `max`)
`[VERIFIED frame_inventory_contract.py:50-51, :221-222]`. There is **no dedicated
`acquisition_mode`/`full_focus` boolean field** — it must be derived from the product columns
`[INFERRED]`.

**4. Missing/unknown conditioning.** No `*_known` flag / sentinel convention for conditioning covariates
`[VERIFIED — grep negative]`. QC has `use_snip`/`qc_fail_reasons` verdicts `[VERIFIED snip_qc
doc/contract]` but nothing marking a covariate as unknown. Plate metadata has documented **empty
genotype/medium/temperature cells** (the "missing tabs / empty cells" audit) but that's null-in-source,
not a typed unknown flag `[VERIFIED METADATA_AUDIT.md]`.

---

## PART 3 — Hygiene traps

**1. Three `dataset_configs` import paths.**
- `src.core.data.dataset_configs` — the **live, canonical** one (full metric machinery). Imported by
  `run_utils.py:429`, `dataset_classes.py:73`, and internally `[VERIFIED]`.
- `data.dataset_configs` → resolves to **`src/data/dataset_configs.py`** (the thin analysis/embedding
  **shim**: `_SnipDataset`-based, `make_metadata` no-op, empty `NTXentDataConfig`) when `src/` is on
  `sys.path`. Imported by `model_configs.py:8`, `_extra_files/ldm_model_configs.py:8`, `src/analyze/*`,
  `src/run_morphseq_pipeline/services/*` `[VERIFIED grep + src/data/dataset_configs.py:69-116 +
  src/analyze/analysis_utils.py:6-9]`.
- `src.data.dataset_configs` — the **same shim file** via the `src.` package path (referenced in the
  pipeline docstring at `legacy_embeddings/snip_source.py:6`) `[VERIFIED]`.
- **Landmine**: `model_configs.py:8` imports `BaseDataConfig, NTXentDataConfig` from the **shim**
  (`data.dataset_configs`), so `metricVAEConfig.dataconfig` defaults to the **empty-`pass` shim
  `NTXentDataConfig`**, not the live one — a config/wiring inconsistency that co-exists with the O7.3
  `target:"BasicDataset"` override `[VERIFIED model_configs.py:8, :92]`.

**2. `src/vae` dead-stack.** `src/core` **never** imports `src/vae` `[VERIFIED — grep negative within
src/core]`. **However, `src/vae` is NOT fully dead**: live analysis code imports it —
`src/analyze/analysis_utils.py:13` and `src/analyze/get_recon_examples.py:11`
(`from vae.models.auto_model import AutoModel`) `[VERIFIED]`. So the doc's "nothing live imports it"
holds **for `src/core` only**; refuted repo-wide. **Legacy-vs-current boundary inside `src/core`**:
current = `src/core/data/` (dataset_classes/configs), `src/core/losses/`,
`src/core/models/legacy_models.py`+`factories.py`, `src/core/lightning/`, `src/core/run/run_utils.py`.
Legacy/stale = `src/core/functions/dataset_utils.py` (old `*DatasetCached` metric samplers, superseded
by `data/dataset_classes.py`), `src/core/models/_extra_files/`, `registry.py` (stub with broken
`vae_loss_basic` refs `[VERIFIED registry.py:1-15]`).

**3. `tv_weight` unsummed** — CONFIRMED (O1.4).
**4. `accumulate_grad_batches` inert** — CONFIRMED (O2.4).
**5. Hard-coded `128×288`** — CONFIRMED, 3+ sites (O3.4).
**6. `logvar` clamp parity** — CONFIRMED absent in `metricVAE` (O1.5).

---

## Consolidated summaries

### Machinery preserve-map (O1–O4)
A manual-optimization Lightning system (`LitModel`) wrapping `VAE`/`metricVAE`, with a **composed loss**
(pixel + KLD + LPIPS + hinge-GAN + NT-Xent-metric), each weight **cosine-ramped by epoch** at runtime.
Metric loss = affine-in-Euclidean-distance NT-Xent on the biological latent sub-block, `L_in` form,
temperature-scaled, margin currently inert. Config is a 4-part pydantic composition assembled by Hydra
defaults → per-model YAML → `deep_merge`. Sweeps are hand-written shell scripts over Hydra CLI
overrides; runs are reproduced from `.hydra/config.yaml` + `arch_spec.json` + `split_indices.pkl` +
`checkpoints/`. **Do not break**: the ramp schedulers, the 3-group optimizer LRs, the
arch_spec/`load_encoder` inference path, and the checkpoint/`arch_spec.json` naming.

### Removal-safety summary (`metric_array`)
Update together: (1) loss read `loss_functions.py:354-369`; (2) sampler read `dataset_classes.py:113-127`;
(3) config plumbing `run_utils.py:339-340`; (4) fields `loss_configs.py:126` + `dataset_configs.py:161`;
(5) builder `dataset_configs.py:216-237`; (6) upstream producer `build04:1615-1657` + on-disk
`perturbation_metric_key.csv`/`metric_key.csv`; (7) legacy samplers `functions/dataset_utils.py:254,364`
(confirm dead). **No** logging/eval/checkpoint/serialization consumer exists (the only such hook is
commented out in `callbacks.py:15-29`). Value semantics to preserve if reimplemented: `+1` positive /
`0` negative / `−1` exclude.

### Core↔pipeline divergence map (Phase 1 compatibility surface)
| # | Pipeline emits | `src/core` loader expects | Divergence |
|---|---|---|---|
| D1 | analysis-ready keyed on `snip_id` + 5-level spine | `embryo_metadata_df_train.csv` with `snip_id, experiment_id, experiment_date, inferred/predicted_stage_hpf, short_pert_name` | different table, filename & schema; loader table is legacy `[VERIFIED dataset_utils.py:29-38 vs analysis_ready/contract.py]` |
| D2 | time column `time_int` (snip) / `time_index` (frame) | loader uses `snip_id[:-6]` string slicing for `embryo_id`, `stage_hpf` from age key | loader re-parses ids by slicing, not the pipeline's minted ids `[VERIFIED dataset_utils.py:67, dataset_configs.py:222-224]` |
| D3 | flat `z_mu_*` / `z_sigma_*`(=logvar), `embedding_model_name` | (embeddings not re-consumed by trainer) + doc-claimed `z_mu_b_*` | no `z_mu_b_*` from pipeline; biological split is a legacy-only concept `[VERIFIED]` |
| D4 | snips `576`-tall @ `7.8µm/px` (config-derived) | model input `288×128`, `_pixel_scale` literal `128×288` | geometry mismatch; needs resize + `_pixel_scale` de-hardcode `[VERIFIED]` |
| D5 | single-image snips (`BasicDataset` wired) | `metricVAE.forward` wants `(B,2,C,H,W)` pairs | metric path non-functional as wired `[VERIFIED]` |
| D6 | perturbation identity via genotype contracts / plate metadata | `perturbation_id`/`short_pert_name` + offline `metric_array` | pair relations live outside pipeline (`build04`); refactor must rebuild this layer `[VERIFIED]` |

### Contract-facing field inventory
**Emittable today** (typed, in pipeline contracts): `snip_id, embryo_id, physical_embryo_id, well_id,
experiment_id, image_id, time_int, channel_id`; `image_micrometers_per_pixel, target_pixel_size_um,
output_height_px, output_width_px, objective_magnification, microscope_id, z_position`;
`predicted_stage_hpf`, mask geometry (`area_um2, length_um, width_um, …`), curvature, `fraction_alive`;
QC (`use_snip, qc_fail_reasons, focus_flag, motion_blur_flag, …`); latent `z_mu_*, z_sigma_*,
embedding_model_name`; plate metadata (`genotype, start_age_hpf, temperature, medium`, + opportunistic).
**Must be built**: `numerical_aperture`; frame-level **best-focus plane + signed δ**; a typed
**acquisition_mode** (FF/slice) boolean (currently only derivable); a **`*_known`/unknown** convention
for conditioning covariates; the **positive/negative pair relation** structure that replaces
`metric_array` (from genotype contracts + age window).

### Open discrepancies (`[DOC-ONLY]`, for correcting the reference doc)
1. `AutoModel.load_from_folder` — not the live loader; live path is `load_encoder`/`arch_spec.json`
   (comment-only in `src/core`; real only in deprecated `src/vae`).
2. `latent_embeddings` emits `z_mu_b_*` — **false** for the pipeline; only `z_mu_*` (flat). `z_mu_b_*` is
   legacy `assess_vae_results` only.
3. `make_metadata` is a no-op shim — **true only of `src/data/dataset_configs.py`**; the live
   `src.core.data.NTXentDataConfig.make_metadata` does substantial work.
4. `initialize_model` raises `AttributeError` on missing `metric_array` — **false**; it's
   `hasattr`-guarded.
5. Metric-loss `margin` applied asymmetrically — **false**; it's shared across all logits and cancels
   (inert).
6. "`src/vae` — nothing live imports it" — **true for `src/core` only**; `src/analyze` imports it live.

### Critical-path improvement notes (gated — suggestions only, no edits)
Only items that **block** the refactor:
1. **D5 paired-input wiring** blocks a functional metric path: the metric hydra config wires
   `dataconfig.target:"BasicDataset"` (single image) while `metricVAE.forward`/`NTXentLoss` require
   `(B,2,…)` pairs. Must be reconciled (wire `NTXentDataset`, or re-architect pairing) for metric
   training to run at all `[VERIFIED metric_vae_timm.yaml:42 vs legacy_models.py:98,
   loss_functions.py:291]`.
2. **Part 3.1 shim import** blocks correct config instantiation: `model_configs.py:8` pulls
   `NTXentDataConfig` from the empty `src/data` shim, so `metricVAEConfig`'s default dataconfig has no
   `metric_array`/`make_metadata` machinery — the metric path can only work by the hydra override fully
   replacing it. Import source should point at `src.core.data.dataset_configs`
   `[VERIFIED model_configs.py:8]`.
3. **D4 `_pixel_scale` hard-code** blocks clean interface with new snip geometry: it's a literal
   `128*288` independent of `input_dim`, so changing geometry silently mis-scales the reconstruction term
   relative to KLD `[VERIFIED loss_functions.py:185-187]`.

The "must be built" Part-2 items (best-focus/δ, optical covariate completion, pair-relation replacement)
are refactor scope, not pre-existing blockers, so they live in the field inventory rather than here.
