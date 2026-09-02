# Planned Pipeline Revisions

Living tracker for deferred pipeline changes surfaced while running real data
(2026-07, the 20260702 hotchem Keyence run). Priority is **throughput** — the pipeline is
richly structured but under-optimized for actually pushing data through efficiently.

Status legend: 🔴 not started · 🟡 partial/workaround in place · 🟢 done

---

## 1. Batching (`run_batch`) for model-heavy steps — 🟢 DONE

> **RESOLVED 2026-07-25 by resident model servers, not by batching.** The cost this section
> targets — thousands of model reloads — is fixed for the two steps where it mattered, via a
> different mechanism. A server loads the model once per run and per-well jobs become thin socket
> clients, so **the DAG is untouched**: still one job per well, same shards, same merge. Batching
> would instead have collapsed 576 jobs into 1.
>
> Measured A/B, all outputs proven identical to the per-well path:
>
> | Step | Per-well | Served | Speedup |
> |---|---|---|---|
> | `snip_auxiliary_masks` (4× UNet) | 280.39s | 97.47s | **2.88×** (2,100 mask PNGs byte-identical) |
> | `frame_detections` (GDINO) | 262.79s | 86.32s | ~3× (CPU floor; cell-for-cell identical) |
> | `frame_masks` (SAM2) | 1963.97s | 1919.76s | 1.02× on long time series; wired opt-in for one-frame SeaHub wells |
>
> Both served steps are behind config toggles (`frame_detections.use_model_server`,
> `unet_snip.use_model_server`), default off pending an end-to-end run. See
> `MODEL_SERVER_WIRING.md` and `MODEL_LOAD_BENCHMARKS.md`.
>
> **`frame_masks` was retagged `EXECUTION_RUN_BATCH` → `EXECUTION_PER_WELL`.** Its load is
> 3.2–3.6s against 621–1343s of real per-well work (ratio ~0.005), so neither batching nor
> serving is worth it for long time series. The earlier `<0.1s/well` benchmark had selected a
> 1-frame well—which is representative of SeaHub. The proven SAM2 adapter is therefore wired
> behind `frame_masks.use_model_server` and enabled only by SeaHub runtime overlays.
>
> **TECH DEBT deliberately not taken on:** the registry has no field recording "this step is
> served" — that fact lives only in the config toggles. `execution` means *job count*, and a
> served step genuinely is `PER_WELL` (N jobs, model elsewhere), so overloading the enum would
> conflate two independent axes. Add a separate field if serving ever becomes a per-step rather
> than per-run choice.
>
> **DONE 2026-07-29 — `latent_embeddings`.** One run-level rule now passes every discovered
> snip inventory to the existing batch entrypoint. It loads the CPU VAE once, atomically writes
> the unchanged per-well parquet + `.validated` files, then writes an experiment-level completion
> gate before the ordinary merge runs.

## Original problem statement (retained for context)

**Problem.** Every model-heavy step runs as one Snakemake job *per well*: `conda run` →
cold-load model → process one well → exit. Across ~576 wells that is thousands of full model
reloads (GroundingDINO, SAM2, 4× UNet), which dominates wall-clock. Observed: a serial
(`--cores 1`) full run did ~⅓ of the wells in ~16 h.

**Historical state before the 2026-07-29 fix — `EXECUTION_RUN_BATCH` was an intent label.** The registry
defines `EXECUTION_RUN_BATCH`, tags `frame_masks` and `latent_embeddings` with it, and even exposes
an accessor `execution_model_for_step()` (paths.py). **But the field is consumed by nothing:**
- `execution_model_for_step()` is *called nowhere* — no rule or code path branches on
  `per_well` vs `run_batch`.
- The code says so itself. `rules/latent_embeddings.smk` DOCTRINE NOTE: *"Realizing a true
  single-process-all-wells batch needs Snakemake's `--batch` mechanism, which no rule in this repo
  uses yet; like frame_masks (also RUN_BATCH in the registry), the rule is declared per-well. The
  `execution` field documents intent; the batch optimization is deferred."*
- Proof: `frame_masks` is *already* tagged `EXECUTION_RUN_BATCH`, yet in the 2026-07 run it
  reloaded SAM2 **per well** (one `conda run … frame_masks` per `{well_id}`).

Consequence: **flipping a step's `execution` to `EXECUTION_RUN_BATCH` is a no-op** — the rule stays
`<step>_per_well` (one job/well) and the model still reloads per well. The field is a TODO marker.

**What proper batching needs.**
1. **Batch-capable entrypoints** — accept a *well list* (or "all discovered"), load the model
   **once**, loop wells. Partly built already: `legacy_embeddings/entrypoint.py` "accepts a list of
   `(inventory, output)` pairs" (load once, iterate) — the rule just hands it a single pair today.
   `cmd_frame_detections` / `cmd_frame_masks` / `cmd_snip_auxiliary_masks` still take one well.
2. **A batch Snakefile rule** — one job producing *all* per-well shards for the run
   (checkpoint-aware; expands over discovered wells like the merge helpers).
3. **Incremental write+validate inside the loop** — stamp each well's shard `.validated` as it
   finishes, so a crashed batch leaves completed wells cached and a restart resumes (preserves the
   presence-based merge).
4. *(optional)* generic dispatcher that chooses per_well vs batch from the `execution` field
   instead of hand-wiring per step.

**Why low-downside (per-well philosophy preserved).** Batching is a compute optimization,
*invisible to the output contract*: it still writes the same per-well shards, merged identically.
Only per-well **job** granularity is traded away — and a single GPU serializes inference anyway,
so no real parallelism is lost. Net: load once, keep every shard, ~order-of-magnitude less
overhead. Steps that benefit: `frame_detections` (GDINO), `frame_masks` (SAM2),
`snip_auxiliary_masks` (4× UNet), `latent_embeddings` (VAE).

**Interim workaround (2026-07).** Run `--cores 8` with GPU steps capped to 1-at-a-time via
`--resources gpu=1 --set-resources <rule>:gpu=1`; parallelizes the many CPU steps across wells.
Helps, but does NOT remove the per-well model reloads — batching is the real fix.

---

## 2. Keyence `z_stack` materialization — 🟡 workaround in place

**Problem.** `motion_blur_qc` and `focus_qc` require the `BF__z_stack` product, but Keyence
`z_stack` materialization is unimplemented — `materialize_well_keyence.py` raises
`NotImplementedError` ("Keyence z_stack fanout deferred — requires per-tile plane handling").
So the full QC path (and anything needing z-planes) cannot complete for Keyence projection-only
data.

**What it needs.** Implement Keyence z_stack materialization: per-tile plane fanout + stitching so
each `(well, channel, z, time)` plane is materialized like the YX1 path, satisfying the
`BF__z_stack` product the z-dependent QC reads.

**Interim workaround (2026-07).** `snip_qc.exclusion_flags` config override drops `focus_flag` and
`motion_blur_flag`, so `snip_qc` aggregates only the z_stack-free QC steps (death_detection,
surface_area, mask_quality). Lets Keyence runs reach a QC verdict without z-planes.

---

## 3. Merge/validation race — 🟢 fixed for snip_inventory · 🔴 frame_detections gap

**Problem.** A merge rule collects only `.validated` per-well shards, but some merge rules declared
their Snakemake `input:` as the per-well `.csv` *without* the `.validated` sentinels. Under
`--cores > 1` the merge can start before validation finishes → `collect_well_shard_paths` sees zero
shards → `ValueError: no shards to concatenate`.

**State.** Audited all merge rules: only two lacked `per_well_validated`.
- `merge_snip_inventory` — 🟢 **fixed** (branch `pipeline-merge-validated-fix`): added
  `_snip_inventory_validated_for_run` + `per_well_validated=` input, mirroring `merge_frame_masks`.
- `merge_frame_detections` — 🟢 **fixed since this was written.** `validate_frame_detections_for_well`
  now exists and `merge_frame_detections` declares `per_well_validated=_frame_detections_validated_for_run`,
  matching the other merge rules. Verified 2026-07-25; this entry was stale.

---

## 4. GPU resource declaration on rules — 🟢 DONE (2026-07-25, commits 28fe3d7f + SGE scripts)

`resources: gpu=1` now declared on the 5 rules that genuinely hold GPU memory: the four known
model-heavy ones plus **`materialize_image_product_for_well`**, which was missing from every prior
list — it runs `LoG_focus_stacker` (real torch conv2d) and was found by searching for GPU work
rather than trusting the model-step names. `encode_latent_embeddings_for_run` deliberately does
NOT declare it (CPU-only stage; claiming the slot would serialize it behind real GPU work).

Two things learned:
- **The declaration alone is inert.** Snakemake only enforces `resources:` when a budget is
  supplied via `--resources gpu=1` at the CLI or a profile; this repo has no profile, and the
  `resources:` block in `config.yaml` is an unread config-dict key, not the CLI mechanism. The
  flag was added to 22 SGE submit scripts and the QUICK_RUN_READ_ME examples.
- **Flag placement matters.** `--resources gpu=1` immediately before a bare positional target
  crashes Snakemake 7.32.4 (`ValueError: not enough values to unpack`) because its greedy
  `nargs="*"` swallows the target token. Keep another `--flag` after it.

**Interaction with model servers:** a served step INVERTS this. The service rule declares `gpu=1`;
the per-well client must NOT — the client holds no GPU memory, and both declaring it means the
service takes the only unit, nothing is schedulable, and Snakemake blocks forever without an
error. Note also that services need `--cores >= 2` (the service holds a core for the whole run).

## 4b. Original §4 note (retained)

Model-heavy rules declare no `resources: gpu=1`, so nothing stops Snakemake from scheduling many
GPU jobs at once and OOM-ing a single card. Currently worked around on the CLI
(`--set-resources`). Bake `resources: gpu=1` into the GPU rules so any `--cores > 1` invocation is
OOM-safe by default (and pairs naturally with the batching work in §1).

---

## 5. Snip image-quality regression — 🟡 knobs restored, acceptance gate open

**Evidence:** `results/nlammers/20260729_morphseq_integration/root_analysis/IMAGE_QUALITY_ROOT_CAUSE_REPORT.md`
(2026-07-30). Full quantitative report — read it before acting on this entry.

**Problem.** Snips regressed against legacy via two independent preprocessing changes:

- **Scale** 6.5 → 7.8 µm/px (commit `93ffd7904b51`, 2026-03-02, no calibration rationale found).
  On the fixed 576×256 canvas this shrinks the embryo 16.7% per linear dimension; predicted area
  ratio `(6.5/7.8)^2 = 0.694` against an observed 0.700.
- **Blend radius** 75 → 20 µm (commit `1c1d7a9f6a00`, 2026-07-06, to stop neighbours bleeding into
  a snip). Blur sigma is `blend_radius_um / pixel_size_um`, so the taper narrowed 11.54 px → 2.56 px.
  **This is the direct cause of the bright/clipped look** — 3.63× the saturated-pixel fraction
  (95% CI 3.19–4.17), more saturated in 89/93 wells.

Excluded as causes: CLAHE (byte-identical across four skimage versions), source/flat-field, polarity,
JPEG-vs-PNG, background noise, and scale-as-saturation-mechanism.

**Downstream impact.** Same `20241107_ds_sweep01_optimum` checkpoint serves both generations.
Current-vs-legacy embeddings: median per-dimension correlation 0.687, pairwise-distance correlation
0.660, temperature R² 0.582 → 0.353. Blast radius is the canonical snips and everything reading their
pixels — auxiliary masks, snip QC, viability, latent embeddings, `analysis_ready` joins. Native-resolution
segmentation geometry, curvature, and surface area are **not** affected and need no recompute.

**State.**
- 🟢 `defaults.py` now reads `DEFAULT_TARGET_PIXEL_SIZE_UM = 6.5` / `DEFAULT_BLEND_RADIUS_UM = 75.0`.
- 🔴 **Embedding acceptance gate.** Restoring the two knobs is NOT proven to restore model behaviour.
  The report is explicit: current-source/current-mask 6.5/75 renders do not recreate legacy embeddings,
  because segmentation (SAM2 vs low-res UNet), rotation (mass-distribution vs yolk-guided), and source
  assets also drifted. In the 2×2 factorial 6.5/20 is *numerically* closest to legacy for several
  metrics — that is **accidental compensation, not evidence**. Compare regenerated vectors, pairwise
  geometry, developmental/temperature signal, and auxiliary QC against legacy references before adopting.
- 🔴 **The neighbour conflict that motivated 20 µm is unresolved.** A 75 µm outward blur can carry
  adjacent embryos. Proposed fix: replace out-of-mask source pixels with synthetic background *before*
  applying a legacy-width taper — then validate or retrain the VAE against the redesigned distribution.
- 🔴 **Version the preprocessing contract per checkpoint:** scale, blend radius, mask/pose method,
  CLAHE settings, background model, frame shape, encoding. Record in every snip inventory.
- 🔴 **Golden-image regression tests:** foreground area, p95, saturation fraction, taper profile,
  exact stored fixtures, checkpoint embedding drift.
- 🔴 **Bulk re-run** of snip processing + snip-dependent products, after the gate passes.

## 6. SeaHub — 🟡 built, unverified end-to-end

**Docs:** `docs/refactors/seahub/` (contract, design, workplan, reconciliation addendum).
**Runbook:** `results/nlammers/20260723_seahub/HANDOFF.md`. **Code:** `src/data_pipeline/acquisition/seahub/`.

**State.** Commit `409d9c83` — *"Patch to resolve issues with SeaHUB embryo image extraction.
**Still needs testing**"* — is the newest pipeline commit. The full-corpus run (reconcile →
GroundingDINO detection → build-bundle → materialize-shard → back half) has not been verified end to end.

**Known non-blockers, carried deliberately** (design §6 + handoff watch-outs):
- **µm/px is a placeholder (7.8).** All physical-scale and surface-area features are **uncalibrated**
  for SeaHub. Any physical-size analysis must exclude `source_scope == 'seahub'`.
- **`build-bundle` must be ONE invocation** over the full manifest — the canvas is corpus-wide per call,
  so splitting it breaks the uniform frame size.
- **Focus/motion QC false-flags ~all SeaHub rows** (single_z). Annotate-only; consumers ignore.
- **"Packing is a fiction"** — a synthetic plate mixes perturbations and stages. Biology is per-well;
  never treat a SeaHub plate as a shared physical condition.
- **RGB→grayscale** discards colour; acceptable for brightfield, note it if any run used colour stains.
- **QC tests cannot run in one env** — morphseq-env lacks skimage, mdcolon env shadows `data_pipeline`.

**Open.** Verify the patched extraction on a real shard; then run the corpus.
