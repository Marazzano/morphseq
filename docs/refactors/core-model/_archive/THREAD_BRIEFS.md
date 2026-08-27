> **ARCHIVED — superseded dispatch. Do not dispatch from this file.**
>
> Briefs for the 2026-08-26 overnight threads. Those threads have run; their doc-reorg task (T1)
> was completed on 2026-08-27 and the layout it describes no longer matches the tree.
>
> **It also propagates the confabulation**: the T5 brief states "Phase 1 is implemented and
> unit-tested". It was not. Several paths are wrong (`docs/refactor/evidence/` — note the missing
> `s`), and `plans/THREAD_BRIEFS.md` no longer exists.
>
> **Verification:** `../reports/GROUND_TRUTH_2026-08-27.md`.

# Dispatch briefs — 2026-08-25

Six threads fire now; one is gated. Branch per thread.

**Verified layout (2026-08-25).** The refactor docs root is **`docs/refactors/core-model/`**, with
`AGENTS.md`, `DECISIONS.md`, `OUTSTANDING_PIPELINE_ISSUES.md`, `SNIP_IMAGE_REGRESSION_STATUS.md`,
`TRAINING_READINESS_REMAINDER.md`, `UPSTREAM_PIPELINE_STATE.md` at that level, plus `audits/`,
`contracts/MANIFEST_SCHEMA.md`, `plans/`, and `reports/` (including `reports/recon_tables/*.csv`).
`docs/refactors/seahub/` is a sibling; the pipeline's own docs are at `docs/data_pipeline/`.

**`PROJECT_STATUS.md`, `PIPELINE_TASKS.md` and `PREFLIGHT_DATA_CHECKS.md` do not exist and are
retired.** Their content now lives in `PLAN.md` and `DECISIONS.md`. If any brief below still names
one, ignore that reference — do not reconstruct the file.

**T1 is moving four documents into `evidence/` concurrently**, so if a path fails, look for the file
by name before concluding it is missing.

**How to dispatch:** point each agent at this file and name its thread — e.g. *"Read
`docs/refactors/core-model/plans/THREAD_BRIEFS.md`. You are T3. Read the header, then execute the T3
section only. Stay inside T3's file-ownership fence."* Everything an agent needs is in this file;
nothing else needs pasting.

**Paths are pre-filled** for the workstation mount
(`/media/nick/gs_cluster/projects/data/morphseq/pipeline/output`, env `morphseq-env`). If an agent
runs on the cluster, correct the root in its instruction. Single-experiment work uses
`20250612_24hpf_ctrl_atf6` — 97 snips, all four artifacts, 93 through the combined gate.

| id | thread | data? | branch | owns |
|---|---|---|---|---|
| T1 | Archive four docs + build status script | no | `thread/status-system` | `docs/refactors/core-model/evidence/` (new), `scripts/status.py`, `conftest.py` marker — **not** `reports/`, `contracts/`, `audits/` |
| T2a | Pipeline preprocessing provenance | no | `thread/snip-provenance` | `src/data_pipeline/object_extraction/snip_processing/**` |
| T3 | Metric unblock + hardenings + acceptance | yes | `thread/phase1-accept` | `src/core/**`, `tests/core/**` |
| S1 | Study: QC / `sa_outlier_flag` | yes | `study/qc-sa-outlier` | `docs/refactors/core-model/reports/**` only |
| S2 | Study: stage estimation lineage | yes | `study/stage-lineage` | `docs/refactors/core-model/reports/**` only |
| A1 | Audit: regeneration scope + timing | yes | `audit/regen-scope` | `docs/refactors/core-model/reports/**` only |
| **T2b** | **Rerender + legacy comparison** | yes | gated on T2a **and** the F3 check | — |

**Run first, by hand, in five minutes:** do legacy snips for `20250612_30hpf_ctrl_atf6` still exist on
disk? T2b's entire design assumes the 93-well paired baseline is available. If it's gone, T2b changes
shape and should not be planned around.

---

## T1 — document restructure and computed status

> Read `docs/refactor/PROJECT_STATUS.md`. This repo's project state currently lives in hand-written
> prose across ~11 documents, several claiming authority over the same facts. That has already caused
> at least one wrong claim to propagate through three documents unchecked. Your job is to replace
> hand-maintained status with computed status.
>
> **1. Archive only — the written-document restructure is already done.** `PLAN.md`, `DECISIONS.md`,
> and `AGENTS.md` have been reconciled and placed. Your only structural task is to create
> `docs/refactors/core-model/evidence/` and move these four narrative state documents into it, updating
> cross-references in whatever still points at them: `OUTSTANDING_PIPELINE_ISSUES.md`,
> `SNIP_IMAGE_REGRESSION_STATUS.md`, `UPSTREAM_PIPELINE_STATE.md`, `TRAINING_READINESS_REMAINDER.md`.
> Also move `plans/AGENT_BRIEFS_PHASE1.md` there — that thread has landed. Leave `audits/`,
> `contracts/`, and `reports/` exactly as they are. Delete nothing.
>
> **Before anything else:** diff the incoming `DECISIONS.md` against the previous version in git and
> **report any decision the old file contained that the new one does not.** The new ledger was
> reconciled outside the repo and may have dropped something.
>
> **2. `scripts/status.py`.** Writes `docs/refactors/core-model/STATUS.md`. Create `scripts/` if absent. Sections:
> - **Git:** current branch, HEAD, dirty tree, unpushed commits (`git log @{u}..HEAD`). For a
>   configured list of commits of interest, whether each is an ancestor of HEAD and of `origin/main`.
> - **Tests:** run pytest, record pass/fail counts and duration.
> - **Decisions:** cross-reference `DECISIONS.md` against pytest markers (below). **Use the decision
>   IDs already in the repo's `DECISIONS.md`** — do not renumber. If an external ledger disagrees,
>   report the divergence rather than merging blind. Report
>   `N decisions · M with tests (K passing) · unverified: [ids]`. **The unverified list is the most
>   important line in the file** — make it prominent.
> - **Environment:** python version, whether `pyarrow` imports, key package versions.
> - **Data (optional, `--with-data`):** cohort artifact availability and row counts. Cache keyed on
>   source-artifact hashes. Must be skippable — git and tests always run and must be instant.
>
> **3. Decision markers.** Add a `decision` pytest marker registered in `conftest.py`:
> ```python
> @pytest.mark.decision("D9")
> def test_margin_term_absent(): ...
> ```
> Give `DECISIONS.md` a `verified by` column. Backfill markers onto **existing** tests wherever one
> already proves a decision. Do not write new tests to chase coverage — report unverified decisions
> honestly; that list is the deliverable.
>
> **4. Evidence convention.** Add a short section to `AGENTS.md`: every factual claim in a written doc
> carries its source inline — a command, a `file:line`, or a measurement date — or is explicitly
> marked as inference. Apply it to the claims you migrate into `PLAN.md`.
>
> **Constraints.** Touch no `src/` file. Do not modify `MANIFEST_SCHEMA.md`. The script must run in
> under 10 seconds without `--with-data`.
>
> **Summary must state:** the unverified-decision list; any claim you migrated that you could not
> source; anything in `PROJECT_STATUS.md` that the script contradicts.

---

## T2a — preprocessing provenance in snip output

> Read `docs/refactor/evidence/SNIP_IMAGE_REGRESSION_STATUS.md`.
>
> **Problem.** Snips are written with no record of how they were rendered. All 133 readable merged
> inventories lack `source_micrometers_per_pixel` and `snip_micrometers_per_pixel`, so 699,505 stored
> snips cannot prove what contract they obey. That is the root of the current corpus crisis.
>
> **This is time-critical.** A full regeneration is imminent. If provenance is not in place first, we
> regenerate ~700k snips and *still* cannot prove their contract, and the work is wasted.
>
> **Task.** Make every snip-processing run record its rendering contract:
> - Two per-row inventory columns: `source_micrometers_per_pixel`, `snip_micrometers_per_pixel`.
> - A per-run sidecar (JSON, beside the merged inventory) capturing: git commit and branch, target
>   µm/px, blend radius µm, mask source and version, orientation policy and evidence source, CLAHE
>   settings, background model and noise scaling, frame shape, resampling kernel, file encoding.
> - A contract-version marker so consumers can detect schema era.
>
> Extend the inventory contract and its validator. Existing artifacts must still read — new columns
> are **optional on read, required on write**. Do not break `MANIFEST_SCHEMA.md`'s reserved names:
> use exactly `source_micrometers_per_pixel` and `snip_micrometers_per_pixel`.
>
> **Constraints.** Do not change rendering behaviour — this records what happens, it does not alter
> it. `DEFAULT_TARGET_PIXEL_SIZE_UM = 6.5` and `DEFAULT_BLEND_RADIUS_UM = 75.0` stay as they are.
> Touch nothing under `src/core`.
>
> **Done when** a single-plate run writes both columns and a complete sidecar, and re-reading an old
> pre-provenance inventory still succeeds with the fields reported absent.
>
> **Summary must state:** every parameter you captured and where you read it from; anything in the
> required list you could not capture and why.

---

## T3 — unblock metric mechanically, harden, then prove the path

> Read `docs/refactor/AGENTS.md`, `docs/refactor/MANIFEST_SCHEMA.md`, and
> `docs/refactor/evidence/TRAINING_READINESS_REMAINDER.md`. Phase 1 is implemented and unit-tested
> (58/58). Nothing here is a redesign.
>
> **1. Throwaway metric mapping — a test fixture, not policy.** The real metric mapping is a pending
> science decision. Create `tests/core/fixtures/metric_map_PLACEHOLDER.csv` and a matching small
> relation matrix, with **two arbitrary groups** and a matrix making some pairs legal and some illegal
> — a single group would make everything compatible and would never exercise the missing-positive
> path. Name the files so nobody mistakes them for policy, and add a header comment saying so.
>
> **2. Named config presets.** Add `basic` and `metric` run configs. The shared Hydra data group
> carries required-no-default values and defaults both `apply_stage_filter` and
> `resolve_metric_groups` to true, so a basic run currently inherits staging and mapping requirements
> it has no use for. The basic preset must set both false explicitly. Each preset names: pipeline
> root, ordered experiment list, `[BF]`, `[projection]`, QC policy, stage policy, split policy, model
> input size, artifact dir, W&B mode.
>
> **3. Decode robustness — required, or the acceptance run dies on the first bad file.** Today a
> corrupt PNG raises inside a dataloader worker and kills the run. Make `__getitem__` non-fatal: catch
> the decode failure, log the `snip_id`, resample from the same split, increment a counter, and fail
> the run only when the counter crosses a configured threshold. Silent skipping is not acceptable —
> the count and the IDs go into the cohort report.
>
> **4. Cohort-wide legal-positive preflight.** `NTXentDataset` currently checks lazily in
> `__getitem__`, so a run can die mid-epoch. After splits and the matrix are known, determine for
> every row in every split whether a legal **in-split** positive exists. Report by split × metric
> group with counts and example IDs. Use the **sampler's** criterion (`delta <= time_window`), not the
> loss's (`time_window + 1.5`) — the wider one under-reports. Do not evaluate per-anchor against all
> candidates; reuse the `(metric_group, age_bin)` index and evaluate per cell, but check **exact
> deltas** for candidates in neighbouring bins or the preflight will pass where `__getitem__` fails.
>
> **5. Split-ratio tolerance.** The manifest reports configured vs achieved fractions and asserts
> nothing. Add a size-aware tolerance (absolute minimum count per split, plus a relative tolerance
> above some size), and **fail unconditionally on any empty required split** — no tolerance knob on
> that one. Achieved values go into provenance. Note: an explicit `test_experiments` list is coming,
> so compute targets over the unpinned pool, not the whole cohort.
>
> **6. End-to-end acceptance on one small complete experiment.** In order, inspecting output at each
> step rather than checking it didn't crash: build the manifest and read the contract/join/cohort/split
> reports → open the selected images and masks → iterate train/eval/test loaders at real worker count
> → one finite basic train/val step through the real Lightning wrapper → same in metric mode with the
> placeholder mapping → a tiny `Trainer.fit` producing a checkpoint, `arch_spec.json`, the provenance
> bundle and the W&B artifact → reload via `load_trained_model` and verify source hashes and the exact
> snip-ID set reproduce → repeat under intended DDP/multiworker/GPU settings and check cluster write
> permissions.
>
> **Pass means** every step completes and the reload reproduces the exact snip-ID set. It does **not**
> mean loss went down. This corpus is known-regressed; nothing here is a science result.
>
> **Summary must state:** which of the eight acceptance steps passed, which failed and how; the
> legal-positive preflight report; achieved split fractions; and any decode failures encountered.

---

## S1 — study: is `sa_outlier_flag` a phenotype filter?

> Read-only. Write to `reports/STUDY_qc_sa_outlier.md` plus tables under `reports/study_tables/`.
> Modify nothing in `src/`.
>
>
> **Read these two first — they exist and are directly on point:**
> `docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md` and
> `docs/data_pipeline/specs/target/specs/quality_control/snip_qc_verdict_and_flag_resolver.md`.
> The first is a standing tech-debt note about surface-area QC being confounded — read it before
> forming any hypothesis, and report whether the confound it documents is pose, phenotype, or both.
>
> `sa_outlier_flag` appears in **69.6% of all QC failures** (241,274 of 346,698 failing snips, 2.36
> flags per failure) and is the only *morphometric* criterion in an otherwise acquisition/segmentation
> flag set. Strict QC removes 65.2% of the corpus. If the flag is a population-level outlier test, it
> preferentially deletes severe phenotypes — which for a morphology model is exactly backwards.
>
> **The decisive test, do this first:** join QC outcomes to plate metadata and report **strict removal
> rate by genotype**, and specifically controls (`wt*`, `inj-ctrl*`, `ctrl-inj*`, `DMSO`, `dmso`)
> versus severe perturbations (crispants, homozygotes). Report per-genotype n, removal rate, and the
> `sa_outlier_flag` share of those removals. If controls sit materially below perturbed genotypes,
> the flag is eating phenotypes and the number is unarguable either way.
>
> **Also report:**
> 1. What `sa_outlier_flag` is computed against — read the implementation, do not infer. Per-track
>    over time (safe: catches segmentation failure) or against a population / stage-matched
>    distribution (harmful)? Quote the code.
> 2. Count of failing snips carrying `sa_outlier_flag` **and no other flag** — the population
>    recoverable by demoting it.
> 3. The same removal-rate-by-genotype breakdown for `focus_flag` and the two death flags, as
>    controls: those *should* be roughly phenotype-independent. If they are not, the confound is
>    broader than sa_outlier.
> 4. Whether removal rate correlates with frames-per-embryo (long timelapses average 34.9% pass,
>    short plate experiments 77.0%) — separate the length effect from the phenotype effect.
> 5. A recommended default QC policy with its expected cohort size.
>
> **Summary must state:** the control-vs-perturbed removal gap with n and effect size, and whether it
> survives controlling for experiment length.

---

## S2 — study: stage estimation lineage

> Read-only. Write to `reports/STUDY_stage_lineage.md`. Modify nothing in `src/`. Check
> `docs/core/metric_loss_overview.md`, `docs/architecture/training_guide.md`, and
> `docs/data_pipeline/METADATA_AUDIT.md` for prior treatment of stage semantics.
>
> Metric pairing is built entirely on stage deltas within `time_window`. Legacy core consumed
> `age_key.csv`'s `inferred_stage_hpf_reg`; the new pipeline emits `predicted_stage_hpf` with a
> `model_version` and statuses including `missing_start_age_hpf`, which suggests it is anchored to
> known start age with a temperature-adjusted rate rather than being a pure image regression. **If the
> two have different scales or biases, `time_window` silently means something different than when it
> was tuned.**
>
> Report:
> 1. What produces `predicted_stage_hpf` — model, inputs, training data, and what `model_version`
>    values appear across the corpus. Quote the code.
> 2. What produced legacy `inferred_stage_hpf_reg`. Does that code still exist in the repo or its
>    history? If it is gone, say so and name the last commit that had it.
> 3. Where both exist for the same embryos, do they agree? Report correlation, bias, and a
>    Bland–Altman-style spread. If no overlap exists, say so plainly rather than substituting a proxy.
> 4. Whether the current `time_window` default was tuned against legacy or new stage units.
> 5. Stage coverage by experiment and the distribution of `stage_prediction_status`, including the
>    20,219 snips with finite stage and no status column (schema S02).
>
> **Summary must state:** whether new and legacy stage are interchangeable, and if not, what it
> implies for `time_window`.

---

## A1 — audit: regeneration scope and timing

> Read-only except `reports/AUDIT_regeneration_scope.md`. Also read
> `docs/data_pipeline/PLANNED_REVISIONS.md` (the canonical pipeline tracker) and
> `docs/data_pipeline/RAW_DATASET_INVENTORY.md`. Read
> `docs/refactor/evidence/SNIP_IMAGE_REGRESSION_STATUS.md` first — it states that upstream image
> materialization and native-resolution segmentation do **not** need to rerun, and the required chain
> is snip processing → snip auxiliary masks and QC → fraction-alive/viability → legacy embeddings →
> consumers. Verify that claim rather than assuming it.
>
> Report:
> 1. **Scope.** Confirm which stages must rerun and which need only verification. Biggest lever — get
>    this right before timing anything.
> 2. **Per-stage wall clock** on one representative plate, measured not estimated.
> 3. **Resident model servers.** `docs/data_pipeline/MODEL_SERVER_WIRING.md` and
>    `docs/data_pipeline/MODEL_LOAD_BENCHMARKS.md` already exist — read both before timing anything;
>    the benchmark work may already answer part of this. Measured at 2.88× for `snip_auxiliary_masks` and ~3× for
>    `frame_detections`, both toggles default **off** pending end-to-end validation. Time one plate
>    with them off and once on. This is plausibly the difference between a day and a week.
> 4. **GPU scheduling.** Rules declare `resources: gpu=1`, inert without `--resources gpu=1` on the
>    CLI, currently baked into SGE scripts rather than a profile. Confirm the bulk path actually gets
>    a GPU.
> 5. **Disk** for a second immutable output root alongside the existing one.
> 6. **Extrapolated wall clock** for the full corpus (699,505 snips, 133 experiments) and for a
>    training-cohort-only subset, both with and without the servers.
> 7. **SeaHub separately** — it needs full reprocessing including the front end, so time it on its own
>    terms.
> 8. **Downstream rebuild vs verify** for each dependent product, with cost.
>
> Also answer, since you are already in the data: does any stored SeaHub row carry a
> `source_micrometers_per_pixel` other than exactly 7.8? If every row is 7.8, no calibration is being
> applied regardless of what code exists. Quote the assignment site.
>
> **Summary must state:** total estimated wall clock with and without the servers, and the single
> biggest cost driver.

---

## T2b — rerender one plate and compare to legacy *(gated: dispatch after T2a lands)*

> **Inputs**
> - legacy snips: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/bf_embryo_snips/`
> - pipeline output root: `/media/nick/gs_cluster/projects/data/morphseq/pipeline/output`
> - experiment: `20250612_30hpf_ctrl_atf6` — chosen because a 93-well paired legacy baseline already
>   exists for it in the root-cause analysis
> - checkpoint for the model gate: `20241107_ds_sweep01_optimum`
>
> Read `docs/refactor/evidence/SNIP_IMAGE_REGRESSION_STATUS.md` in full before starting.
>
> **Confirm three things first and stop if any fails:** that legacy snips for this experiment exist
> under that root and how they are laid out (expect a legacy training-bundle structure, and expect
> **JPG** — the new pipeline writes PNG; the root-cause analysis already ruled out encoding as a
> cause, so cross-format comparison is valid, but your tooling must handle it); that both roots are
> reachable from the host you are on (recon ran against a `/media/nick/gs_cluster/...` workstation
> mount, while the legacy path is `/net/trapnell/...` — these may be different mounts with very
> different latency); and that T2a's provenance recording is present in the code you are about to run.
>
> **Task.** Regenerate this one plate under a **fresh, immutable output root** at an explicitly named
> commit. Never write into the existing tree. Record the commit in the report title.
>
> **Declare your pass thresholds before you run and put them at the top of the report.** This is not
> optional process. In the original controlled analysis a 6.5/20 arm scored *closest to legacy* on
> several metrics — not because 20 µm was right, but because its harder blend accidentally compensated
> for unrelated drift. Metric proximity alone is not evidence, and post-hoc thresholds will rationalise
> whatever you get.
>
> **Expect imperfect parity, and do not treat that as failure.** Restoring 6.5/75 does not reproduce
> legacy exactly: the mask source (full-frame SAM2 vs lower-resolution UNet/JPEG-era), the orientation
> evidence (mass-distribution fallback vs yolk-guided), and the source/crop assets all drifted and were
> not reverted. The question is whether the new contract is **acceptable**, not whether it is identical.
>
> **Image gate**, paired by well/embryo: foreground area and linear extent · mean, p95, and fraction of
> pixels ≥250 · radial taper profile across the boundary · orientation disagreement including explicit
> 180° flips · neighbouring-embryo and debris leakage into the 75 µm halo · contact sheets showing the
> **largest outliers**, not only medians.
>
> **Model gate**, both sets encoded through the same checkpoint: per-dimension correlation and
> standardized RMSE · pairwise-distance correlation · nearest-reference self-match · preservation of
> developmental and temperature signal · auxiliary-mask and QC changes caused by the regenerated snips.
>
> **Reference points.** The regressed corpus gave per-dimension correlation 0.687, pairwise-distance
> correlation 0.660, self-match 18.3% (chance ≈1.1%), temperature R² 0.353 against a legacy 0.582.
> Recovery toward the legacy values is the signal. Predicted foreground-area ratio for the scale
> regression alone was 0.694; measured was 0.700.
>
> **Provenance gate.** The run must emit T2a's full sidecar. If any field is missing, that is a T2a
> defect — report it rather than filling it in by hand.
>
> **Summary must state:** your pre-declared thresholds and whether each was met; the commit tested;
> every metric above with its legacy counterpart; and an explicit accept / do-not-accept
> recommendation for this rendering contract.
