> **Recovered 2026-08-27** from the untracked working tree of the `audit/regen-scope` worktree,
> which was removed during housekeeping. Measured 2026-08-24, **before** the `origin/main` merge
> (`19061cbf`) landed 116 `src/data_pipeline` commits including a snip-rendering rewrite. Its
> storage/timing/GPU measurements are likely still directionally valid; re-check any claim that
> depends on the renderer before planning the rerun from it.
>
> The `plans/THREAD_BRIEFS.md` path it cites is now `_archive/THREAD_BRIEFS.md`.

# Audit: regeneration scope, timing, GPU, and storage

**Audit date:** 2026-08-24. **Branch:** `audit/regen-scope`. **Mode:** read-only except this
report, as required by `docs/refactors/core-model/plans/THREAD_BRIEFS.md:241-271`. Measurement labels such as **M1** refer to
the exact commands and artifacts in [Measurement ledger](#measurement-ledger); each label is an
inline evidence citation.

## Decision

**Conditional NO-GO for launching all 133 experiments today.** This is an audit recommendation,
not a measured fact. The ordinary 133-experiment corpus does **not** need image materialization,
GroundingDINO, native-resolution segmentation, or physical-embryo registry recomputation; all
10,833 wells have their three required upstream shards and `.validated` sentinels, and all 529,171
distinct images referenced by the 699,505 current snips exist (M1, M2, M8). The rebuild boundary is the
snip renderer and everything whose values or pixels derive from its output
(`src/data_pipeline/pipeline_orchestrator/rules/snip_processing.smk:51-127`; `SNIP_IMAGE_REGRESSION_STATUS.md:295-298`).

The blockers to committing the whole corpus are operational:

1. There is no clean, current-DAG, **same-plate** full A/B with the resident server off and on. The
   available full-plate measurements are real, but use different plates and some served outputs may
   have been reused (M3-M6). Therefore the 23.17-task-day/14.15-task-day totals below are explicitly
   **extrapolations**, not acceptance timings.
2. The only safe bulk launch path is a copied array template with both the SGE GPU request and the
   Snakemake resource budget intact; the template itself is guarded and cannot be submitted
   (`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_array_TEMPLATE.sge:23-39,119-136`).
3. The repository is not installable for the pipeline import from its root: packaging includes only
   `src` and `src.*`, while the workflow imports top-level `data_pipeline`
   (`pyproject.toml:11-14`; M7). Existing SGE scripts work around that with `PYTHONPATH`, which the
   binding working agreement forbids (`AGENTS.md:69`; this is an observed packaging blocker, M7).
4. The filesystem is at 99% utilization. The scoped new root fits, but only if it copies or links the
   529,171 explicitly referenced source images—not whole acquisition directories (M8-M10).

**Inference / recommended gate:** run the representative `20230613` plate into the intended fresh
root once with `unet_snip.use_model_server: false` and once with it `true`, using the exact production
revision and copied array submit script. Do not release the 133-task array until that pair is green
and byte/schema comparison passes. This is the outstanding literal requirement in
`docs/refactors/core-model/plans/THREAD_BRIEFS.md:252-255`.

## Summary numbers

The timed representative plate is `20230613`: 5,421 snips, 75 wells, and 3.1% more snips than the
5,259.44 corpus mean. Its complete server-off DAG ran in **2:59:03** on four cores (M1, M3). That
historical run did not have a global `gpu=1` budget and therefore overlapped GPU rules unsafely. The
planning model corrects for the production template: non-auxiliary measured job work is divided over
four cores, auxiliary-mask work is serialized by `--resources gpu=1`, and the measured 2.88x A/B is
applied for the served case (M3, M6, M11). This is an inference model and does not include queue wait,
filesystem contention, retries, or a safety factor.

| Scope | Runnable unit | Server off | Server on | Ideal elapsed at four concurrent array tasks |
|---|---:|---:|---:|---:|
| Full corpus | 133 experiments; 10,833 wells; 699,505 snips | **23.17 task-days** | **14.15 task-days** | 139.01 h off / **84.92 h on** |
| QC-available experiments | 105 experiments; 8,506 wells; 531,902 snips | 18.19 task-days | 11.11 task-days | 109.15 h off / 66.68 h on |
| Provisional survivor-bearing subset | 94 experiments; 7,694 wells; 511,022 snips | **16.45 task-days** | **10.05 task-days** | 98.73 h off / **60.31 h on** |

Sources and formula: corpus membership/counts are M1; well counts are M2; timings and calculations
are M3, M6, and M11. A task-day is one experiment array task occupying its allocated GPU for 24
hours. “Ideal elapsed” is explicitly an inference equal to task-work divided by four; it is not a
cluster forecast.

The 176,466 currently selected training rows are **not** a runnable pipeline subset: the pipeline
fans out by explicit experiment/well, and `use_snip` is a downstream QC result that can change after
rerendering (`AGENTS.md:23-30`; `src/data_pipeline/pipeline_orchestrator/rules/snip_qc.smk:82-123`). The conservative provisional subset is
therefore all 511,022 rows in the 94 experiments that currently contain at least one survivor (M1).
The actual phase-one curated experiment list remains an open decision (`DECISIONS.md:79-92`), so the
94-experiment number must not silently become policy.

**Single biggest cost driver:** without serving, each well starts a new process and loads four UNet
checkpoints. On the representative plate this consumed **8,260 measured GPU rule-seconds**, median
105 seconds per well—the largest rebuild-family total (M3). With the production `gpu=1` budget that
work is serial: it contributes 2:17:40 to the representative plate model and 13.81 task-days to the
full corpus. Applying the measured 2.88x served A/B reduces the full-corpus auxiliary allowance to
4.79 task-days, a 9.01-task-day saving (M6, M11). The underlying benchmark attributes 69.94 seconds
per process to model loading and measured byte-identical served outputs
(`MODEL_LOAD_BENCHMARKS.md:229`).

## 1. Verified rerun boundary

The claim in `SNIP_IMAGE_REGRESSION_STATUS.md:295-298` is correct for the ordinary 133-experiment
corpus. The renderer consumes validated per-well `frame_masks`, `frame_inventory`, and
`physical_embryo_registry`, then writes the new snip inventory and pixels
(`src/data_pipeline/pipeline_orchestrator/rules/snip_processing.smk:51-127`). None of those three inputs depends on the changed crop
rendering. The measured on-disk preflight found all 10,833 payloads and all 10,833 validation
sentinels for each of the three input families (M2); the explicit snip manifests referenced 529,171
unique materialized images, all present and metadata-accessible via `stat` (M8).

| Product family | Action | Reason / evidence |
|---|---|---|
| Acquisition materialization and frame inventory | **Verify only** | Direct renderer input; 10,833/10,833 shards and sentinels present, and 529,171/529,171 referenced images exist (M2, M8). |
| GroundingDINO frame detections | **Verify only** | It feeds native frame masks, not the changed snip presentation; rerunning it would pull the excluded front of the DAG (`src/data_pipeline/pipeline_orchestrator/rules/frame_masks.smk:1-7,190-202`; `SNIP_IMAGE_REGRESSION_STATUS.md:295-298`). |
| Native-resolution frame masks / segmentation | **Verify only** | Direct renderer input; 10,833/10,833 shards and sentinels present (M2; `src/data_pipeline/pipeline_orchestrator/rules/snip_processing.smk:51-127`). |
| Physical-embryo registry | **Verify only** | Direct renderer input; 10,833/10,833 shards and sentinels present (M2; `src/data_pipeline/pipeline_orchestrator/rules/snip_processing.smk:51-127`). |
| Snip inventory, processed snips, and snip-space embryo masks | **Rebuild** | This is the changed boundary and the exact acceptance target (`SNIP_IMAGE_REGRESSION_STATUS.md:257-295`). |
| Four auxiliary masks | **Rebuild** | Their inputs are the validated regenerated snips (`src/data_pipeline/pipeline_orchestrator/rules/snip_auxiliary_masks.smk:93-115,120-153`). |
| Mask geometry, pose/curvature, focus/motion/mask-quality QC | **Rebuild** | Their values derive from changed snip or mask pixels; treating old values as current would mix generations. This is an inference from their placement downstream of the snip and auxiliary-mask contracts (`src/data_pipeline/pipeline_orchestrator/rules/analysis_ready.smk:20-45`). |
| Fraction alive, death/viability, stage prediction, and resolved snip QC | **Rebuild** | These are downstream policy/data products, including the required fraction-alive/viability chain (`SNIP_IMAGE_REGRESSION_STATUS.md:295-298`; `src/data_pipeline/pipeline_orchestrator/rules/snip_qc.smk:82-123`). |
| Legacy latent embeddings | **Rebuild** | Pixel-derived. The current rule loads once per experiment and is CPU-only (`src/data_pipeline/pipeline_orchestrator/rules/latent_embeddings.smk:75-118`). |
| Merged tables, reports, `analysis_ready`, and core manifests | **Rebuild / re-resolve** | They join downstream products and must identify one coherent generation (`src/data_pipeline/pipeline_orchestrator/rules/analysis_ready.smk:20-45`; `AGENTS.md:17-20`). |

The raw-data inventory is a broader 181-experiment acquisition scan, including 115 Keyence and 66
YX1 entries (`RAW_DATASET_INVENTORY.md:1-5`); it is not the operational 133-experiment list (M1).
Because every explicitly referenced materialized renderer input exists, raw re-ingestion is outside
this rerun boundary (inference from M2 and M8). This scope is not SeaHub's scope; see
[SeaHub](#7-seahub-separate-scope).

## 2. Measured plate timings

### Representative server-off plate

`20230613` contains 5,421 snip rows across 75 wells, versus a 5,259.44-row corpus mean (M1). Its
complete production log spans 2026-07-28 11:16:13 to 14:15:16: **2:59:03 measured wall clock** and
2,638/2,638 completed steps (M3). It used four cores and did not print a global `Provided resources:
gpu=1`, so the run is a valid elapsed-time observation but not proof of safe GPU serialization (M3).
A separate mostly-cached server-off run of the same plate completed in **2:12:37**, but still ran 22
frame-detection, 40 frame-mask, and 73 snip jobs; it is corroboration, not a clean downstream-only
timing (M3).

The following are measured job-time sums and per-well medians from that log. Job-time sums are not
additive wall time because Snakemake overlaps rules; this distinction is why the end-to-end envelope
is reported separately (M3).

| Stage | Jobs | Sum of measured job durations | Median per well | Current decision |
|---|---:|---:|---:|---|
| Snip processing | 75 | 1,216 s | 16 s | Rebuild |
| Auxiliary masks, server off | 75 | **8,260 s GPU** | **105 s** | Rebuild; serve |
| Mask geometry | 75 | 818 s | 10 s | Rebuild |
| Pose kinematics | 75 | 994 s | 12 s | Rebuild |
| Curvature metrics | 75 | 1,034 s | 13 s | Rebuild |
| Focus QC | 75 | 1,289 s | 16 s | Rebuild |
| Motion-blur QC | 75 | 2,536 s | 32 s | Rebuild |
| Mask-quality QC | 75 | 566 s | 6 s | Rebuild |
| Fraction alive | 75 | 573 s | 7 s | Rebuild |
| Death/viability | 75 | 582 s | 7 s | Rebuild |
| Stage predictions | 75 | 336 s | 4 s | Rebuild |
| Surface-area QC | 75 | 407 s | 5 s | Rebuild |
| Resolved snip QC | 75 | 710 s | 8 s | Rebuild |
| Legacy embeddings, old per-well rule | 75 | 2,976 s | 40 s | Do not extrapolate as current embedding cost; rule is now run-batched |

Source: M3. The old embedding row is deliberately retained in the full-corpus estimate as a
conservative allowance; the current rule is one CPU batch per experiment and its average-size-plate
cost remains unmeasured (`src/data_pipeline/pipeline_orchestrator/rules/latent_embeddings.smk:75-118`).

### Server-on production observations

Both resident-service toggles still default off
(`src/data_pipeline/pipeline_orchestrator/rules/frame_detections.smk:24`;
`src/data_pipeline/pipeline_orchestrator/rules/snip_auxiliary_masks.smk:23`). Only the auxiliary-mask
toggle affects the ordinary rerun boundary established above (M2). The A1 brief records an
approximately 3x GroundingDINO result (`docs/refactors/core-model/plans/THREAD_BRIEFS.md:252-255`), but it cannot reduce this
rerun because GroundingDINO is verify-only.

The most current downstream-only run is `20250612_24hpf_ctrl_atf6` (97 snips, 96 wells). Its main
attempt occupied 40:47, failed late in QC/schema handling, and its resume occupied 12:07; total
scheduled job occupancy was **52:54**, or **56:41** from the first start through final completion including the
resubmission gap (M4). The DAG contained snip processing onward and merged the existing upstream
frame inventories, masks, and registries rather than rebuilding them (M4). Measured current-rule
work included 511 job-seconds for snip processing, a 126-second served auxiliary-mask envelope, and
41 seconds for the single run-batched legacy embedding job (M4).

A second served run, `20260408_pbx`, processed a 96-well plate whose merged inventory has 17,639
snips. The selected 866-job slice (GroundingDINO service, native masks, snips, and auxiliary service)
completed in **21:59 outer wall / 21:50 Snakemake wall**, with `Provided resources: gpu=1` (M5).
GroundingDINO loaded once in 24.46 seconds; the four-UNet adapter loaded once in 68.43 seconds; the
served auxiliary group spanned about 104 seconds (M5). This is valuable scale evidence, but is not a
clean rerender benchmark because some stored pixel payloads may have been reused; that caveat is an
inference from the very low request durations and existing on-disk products.

The controlled three-well benchmark remains the clean equivalence evidence: auxiliary masks were
280.39 seconds without serving versus 97.47 seconds served (**2.88x**) and all 2,100 mask PNGs were
byte-identical (`MODEL_LOAD_BENCHMARKS.md:229`; M6). GroundingDINO
is outside the ordinary rerun boundary because its validated native-mask descendants already exist
(M2; `src/data_pipeline/pipeline_orchestrator/rules/frame_masks.smk:1-7`).

**Unfinished:** no log contains a clean full-plate run of the same representative plate under both
settings. The server-on values therefore support the planning model but do not close the A/B gate.

## 3. Extrapolation method and uncertainty

The complete plate produced 42,678 seconds of summed job work on four cores in 10,743 seconds of wall
clock; `42,678 / 4 = 10,669.5`, within 73.5 seconds of the observed envelope (M3). That measured
near-saturation supports using four-core-equivalent job work for the scoped model. Removing verified-
only GroundingDINO, native masks and their validation, and registry build/validation leaves 30,653
measured downstream job-seconds (M3, M11).

For each scope, non-auxiliary work is
`(30,653 - 8,260) / 4 / 75 wells * scope_wells`. With the production `gpu=1` resource budget,
server-off auxiliary work is serialized as `8,260 / 75 * scope_wells`. The server-on scenario applies
the clean measured factor: `(8,260 / 2.88) / 75 * scope_wells` (M3, M6, M11). This deliberately uses
measured plate work and the measured A/B rather than inventing a per-snip rate.

The model adds the serialized GPU term to four-core-equivalent non-auxiliary work and takes no credit
for CPU/GPU overlap, so it is a conservative task-occupancy model in that respect (inference). The
array's GPU is allocated for the entire experiment task even while CPU rules run
(`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_array_TEMPLATE.sge:23-30,121-136`).

For the representative plate itself, that safe-schedule model is **3:50:58 off / 2:21:06 on**
(M11). These are planning calculations, not substitutes for the missing same-plate production A/B;
the directly observed complete historical wall clock remains 2:59:03 under the unsafe resource setup
(M3).

The model is conservative because it retains the old per-well embedding work even though the current
rule is batched once per experiment (M3; `src/data_pipeline/pipeline_orchestrator/rules/latent_embeddings.smk:75-118`). It may be optimistic
because it assumes the three-well 2.88x factor transfers to a 5,421-snip plate and does not model queue
delay, I/O contention, or retries (inference). Consequently the totals are appropriate for planning,
but not for booking a completion date before the same-plate A/B.

The array template gives each experiment task its own one-GPU budget; `-tc 4` permits four such tasks
to run on four allocated GPUs, not four GPU rules on one card
(`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_array_TEMPLATE.sge:2-15,121-136`). The four-task column in the
summary assumes uninterrupted capacity and perfect load balance (inference).

## 4. GPU scheduling audit

The current array template requests an SGE GPU with `#$ -l gpgpu=TRUE,cuda=1`, four CPU slots, and
passes `--resources gpu=1` to Snakemake
(`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_array_TEMPLATE.sge:23-30,121-136`). The single-run template has
the same two controls (`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_TEMPLATE.sge:16-23,131-151`). Served
clients correctly claim no GPU while the service holds the one unit
(`src/data_pipeline/pipeline_orchestrator/rules/snip_auxiliary_masks.smk:66-115`); services also need at least two cores
(`MODEL_SERVER_WIRING.md:80-92`).

This proves the **template**, not an upcoming concrete submission. The bulk launch must therefore
retain all of the following:

- the explicit ordered experiment file and matching `-t 1-N` range
  (`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_array_TEMPLATE.sge:2-21`);
- `#$ -l gpgpu=TRUE,cuda=1` and `--resources gpu=1`
  (`src/data_pipeline/pipeline_orchestrator/sge_job_submissions/submit_snakemake_array_TEMPLATE.sge:23-30,121-136`);
- at least two cores for a served run (`MODEL_SERVER_WIRING.md:80-92`);
- the resource inversion: GPU on the service, not its clients
  (`MODEL_SERVER_WIRING.md:18-39`).

The historical representative server-off log had four cores but no global GPU budget (M3), while
the current served logs printed `Provided resources: gpu=1` and ran on A100/L40S GPU nodes (M4, M5).
Thus old elapsed logs must not be treated as proof that every legacy SGE script is safe.

## 5. Disk for a second immutable root

At measurement time the shared filesystem had 4,378,673,020,928 bytes free and reported 99%
utilization (M8). A full `du` of the existing pipeline root was stopped after 1,365.94 seconds without
a result because the metadata walk had not completed (M9); no size is invented for that tree.

The scoped measurement is more relevant:

| Scope / component | Measured or extrapolated bytes | Evidence |
|---|---:|---|
| Full corpus referenced upstream images | **899,282,038,198 B (837.52 GiB)** | Exact `stat` sum of 529,171 ordered, manifest-referenced paths; zero missing (M8). |
| Provisional 94-experiment subset referenced images | **621,413,812,615 B (578.74 GiB)** | Exact `stat` sum of 375,700 referenced paths; zero missing (M8). |
| Storage sample (`20251020`) regenerated downstream tree | **925,110,784 B** | Exact `du` of snips, aux manifests/masks, feature extraction, QC, and analysis-ready for 5,216 snips (M10). |
| Full downstream products | **124,064,344,126 B (115.54 GiB)** | Explicit inference: representative bytes per snip multiplied by 699,505 (M10). |
| Full scoped root before margin | **1,023,346,382,324 B (0.931 TiB)** | Inference: exact input bytes plus downstream extrapolation (M8, M10). |
| Full scoped root with 20% margin | **1,228,015,658,789 B (1.117 TiB)** | Inference; leaves about 2.87 TiB against the measured free-space value (M8, M11). |

The `20251020` storage sample's entire acquisition directory occupies 73,446,961,664 bytes, but only
8,282,942,629 bytes are the distinct images referenced by its snip inventory (M10). The difference
is unrelated products such as additional materializations; copying whole acquisition experiment
directories would therefore badly overbudget and violate the explicit-product scope (inference;
`AGENTS.md:17`).

**Storage conclusion:** a scoped copy fits; a blind clone is unaudited and must not be launched.
The new root needs an explicit ingress design that copies, hard-links, or otherwise immutably
references only the verified renderer inputs. Because the current DAG constructs upstream artifact
paths under one `DATA_ROOT`, merely pointing `output_root` at an empty directory is not sufficient
(`src/data_pipeline/pipeline_orchestrator/rules/snip_processing.smk:51-127`). Choosing copy versus hard-link/reference is an implementation
decision outside this read-only audit.

## 6. Downstream rebuild cost table

Costs below are observed representative-plate job durations, not a claim that the stages execute
serially (M3). “Verify” cost is the measured metadata audit: the three upstream shard families plus
529,171 source-image stats completed without a missing artifact (M2, M8).

| Dependency | Rebuild or verify | Measured plate cost / note |
|---|---|---|
| Materialized projection inputs, frame inventory, frame masks, registry | Verify | 10,833 wells checked for three artifact+sentinel pairs; 529,171 files checked (M2, M8). |
| Snip pixels + snip mask | Rebuild | 1,216 job-seconds, median 16 s/well (M3). |
| Auxiliary masks | Rebuild | 8,260 measured GPU rule-seconds off, median 105 s/well; controlled served A/B is 2.88x (M3, M6). |
| Geometry + pose + curvature | Rebuild | 818 + 994 + 1,034 job-seconds (M3). |
| Focus + motion + mask-quality QC | Rebuild | 1,289 + 2,536 + 566 job-seconds (M3). |
| Fraction alive / viability | Rebuild | 573 + 582 job-seconds on the representative plate (M3). |
| Legacy embeddings | Rebuild | Current batch rule measured 41 s on the 97-row plate; average-size-plate cost remains unmeasured (M4; `src/data_pipeline/pipeline_orchestrator/rules/latent_embeddings.smk:75-118`). |
| Snip QC, merges, reports, analysis-ready, core resolved manifests | Rebuild / re-resolve | Current run reached analysis-ready after late failure+resume; 52:54 occupied wall in total (M4). |

## 7. SeaHub: separate scope

SeaHub is not part of the 699,505-row/133-experiment total (M1). The canonical tracker says it is
“built, unverified end-to-end” and that every scale is a 7.8 placeholder
(`PLANNED_REVISIONS.md:202-220`), but the stored 20260804 production artifacts are newer and disagree.

The production bundle contains 92 shards and 8,064 analysis-ready rows. A read-only reopen in the
`segmentation_grounded_sam` environment read all 92 Parquets, all 8,064 rows, and confirmed
`source_scope == "seahub"` everywhere (M12). The original verifier did run and failed only because
its selected environment had neither `pyarrow` nor `fastparquet`; its report recorded 92/92
unreadable artifacts (M12). This is an environment failure, not evidence that the Parquets are bad.

Measured SeaHub work must remain separate:

| SeaHub stage | Measured scope | Measured time | Evidence / caveat |
|---|---:|---:|---|
| GroundingDINO detection | 1,019 FOVs, 20 source experiments | 13:00 array wall; 1,918 GPU task-seconds | Actual 2026-08-02 job; the 20260804 run root appears to reuse/copy this detection generation, so provenance linkage needs verification (M13; inference for reuse). |
| SAM2 source-FOV mask persistence | 1,008 accepted FOVs, 8,064 masks | about 16:30 job occupancy | Actual 2026-08-04 job reached `DONE`, then its postflight reported a 8,064-vs-12,064 count error; the CSV now parses as exactly 8,064 logical records and `_SUCCESS` was written later. This is not a clean completion and must be reconciled (M13). |
| Corpus-wide bundle planning | 8,064 embryos, 92 shards | 1:10 | Actual 2026-08-06 log (M13). |
| Materialize 92 shards | 8,064 one-frame wells | 4:04 array wall; 1,597 task-seconds | Actual 2026-08-06 array (M13). |
| Standard pipeline back half | 92 shards / 8,064 rows | **40:09:46 array wall**; 420,929 task-seconds (116.92 task-hours); median task 1:01:19.5 | Actual 2026-08-06 through 2026-08-08 array; tasks overlapped (M13). |
| Verification | 92 analysis-ready Parquets | Failed in job environment; passes read-only in environment with PyArrow | M12. |

Thus the tracker is stale about both end-to-end state and the stored numeric scale, but the source-mask
sentinel/provenance anomaly means SeaHub is **verify/reconcile first**, not “rerun the front blindly”
(audit inference based on M12-M14).

### SeaHub pixel-scale question

No stored 20260804 SeaHub frame-inventory row has a column named
`source_micrometers_per_pixel`; the current contract column is `image_micrometers_per_pixel`. Across
all 92 stored inventories, all **8,064/8,064** rows have that column, **0 equal 7.8 exactly**, and
8,064 differ; the range is 3.6924676700045973 to 9.0 and the median is 7.092735014403527 (M14).

The assignment chain is:

1. `src/data_pipeline/acquisition/seahub/scale_calibration.py:421-465` computes and bounds a per-FOV
   value; it writes `image_micrometers_per_pixel: final_scale` at
   `src/data_pipeline/acquisition/seahub/scale_calibration.py:466-476`.
2. `_attach_fov_scale` broadcasts one FOV row to its embryos
   (`src/data_pipeline/acquisition/seahub/integration.py:498-530`).
3. `_frame_inventory` reads that value, falling back to configured 7.8 only if absent, at
   `src/data_pipeline/acquisition/seahub/integration.py:1110-1124`, then assigns it to the stored row
   at `src/data_pipeline/acquisition/seahub/integration.py:1128-1149`.

The values are still explicitly marked `calibration_status: "placeholder"` because they are
mask-area inference rather than direct physical metrology
(`src/data_pipeline/acquisition/seahub/scale_calibration.py:471-478`). Therefore
the tracker is right that absolute-size analysis remains uncalibrated, but wrong that the stored
numeric values are all 7.8 (`PLANNED_REVISIONS.md:211-213`; M14).

## 8. Disagreements and launch hazards

1. `PLANNED_REVISIONS.md:202-220` says SeaHub has not run end to end and uses 7.8 everywhere; stored
   20260804 artifacts instead contain 92 readable analysis-ready Parquets and 8,064 non-7.8 frame
   rows (M12, M14). The final verifier's missing-Parquet-engine failure and the SAM2 sentinel anomaly
   still need reconciliation (M12, M13).
2. Binding `AGENTS.md:43-45` says core optical covariates do not exist and must not be derived or
   imputed, while `DECISIONS.md:25-26,61` says they are present and should be added. `AGENTS.md` wins.
   The SeaHub `image_micrometers_per_pixel` audit above is pipeline evidence; it is not permission to
   add a core covariate.
3. Binding `AGENTS.md:84-85` says PyArrow 25.0.1 is installed in `morphseq-env`; the 2026-08-24
   command `conda run -n morphseq-env conda list pyarrow` returned no package, and a Parquet read
   failed. `segmentation_grounded_sam` has PyArrow 12.0.1 and read all SeaHub outputs (M12). Any bulk
   preflight must test the actual execution environment and fail specifically; it must not skip QC.
4. The SGE templates and Snakefile inject `PYTHONPATH`, while the binding agreement forbids path
   manipulation (`AGENTS.md:69`; `src/data_pipeline/pipeline_orchestrator/Snakefile:62-72`; M7). This packaging disagreement should be fixed
   before treating the branch as a reproducible production launcher.
5. Model-server clients have no retry/backoff, so a server transport failure exits rather than
   recovering (`MODEL_SERVER_WIRING.md:96-100`). This is a known operational risk, not a reason to
   disable the server and accept the planning model's nine extra task-days.

## Measurement ledger

All commands were read-only and run on 2026-08-24 PDT.

- **M1 — explicit corpus/cohort census.** Python `csv.DictReader` over tracked
  `docs/refactors/core-model/reports/recon_tables/availability_schema.csv` and
  `docs/refactors/core-model/reports/recon_tables/combined_metric_gate_by_experiment.csv`, restricted to readable inventory
  rows. Output: 133 experiments / 699,505 snips; 105 / 531,902 QC-available; 94 / 511,022
  survivor-bearing; 176,466 survivors; mean 5,259.436090225564 snips/experiment. This uses the
  explicit tracked table, not output-tree experiment discovery.
- **M2 — upstream shard preflight.** For the ordered M1 experiment list, read each merged inventory,
  carried each opaque `well_id` unchanged, and checked the exact per-well paths for
  `frame_inventory`, `frame_masks`, and `physical_embryo_registry`, plus each `.validated` path.
  Output: 133 experiments / 10,833 wells; 10,833 payloads and 10,833 sentinels present for each of
  all three families; zero missing.
- **M3 — representative server-off timing.** Parsed timestamped rule/jobid/`Finished job` events in
  `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output/work_directories/20230613/.snakemake/log/2026-07-28T111600.007018.snakemake.log`
  with Python `datetime`; M1 supplies the row count. Output: 5,421 rows, 75 wells, 10,743-second
  complete envelope, 2,638/2,638 completed steps, 42,678 summed job-seconds, and the stage
  sums/medians reported above. The same parser checked the completed mostly-cached run
  `2026-07-26T151012.070106.snakemake.log`: 7,957-second envelope, 2,528/2,528 completed steps, with
  22 detection, 40 frame-mask, and 73 snip jobs.
- **M4 — current downstream served timing.** Parsed
  `/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/logs/snipfix_20250612_p01.23311495.{out,err}`
  and the adjacent `snipfix_20250612_resume.23314581.{out,err}`. Output: main 15:58:56-16:39:43; resume
  16:43:30-16:55:37; both print `Provided resources: gpu=1`; stage event parsing produced 511
  snip-processing job-seconds and 41 seconds for `encode_latent_embeddings_for_run`; harness events
  give the served auxiliary envelope.
- **M5 — large served slice.** Parsed
  `/net/trapnell/vol1/home/mdcolon/proj/morphseq/logs/msrv_pbx_all.22833499.{out,err}` and counted
  17,639 data rows with `wc -l .../20260408_pbx_snip_inventory.csv` minus header. Output: 21:59
  outer, 21:50 Snakemake, one 24.46-second GroundingDINO load, one 68.43-second UNet load, and
  `Provided resources: gpu=1`.
- **M6 — controlled service A/B.** `MODEL_LOAD_BENCHMARKS.md:229`; this is a dated direct SGE
  measurement, not an estimate.
- **M7 — packaging/import check.** From repository root:
  `conda run -n morphseq-env python -c 'import data_pipeline'` ->
  `ModuleNotFoundError: No module named 'data_pipeline'`; package discovery is
  `pyproject.toml:11-14`. No `PYTHONPATH` was set for the audit command.
- **M8 — free space and explicit referenced-image size.** `df -B1 <pipeline-output-root>` ->
  4,378,673,020,928 bytes available, 99% used. Python read the ordered M1 inventories, deduplicated
  `image_path` without parsing IDs, sorted paths for metadata locality, and summed `os.stat().st_size`:
  full 529,171 files / 899,282,038,198 bytes / zero missing; QC-available 386,215 /
  651,934,466,635; survivor-bearing 375,700 / 621,413,812,615.
- **M9 — whole-root size attempt.** `/usr/bin/time du -sx --block-size=1
  <pipeline-output-root>` was interrupted after 1,365.94 seconds without a result. A subsequent
  component-wise `du` also produced no result and was terminated during final process cleanup. No
  files changed.
- **M10 — representative storage.** `du -s --block-size=1` on the five `20251020` downstream
  product roots -> 925,110,784 bytes. The whole acquisition experiment root was 73,446,961,664
  bytes. `os.stat` over distinct `image_path` values in that plate's inventory -> 5,190 files /
  8,282,942,629 bytes; the inventory has 5,216 snips. Full downstream extrapolation is the measured
  925,110,784 / 5,216 byte rate times 699,505 rows.
- **M11 — timing/storage arithmetic.** Python arithmetic using M1-M3, M6, and M8-M10. The complete
  plate's 42,678 job-seconds divided by four differs from its measured 10,743-second envelope by
  73.5 seconds. Excluding upstream verify-only rules leaves 30,653 job-seconds, of which 8,260 are
  auxiliary masks. Representative safe-schedule model: 13,858.25 seconds off and 8,466.31 seconds
  on. Full safe-schedule off: 2,001,685.63 task-seconds; full served scenario:
  1,222,873.17 task-seconds; survivor-bearing off: 1,421,671.67; on: 868,530.07. Scoped disk before
  margin: 1,023,346,382,324 bytes; with 20%: 1,228,015,658,789 bytes.
- **M12 — SeaHub output verification.** Read
  `.../seahub/derived/20260804_prod01/bundle/integration/experiment_manifest.csv` and all explicitly
  listed analysis-ready Parquets with `segmentation_grounded_sam` (PyArrow 12.0.1): 92/92 readable,
  8,064 rows, all rows `source_scope == "seahub"`. The original
  `/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/logs/seahub_verify.23494609.err`
  records the missing-engine failure.
- **M13 — SeaHub timing.** Parsed START/FINISHED timestamps in jobs
  `seahub_gdino.23408651/23408652`, `seahub_sa_masks.23438008`,
  `seahub_plan.23494587`, `seahub_mat.23494607`, and `seahub_back.23494608`. The 92 back-half
  task durations sum to 420,929 seconds; array envelope is 144,586 seconds; all 92 tasks have a
  finish marker. The SAM2 stderr contains its postflight count error.
- **M14 — SeaHub scale census.** Read the 92 `frame_inventory_csv` paths from the explicit production
  experiment manifest with Python `csv`: 8,064 rows; zero inventories contain
  `source_micrometers_per_pixel`; all contain `image_micrometers_per_pixel`; 0/8,064 values equal
  7.8 exactly. The 1,019-row `fov_scale_calibration.csv` has 1,019/1,019 values unequal to 7.8.

## Work deliberately not done

- No pipeline output was written, overwritten, chmodded, or deleted; no SGE job was submitted.
- No source, test, config, tracker, status, or contract file was changed.
- No clean same-plate server A/B was fabricated from unlike logs; that gate remains open.
- No exact size is claimed for the existing full output tree because its read-only `du` did not
  finish in 22:45.94.
- No phase-one experiment curation was invented; the 94-experiment subset is labeled provisional.
- No SeaHub success sentinel was repaired and no failed verifier report was overwritten.
