# Decision ledger — core-model refactor

**Updated:** 2026-08-25 · **Location:** `docs/refactors/core-model/DECISIONS.md`

This is the ledger of things a human decided. It contains no computed facts — those live in
`STATUS.md`, which is script-generated. Plan, issues, and next steps live in `PLAN.md`.

**Every claim in a written document carries its source inline** — a command, a `file:line`, or a
measurement date — or is explicitly marked as inference. Claims that cannot carry a source belong in
generated `STATUS.md`.

⚠️ **Before replacing the previous version of this file, diff it and report anything it contained
that is not carried over here.** This ledger was reconciled outside the repo and may not include
decisions recorded only in the prior copy.

---

## Ratified

| # | Decision | Verified by |
|---|---|---|
| D1 | `analysis_ready` is **not** the training boundary; the manifest is built from `snip_inventory` + `stage_predictions` + `snip_qc` + `plate_metadata` | — |
| D2 | Replace `metric_array` with a **rule-based relation function** at class×class granularity; curation revision pending | — |
| D4 | Carry `image_product_type`; phase one filters FF-only via an explicit list. **No ID-format change** — product type and z belong in columns, never in `snip_id` | — |
| D5 | Fix both metric mis-wiring causes: the `src/data` shim import at `model_configs.py:8`, **and** `dataconfig.target: "BasicDataset"` in the metric Hydra config | — |
| D6 | Keep `[1, 288, 128]` as the default model input; image size becomes a config parameter | — |
| D7 | Derive `_pixel_scale` from `input_dim` rather than the literal `128*288` at `loss_functions.py:185-187`. Forced by D6 | — |
| D8 | **On-the-fly** downsampling at the dataset boundary; no exported training set. A cache only if a real-host I/O measurement demands it | — |
| D9 | Delete the margin term from the metric loss entirely — confirmed inert (shared across all logits, cancels in the softmax) | — |
| D10 | Delete `accumulate_grad_batches` — never passed to Trainer, ignored under manual optimization | — |
| D11 | Splits group-disjoint at `physical_embryo_id`, **content-hash based** (`blake2b`), stable under cohort growth, persisted by ID. Ratio tolerance asserted; empty required split fails unconditionally | — |
| D12 | Decoder δ-pathway: thin, injected late | — |
| D13 | Predictive z-slice reconstruction: **excluded** ("not worth the lift") | — |
| D15 | Metric path wired now; metric **policy** revisited separately | — |
| D16 | Run provenance = one directory per run: `resolved_config.yaml`, gzipped `snip_ids`, `split_assignments.csv` keyed by `physical_embryo_id`, `metric_group_map.csv`, `sources.json` (path/size/mtime/row-count per source artifact), `cohort_report.json`, plus the adapter's git SHA. **Not** the full manifest — IDs + config regenerate it, `sources.json` detects when they can't | — |
| D17 | The provenance bundle is a **W&B Artifact**; the local directory is staging, addressed by one config key (`run_artifacts_dir`). No coupling to Hydra's cwd or Lightning's internals | — |
| D18 | The manifest carries the **individual QC flag columns**, not just `use_snip`; cohort policy is a named, versioned predicate over flags | — |
| D19 | Do **not** hash image content (~28 GB, hours). Hash source artifacts (~500 MB, seconds) plus the sorted `snip_id` list | — |
| D20 | `ImageFolder` is **deleted**, not deprecated. Datasets are plain `torch.utils.data.Dataset` over manifest rows; view generation lives in `__getitem__`, which holds the whole row including `embryo_mask_snip_path`. No separate abstraction layer | — |
| D21 | **6.5 µm/px and 75 µm blend radius are the compatibility baseline.** Pre-August stored snips are not checkpoint-compatible inputs | — |
| D22 | **Absence is never failure.** `qc_status` and `stage_status` are three-state. `no_artifact` ≠ failed QC (28 experiments, 167,603 snips); `unavailable` ≠ failed staging (2 experiments, 20,219 snips with finite stage and no status column) | — |
| D23 | Adopt SupCon **`L_out`** (sum over positives outside the log) rather than `L_in`. Decided previously and relitigated through documentation entropy — treat as settled. Requires a λ_metric/τ retune | — |
| D24 | Test holdout supports an explicit `test_experiments` list: listed experiments go wholesale to test, the remainder hash-assigns into train/eval. Tolerance targets computed over the unpinned pool | — |
| D25 | Decode failures are **non-fatal**: log the `snip_id`, resample from the same split, count, and fail the run only above a configured threshold. Counts and IDs go into the cohort report | — |

`verified by` is filled in by the pytest `decision` marker. An empty cell means **asserted but
unproven** — the count of empty cells is the headline line in generated `STATUS.md`.

## Superseded

**D3 / D14 — carry optical covariates, add explicit optical conditioning.** Not actionable.
`micrometers_per_pixel`, `microscope_id`, `objective_magnification`, and `z_position` exist in legacy
metadata but in **no pipeline schema** — *verified 2026-08-19, `reports/PIPELINE_RECON.md` §9*.
Reserve the names; never derive, default, or impute. Revives only if the pipeline emits them.

**O9 — "FF is done" convergence criterion.** Deleted permanently.

## Corrections to prior records

**The 7.8 µm/px figure was struck in error on 2026-08-18 and is reinstated.** 7.8 was the value the
replacement pipeline was actually writing; the 6.5→7.8 drift together with the 75→20 blend-radius
drift *is* the snip regression. The earlier strike would have caused a reader to dismiss the most
important open issue in the project.

**Optical covariates were recorded inconsistently** — "present" was true of legacy metadata,
"absent" true of pipeline output, and no document said which it meant. See Superseded.

**Two closed items sat in the open list**: the ImageFolder boundary (closed by D20) and run-artifact
layout (closed by D16/D17).
