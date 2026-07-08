# Handoff: Consolidate All QC into `0_load_and_clean_datasets.py`

## The Goal

All QC — imaging pipeline QC and sequencing QC — should be computed and written
by **`0_load_and_clean_datasets.py`**. Scripts `1_fit`, `2_predict`, and the `3x`
plots should only see embryos that **pass both gates** (`final_usable = True`).
The analysis pipeline should be self-documenting: run `0` and you get a single
authoritative table of who passed, who failed, and why.

---

## Current State (as of end of session 2026-06-16)

### What `0` does today
- Loads build06 latent CSVs for all experiments → `query_all_rows_clean.csv`
- Reads the Excel `sequenced` sheet per plate → `query_sequenced_embryos.csv`
- Writes per-embryo metadata (genotype, collection_time_hpf, physical_embryo_id, etc.)
- **Does NOT know about imaging QC outcomes** (build04 `use_embryo_flag`, QC flags)
- **Does NOT know about sequencing QC** (failed_sequencing_qc CSV)

### What `3a` does today
- Reads Excel `sequenced` sheets + build04 QC CSVs → `sequenced_coverage_audit.csv`
- Classifies each sequenced well: `OK / EXCLUDED / ABSENT_IMAGED / ABSENT_NO_IMAGE / QC_NOT_RUN`
- Reads `tables/failed_sequencing_qc/missing_embryos_due_to_sequencing.csv` →
  resolves seq-failures via `SEQ_QC_IMAGING_PLATE_MAP` dict
- Writes `embryo_loss_map.csv` with `seq_qc_status` + `final_usable` columns
- Writes `embryo_registry.csv` — **CURRENTLY WRONG**: built from `query_all_rows_clean.csv`
  (pipeline survivors only), so imaging-failed embryos are invisible. The Venn/counts
  undercount because 13 crispant non-OK wells never appear.
- Writes plots: Venn, stacked bars, well grids

### What `2_predict` does today
- Filters on `sequenced > 0` from `query_sequenced_embryos.csv`
- **Does NOT filter on imaging QC** (imaging-failed embryos have no build06 row so they
  disappear naturally, but this is invisible/implicit — not documented)
- **Does NOT filter on seq QC** — seq-failed embryos that passed imaging still go into
  predictions. This is the active bug: seq-failed embryos are being predicted.

### The gap
`query_all_rows_clean.csv` = "passed pipeline (build06)"  
`embryo_loss_map.csv` = "passed imaging QC AND sequencing QC" ← the right filter  
`2_predict` uses the former; it should use the latter.

---

## Migration Plan

### Step 1: Move imaging-QC reading into `0`

Add a function `read_imaging_qc(experiment: str) -> dict[str, str]` to `0` that reads
the build04 `qc_staged_{experiment}.csv` and returns `{well: status}` where status ∈
`{OK, EXCLUDED, ABSENT_IMAGED, ABSENT_NO_IMAGE, QC_NOT_RUN}`.

The logic already exists in `3a`'s `audit()` function — **copy it into `0` as a helper**.
Key paths already defined in `3a` (add them to `0`):
```python
BUILD04_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build04_output"
STITCHED_FF  = PROJECT_ROOT / "morphseq_playground" / "built_image_data" / "stitched_FF_images"
REAL_QC_FLAGS = ["frame_flag", "sam2_qc_flag", "no_yolk_flag", "focus_flag",
                 "bubble_flag", "dead_flag", "dead_flag2", "sa_outlier_flag"]
```

The human-curated disposition sidecar (`tables/well_dispositions.csv`) stays in `3a`
for now — it is optional narrative, not needed for the `final_usable` gate.

### Step 2: Move seq-QC reading into `0`

Move the following from `3a` into `0`:
- `SEQ_QC_IMAGING_PLATE_MAP` dict (the (imaging_plate, perturbation) → [experiment_id] mapping)
- `SEQ_QC_DATE_WARNINGS` dict
- `_norm_well()` helper
- `load_seq_qc_failures(query_all_rows)` function

`0` already reads `tables/failed_sequencing_qc/missing_embryos_due_to_sequencing.csv`
path is `RUN_DIR / "tables" / "failed_sequencing_qc" / "missing_embryos_due_to_sequencing.csv"`.

### Step 3: Write `tables/qc_registry.csv` from `0`

At the end of `0`'s `main()`, build a single QC registry:

**Grain: one row per sequenced well** (same grain as `embryo_loss_map.csv` today, but
built from the Excel `sequenced` sheet — not from build06 survivors).

The spine is the Excel `sequenced` sheet (all wells with `sequenced > 0`), joined to:
- build04 QC outcome → `imaging_qc_status`
- build06 data existence → `has_latents` (True if embryo made it to build06)
- `embryo_id` from build04 (or synthetic `{exp}_{well}` if absent)
- `physical_embryo_id` from `0`'s existing logic
- seq-QC outcome → `seq_qc_status`
- `final_usable = (imaging_qc_status == "OK") & (seq_qc_status == "seq_ok")`

Columns:
```
embryo_id, physical_embryo_id, experiment, gene, well, sequenced, sequenced_stratum,
genotype_clean, zygosity, collection_time_hpf, data_source,
imaging_qc_status, exclusion_flags, has_latents,
seq_qc_status, final_usable
```

Write to `tables/qc_registry.csv`. This replaces `embryo_loss_map.csv` as the
authoritative single-source QC table.

Also write `tables/query_sequenced_embryos.csv` to contain **only `final_usable=True`
rows** (currently it is all `sequenced > 0` rows, which includes seq-failed and
imaging-failed embryos). This is the table `2_predict` and `1_fit` read — changing
what goes into it automatically gates the whole downstream pipeline.

### Step 4: Update `3a` to READ from `0`, not recompute

After Step 1–3, `3a` becomes a **plot-only script**:
- Remove `audit()`, `SEQ_QC_IMAGING_PLATE_MAP`, `load_seq_qc_failures()`,
  `build_embryo_registry()` from `3a`
- `3a` reads `tables/qc_registry.csv` (written by `0`) and makes plots:
  - Coverage heatmap, stacked bar, well grids (with SEQ_FAILED cyan)
  - Venn diagram, final_usable_by_gene bar
  - Loss reasons bar
  - Writes `MISSING_SEQUENCED_AUDIT.md` from the registry

The markdown audit and dispositions sidecar logic can stay in `3a` (it is narrative,
not a filter gate).

### Step 5: `1_fit` and `2_predict` need no changes

Because `query_sequenced_embryos.csv` (Step 3) will only contain `final_usable=True`
rows, scripts `1` and `2` automatically get the correct filtered set. No code changes
needed there — the gate is upstream.

---

## Run order after migration

```
0_load_and_clean_datasets.py     ← computes ALL QC; writes qc_registry.csv
                                    and filtered query_sequenced_embryos.csv
1_fit_reference_models.py        ← reads filtered tables; no QC logic
2_predict_sequenced_embryos.py   ← reads filtered tables; no QC logic
3a_audit_sequenced_coverage.py   ← reads qc_registry.csv; plots only
3b / 3c / 3d / ...               ← plots; no QC logic
```

---

## Files touched

| File | Change |
|---|---|
| `0_load_and_clean_datasets.py` | Add imaging QC + seq QC reading; write `qc_registry.csv`; filter `query_sequenced_embryos.csv` to `final_usable` only |
| `3a_audit_sequenced_coverage.py` | Remove audit/seq-QC computation; read `qc_registry.csv`; keep plots + markdown |
| `cilia_qc_helpers.py` | No change (select_for_label_transfer stays) |
| `1_fit_reference_models.py` | No change |
| `2_predict_sequenced_embryos.py` | No change (benefits automatically via filtered input) |
| `tables/qc_registry.csv` | New file (replaces embryo_loss_map.csv as authoritative QC) |
| `tables/embryo_registry.csv` | Can be deleted after migration (was built incorrectly anyway) |

---

## Key constants to move from `3a` → `0`

Already in `3a`, need to move verbatim:

```python
BUILD04_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build04_output"
STITCHED_FF  = PROJECT_ROOT / "morphseq_playground" / "built_image_data" / "stitched_FF_images"
REAL_QC_FLAGS = ["frame_flag", "sam2_qc_flag", "no_yolk_flag", "focus_flag",
                 "bubble_flag", "dead_flag", "dead_flag2", "sa_outlier_flag"]
SEQ_QC_IMAGING_PLATE_MAP = { ... }   # the full dict from 3a (verified correct)
SEQ_QC_DATE_WARNINGS     = { ... }   # the warnings dict from 3a
```

The `SEQ_QC_IMAGING_PLATE_MAP` dict in `3a` is already verified correct and should be
copied exactly — do not re-derive it.

---

## What NOT to move

- `well_dispositions.csv` loading and disposition logic → stays in `3a` (human narrative,
  not a filter gate)
- `write_schema_sidecar()` → stays in `3a`
- `write_markdown()` / `MISSING_SEQUENCED_AUDIT.md` generation → stays in `3a`
- All plot functions → stay in `3a`

---

## Stale dispositions note

`well_dispositions.csv` has 9 stale entries (wells that were `truncated_acq` but are
now OK in build04 after reprocessing). These should be pruned from the sidecar:
```
20260324_cep290_18hpf_24hpf_plate02: G11, H01, H05, H06, H08
20260324_cep290_18hpf_plate01:       G09, H02, H06, H09
```
Not blocking for the migration, but should be cleaned up.
