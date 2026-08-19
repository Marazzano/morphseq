# Resolved training manifest — shared schema contract

**Version:** 1.0 · **Date:** 2026-08-19
**Status:** Binding interface between the manifest adapter (slice 1A) and its consumers (1B, 1C).
All three slices are written against this document so they can be developed in parallel.

One row per `snip_id`. Row order **is** dataset order. No second glob, no `ImageFolder`, no filename
parsing, no parallel array indexed alongside it.

---

## Provenance of every column

Measured against the real cohort on 2026-08-19: 699,505 inventory rows across 133 experiments.

### From `snip_inventory` (schemas I01/I02 — same 22 columns, order differs only)

| column | requirement | notes |
|---|---|---|
| `snip_id` | always | unique; 0 duplicates measured across 699,505 rows |
| `embryo_id`, `physical_embryo_id`, `experiment_id`, `well_id` | always | 0 nulls, 0 grammar violations measured |
| `image_id`, `time_index`, `channel_id` | always | frame provenance; `channel_id` is `BF` for the entire cohort |
| `processed_snip_path` | always | relative to `output_root`; 100% resolve and open |
| `embryo_mask_snip_path` | always | **the mask path — it is a column, do not go looking for it.** Convention `<snip_id>_embryo.png`, colocated; 699,505/699,505 coverage |
| `image_path` | always | source frame |
| `is_valid_snip` | always | **True for every row in the cohort — a no-op gate.** Keep it defensively; do not remove; do not assume it filters anything |
| `mask_id`, `track_id`, `embryo_mask` | carry | provenance |
| `crop_{x,y}_{min,max}_px`, `crop_width_px`, `crop_height_px` | carry | crop geometry |
| `error_message` | carry | |

### From `stage_predictions` (S01 has `stage_prediction_status`; S02 does not)

| column | requirement | notes |
|---|---|---|
| `predicted_stage_hpf` | metric mode | 531,337/699,505 finite |
| `model_version` | carry | |
| `stage_status` | **derived, three-state** | see below |

**`stage_status` must be three-state, not boolean.** Schema S02 (`20260417_irx_pilot`,
`20260418_irx_pilot`) carries finite stages but *no status column*. Treating that as failure
discards 20,219 snips that have perfectly good stages, 8,001 of which also pass QC. Values:

- `predicted` — status column present and equal to `"predicted"` (511,118 rows)
- `unavailable` — **no status column in the source schema**, stage may still be finite (20,219 rows)
- anything else — the source's literal status string (e.g. `missing_start_age_hpf`, 898 rows)

The default metric cohort requires `predicted`. `unavailable` is a named, configurable inclusion —
not a silent one, and not a silent exclusion either.

### From `snip_qc` (Q01 = 101 experiments, Q02 = 2, Q03 = 2)

| column | requirement | notes |
|---|---|---|
| `use_snip` | when QC present | 0 nulls measured; 185,204/531,902 pass (34.8%) |
| `qc_fail_reasons` | when QC present | pipe-delimited |
| `persistence_dead_flag`, `viability_dead_flag`, `focus_flag`, `discontinuous_mask_flag`, `edge_flag`, `overlapping_mask_flag`, `motion_blur_flag`, `sa_outlier_flag` | Q01/Q03 only | **carry all of them** |
| `focus_qc_applicability`, `motion_blur_qc_applicability`, `surface_area_qc_applicability` | Q01/Q03 | |
| `death_detection_qc_applicability` | Q03 only | |
| `qc_status` | **derived, three-state** | see below |

**Carry the individual flags, not just the verdict.** `sa_outlier_flag` appears in 69.6% of all
failures and is the only *morphometric* criterion in an otherwise acquisition/segmentation set; it
is suspected of discarding usable images (`PIPELINE_TASKS.md` G1). Cohort policy must therefore be a
named predicate over flags that can change without re-plumbing the manifest.

**`qc_status` must be three-state:** `evaluated` (105 experiments, 531,902 rows) · `no_artifact`
(28 experiments, 167,603 rows) · `row_missing` (0 measured, but must not be silently conflated).
`no_artifact` is not the same as failing QC and must never be collapsed into it.

**Schema Q02 carries only `use_snip` and `qc_fail_reasons`** — no per-flag columns. A per-flag policy
applied to a Q02 experiment must **fail loudly**, naming the experiment. It must not silently fall
back to `use_snip`.

### From `plate_metadata` (38 distinct schemas, P01–P38)

Join many-to-one on `well_id`; exactly one plate row per selected well.

| column | requirement | notes |
|---|---|---|
| `genotype` | carry | 10.21% null, 99 values, known duplicate spellings — do **not** normalise in core |
| `start_age_hpf`, `temperature`, `medium` | carry | |
| `strain` | carry when present | 63.66% null, 3 values |
| `chem_perturbation` | carry when present | 72.40% null, 53 values |
| all other plate columns | carry opportunistically | declare which are optional; never require one that only some schemas have |

`short_pert_name` **does not exist anywhere** in this cohort. Do not look for it, do not synthesise it.

### Derived by the adapter

| column | requirement | notes |
|---|---|---|
| `stage_hpf` | metric mode | materialised from `predicted_stage_hpf` |
| `metric_group` | metric mode | from a **required config-supplied mapping table**; see below |
| `split` | always | `train`/`eval`/`test`, assigned by `physical_embryo_id` |
| `image_product_type` | always | **currently path-inferred, not a column** — every row is `projection`/`focus_stack`. Emit `image_product_type_source` alongside it recording that it was inferred |
| `qc_status`, `stage_status` | always | three-state, above |

### Absent — do not fabricate

`source_micrometers_per_pixel`, `snip_micrometers_per_pixel`, `microscope_id`,
`objective_magnification`, `z_position`, `numerical_aperture`. **None exist in any pipeline schema.**
Reserve the names in the schema declaration so the columns can be adopted without a migration, but
do not derive, default, or impute them.

---

## Cohort filtering

Every filter is explicit, configured, and counted. No implicit defaults.

- **channel** — list, no default. Cohort is entirely `BF`.
- **`image_product_type`** — list, no default. Phase one: `["projection"]`. A cohort containing more
  than one type must fail unless the list names them all.
- **validity** — require `is_valid_snip`. Known no-op; keep it.
- **QC** — a **named, versioned policy**. Default `strict_use_snip`: require `qc_status == "evaluated"`
  and `use_snip`. Yields 185,204 rows. Per-flag policies are first-class alternatives.
- **stage** — metric mode requires finite `stage_hpf` and `stage_status == "predicted"` by default.

**Expected result of the default cohort: 176,466 rows.** An adapter that produces a materially
different number against the same experiment list has a bug — treat this as a regression check.

Every filter must report rows in, rows out, and reason, per experiment, into `cohort_report.json`.

---

## Splits

Assigned at `physical_embryo_id` (13,194 unique). Use a **deterministic content hash**, not a seeded
shuffle: `blake2b(physical_embryo_id) → uint64 → [0,1) → bucket by configured ratios`.

The property that matters is stability **under cohort growth**, not just under row reordering. Adding
an experiment must not move any existing embryo across splits, or every prior checkpoint's validation
set is retroactively contaminated. A seeded shuffle does not give this; a content hash does, for free.

Exact ratio control is lost — hashing gives approximately 80/10/10, not exactly. At this scale that is
immaterial; assert it lands within tolerance and record the achieved fractions.

---

## `metric_group`

Core **must not** infer this. It comes from a required config-supplied mapping table at well or
sample grain, and any selected non-null value not covered by the table is a **hard failure naming the
uncovered values**.

Measured reality: no plate column maps onto the existing curated class vocabulary. Exact string
matching gives 0–1 of 48; order-insensitive token matching recovers 11. The curated vocabulary is
composite (`{perturbation}_{background}`, and `{chem}_{dose}_{start_age}` for chemical classes) and
the background field is 63.66% null. **The mapping table's content is a pending science decision.**
Build the mechanism; ship a stub table; do not guess the semantics.
