# Tonight's second prompt — real-data reconnaissance (read-only, parallel-safe)

Fill `<PIPELINE_OUTPUT_ROOT>` and `<EXPERIMENT_IDS>`. Branch `slice/recon`. Touches no `src/` file,
so it runs safely alongside Phase 0.

---

Read `docs/refactor/AGENTS.md` and `docs/refactor/DECISIONS.md`. This is a **read-only survey**. Write nothing
under `<PIPELINE_OUTPUT_ROOT>`, modify nothing in `src/`. Deliver a script under `scripts/recon/`
and a report at `reports/PIPELINE_RECON.md`.

Pipeline outputs for experiment `{exp}` live at:
`object_extraction/{exp}/snips/{exp}_snip_inventory.csv` ·
`feature_extraction/{exp}/stage_predictions/{exp}_stage_predictions.csv` ·
`quality_control/{exp}/snip_qc/{exp}_snip_qc.parquet` ·
`acquisition/{exp}/ingest_metadata/plate_metadata.csv`.
Treat `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` as authoritative over this list.

Report, with numbers, for `<EXPERIMENT_IDS>`:

1. **Availability and schema drift.** Per experiment × source: present/absent, row count, column
   list, and columns differing from the newest experiment. Flag any inventory lacking
   `source_micrometers_per_pixel` / `snip_micrometers_per_pixel`.
2. **Cohort size.** Total snips; counts by experiment, channel, and `physical_embryo_id`;
   distribution of frames per `physical_embryo_id`. Exact numbers.
3. **Identity integrity.** Duplicate `snip_id`s; nulls in the identity spine; disagreement with the
   canonical grammar (`physical_embryo_id = {well_id}_e{n}`, `embryo_id = {pid}_{channel_id}`,
   `snip_id = {embryo_id}_t{t}`). Report, do not fix.
4. **Image product type.** Value sets for `image_product_type` and `projection_method` across the
   cohort. **Critical:** if more than one product type is present, check whether `snip_id` remains
   unique — the ID grammar has no focus axis, so collisions are possible. Also report the
   `z_position` distribution.
5. **Masks.** Masks are colocated with their snips. Discover the naming convention, report it
   explicitly, and give per-experiment coverage — how many snips have no locatable mask.
6. **Paths.** Fraction of `processed_snip_path` values resolving under `output_root`, absolute, or
   missing on disk. Open ~100 and confirm 8-bit non-interlaced grayscale at pipeline (H, W) = (576, 256).
7. **QC.** `use_snip` pass rate overall and per experiment; `qc_fail_reasons` token histogram;
   `is_valid_snip` × `use_snip` cross-tab; count of inventory snips with **no** QC row. State
   explicitly whether strict `is_valid_snip & use_snip` removes an *experiment-correlated* fraction
   rather than a uniform one.
8. **Staging.** `stage_prediction_status` distribution; finite `predicted_stage_hpf` coverage overall
   and per experiment; the stage histogram; and survivors of the combined metric gate
   (valid ∧ QC-pass ∧ status=="predicted" ∧ finite stage).
9. **Metric-group candidates.** Every plate-metadata column present, with null rate, cardinality, and
   value set. `short_pert_name` is **not** guaranteed — report whether it exists at all. For each
   plausible candidate, report how the cohort's values map onto the existing curated class labels:
   matched, cohort values with no class, classes unused by the cohort.
10. **Pixel scale and intensity.** Distribution of the µm/px columns where present; flag any **mixed**
    scales. On ~200 random snips per experiment, per-image mean, std, min, max, and saturated-pixel
    fraction, presented **per experiment side by side**. The question being answered: are intensity
    statistics experiment-correlated strongly enough to be learnable as a batch effect?
11. **I/O throughput.** This decides whether we downsample on the fly or need a cache. Sample a few
    hundred snips from the real mount and report: cold read+decode time per image, warm read+decode
    time, and the 576×256 → 288×128 resize cost measured separately. Then state images/sec/worker and
    what that implies for a batch of 64 in metric mode (**two views per item = 128 reads per batch**).
    Call out specifically whether per-file latency or decode CPU dominates.
12. **Group-split feasibility.** With group-disjoint splits at `physical_embryo_id` and an ~80/10/10
    target: achievable ratios, and whether any candidate metric group would have too few embryos in a
    split to form legal positive pairs.

`snip_qc` is Parquet. If no Parquet engine is available, report that as a blocking finding rather
than skipping QC analysis.

**In your summary:** the three findings most likely to change our design; anything contradicting
`docs/refactor/NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md`; and any item you could not measure, with the reason.
