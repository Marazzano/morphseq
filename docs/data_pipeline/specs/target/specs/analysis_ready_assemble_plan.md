# analysis_ready — assemble plan (final-merge product)

Status: PLAN (2026-07-03). Wires the existing stub
`src/data_pipeline/analysis_ready/__init__.py` into a real DAG product. No code yet.

## Goal

One **analysis-ready table, one row per `snip_id`**, that a notebook / embedding job can load
without re-doing joins. It fans in the wide per-snip feature columns + the `snip_qc` verdict + the
plate metadata, all keyed on the identity spine. `snip_qc` stays the proven through-line terminal;
this is an **optional downstream product**, not on the critical path.

## Doctrine (from the stub — do not violate)

- **Import spine & payload columns from their mint sites; never re-declare a column list here.**
- The spine is carried by every source table already; joins are spine-keyed, not ad-hoc.

## Confirmed identifiers (read from source, not guessed)

- Spine grain `snip_id` expands to: `snip_id, embryo_id, physical_embryo_id, well_id, experiment_id`
  (`snip_identity_contract.SNIP_ID_SPINE_COLUMNS`, additive one-ID-per-level), plus frame provenance
  `image_id, time_index, channel_id` (`SNIP_FRAME_PROVENANCE_COLUMNS`).
- snip_qc payload: `use_snip` (bool), `qc_fail_reasons` (str) — `snip_qc.contract.SNIP_QC_PAYLOAD_COLUMNS`.
- curvature payload includes `baseline_deviation_normalized`, `total_length_um`,
  `baseline_deviation_um`, … (`curvature_metrics.contract.CURVATURE_PAYLOAD_COLUMNS`).
- z_mu_b embeddings: `legacy_embeddings` table, keyed on `snip_id`.
- plate_metadata required fields: `genotype, start_age_hpf, temperature` — keyed on **`well_id`**
  (`plate_metadata_contract.REQUIRED_PLATE_METADATA_FIELDS`).

## Exhaustive column list (one row per snip_id)

Spine (always): `snip_id, embryo_id, physical_embryo_id, well_id, experiment_id, image_id,
time_index, channel_id`. NOTE: the spine is 5 IDs, additive one-per-level — `embryo_id` sits
between `physical_embryo_id` and `snip_id` and travels with every snip.

Feature payloads (left-join on `snip_id`):
- curvature_metrics: `total_length_um, mean_curvature_per_um, baseline_deviation_um,
  baseline_deviation_normalized, max_baseline_deviation_um, baseline_deviation_std_um,
  arc_length_ratio, chord_length_um, keypoint_deviation_q1_um, keypoint_deviation_mid_um,
  keypoint_deviation_q3_um, centerline_point_count`
- stage_predictions: `predicted_stage_hpf, model_version`
- mask_geometry: `area_um2, perimeter_um, length_um, width_um, centroid_x_um, centroid_y_um`
- pose_kinematics: `orientation_angle, bbox_width_um, bbox_height_um, displacement_um,
  speed_um_per_s, delta_x_um, delta_y_um, delta_time_s`
- fraction_alive: `fraction_alive`
- legacy_embeddings: `embedding_model_name` + dynamic `z_mu_*` block (`z_mu_0 … z_mu_b …`, count is
  model-dependent; matched by `z_mu_` prefix, NOT a fixed list) + optional `z_sigma_*`.

QC (on `snip_id`): `use_snip, qc_fail_reasons`.

Plate metadata (broadcast `well_id`→snips; real 12-col schema minus spine dups):
`well_index, medium, mold_type, genotype, chem_perturbation, start_age_hpf, series_number_map,
start_age_morph, embryos_per_well, temperature`.

### Real broadcast example
From `data_pipeline_output/acquisition/20250912/ingest_metadata/plate_metadata.csv`, well
`20250912_A01`: `genotype=wik-ab, chem_perturbation=tri_1-15, start_age_hpf=11, temperature=30,
medium=MC05`. That single plate row copies onto EVERY snip with `well_id == 20250912_A01` (each
embryo × timepoint). A snip `20250912_A01_e01_t0007` carries its own features
(`baseline_deviation_normalized`, `predicted_stage_hpf`, `z_mu_b`, …) plus the broadcast plate row.

## The merge (fan-in on the spine)

All feature/QC sources are **1 row per `snip_id`** → straight left-joins on `snip_id`:

```
spine (snip grain)                       ← base frame, defines the snip_id universe
  ⟵ curvature_metrics      on snip_id    ← baseline_deviation_normalized, total_length_um, ...
  ⟵ legacy_embeddings      on snip_id    ← z_mu_b columns
  ⟵ mask_geometry          on snip_id    ) optional, per Feature-scope decision
  ⟵ pose_kinematics        on snip_id    )
  ⟵ fraction_alive         on snip_id    )
  ⟵ stage_predictions      on snip_id    )
  ⟵ snip_qc                on snip_id    ← use_snip, qc_fail_reasons
  ⟵ plate_metadata         on WELL_ID    ← genotype/age/temp/... broadcast well→snips
```

**The one structural wrinkle:** plate_metadata is keyed on `well_id`, not `snip_id`. It's a
`well_id → snip_id` broadcast (one well's row copied to every snip in that well). The spine already
carries `well_id`, so this is a clean left-join on `well_id`. **Precedent to copy exactly:**
`stage_predictions/compute.py` already does `plate_metadata_df.set_index("well_id")` and joins by
well — reuse that pattern; don't invent a new one.

## Join contract / invariants

- **Base = spine** (the snip_qc through-line output owns the authoritative snip_id set). Every join
  is a LEFT join onto the base — no feature source may add or drop snips.
- After each join: assert row count unchanged and `snip_id` still unique (fail loud if a source
  table has dup snip_ids — its own contract should already forbid this, so this is a belt check).
- Feature columns are NULLABLE by their own contracts (e.g. curvature nulls when centerline too
  short); analysis_ready preserves the nulls, does not fill.
- plate_metadata join: every `well_id` present in the spine MUST have a plate row (same hard-fail
  stage_predictions enforces) — a missing well is a data error, not a silent NaN.

## DAG wiring (how it enters the Snakemake graph)

analysis_ready is a **merged-level fan-in join**, NOT a per-well build. Every input is already at
experiment grain (merged feature CSVs, merged snip_qc, merged latents parquet) or is broadcast from
experiment grain (plate_metadata by well_id). So it does NOT follow the build-per-well→validate→merge
pattern — faking a per-well pass would re-read the experiment-level latents/plate for every well.
Shape: ONE rule, merged inputs → one `{experiment}_analysis_ready.parquet`.

Wiring steps (each mirrors an existing precedent):

1. **Register the step in `orchestration/paths.py`** — add an `analysis_ready` entry to
   `PIPELINE_STEPS` with `product_dir: "analysis_ready"`, a single artifact
   `analysis_ready: {PATH_MODE_MERGED: "{experiment_id}_analysis_ready.parquet"}`. Only MERGED mode
   (no per-well). This is what makes `rule_artifact("analysis_ready", ...)` resolve.

2. **`tasks.py` subcommand** — add an `analysis-ready` dispatch that calls
   `analysis_ready.entrypoint`, taking `--curvature-csv --stage-predictions-csv --mask-geometry-csv
   --pose-kinematics-csv --fraction-alive-csv --latents-parquet --snip-qc-parquet
   --plate-metadata-csv --output-parquet`. (Mirrors how `stage-predictions` dispatches.)

3. **`rules/analysis_ready.smk`** — one rule `build_analysis_ready`:
   - `input:` the MERGED artifact of each feature step
     (`rule_artifact("curvature_metrics", "curvature_metrics", "{experiment}", path_mode=PATH_MODE_MERGED)`,
     same for stage_predictions/mask_geometry/pose_kinematics/fraction_alive/latents), the merged
     snip_qc verdict, and `PLATE_METADATA_CSV`. Each merged input's build already gates on its own
     per-well validate→merge chain, so analysis_ready transitively waits for all features.
   - `output:` `rule_artifact("analysis_ready", "analysis_ready", "{experiment}", path_mode=PATH_MODE_MERGED)`.
   - `shell:` the `analysis-ready` task.
   - (optional) a `validate_analysis_ready` rule if we want a `.validated` sentinel — but since every
     input is already validated upstream, the join's own row-count/uniqueness asserts may suffice.

4. **`include:` in the Snakefile** — add
   `include: str(WORKFLOW_DIR / "rules" / "analysis_ready.smk")` after `stage_reports.smk` (line ~348).

5. **Target membership** — do NOT add to `through_line` (snip_qc stays the proven terminal). Add an
   OPTIONAL named target `analysis_ready` (like `reports`) that requests the merged parquet per
   discovered experiment, so a user opts in with `snakemake analysis_ready`.

## Files to create

1. `analysis_ready/contract.py` — composes the column set by IMPORTING each product's contract
   tuple (spine + chosen feature payloads + `SNIP_QC_PAYLOAD_COLUMNS` + plate fields). Extend
   `ANALYSIS_READY_SPINE_COLUMNS` in `__init__.py`; do not inline a literal list.
2. `analysis_ready/assemble.py` — the fan-in join described above; pure pandas, reads each product's
   materialized table, returns the wide frame.
3. `analysis_ready/entrypoint.py` — thin CLI wrapper (mirror `stage_predictions/entrypoint.py`).
4. Snakemake rule `rules/analysis_ready.smk` — depends on all included feature-step outputs +
   snip_qc + plate_metadata; NOT in the `through_line` target (optional product).

## Open decisions (deferred to user — asked, not yet answered)

- **Feature scope**: minimum is curvature_metrics + legacy_embeddings (the columns you named) +
  snip_qc verdict. Full set adds mask_geometry / pose_kinematics / fraction_alive / stage_predictions.
- **Plate columns**: "all plate cols broadcast by well_id" (widest, matches your ask) vs curated
  subset vs well_id-only. Default assumption for this plan = **all plate columns, broadcast**.

## Non-goals

- No row filtering by `use_snip` — carry the verdict, let notebooks filter. (Keeps the table honest
  and re-usable.)
- No column re-declaration, no new identity vocabulary, no plate reshaping beyond the well-broadcast.
