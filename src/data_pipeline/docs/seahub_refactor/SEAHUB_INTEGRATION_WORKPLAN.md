# SeaHub → morphseq pipeline: WORKPLAN (executable)

**Read `SEAHUB_INTEGRATION_DESIGN.md` first.** It holds the frozen decisions and contracts this plan
executes. Where this plan and the design doc disagree, the design doc wins (or stop and flag it).

**Guiding constraints (from the repo owner — do not violate):**
- Work on `main`; do NOT create branches; do NOT push (the owner pushes).
- Never touch raw image data or source SeaHub data. All new outputs go under
  `results/nlammers/20260723_seahub/` and the pipeline `acquisition/` output tree.
- Do NOT touch mdcolon-owned files/dirs.
- Do NOT run materialization on a login node (OOM risk) — submit to the cluster (see Phase 5).
- The back half must remain **unmodified** (Scheme A). If you find yourself editing back-half code,
  stop — that means a contract was misread.

**Architecture in one line:** build a standalone front-half materializer that turns each SeaHub FOV
into 8 single-frame synthetic wells (detect → pad → write + per-well plate-map + scaffolded
inventories), then run the existing back half over the resulting synthetic experiments.

---

## Phase 0 — Confirm contracts & reproduce the ingest table
**Goal:** no surprises later; a single per-embryo ingest table drives everything.

0.1 Read every file in DESIGN §7. Confirm the id grammar and the `broadcast_plate_columns` allow-list.
0.2 Reproduce/parse `outputs/image_metadata_reconciliation.csv`; filter to `image_role ==
    'eight_embryo_fov'`. Join to the per-embryo boxes (`embryo_manifest.csv` box columns
    `crop_x1_px..crop_y2_px`, `embryo_position`). For FOVs not yet detected, detection runs in Phase 2 —
    Phase 0 only needs the FOV-level table + the box schema.
0.3 Apply the ingest-scope rule (DESIGN §3.2): keep FOVs with `(experiment_id AND stage_hpf AND
    perturbation_parsed)`; classify each as `has_seq_link` true/false. Write the true drop set to
    `outputs/integration/dropped_fovs.csv` with a reason column (no silent drops).
0.4 Apply the owner-supplied **somite→hpf table** to recover the `12s`-style FOVs (DESIGN §3.4).
    If the table is not yet available, ingest them with `stage_hpf` null + raw token retained, and
    list them in `dropped_fovs.csv` as `deferred_somite` so they're trivially re-includable.

**Acceptance:** `outputs/integration/embryo_ingest.csv` — one row per embryo to be materialized,
carrying every source field in DESIGN §3.3, plus the true drop list. Row count ≈ 8 × ingested FOVs
(~10,900 expected). Print the tier counts and confirm they match DESIGN §3.2 (±somite recovery).

---

## Phase 1 — Mint identity & build pipeline-standard metadata
**Goal:** turn `embryo_ingest.csv` into per-experiment metadata tables that pass the ingest validators.

1.1 Group by SeaHub experiment; pack into ≤96-well plates in the deterministic order (DESIGN §2.4).
    Mint `experiment_id`, `well_index`, `well_id`, `physical_embryo_id`, `image_id` **only** via the
    `shared/identifiers` constructors — never string-format by hand.
1.2 For each synthetic `experiment_id`, write to `acquisition/{experiment_id}/`:
    - `ingest_metadata/scope_metadata_mapped.csv` — one row per `image_id`, matching the 29-column
      schema of a real `scope_metadata_mapped.csv` (diff columns against `20240813_24hpf`). Single
      frame: `time_int=0`, `channel_id=BF`. Fill unknowns explicitly (µm/px null per DESIGN §6).
    - `ingest_metadata/plate_metadata.csv` — one row per `well_index` with the per-well biology map
      (DESIGN §3.3), including `image_kind='single_z'`, `z_position` null, `source_scope='seahub'`,
      `collection_name` + seq-link fields, `metadata_match_status`.
    - `well_identities/position_well_mapping.csv` and `discovered_wells.txt` (global well_ids).
1.3 Persist the full reverse map `outputs/integration/well_provenance.csv`
    (`well_id → source FOV, filename, fov_position, box px`) for auditing/debugging.

**Acceptance:** for one sample experiment, the metadata tables **pass the existing validators**
(run the ingest/plate/position validators against the written files; `.validated` sidecars appear).
`validate_well_id`/`validate_well_index`/`parse_image_id` succeed on a random sample of 50 rows.

---

## Phase 2 — Materialize the padded embryo frames
**Goal:** write the "dummy FF" grayscale frames the back half will consume.

2.1 Reuse the GroundingDINO detection + within-FOV position assignment from `seahub_workflow.py`
    (thresholds box=0.15, text=0.10, NMS→8). Do NOT reimplement detection. Run on GPU for the full
    corpus (CPU is fine only for the 3-experiment smoke test).
2.2 Compute the standard padded size once over ALL ingested boxes (DESIGN §5): max box W/H → round up.
    Pad (never downscale) each crop centered to that `(H, W)`, grayscale, neutral background fill.
2.3 Write each embryo to
    `acquisition/{experiment_id}/materialized_images/{well_id}/BF/projection/{well_id}_BF_t0000.jpg`
    (JPEG quality from `config.yaml`). Projection slot, projection-grammar image_id (no `_z` token) —
    z-slice truth lives only in metadata (DESIGN §4).
2.4 FOVs whose detection ≠ 8 embryos: write a QC preview, do NOT emit a partial well set, and log them
    to `outputs/integration/detection_failures.csv` for review (mirrors existing count-mismatch QC).

**Acceptance:** materialized frame count == expected embryo count (minus detection failures).
Spot-check 10 frames: correct dims, grayscale, embryo centered with margin, background clean.
Contact sheet under `outputs/integration/materialization_qc/` for visual review.

---

## Phase 3 — Wire into the front half (minimal change) & validate
**Goal:** make the front-half validator accept the drop-in without editing back-half code.

3.1 **Preferred (minimal-change) route — drop-in scaffold:** use
    `acquisition/metadata_ingest/frame_inventory/scaffold_dropin_inventory.py` to generate the
    `frame_inventory` product inventories for the externally-materialized images, then run the
    front-half validation for one synthetic experiment.
    - If the scaffolder needs fields (`experiment_id`/`well_index`/`µm_per_pixel`), fill from Phase 1
      (µm/px null is acceptable per DESIGN §6; confirm the validator tolerates null — if it hard-requires
      a number, set a documented sentinel and record it in DESIGN §6).
3.2 **Fallback route — real scope:** only if drop-in cannot satisfy the front-half product contract,
    add a thin `seahub` materialization scope modeled on `scope/yx1/materialize_well_yx1.py` and
    register it in `scope_resolver_for_materialization_plan.py`. This is heavier; prefer 3.1.
3.3 Run `front_half_products` for ONE synthetic experiment end-to-end (materialize → validate). Confirm
    it reports success and produces the same product/inventory shape as a real experiment (diff the
    tree against `20240813_24hpf`).

**Acceptance:** `front_half_products` completes for one synthetic experiment with a passing validation
and **zero edits to back-half code**. Capture the exact invocation for Phase 5.

---

## Phase 4 — Back-half smoke test (3 experiments, prove seamlessness)
**Goal:** prove an embryo flows all the way to `analysis_ready` with correct metadata, on GPU.

4.1 Use the 3 validation experiments already vetted (GENE6/24hpf, GENE13/48hpf, CHEM17/72hpf).
    Run the **full back half** (detection → segmentation → snips → features → embeddings →
    analysis_ready) via the standard orchestrator, unmodified, on GPU.
    - Embeddings must run on GPU — confirm the embeddings `device: cuda` config (the one-line fix noted
      in prior sessions) is set before this run.
    - Use the working VAE env (`vae-env-cluster`) per the recorded env contract.
4.2 Tune the Phase-2 background fill value here if segmentation grabs the padding instead of the embryo.
4.3 Open the resulting `analysis_ready` parquet and verify per-embryo rows carry:
    `source_scope='seahub'`, `image_kind='single_z'`, `z_position` null, correct `genotype`/
    `chem_perturbation`, `stage_hpf`, `collection_name`, and a latent embedding (`z_mu_*`).

**Acceptance:** ≥ (24 − detection_failures) SeaHub snips present in `analysis_ready` with valid latents
and the metadata above. `focus_flag`/`motion_blur_flag` may be True (expected; annotate-only).
Save a one-page result summary to `outputs/integration/smoke_test_report.md`.

---

## Phase 5 — Scale to the full corpus
**Goal:** run all ~114 synthetic plates through, cluster-submitted, resumable, cheap-to-kill.

5.1 Generate the full experiment manifest (one synthetic `experiment_id` per line).
5.2 Front half: materialization is GPU (GroundingDINO) — submit as an SGE array (one task per
    experiment), NOT on a login node. Size `mfree` from the smoke-test peak RSS; GPU per task.
5.3 Back half: submit as an SGE array over the synthetic experiments, matching the existing back-half
    submission pattern (6 GPUs max unless told otherwise), `--keep-going`, per-experiment work dirs
    with the stale-lock guard.
5.4 Prioritize the 3 smoke-test experiments first so a systemic failure is caught cheaply before the
    full fan-out.

**Acceptance:** all non-failed synthetic experiments reach `analysis_ready`; a final
`outputs/integration/integration_summary.csv` reports per-experiment embryo counts in vs. snips out,
detection failures, and drop reasons. No silent coverage gaps.

---

## Cross-cutting acceptance (the definition of done)
- SeaHub embryos are queryable in `analysis_ready` alongside Keyence/YX1 data, keyed by the standard
  identity spine, with per-embryo genotype/perturbation/stage and the sequencing `collection_name`.
- The training loader can select SeaHub via `source_scope=='seahub'`, read `image_kind`/`z_position`,
  and ignore the false `focus_flag`/`motion_blur_flag`.
- **No back-half code was modified.** Front-half additions are isolated to the drop-in materializer
  (or a thin new scope) + this results directory.
- Every dropped/failed FOV is enumerated with a reason (no silent truncation).

## Open items to confirm with the owner (non-blocking to start Phase 0–2)
- Somite→hpf conversion table (Phase 0.4).
- Whether the refactored VAE loader's z-metadata column names are exactly `image_kind`/`z_position`
  (rename at Phase 1 if different — values are fixed, names may not be).
- Final `{DATE}` token for `experiment_id` if not `20260723`.
