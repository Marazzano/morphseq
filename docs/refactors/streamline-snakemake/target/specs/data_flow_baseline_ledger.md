# Data-Flow Baseline Ledger (Tier 0) — "what does green mean today?"

**Status:** Tier-0 deliverable of `data_flow_test_plan.md` §2. Consolidates the last real-data proof
per spine step from the dated snapshots in `current_state_and_next_steps.md` — **not re-derived**.
**Compiled:** 2026-06-26, against the live Snakefile + on-disk `data_pipeline_output/`.

The ledger's job (§2) is to make the **true gaps** explicit and stop us re-proving what's proven.
"Last real-data proof" = the most recent commit/snapshot that ran the step on real bytes (not a
unit test, not `snakemake -n`). "Scale" = experiment / wells / timepoints it ran at.

---

## The spine ledger (ingest_* → … → snip_qc)

| step_key | rule(s) | last real-data proof | scale (exp / wells / tp) | scope(s) | CPU/GPU | gap |
|---|---|---|---|---|---|---|
| `ingest_plate_metadata` | `ingest_plate_metadata` | front_half smokes (≤2026-06-26) | 20250912 / 95 disc / — | YX1 | CPU | none for YX1; Keyence proven on `20250612_24hpf` (5184e2b2) |
| `ingest_scope_metadata` | `ingest_scope_metadata` | front_half smokes; Keyence 5184e2b2 | 20250912 / exp-grain; 20250612 / exp-grain | YX1+Keyence | CPU | none |
| `map_positions_to_wells` | `map_positions_to_wells` | front_half smokes | 20250912 / 95 discovered | YX1+Keyence | CPU | none — 95 wells discovered, global `well_id` |
| `apply_position_to_well_mapping` | `apply_position_to_well_mapping` | front_half smokes | 20250912 / exp-grain | YX1+Keyence | CPU | none |
| `discover_wells` (checkpoint) | `discover_wells` | front_half smokes | 20250912 / 95 / — | agnostic | CPU | none — `discovered_wells.txt`, global `well_id` |
| `materialize_image_product_for_well` | `materialize_image_product_for_well` | z_stack/projection smokes (5246407b, 5184e2b2) | 20250912 / B01,C01 / 1–3 tp; 20250612 / A01,A02 / 1 tp (mosaic) | YX1+Keyence | GPU for focus_stack proj; z_stack CPU | YX1 full-scale (95 wells/all tp) never run; Keyence focus_stack OOM'd on CPU (needs GPU) |
| `assemble_well_frame_inventory` / `validate_frame_inventory_for_well` | same | 5246407b additive smoke | 20250912 / B01 / 1 tp | agnostic | CPU | none for B01 |
| `merge_frame_inventory` | `merge_frame_inventory` | multi-well mixed-product test f6e4abe3 (integration, real fns) | 20250912 / B01+C01 | agnostic | CPU | merge seam proven 2-well |
| `frame_detections_per_well` | `frame_detections_per_well` | prior B01 smoke (on-disk shard exists) | 20250912 / B01 / multi-tp (stale) | agnostic | GPU (GroundingDINO) | **no continuous run to snip_qc**; CPU-at-1tp unproven until Tier 1 |
| `frame_masks_per_well` / `validate_frame_masks_for_well` / `merge_frame_masks` | frame_masks.smk | a4f284d9 + validate fix (≤2026-06-22) | 20250912 / B01 / multi-tp (stale) | agnostic | GPU (SAM2) | **`rule all` STOPS here**; everything past is unproven on continuous real data |
| `build_physical_embryo_registry_for_well` (+validate/merge) | physical_embryo_registry.smk | registry shipped (S1–S4, ≤287f5eb0) | 20250912 / B01 (shard on disk) | agnostic | CPU | unit-green + shard exists; no continuous run past frame_masks |
| `snip_processing_per_well` (+validate/merge) | snip_processing.smk | B01 shard on disk | 20250912 / B01 / 3 snips (stale) | agnostic | CPU | same — no continuous real-data run |
| `build_snip_auxiliary_masks_for_well` (+validate) | snip_auxiliary_masks.smk | B01 shard on disk | 20250912 / B01 | agnostic | GPU (UNet; `unet_snip.device`) | CPU-forced via config; continuous run unproven |
| `build_mask_geometry_for_well` (+validate/merge) | mask_geometry.smk | B01 shard on disk | 20250912 / B01 | agnostic | CPU | unit-green; no continuous run |
| `build_pose_kinematics_for_well` | pose_kinematics.smk | B01 shard on disk | 20250912 / B01 | agnostic | CPU | unit-green; no continuous run |
| `build_fraction_alive_for_well` (+validate) | fraction_alive.smk | B01 shard on disk | 20250912 / B01 | agnostic | CPU | unit-green; depends on aux masks (snip migration) |
| `build_stage_predictions_for_well` (+validate) | stage_predictions.smk | B01 shard on disk | 20250912 / B01 | agnostic | CPU/GPU (classifier) | unit-green; no continuous run |
| `consolidated_features` (+validate/merge) | consolidated_features.smk | B01 shard on disk | 20250912 / B01 | agnostic | CPU | unit-green; no continuous run |
| `build_surface_area_qc_for_well` (+validate) | surface_area_qc.smk | ffd47280 (built) | — / unit | agnostic | CPU | unit-green only; **never on continuous real data** |
| `build_mask_quality_qc_for_well` (+validate) | mask_quality_qc.smk | 89e3b3fa (built) | — / unit | agnostic | CPU | unit-green only; never on continuous real data |
| `build_death_detection_for_well` (+validate) | death_detection.smk | fd729289 (built) | — / unit | agnostic | CPU | unit-green only; never on continuous real data |
| `snip_qc` (resolved_sources → build → validate → merge) | snip_qc.smk | 91a05ad6 (built) + `snakemake -n` 43-job plan | — / unit + dry-run | agnostic | CPU | **TERMINAL — never reached on real data.** No named target requested it until `through_line` (2026-06-26). |

---

## The true gaps (§2 known-gaps list — confirmed against disk)

1. **Everything past merged `frame_masks` has NO continuous real-data run.** ✅ CONFIRMED.
   `rule all` stops at merged `frame_masks` (Snakefile). Registry → snip → features → QC → snip_qc
   are all unit-green + `snakemake -n`-green, and B01 per-well *shards exist on disk* from prior
   ad-hoc smokes — but no single run ever flowed raw → `snip_qc` continuously. The Tier-1
   `through_line` run (2026-06-26) is the first.
2. **Full YX1 at scale (95 wells / all timepoints) never run.** ✅ CONFIRMED — only 1–2 wells / 1–3 tp.
   This is the Tier-2 (WIDTH) gap, behind the GPU gate.
3. **Keyence proven only on the front half (A01+A02), never through the agnostic back half.** ✅
   CONFIRMED (5184e2b2 stops at front-half materialization). This is the Tier-3 (AGNOSTIC) gap.

### On-disk inconsistency to be aware of (the §2 "may have been overwritten" warning, made concrete)
The B01 per-well shards on disk as of 2026-06-26 are **internally inconsistent** (laid down by
different earlier smokes): `frame_inventory` carries 48 frame rows (a multi-timepoint smoke), while
`frame_detections` / `frame_masks` / `snip_inventory` carry only 3 rows each (a 1-tp smoke). This is
exactly why Tier 1 is run as a **continuous from-raw rebuild** (mdcolon, 2026-06-26) rather than a
tail-only top-up — so every shard is regenerated coherently at one timepoint in a single pass, and a
green snip_qc verdict actually reflects bytes that flowed the whole length together.

### Scope notes
- The microscope dissolves at `apply_position_to_well_mapping` (scope metadata converges); everything
  from `discover_wells` onward is **microscope-agnostic** ("scope = agnostic" above) — the invariant
  Tier 3 exists to prove on real Keyence bytes.
- "GPU" entries are the *production* device; at Tier-1 scale (1 well / 1 tp) the plan (§73) asserts
  these legs run on CPU, and the config forces `unet_snip.device: cpu`. The GPU gate guards the
  Tier-2 width run, where parallel wells make these legs GPU-bound.
