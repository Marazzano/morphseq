# HANDOFF — curvature_metrics_report wiring (overlay bug FIXED)

**Status:** ✅ Report wired into DAG + overlay bug fixed and visually confirmed. Branch
`mdcolon/20260222_docs_snakemake_remake`. Not yet committed. Snakemake dry-run still blocked (no
conda env has snakemake — see memory `project_two_tree_reconciliation`); validated via direct
Python/CLI instead.

## What this task was
Wire `curvature_metrics_report` into the Snakemake DAG (only `compute` existed before). Prototype at
`results/mcolon/20260702_qc_reporting_v1/report_scripts/curvature_metrics/report.py` became the real
`src/data_pipeline/feature_extraction/curvature_metrics/report.py`.

## The bug that was fixed (was: gallery rendered but spine overlay never drew)
**Root cause:** report decoded the full-frame RLE mask (`frame_masks.mask_rle`, ~2189²) and tried to
crop+resize it into snip space. But `snip_processing` doesn't just crop — image AND mask go through
`extract_embryo_crop → apply_rotation_to_snip → crop_to_embryo_bounds` TOGETHER
(`run_snip_processing.py:189-210`). The crop+resize path ignored the **rotation** and re-skeletonized
an edge-clipped mask → different/degenerate centerline → spline landed out of frame → nothing drew.

**Fix:** report now reads `embryo_mask_snip_path` from snip_inventory — the per-snip embryo mask
snip_processing already carried through the SAME crop+rotate as the snip image, pixel-aligned by
construction (same source the prototype used and `fraction_alive/compute.py` consumes). Spine computed
on that snip-space mask lands directly on the snip: no offset/rotate/resize math.

## Files changed (all on branch, uncommitted)
- `src/data_pipeline/feature_extraction/curvature_metrics/report.py` — `_overlay_fn` now loads
  `embryo_mask_snip_path` directly (`_load_snip_mask`); dropped `frame_masks_csv` param + the
  `json`/`decode_binary_mask_rle`/`resize_binary_mask_to_shape` imports + crop-box logic; new
  `_resolve_snip_paths` resolves both image AND mask path.
- `src/data_pipeline/pipeline_orchestrator/tasks.py` — `cmd_curvature_metrics_report` +
  `curvature-metrics-report` subparser no longer take `--frame-masks-csv`.
- `src/data_pipeline/pipeline_orchestrator/rules/curvature_metrics.smk` — report rule dropped
  `frame_masks` input + shell arg; docstring updated.
- `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` — `curvature_metrics_report` entry
  (already conformant to `pipeline_file_philosophy.md`, matches `surface_area_qc_report` template
  exactly). **No change needed — verified.**

## Verification done
Real 20250912_C03 data, 12 snips: computed real `curvature_metrics.csv` via
`compute_curvature_features`, ran report, **12/12 snips produce non-empty 200-pt spline**, gallery
visually shows red spine + green outline correctly on every embryo (round→curled tracks with
baseline_deviation). Curvature tests 3/3 pass; `tasks.py` imports clean. `--frame-masks-csv` gone
from CLI `--help`.

Test driver + outputs (may be gc'd — session-specific scratch):
`.../scratchpad/curvreport_test/run_test.py`, `gallery.png`, `feature_grid.png`,
`small_snip_inventory.csv`, `curvature_metrics.csv`.

## Design note (in the report.py docstring, keep it)
The report's spine is a **redraw for display** on the rotated snip-space mask; the merged *numbers*
come from the full-frame RLE mask. Curvature is rotation/scale-covariant so the drawn spine faithfully
shows the same animal — but it's the display path, matching prototype + `report_world.md`.

## Real-tree data layout (differs from earlier handoff assumption!)
Not `object_extraction/snips/20250912/...`. Actual:
- snip_inventory: `data_pipeline_output/object_extraction/20250912/snips/per_well/{well}/{well}_snip_inventory.csv`
- embryo masks: `.../snips/per_well/{well}/snips/{physical_embryo_id}/{snip_id}_embryo.png`
- frame_masks: `data_pipeline_output/object_extraction/20250912/frame_masks/per_well/{well}/...`
- frame_inventory: `data_pipeline_output/acquisition/20250912/frame_inventory/per_well/{well}/...`
- **No materialized `curvature_metrics.csv` in the real tree yet** — the merged artifact the report
  rule consumes doesn't exist on disk; it's produced by `merge_curvature_metrics`. Full curvature
  compute is slow (~2min single-threaded per well; SGE fans out 50 slots per
  `submit_curvature_report.sge`).

## Open / next
1. **Commit** the 4 changed files (user hasn't asked yet — don't commit unprompted).
2. The user asked "where's the report" under `data_pipeline_output/feature_extraction` — it is NOT
   materialized there yet (needs a real DAG run; snakemake-not-installed blocker). To show a report,
   run the report builder directly on a well as in `run_test.py`.
3. **User is about to run two experiments: an `irx` experiment + a repeat of this (20250912-style)
   experiment.** The report wiring should carry over unchanged — it's experiment-agnostic (keyed on
   `{experiment_id}`). Watch that both new experiments produce `embryo_mask_snip_path` (they will, via
   standard snip_processing).
4. Snakemake dry-run of the report rule still unverified end-to-end (env blocker).
