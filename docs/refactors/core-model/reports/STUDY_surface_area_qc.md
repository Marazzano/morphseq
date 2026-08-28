# Study: surface-area QC lower-bound sensitivity

**Measurement date:** 2026-08-27 local / 2026-08-28 UTC. The reproducible run records
`2026-08-28T02:00:25.689222+00:00`, Python 3.10.16, pandas 2.2.3, NumPy 1.26.4, base revision
`aea86ff170f042355b54edf096e9a3855bdecd4d`, and 64.35 s runtime
(`surface_area_qc/run_summary.json`).

**Verdict:** this bounded SeaHub snapshot reproduces the stored surface-area flag exactly, but it
cannot establish that the lower gate rejects usable embryos. The available rows are all
single-snapshot, single-z, placeholder-calibrated observations whose surface-area applicability is already
`diagnostic_only`; no visual usability labels were assigned, and the current reruns did not finish
(`surface_area_qc/stratification.csv`; `surface_area_qc/track_persistence.csv`;
`surface_area_qc/boundary_review_set.csv`; rerun-log checks under **Availability boundary**).

The resulting policy recommendation is therefore to retain a diagnostic surface-area signal for
this source, make no production threshold or shape-rule change from these counts, and defer any
exclusion decision until a coherent regenerated corpus supports complete assets, longitudinal
persistence, and blinded boundary review (`surface_area_qc/source_inventory.csv`;
`surface_area_qc/policy_counterfactual.csv`; inference from the measured limitations above).

## Scope and provenance

The study uses the first 20 non-empty entries of the explicit ordered authority
`/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiments.txt`;
the exact IDs, in authority order, are preserved in `surface_area_qc/selected_experiments.txt`
(`surface_area_qc/run_summary.json`). The authority contains 92 entries and is the path selected by
the tracked SeaHub submission script at
`results/nlammers/20260723_seahub/submit_seahub_back_half.sge:23-24`; its SHA-256 is
`67e56843cd661e2a787b8edb8b228948f5cc1151cbc1e22c5a971ef4d5cd8677`
(`surface_area_qc/source_inventory.csv`;
`sha256sum /net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiments.txt`;
`surface_area_qc/run_summary.json`, measured 2026-08-27).

The declared bundle manifest is
`/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiment_manifest.csv`,
SHA-256 `bbdc1d634be90abe167c1d957a719c13cc757a9a4c19cf423847eda41b2f7a17`;
the pipeline output authority is
`/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output`, which is declared by
`results/nlammers/20260723_seahub/submit_seahub_verify.sge:22-29`
(`surface_area_qc/run_summary.json`). Experiment membership was never inferred from the output tree,
and the study script contains no output-tree glob (`scripts/recon/analyze_surface_area_qc.py:3-12,145-176`).

The exact study command was:

```bash
cd /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-core-d
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate morphseq-env
python scripts/recon/analyze_surface_area_qc.py \
  --experiment-list /net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiments.txt \
  --experiment-limit 20 \
  --experiment-manifest /net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/derived/20260804_prod01/bundle/integration/experiment_manifest.csv \
  --pipeline-output-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output \
  --output-dir docs/refactors/core-model/reports/surface_area_qc \
  --reference-csv src/data_pipeline/quality_control/surface_area_qc/references/surface_area_reference_v1.csv \
  --review-size 300
```

The run joined 1,752 observations and 1,752 distinct `physical_embryo_id` values. Each physical
embryo has exactly one observation in this snapshot (`surface_area_qc/run_summary.json`;
`surface_area_qc/track_persistence.csv`). IDs were joined as opaque strings; the script never slices,
parses, or reconstructs them (`scripts/recon/analyze_surface_area_qc.py:145-176,398-603`).

### Availability boundary

The available output is a partial/incoherent snapshot, not current regenerated-corpus evidence. On
2026-08-27, the following command emitted `NOT_FINISHED` for all eight SeaHub rerun tasks:

```bash
cd /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq
for task_id in 1 2 3 4 5 6 7 8; do
  rg -q 'SeaHub FULL re-run FINISHED' "logs/seahub_rerun.24123096.${task_id}.out" || echo NOT_FINISHED
done
```

The first error log records `build-collection-provenance ... --microscope SeaHub` followed by
`invalid choice: 'SeaHub' (choose from 'Keyence', 'YX1')`, and all eight error logs contain
`Error in rule build_collection_provenance`
(`logs/seahub_rerun.24123096.1.err:431,467-472,491`;
`rg -n 'Error in rule build_collection_provenance' logs/seahub_rerun.24123096.{1..8}.err`, measured
2026-08-27).

The broader five-task rerun also had no `FINISHED` marker:

```bash
cd /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq
for task_id in 1 2 3 4 5; do
  rg -q 'FINISHED' "logs/snip_full_rerun_20260827.24123147.${task_id}.out" || echo NOT_FINISHED
done
```

Tasks 1-3 and 5 report `unsupported resolved product plan schema_version 1; expected 2`, while task
4 reports a `MissingInputException` for the curvature-metrics input
(`logs/snip_full_rerun_20260827.24123147.1.err:560`;
`logs/snip_full_rerun_20260827.24123147.2.err:560`;
`logs/snip_full_rerun_20260827.24123147.3.err:561`;
`logs/snip_full_rerun_20260827.24123147.5.err:885`;
`logs/snip_full_rerun_20260827.24123147.4.err:4`, measured 2026-08-27).

For each of the 20 selected IDs, canonical merged `frame_inventory`, `collection_provenance`,
`frame_masks`, `physical_embryo_registry`, `surface_area_qc`, `mask_quality_qc`,
`death_detection_qc`, `focus_qc`, and `motion_blur_qc` artifacts are absent: 180 explicit missing
paths in total (`surface_area_qc/missing_artifacts.csv`; `surface_area_qc/source_inventory.csv`).
Canonical plate metadata, snip inventory, mask geometry, stage predictions, and snip-QC Parquet are
present for all 20; the declared bundle manifest also names an ingress frame inventory and ingress
frame-mask file for every selected ID (`surface_area_qc/source_inventory.csv`). The ingress files
were used only as explicitly declared evidence for acquisition/calibration and segmentation
provenance; their presence is not treated as a replacement for the absent canonical merged files
(`scripts/recon/analyze_surface_area_qc.py:398-603`; `surface_area_qc/source_inventory.csv`).

The canonical snip inventory lacks `snip_product_key`, so zero rows have a complete
`(snip_id, snip_product_key, z_index)` asset key. The intermediate and review tables retain the
source `snip_id`, leave product and plane null, label the key
`unavailable_missing_snip_product_key`, and carry the declared processed-snip and mask paths rather
than fabricating a key (`surface_area_qc/run_summary.json`;
`surface_area_qc/surface_area_intermediate_rows.csv`;
`surface_area_qc/boundary_review_set.csv`).

## Measured facts

### Rule parity

The live defaults use reference v1 on `predicted_stage_hpf`, with `k_lower=0.9` and
`k_upper=1.4`; the flag is `area < k_lower*p5(stage)` or `area > k_upper*p95(stage)`
(`src/data_pipeline/quality_control/surface_area_qc/config.py:20-31`;
`src/data_pipeline/quality_control/surface_area_qc/compute.py:36-48`). The study interpolated the
packaged v1 p5/p95 curves, recorded both bounds and normalized boundary margin per observation, and
recomputed all 1,752 flags (`surface_area_qc/surface_area_intermediate_rows.csv`).

Stored-versus-recomputed parity is exact: zero mismatches. The stored flag is available through
snip QC even though the separate canonical `surface_area_qc` table is absent
(`surface_area_qc/run_summary.json`; `surface_area_qc/parity_mismatches.csv`;
`surface_area_qc/source_inventory.csv`). This parity result is limited to the available partial
snapshot and its recorded reference/config fingerprint; it does not certify the unfinished reruns
(inference from **Availability boundary**).

### Failure decomposition and cohort consequence

At the production multipliers, 223/1,752 rows and physical embryos are flagged (12.73%): 211
too-small (12.04%) and 12 too-large (0.68%)
(`surface_area_qc/failure_decomposition.csv`). Because every observation is `diagnostic_only`, all
1,752 currently have `use_snip=True`; making surface area non-excluding changes this available
cohort by zero rows and zero physical embryos
(`surface_area_qc/failure_decomposition.csv`; `surface_area_qc/stratification.csv`). The production
implementation intentionally assigns `diagnostic_only` to single-z acquisitions, while retaining
exclusion for other acquisitions (`src/data_pipeline/quality_control/surface_area_qc/compute.py:62-66,116-143`).

Of the 223 surface-area flags, 104 have no other available source QC flag. Focus co-occurs on 84
surface-area flags (37.67%), viability/death on 37 (16.59%), and discontinuity, edge, motion blur,
overlap, and persistence/death each co-occur on zero
(`surface_area_qc/failure_decomposition.csv`; `surface_area_qc/qc_flag_cooccurrence.csv`). These are
signal-isolation counts, not exclusion-only counts: surface area is not added to `qc_fail_reasons`
when applicability is diagnostic, and this snapshot has zero surface-area-only exclusion reasons
(`src/data_pipeline/quality_control/snip_qc/build.py:60-85`;
`surface_area_qc/failure_decomposition.csv`).

### Composition

The observed flag rate is heterogeneous across the explicit experiments: it ranges from 0/96 in
several shards to 41/96 in `20260804_seahub_CHEM13_shard001`; this is an experiment label, not a
parsed identifier (`surface_area_qc/stratification.csv`). Chemical-domain rows contribute 220/223
flags among 1,464 observations, while genetic-domain rows contribute 3/223 among 288 observations
(`surface_area_qc/stratification.csv`). This is a composition measurement only; the study does not
normalize genotype vocabulary or infer biology from the labels (study design;
`scripts/recon/analyze_surface_area_qc.py:791-826`).

By nominal `predicted_stage_hpf` bin, flags are 35/400 below 18 hpf, 61/376 at 18-24, 0/88 at
30-36, 10/424 at 36-48, 69/296 at 60-72, and 48/168 above 72; no rows occur in the other configured
bins (`surface_area_qc/stratification.csv`). The nominal stage axis is a pipeline prediction, not
independently validated biological truth (`src/data_pipeline/quality_control/surface_area_qc/config.py:20-31`;
study limitation).

All 1,752 rows share source scope `seahub`, acquisition mode `snapshot`, temperature 28.5 C,
calibration status `placeholder`, calibration method
`strict_wt_p50_over_fov_median_mask_area`, segmentation backend `sam2_image_precomputed`, model ID
`sam2.1_hiera_large:seahub_source_fov`, and surface-area applicability `diagnostic_only`
(`surface_area_qc/stratification.csv`). Consequently, this study measures no between-source,
between-temperature, between-calibration, between-segmentation-model, or snapshot-versus-time-series
contrast (inference from the single-level strata in `surface_area_qc/stratification.csv`).

There is no explicit control-status or `is_control` column in the joined sources, so all 1,752
control-status values are `unavailable`. Strings such as genotype or perturbation labels were not
converted into control status (`surface_area_qc/run_summary.json`;
`surface_area_qc/stratification.csv`; `scripts/recon/analyze_surface_area_qc.py:520-536`).

### Track persistence

All 1,752 physical embryos have one observation. The track output classifies 211 low-area cases as
`single_observation_low_unassessable` and 1,541 other cases as
`single_observation_no_low_area`; persistent failures, isolated dips, run lengths beyond one, and
neighbor recovery are unavailable (`surface_area_qc/track_persistence.csv`). This cohort therefore
cannot test the documented pose/mask-confound hypothesis longitudinally
(`docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md:8-23`;
inference from `surface_area_qc/track_persistence.csv`).

### Offline counterfactuals

With `k_upper=1.4` fixed, the lower-multiplier sweep flags 61 rows/embryos at 0.70, 73 at 0.75, 87
at 0.80, 129 at 0.85, and 223 at 0.90. Relative to 0.90, the lower settings recover 162, 150, 136,
and 94 rows/embryos respectively (`surface_area_qc/k_lower_counterfactual.csv`). Required-stratum
composition for every sweep point is in
`surface_area_qc/k_lower_counterfactual_stratification.csv`; these are offline rule outputs, not
usability labels.

Strict exclusion would remove 223 observations/embryos; diagnostic-only removes zero. A candidate
aspect-only rescue rule—exclude upper failures plus lower failures whose shape is unavailable or
`aspect_ratio < 2`—removes 113. A broader candidate aspect/circularity rescue—exclude upper failures
plus lower failures whose shape is unavailable or both `aspect_ratio < 2` and
`circularity >= 0.35`—removes 77 (`surface_area_qc/policy_counterfactual.csv`). These rules are
deliberately named candidates and are not validated classifiers; the thresholds are exploratory
proxies derived from already available fields (`surface_area_qc/run_summary.json`;
`docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md:27-45,87-106`).

The counterfactual composition is not uniform. Under strict exclusion the chemical/genetic counts
are 220/3; under the aspect-only candidate they are 110/3; under the aspect/circularity candidate
they are 74/3. Across nominal stage bins, the aspect-only candidate reduces strict counts from 69 to
12 at 60-72 and from 48 to 5 above 72, but only from 35 to 35 below 18; the broader candidate reduces
them to 2, 5, and 35 respectively
(`surface_area_qc/policy_counterfactual_stratification.csv`). These changes support the hypothesis
that shape correlates with lower-bound behavior, but without visual labels they do not measure
false-positive recovery (inference from the cited counterfactual table).

### Boundary review inputs

The deterministic review sheet contains 300 unique observations with 300 explicit processed-snip
paths and 300 explicit mask paths (`surface_area_qc/boundary_review_set.csv`). Selection round-robins across boundary, side,
stage bin, perturbation domain, shape class, and track class, takes nearest normalized-boundary
distance first, and uses `SHA-256(seed|snip_id)` only as a tie-break
(`surface_area_qc/run_summary.json`; `scripts/recon/analyze_surface_area_qc.py:1098-1235`).

The sheet has 164 lower-bound and 136 upper-bound candidates; 183 are below their selected boundary
and 117 above it. It spans all six observed stage bins, 254 chemical and 46 genetic observations,
158 elongated/thin and 142 compact shapes
(`surface_area_qc/boundary_review_set.csv`, grouped measurement 2026-08-27). Requested control and
persistence balance is unavailable: all 300 have control status `unavailable`, and all are
single-observation track classes. All 300 asset keys are incomplete because product key is absent,
though image and mask paths are explicit
(`surface_area_qc/boundary_review_set.csv`; `surface_area_qc/run_summary.json`). No images were
visually classified in this study, so the sheet contains no good-mask/bad-mask or usable/unusable
labels (study procedure).

### Stage-axis seam for Track E

`surface_area_qc/stage_axis_sensitivity_input.csv` is the shared, observation-level seam. It carries
opaque observation/embryo IDs, area/perimeter/length/width/aspect/circularity, nominal stage and
stage status/version, elapsed time and temperature, interpolated reference/bounds, direction and
continuous margins, source/calibration/segmentation strata, incomplete asset-key status, and review
paths (`surface_area_qc/stage_axis_sensitivity_input.csv`). Track E can replace or perturb the stage
axis without editing this study's outputs (file contract coordinated through the lead on
2026-08-27).

As a diagnostic sensitivity calculation, applying uniform offsets of -3, -1.5, +1.5, and +3 hpf to
the nominal stage changes 93 (5.31%), 48 (2.74%), 119 (6.79%), and 178 (10.16%) row classifications
respectively; aggregate flagged counts become 218, 185, 326, and 377 versus 223 at zero offset
(`surface_area_qc/stage_axis_sensitivity_summary.csv`). The plate-metadata `stage_hpf` values happen
to reproduce all 223 nominal classifications in this snapshot, but that is not an independent
staging validation (`surface_area_qc/stage_axis_sensitivity_summary.csv`; inference from the shared
source values in `surface_area_qc/stage_axis_sensitivity_input.csv`).

## Hypotheses, not findings

- The concentration of candidate shape-rule recoveries at later nominal stages is compatible with
  the documented thin/dorsal-pose confound, but could also reflect source, treatment, calibration,
  segmentation, or stage-estimation effects that this single-source snapshot cannot separate
  (`surface_area_qc/policy_counterfactual_stratification.csv`;
  `docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md:8-23`).
- The 104 flags without another available QC flag are candidates for visual false-positive review,
  not evidence that they are usable embryos (`surface_area_qc/failure_decomposition.csv`;
  `surface_area_qc/boundary_review_set.csv`).
- Stage-axis perturbation sensitivity could contribute to boundary instability, but uniform offsets
  are stress tests rather than a replacement stage estimator
  (`surface_area_qc/stage_axis_sensitivity_summary.csv`; study design).

## Unavailable determinations

- Whether any lower-bound failure is a usable embryo is unavailable because no blinded visual
  labels were produced (`surface_area_qc/boundary_review_set.csv`; study procedure).
- Persistent low area versus isolated dips and neighboring-frame recovery are unavailable because
  every physical embryo has one observation (`surface_area_qc/track_persistence.csv`).
- The prospective full training-cohort delta is unavailable because both current reruns are
  unfinished; the measured zero-row delta applies only to this diagnostic-only partial snapshot
  (**Availability boundary**; `surface_area_qc/failure_decomposition.csv`).
- Control-versus-perturbation balance is unavailable because no explicit control-status field is
  present; no status was inferred from vocabulary (`surface_area_qc/stratification.csv`).
- Cross-source, temperature, calibration, acquisition-mode, and segmentation-version robustness is
  unavailable because each corresponding stratum has one observed value
  (`surface_area_qc/stratification.csv`).
- Complete review asset keys and canonical merged acquisition/provenance are unavailable
  (`surface_area_qc/source_inventory.csv`; `surface_area_qc/run_summary.json`).

## Recommendation

1. Retain the existing `diagnostic_only` surface-area behavior for this single-z SeaHub source and
   carry the raw flag, direction, applicability, and continuous boundary margin into downstream
   tables. This preserves evidence without excluding rows on the basis of this incomplete study
   (`src/data_pipeline/quality_control/surface_area_qc/compute.py:62-66,116-143`; inference from the
   unavailable determinations).
2. Do not lower `k_lower`, promote strict exclusion, or adopt either candidate shape rule from the
   aggregate recovery counts. The candidates change group composition materially and have no
   usability labels (`surface_area_qc/k_lower_counterfactual_stratification.csv`;
   `surface_area_qc/policy_counterfactual_stratification.csv`;
   `surface_area_qc/boundary_review_set.csv`).
3. After a coherent regenerated run exists, regenerate the same tables from an explicit ordered
   authority, require canonical frame/calibration and complete product-aware asset keys, perform a
   blinded review of the boundary sheet, and evaluate row plus physical-embryo error/composition by
   source and acquisition mode before proposing a named/versioned policy (inference from the
   failures documented under **Availability boundary** and the missing fields in
   `surface_area_qc/source_inventory.csv`).

This recommendation does not alter production QC configuration, choose a final QC policy, or claim
that a candidate shape rule is scientifically valid (repository diff; study scope).

## Output index

- `surface_area_qc/run_summary.json`: environment, runtime, authority hashes, selected experiments,
  and headline counts.
- `surface_area_qc/source_inventory.csv` and `missing_artifacts.csv`: exact paths, fingerprints,
  row counts where read, and missing canonical artifacts.
- `surface_area_qc/surface_area_intermediate_rows.csv` and `parity_mismatches.csv`: observation-level
  recomputation seam and parity exceptions.
- `surface_area_qc/failure_decomposition.csv`, `qc_flag_cooccurrence.csv`, and
  `physical_embryo_outcomes.csv`: row- and embryo-level failure accounting.
- `surface_area_qc/stratification.csv`: all required available strata.
- `surface_area_qc/track_persistence.csv`: track/run-length results and explicit unassessable classes.
- `surface_area_qc/k_lower_counterfactual.csv`,
  `k_lower_counterfactual_stratification.csv`, `policy_counterfactual.csv`, and
  `policy_counterfactual_stratification.csv`: offline sweeps and candidate-rule composition.
- `surface_area_qc/boundary_review_set.csv`: deterministic review sheet with image/mask paths and
  incomplete asset-key status.
- `surface_area_qc/stage_axis_sensitivity_input.csv` and
  `stage_axis_sensitivity_summary.csv`: Track E seam and uniform-offset diagnostic.
- `surface_area_qc/surface_area_boundary_overview.png` and
  `k_lower_counterfactual.png`: descriptive figures generated from the cited tables.

All output-index descriptions are defined by `scripts/recon/analyze_surface_area_qc.py:1288-1468`
and were materialized by the exact command above on 2026-08-27.
