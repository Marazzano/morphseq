# Surface-area QC investigation

**Agent:** refactor agent D

**Measurement date:** 2026-08-27 PDT / 2026-08-28 UTC (`surface_area_qc/study_metadata.json:18`)

**State:** `ready_for_integration`; this report does not claim integration or acceptance.

## Recommendation

Keep the production surface-area rule unchanged for now: the primary prospective Track A cohort
should require `is_valid_snip`, apply the stored `sa_outlier_flag` only where applicability is
`exclusion`, preserve `not_applicable` as a third state, and retain the current strict lower/upper
band (`k_lower=0.90`, `k_upper=1.40`). This is an interim cohort policy, not a finding that 0.90 is
optimal (`src/data_pipeline/quality_control/surface_area_qc/config.py:20-31`;
`surface_area_qc/boundary_visual_triage_examples.csv:2-43`; measured 2026-08-27).

Predeclare `lower_diagnostic_upper_strict` as a named sensitivity cohort: ignore the lower flag for
eligibility but keep the upper flag and every other QC exclusion strict. It recovers 50,769 rows and
760 physical embryos with at least one eligible row relative to the strict cohort, but those counts
are not evidence that the recovered observations are usable
(`surface_area_qc/counterfactual_policy_summary.csv:2-3`; measured 2026-08-27). Do not promote that
sensitivity cohort to the primary cohort until Nick or a delegated domain expert completes blinded
review of the 300-case boundary set, with decisions evaluated by experiment, nominal stage,
biology, and persistence strata (`surface_area_qc/boundary_review_set.csv:2-301`;
`scripts/recon/analyze_surface_area_qc.py:1132-1237`).

Do not select a new scalar `k_lower` from this sweep. Continuous head-to-tail silhouettes and
obvious fragmented/artifactual shapes were both observed above and below the current lower bound,
so the reviewed morphology does not supply a scalar separating boundary; these are explicitly
nonexpert mask-morphology observations, not biological-usability labels
(`surface_area_qc/boundary_visual_triage_examples.csv:2-43`; measured 2026-08-27). A future candidate
should be validated as a shape- and/or track-aware policy; the aspect/circularity rules here remain
diagnostic-only because their cutoffs have not been validated
(`docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md:87-107`;
`scripts/recon/analyze_surface_area_qc.py:979-1006`).

Rebuild or freeze a source-coherent QC bundle for `20260304` before using parity as a regenerated
corpus acceptance check. Its stored `snip_qc` predates the geometry and stage tables and contains the
only parity mismatch (`surface_area_qc/source_inventory.csv:1788-1795`;
`surface_area_qc/parity_mismatches.csv:2`; measured 2026-08-27).

## Reproduction and source authority

The successful run used Python 3.10.16 from `morphseq-env`, git branch `core-model-refactor` at input
HEAD `d830391f9f7132d8086a53b5470129847e07e9d2`, and completed in 266.91 seconds
(`surface_area_qc/study_metadata.json:7-8,18,23,27`). The report timestamp is after the measurement
run; the study metadata records the input commit rather than the later Agent D report commit
(`surface_area_qc/study_metadata.json:7-8,18`).

The exact successful command was:

```bash
eval "$(conda shell.bash hook)" && conda activate morphseq-env && \
python -u scripts/recon/analyze_surface_area_qc.py \
  --output-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output \
  --experiments-file docs/refactors/core-model/reports/recon_tables/qc_availability_by_experiment.csv \
  --experiment-column experiment_id \
  --reference-csv src/data_pipeline/quality_control/surface_area_qc/references/surface_area_reference_v1.csv \
  --report-dir docs/refactors/core-model/reports/surface_area_qc \
  --k-lower 0.90 --k-upper 1.40 --boundary-size 300 --seed 20260827
# exit 0: Wrote .../surface_area_qc in 266.9 s
```

The ordered experiment authority contains 148 IDs and has SHA-256
`b0c7d9606a328355a4df5ef849497ee276e5e3357e5e79247ce90c412f325b41`; the v1 reference has SHA-256
`3bffedb6a9ef96ec58e7e85b1e37ffb4983b2120d1a7607563a58ae64bb6c960`
(`surface_area_qc/study_metadata.json:4-6,19,24-25`). The script reads only that ordered list and
resolves declared artifact paths without discovering experiments or samples by glob
(`scripts/recon/analyze_surface_area_qc.py:2-7,34-68,126-180`;
`src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:299-308,340-372,411-427,559-568,648-688,710-719,753-778,799-808,864-873`).

Every declared source path, presence bit, byte size, modification time, SHA-256, row count, column
list, and schema signature is recorded in `surface_area_qc/source_inventory.csv:1-1925`; all
counterfactuals were written only under this tracked report directory
(`scripts/recon/analyze_surface_area_qc.py:1346-1390`;
`surface_area_qc/study_metadata.json:2-3,20,26`).

The live rule interpolates the v1 p5/p95 curves with `numpy.interp`, clamps outside the reference
axis, and flags `area_um2 < 0.90*p5` or `area_um2 > 1.40*p95`; a missing stage is not applicable
(`src/data_pipeline/quality_control/surface_area_qc/reference.py:40-52`;
`src/data_pipeline/quality_control/surface_area_qc/compute.py:36-48,92-113`;
`src/data_pipeline/quality_control/surface_area_qc/config.py:20-31`). The study reproduced those
operators, saved lower/upper bounds, direction, and normalized distances, and did not write a
counterfactual verdict to a pipeline artifact (`scripts/recon/analyze_surface_area_qc.py:679-733`;
`surface_area_qc/study_metadata.json:2-3`).

## Measured facts

### Coverage and parity

Of 148 explicitly listed experiments, 105 had the minimum four sources—snip inventory, mask
geometry, stage predictions, and snip QC—and 43 could not be evaluated from current explicit
sources (`surface_area_qc/experiment_coverage.csv:2-149`; command below, measured 2026-08-27).
Direct artifact availability was sparse: frame inventory 3, frame masks 3, physical registry 3,
surface-area QC 3, plate metadata 144, snip inventory 133, geometry 133, stage predictions 109, and
snip QC 105 (`surface_area_qc/experiment_coverage.csv:2-149`; command below, measured 2026-08-27).

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv("docs/refactors/core-model/reports/surface_area_qc/experiment_coverage.csv")
print(d.evaluable.value_counts())
for c in d.columns[2:]: print(c, int(d[c].sum()))
PY
# True 105; False 43; direct-source counts reported above.
```

The study compared 531,902 rows representing 10,431 physical embryos. It reproduced 531,901 stored
flags exactly: all 20,316 rows in the three direct `surface_area_qc` artifacts matched, while one of
511,586 flags carried through `snip_qc` did not
(`surface_area_qc/overall_failure_summary.csv:2`;
`surface_area_qc/parity_summary.csv:2-106`; measured 2026-08-27).

The mismatch is `20260304_A01_e01_BF_t0058`: stored `True`, recomputed `False`, area
656,256.32 um2, nominal stage 37.796 hpf, and recomputed lower bound 649,012.82 um2, a 1.116% pass
margin (`surface_area_qc/parity_mismatches.csv:2`; command
`python -c "import pandas as pd; r=pd.read_csv('docs/refactors/core-model/reports/surface_area_qc/parity_mismatches.csv').iloc[0]; print((r.area_um2-r.lower_bound_um2)/r.lower_bound_um2)"`, measured 2026-08-27).
The source bundle has no direct surface-area artifact; its carried verdict was modified at
2026-08-13 16:35 UTC, before current geometry and stage inputs at 20:48 UTC, so source incoherence is
the leading explanation but cannot be proven without the old geometry/stage inputs
(`surface_area_qc/source_inventory.csv:1788-1795`; inference, measured 2026-08-27). The row also has
viability, persistence, and focus exclusions, so changing this surface verdict alone would not make
it eligible (command below, measured 2026-08-27).

```bash
python - <<'PY'
import pandas as pd
p = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output/quality_control/20260304/snip_qc/20260304_snip_qc.parquet"
d = pd.read_parquet(p)
print(d.loc[d.snip_id.eq("20260304_A01_e01_BF_t0058"),
            ["use_snip", "qc_fail_reasons"]].to_string(index=False))
PY
# False  viability_dead_flag|persistence_dead_flag|sa_outlier_flag|focus_flag
```

### Materialization and current verdicts

`is_valid_snip` and `use_snip` are not interchangeable in this corpus: 98 rows are invalid
materializations, of which 85 nevertheless have `use_snip=True`; the remaining 531,804 rows are
valid, of which 185,119 have `use_snip=True`
(`surface_area_qc/materialization_use_crosstab.csv:2-5`; measured 2026-08-27). The invalid rows are
confined to `20240813_24hpf` and `20240813_30hpf`
(`surface_area_qc/invalid_materializations_by_experiment.csv:2-3`; measured 2026-08-27). All study
cohort policies therefore require
`is_valid_snip=True` independently of QC (`scripts/recon/analyze_surface_area_qc.py:1009-1018,1064-1076`).

### Failure decomposition and cohort effect

The current band flags 241,273 rows across 4,648 physical embryos: 212,778 are too small and 28,495
are too large. Exactly 52,637 rows fail surface area alone among the stored QC reasons; 185,204 rows
have stored `use_snip=True` (`surface_area_qc/overall_failure_summary.csv:2`; measured 2026-08-27).

After independently requiring a valid materialization and holding every non-surface exclusion
fixed, strict surface exclusion yields 185,119 eligible rows and 7,983 physical embryos with at
least one eligible row. Making both surface directions diagnostic-only yields 237,748 eligible rows
and 8,812 physical embryos, a recovery of 52,629 rows and 829 physical embryos; keeping the upper
gate strict while making only the lower gate diagnostic recovers 50,769 rows and 760 physical
embryos (`surface_area_qc/counterfactual_policy_summary.csv:2-4`; measured 2026-08-27).

Surface failures co-occur with persistence-dead on 91,923 rows (38.10%), viability-dead on 92,164
(38.20%), focus on 127,993 of the 230,798 surface-failed rows where focus is available (55.46%),
discontinuous mask on 5,505 (2.28%), edge on 31,258 (12.96%), overlap on 27,618 (11.45%), and motion
blur on 35,393 of 240,848 rows where motion is available (14.70%)
(`surface_area_qc/qc_flag_cooccurrence.csv:2-8`; measured 2026-08-27).

Applicability is `exclusion` for 531,004 rows and `not_applicable` for 898 rows; none are
`diagnostic_only` in the evaluable stored/resolved view
(`surface_area_qc/stratified_failure_summary.csv:275-276`; measured 2026-08-27). This distinction is
operational: only a true flag with `exclusion` applicability contributes to `qc_fail_reasons`, while
`diagnostic_only` and `not_applicable` do not (`src/data_pipeline/quality_control/applicability.py:5-23`;
`src/data_pipeline/quality_control/snip_qc/build.py:60-85`).

### Track persistence

Of 10,431 physical-embryo tracks, 4,182 contain at least one too-small row; 3,160 have a maximum
consecutive run of at least two, 1,022 have a maximum run of one, and 1,482 contain at least one
single-frame dip with valid nonfailed neighbors on both sides
(`surface_area_qc/physical_embryo_track_persistence.csv:2-10432`; command below, measured 2026-08-27).
At row grain, 206,537 too-small rows occur in persistent runs, 5,039 are isolated dips with recovery,
and 1,202 are single failures at an edge or time gap
(`surface_area_qc/stage_axis_sensitivity_input.parquet`; command below, measured 2026-08-27).

```bash
python - <<'PY'
import pandas as pd
p = "docs/refactors/core-model/reports/surface_area_qc/"
t = pd.read_csv(p + "physical_embryo_track_persistence.csv")
print(len(t), (t.n_too_small_rows > 0).sum(), (t.max_consecutive_run >= 2).sum(),
      (t.max_consecutive_run == 1).sum(), (t.neighbor_recovery_rows > 0).sum())
s = pd.read_parquet(p + "stage_axis_sensitivity_input.parquet")
print(s.too_small_track_class.value_counts())
print(s.loc[s.recomputed_too_small].groupby("too_small_track_class")
       .lower_normalized_distance.agg(["count", "mean", "median"]))
PY
# tracks: 10431 4182 3160 1022 1482
# persistent_run 206537; isolated_dip_with_recovery 5039; edge_or_gap 1202
# failed median normalized distance: persistent -0.534646; isolated -0.084713; edge/gap -0.249491
```

The narrow rule that makes only recovered single-frame dips diagnostic adds 1,155 eligible rows but
only 3 physical embryos with any newly eligible row, so isolated dips do not explain the bulk cohort
effect (`surface_area_qc/counterfactual_policy_summary.csv:5`; measured 2026-08-27).

### Offline threshold and shape diagnostics

With `k_upper=1.40` fixed, lowering `k_lower` from 0.90 to 0.85, 0.80, 0.75, and 0.70 recovers,
respectively, 4,219, 7,385, 9,731, and 11,510 eligible rows; it adds 159, 269, 336, and 381 physical
embryos with at least one eligible row (`surface_area_qc/threshold_sweep.csv:2-6`; measured
2026-08-27). Full experiment, nominal-stage, raw-genotype, raw-perturbation, temperature,
source-scope, calibration, segmentation, acquisition-mode, and applicability composition deltas are
in `surface_area_qc/threshold_sweep_strata.csv:2-1331` (measured 2026-08-27).

The diagnostic rule that excludes only compact (`aspect_ratio < 2`) too-small masks recovers 11,336
rows and 201 physical embryos; additionally requiring circularity at least 0.5 recovers 22,841 rows
and 545 physical embryos (`surface_area_qc/counterfactual_policy_summary.csv:6-7`; measured
2026-08-27). These are intentionally unvalidated counterfactuals: the aspect proxy is documented as
rough, and circularity is a candidate signal rather than a ratified cutoff
(`docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md:27-45,87-107`).

### Stratification

`predicted_stage_hpf` is treated only as the pipeline's nominal stage axis, not validated biological
truth (`docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md:117-127`;
`surface_area_qc/study_metadata.json:22`). On that axis, too-small rates are 7.84% below 18 hpf,
28.15% at 18-<24, 40.39% at 24-<36, 50.60% at 36-<48, 49.98% at 48-<72, and 38.39% at >=72; 898
rows have no stage and no surface failure
(`surface_area_qc/stratified_failure_summary.csv:107-113`; measured 2026-08-27).

Raw source biology was preserved without vocabulary normalization: the table contains 84 genotype
strata and 48 chemical-perturbation strata, while explicit control status is unavailable on all
531,902 rows (`surface_area_qc/stratified_failure_summary.csv:114-246`; measured 2026-08-27).
Temperature is available on 531,594 rows across 14 observed values and unavailable on 308; the three
largest strata are 30.0 C (462,687 rows, 46.31% surface failures), 22.0 C (44,097, 35.54%), and
28.5 C (21,966, 49.08%) (`surface_area_qc/stratified_failure_summary.csv:247-261`; measured
2026-08-27). These are compositions, not temperature effects, because experiment and biology are
not controlled by this observational comparison (inference from the stratified design, measured
2026-08-27).

Explicit source scope, calibration status/method, and declared snapshot/time-series mode are
unavailable for all evaluable rows (`surface_area_qc/stratified_failure_summary.csv:262-264,272`;
measured 2026-08-27). Numeric image scale is available for 20,316 rows and unavailable for 511,586;
the two observed values are recorded without deriving optical covariates
(`surface_area_qc/stratified_failure_summary.csv:265-267`; measured 2026-08-27). Segmentation
backend/model is available for the same 20,316 rows (`sam2_video`, `sam2.1_hiera_s`) and unavailable
for 511,586 (`surface_area_qc/stratified_failure_summary.csv:268-271`; measured 2026-08-27).

Observed identity cardinality—not declared acquisition mode—shows 526,097 rows in 4,626
multi-observation physical tracks and 5,805 single-observation rows/physical embryos
(`surface_area_qc/stratified_failure_summary.csv:273-274`; measured 2026-08-27). No snapshot versus
time-series conclusion is drawn from that proxy (`surface_area_qc/stratified_failure_summary.csv:272-274`).

## Boundary review

The deterministic review set contains 300 valid rows nearest the lower boundary: 150 below and 150
above, sampled round-robin over nominal stage bin, compact versus elongated/thin shape, persistence
class, raw genotype/perturbation tuple, and experiment with seed 20260827
(`scripts/recon/analyze_surface_area_qc.py:1132-1197`;
`surface_area_qc/boundary_review_set.csv:2-301`). It covers 75 experiments, all six populated stage
bins, 156 compact and 144 elongated/thin masks; the below-boundary half contains 74 persistent-run,
39 isolated-recovery, and 37 edge/gap cases (command below, measured 2026-08-27).

```bash
python - <<'PY'
import pandas as pd
b = pd.read_csv("docs/refactors/core-model/reports/surface_area_qc/boundary_review_set.csv")
for c in ["boundary_side", "stage_bin", "shape_class", "too_small_track_class"]:
    print(c, b[c].value_counts())
print("experiments", b.experiment_id.nunique())
PY
```

Raw genotype is present for all 300 selections; chemical perturbation is explicit on 79 and absent
on 221, while explicit control status is unavailable for all 300 (`surface_area_qc/boundary_review_set.csv:2-301`;
command `python -c "import pandas as pd; b=pd.read_csv('docs/refactors/core-model/reports/surface_area_qc/boundary_review_set.csv'); print(b.genotype.notna().sum(), b.chem_perturbation.notna().sum(), b.control_status_raw.value_counts())"`, measured
2026-08-27). Thus the set is balanced over available raw biology but cannot satisfy a labeled
control-versus-perturbation balance without inferring control semantics (unavailable determination,
measured 2026-08-27).

All 300 processed-image and mask paths resolved and were rendered as raw/overlay pairs on 15 contact
sheets (`surface_area_qc/overall_failure_summary.csv:2`;
`scripts/recon/analyze_surface_area_qc.py:1243-1301`; command
`python -c "from pathlib import Path; import pandas as pd; r=Path('/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output'); b=pd.read_csv('docs/refactors/core-model/reports/surface_area_qc/boundary_review_set.csv'); f=lambda x: ((Path(x) if Path(x).is_absolute() else r/Path(x)).is_file()); print(len(b), b.processed_snip_path.map(f).sum(), b.embryo_mask_snip_path.map(f).sum())"`
-> `300 300 300`, measured 2026-08-27). The legacy source schema lacks compound asset
keys for every selected row, so `asset_key_status=unavailable_source_schema` is recorded instead of
fabricating product or plane identity (`surface_area_qc/boundary_review_set.csv:2-301`; measured
2026-08-27).

Refactor agent D visually inspected all 15 sheets and recorded only 42 unambiguous examples: 10
continuous and 10 obvious fragment/disconnected/artifactual observations above the boundary, plus
11 of each below it (`surface_area_qc/boundary_visual_triage_examples.csv:2-43`; measured
2026-08-27). The remaining 258 cases are deliberately unlabeled, and all biological-usability,
mask-quality, pose, and reviewer-note fields in the review set remain blank pending expert review
(`surface_area_qc/boundary_review_set.csv:2-301`; measured 2026-08-27).

## Hypotheses

The documented hypothesis is that scalar area conflates bad yolk-only masks with real embryos in a
thin/dorsal pose (`docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md:8-23`).
The present review is consistent with overlap because continuous silhouettes and obvious fragments
occur on both sides, but it does not establish embryo health, pose, or downstream morphology value
(`surface_area_qc/boundary_visual_triage_examples.csv:2-43`; hypothesis, measured 2026-08-27).

Persistent failures are much more numerous and farther below the boundary than isolated recovered
dips, suggesting persistent low area is a distinct diagnostic population rather than repeated
single-frame noise; whether it represents stable pose, systematic under-segmentation, or true size
cannot be determined from these tables (`surface_area_qc/physical_embryo_track_persistence.csv:2-10432`;
track query under **Track persistence**; hypothesis, measured 2026-08-27).

The wide nominal-stage, temperature, raw-biology, and segmentation-provenance differences could
reflect biology, acquisition/calibration, segmentation, the nominal stage axis, or cohort mix; the
current observational strata do not identify which mechanism dominates
(`surface_area_qc/stratified_failure_summary.csv:2-278`; hypothesis, measured 2026-08-27).

## Unavailable determinations

- Biological usability and health were not expert-labeled; the visual triage is mask-morphology
  description only (`surface_area_qc/boundary_visual_triage_examples.csv:1-43`; measured 2026-08-27).
- Control status cannot be derived safely because no explicit control field is available, although
  raw genotype and perturbation strings are retained (`surface_area_qc/stratified_failure_summary.csv:114-246`;
  measured 2026-08-27).
- Source scope, calibration status/method, and declared snapshot/time-series mode are unavailable;
  only numeric scale for 20,316 rows and observed physical-track cardinality can be reported
  (`surface_area_qc/stratified_failure_summary.csv:262-274`; measured 2026-08-27).
- Segmentation provenance is unavailable for 511,586 rows; comparisons of the available 20,316
  `sam2_video` rows against the unavailable group are not backend effects
  (`surface_area_qc/stratified_failure_summary.csv:268-271`; measured 2026-08-27).
- The precise old geometry/stage/reference inputs that produced the one `20260304` stored verdict
  are unavailable in the declared source bundle (`surface_area_qc/source_inventory.csv:1788-1795`;
  `surface_area_qc/parity_mismatches.csv:2`; measured 2026-08-27).
- Track E stage-axis sensitivity has not been interpreted here. Agent D preserved a unique
  531,902-row, 34-column handoff with IDs, area, nominal stage/status/version, reference bands,
  directions/distances, applicability, track class, raw biology/temperature, and available
  provenance in `surface_area_qc/stage_axis_sensitivity_input.parquet` (command
  `python -c "import pandas as pd; d=pd.read_parquet('docs/refactors/core-model/reports/surface_area_qc/stage_axis_sensitivity_input.parquet'); print(d.shape, d[['experiment_id','snip_id','physical_embryo_id']].duplicated().sum())"` ->
  `(531902, 34) 0`, measured 2026-08-27).

## Tracked outputs

- Source and parity audit: `surface_area_qc/ordered_experiments.csv:1-149`,
  `surface_area_qc/source_inventory.csv:1-1925`, `surface_area_qc/experiment_coverage.csv:1-149`,
  `surface_area_qc/parity_summary.csv:1-106`, and `surface_area_qc/parity_mismatches.csv:1-2`
  (generated by the successful command above, measured 2026-08-27).
- Decomposition and persistence: `surface_area_qc/overall_failure_summary.csv:1-2`,
  `surface_area_qc/stratified_failure_summary.csv:1-278`,
  `surface_area_qc/qc_flag_cooccurrence.csv:1-8`, and
  `surface_area_qc/physical_embryo_track_persistence.csv:1-10432`,
  `surface_area_qc/materialization_use_crosstab.csv:1-5`, and
  `surface_area_qc/invalid_materializations_by_experiment.csv:1-3` (generated by the successful
  command above, measured 2026-08-27).
- Offline counterfactuals: `surface_area_qc/threshold_sweep.csv:1-6`,
  `surface_area_qc/threshold_sweep_strata.csv:1-1331`,
  `surface_area_qc/counterfactual_policy_summary.csv:1-7`, and
  `surface_area_qc/counterfactual_policy_strata.csv:1-1579` (generated by the successful command
  above, measured 2026-08-27).
- Review and Track E exchange: `surface_area_qc/boundary_review_set.csv:1-301`,
  `surface_area_qc/boundary_visual_triage_examples.csv:1-43`, 15 PNGs under
  `surface_area_qc/boundary_contact_sheets/`, and
  `surface_area_qc/stage_axis_sensitivity_input.parquet` (generated/reviewed 2026-08-27).
- Diagnostic figures: `surface_area_qc/figures/failure_direction_by_nominal_stage.png`,
  `surface_area_qc/figures/threshold_sweep.png`, and
  `surface_area_qc/figures/too_small_run_lengths.png` (generated by the successful command above,
  measured 2026-08-27).

No production QC code/configuration, pipeline artifact, core-model code, manifest/plan/decision
document, or Track E output was edited by refactor agent D (`git status --short --branch` and owned
file list in the handoff; checked 2026-08-27).
