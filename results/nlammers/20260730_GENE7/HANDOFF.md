# HANDOFF — GENE7 morphospace analysis

Written 2026-07-30. Synopsis of completed work plus the next task, for an agent picking this up.

## 0. Environment (read first)

Only `points-ml` has the full stack (pandas + pyarrow + sklearn + plotly + nbclient):

```bash
PYTHONPATH=/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src:. \
  /net/trapnell/vol1/home/nlammers/micromamba/envs/points-ml/bin/python <script>
```

Run from `results/nlammers/20260730_GENE7/`. Do not use bare `python`. `kaleido` is NOT installed, so
plotly static PNG export fails — `plotting.save_figure` writes HTML always and skips PNG with a note.

## 1. Working style the user has asked for (non-negotiable)

- **Do not write code until told to.** Asking "should I do X?" is a real question — stop and wait.
  Design discussion is not a green light. The user has stated this matters to them.
- **Do not pause on non-issues.** The user was (rightly) annoyed at being asked to arbitrate a
  question that could have been resolved by reading. Bias toward investigating and reporting, and
  reserve questions for things only they can decide (curated facts, analysis intent).
- **Never rename/move/edit files without explicit consent.** Surface it and let them decide.
- **When told to run something, do only that** — no unrequested dry-runs or side checks.
- Reading and investigating to answer a question is always fine.
- Work on `main`; do not create branches; do not push (the user pushes).

## 2. What exists — the identity bridge (DONE, verified)

`src/morphseq_integration/` — peer package to `data_pipeline`, joins morphology to sci-PLEX
sequencing at well resolution. **139 tests, all passing** (`tests/morphseq_integration/`).

```python
from morphseq_integration import build_master_table, coverage_summary
master = build_master_table()   # 576 wells x 235 cols for GENE7
```

Key facts:

- **GENE7 = 20250612**, six plates, hash plates P01/P02/P18/P04/P05/P06 (24/24/30/30/36/36 hpf).
- **576 wells, 567 with legacy morphology** (98.4%); 9 wells the legacy build excluded.
- Sample-id grammar is **verified**: the 576 minted ids exactly equal the 576 `imaging=True` embryos
  in `GENE7_embryo_metadata.tsv`, both directions. `verify_seq_pairing()` regression-tests this.
- **Morphology is the LEGACY embeddings** (`20241107_ds_sweep01_optimum`), not the pipeline's. The
  pipeline's snip raster is degraded (`target_pixel_size_um` 7.8 vs 6.5 + a saturation difference),
  so its latents are not interchangeable.
- **No pipeline QC in the master table** by explicit user instruction — `use_snip`/`sa_outlier_flag`
  were computed on the degraded raster. Exclusions are curated by image inspection. There is a test
  asserting those columns are absent.
- Latent naming: legacy numbers **continuously 0–99** with the prefix marking the split at 20
  (`z_mu_n_00–19`, `z_mu_b_20–99`). Flat pipeline `z_mu_ii` <-> legacy dim `ii`. (An earlier doc
  claimed `z_mu_b_00–79` — that was wrong.)

## 3. What exists — this sandbox (DONE, notebook executed)

| file | role |
|---|---|
| `legacy_reference.py` | reference-set QC + the 20240813 temperature repair |
| `morph_pca_spline.py` | shared PCA basis + reference spline fitting |
| `plotting.py` | figure grammar (`save_figure` handles missing kaleido) |
| `run_processing.py` | runs the chain, writes `data/` |
| `gene7_morphospace.ipynb` | executed notebook with real outputs |
| `data/` | cached artifacts (see below) |
| `figures/` | 5 interactive HTML figures |

Regenerate everything with `python run_processing.py` (~2 min; the spline fits dominate).

### Cached artifacts in `data/`

| file | contents |
|---|---|
| `gene7_pca.csv` | 567 GENE7 wells, projected; `PCA_00_bio`..`PCA_04_bio` + metadata |
| `reference_pca.csv` | 14,200 wildtype reference snips, projected |
| `hotfish_pca.csv` | 141 20240813 snips, projected, workbook temperatures attached |
| `spline_weighted.csv` / `spline_unweighted.csv` | 2500-point reference curves + `_se` columns |
| `pca_variance.csv`, `pca_basis.joblib` | the fitted 5-component basis |

### Recipes ported from the 2025-Q1 chain

Source: `results/nlammers/20250310/fit_morph_spline_v2.ipynb` (the user chose 0310 over 0312).

- **Reference set** (`select_reference_embryos`): `short_pert_name == "wt_ab"` AND embryo stage span
  `min <= 18 & max >= 42`. Yields **180 embryos / 14,200 snips**, 12.0–59.6 hpf, drawn from
  `20240812` (6,770), `20240626` (4,092), `20250215` (3,338). **It is a genotype+longevity filter,
  not a date filter** — the reference is NOT synonymous with 20240812.
- **PCA**: 5 components on `z_mu_b_*` (80 biological latents), fit on reference + 20240813 pooled.
  98.1% cumulative; first 3 = 85.1%. GENE7 is **transformed only, never fit**.
- **Spline**: `spline_fit_wrapper` from `src/core/functions/spline_fitting_v2.py` — a
  `LocalPrincipalCurve` (NOT a classical spline), 50 bootstraps x 1000 resampled points, averaged to
  2500 points. Two variants: unweighted, and weighted with `ALPHA=0.25` giving the 24 20240813@28.5C
  control snips a quarter of the sampling mass. The two curves differ by only 0.129 mean 5D distance.
- **Polynomial surface / `mdl_stage_hpf` deliberately OMITTED** per user instruction.

### The 20240813 temperature repair (important, non-obvious)

`embryo_stats_df.csv` on the cluster stores **temperature = 22.0 for every 20240813 well**, which is
not a real value. Cause: `results/nlammers/20250217/add_temperature_info.ipynb` patched temperature in
place and re-saved the CSV, but ran against the Dropbox copy — the cluster copy never got it.

The authoritative source is the `temperature` sheet in each plate workbook, giving the real hotfish
series **19 / 25 / 28.5 / 32 / 33.5 / 35 C**. `legacy_reference.attach_workbook_temperature` reads it
at load time and **does not write back** to the shared 415 MB CSV. Do not reintroduce in-place
mutation of that file — that is what made the discrepancy invisible.

Also: 3 hand-curated outlier snips are dropped (`HOTFISH_OUTLIER_SNIPS`), and `20240813_extras` is
excluded (no sequencing, no hash map).

## 4. Open issues (none blocking the next task)

1. **`20240813` sequencing crosswalk is BLOCKED.** Hash maps exist and are readable, but
   `morphseq_integration/hash_map.py` reports `None` for them: their `image_to_hash_map` sheet is
   **8x10, not 8x12**, so the pipeline's grid validator drops it. Real content: 48 authored wells
   each; 24hpf and 30hpf both on **hash plate 5** at disjoint wells (`A01–A06` / `A07–A12`), 36hpf on
   **plate 6**. Two blockers remain: (a) the loader needs to tolerate the narrower grid, (b) **no
   sequencing experiment on seahub matches** — all 40 were swept; none has plates 5/6 with 144
   imaging embryos in the 19–35 C series, and `HF`/`HF5` have zero `imaging=True`. **The `sci_expt`
   name must come from the user.** This did not block the spline, which needs only morphology +
   workbook temperature.
2. **GENE7 has no per-embryo morphological stage.** `predicted_stage_hpf` in `analysis_ready` is
   constant per plate and exactly equals `start_age_hpf`, despite `stage_prediction_status ==
   "predicted"`. The predictor appears to pass through nominal age. Consequence: there is no "where
   along the trajectory" measure, only "how far from it."
3. **GENE7 sits systematically off the reference curve.** Its 28 C controls are ~1.43 from the spline
   vs ~0.23 for 20240813's 28.5 C controls. **The user has assessed this as a batch artifact and
   explicitly tabled it.** Do not spend time on it.
4. **35 C separates strongly in both experiments** independently (GENE7 ~2.88 vs ~1.4 for other arms;
   20240813 1.05 vs 0.20–0.38). Likely real heat-stress morphology.
5. Sequencing expression data is **tabled by the user**. It is an R/Monocle CDS (372 MB, BPCells
   binary) at `annotated/v2.2.1/GENE7/`; there is no R on this machine and no exported counts table.
   Only metadata is in Python.

## 5. NEXT TASK — cohort-specific axes of intra-cohort variability

The user's goal, in their words: *"quick-and-dirty plots that find the ~2-3 strongest axes of
variability within each cohort, and then show embryo images ordered along those axes to enable
interpretation."*

**This is NOT about global PCs.** The 10D space is only a denoised coordinate system; the object of
interest is each cohort's own covariance structure within it.

### Settled design decisions

- **Basis: one 10-component PCA fit on all 567 GENE7 wells** (`z_mu_b_*`). GENE7-native, so no
  perturbation direction is pre-filtered out, and the batch offset falls into the mean. This is a
  NEW basis — do not reuse the 5-component reference basis in `data/pca_basis.joblib`.
- **Cohort unit: target x temperature x timepoint** — **48 cohorts, 9–12 wells each** (`min 9, max
  12`, verified). The user confirmed per-timepoint explicitly: pooling timepoints would reintroduce
  stage as the dominant intra-cohort axis.
- **Whiten the 10D space** (scale each axis to unit variance) before per-cohort PCA. User agreed.
  Rationale: unwhitened, every cohort's PC1 drifts toward the highest-variance global axis, making
  cohorts look alike for uninteresting reasons. Cost: axes are no longer in morphology-variance units.
- **Center per cohort, not globally.** Otherwise the cohort's offset from the GENE7 centroid leaks
  into its PC1 and you recover cohort *position* instead of *intra-cohort variability*.
- **Report the top 2–3 axes only.** At n~12 in 10D you get ~11 non-trivial components and the tail is
  guaranteed noise.
- **Compare cohorts by principal angles between top-2 subspaces**, not PC1-to-PC1 pairing —
  eigenvalue gaps are small at this n so component ordering is unstable.
- **Include a null.** Run the identical procedure on random 12-well draws from within a single
  temperature arm, so "what a cohort-specific axis looks like when there is none" is visible. At
  n=12 you will find apparent structure whether or not it exists; without the null it is
  uninterpretable.
- Bootstrap error bars on each cohort's eigenvalue spectrum so the drop-off to noise is visible.

### Image strips

Snips ARE available and keyed by `snip_id`:

```
<pipeline>/output/object_extraction/{experiment_id}/snips/per_well/{well_id}/snips/
    {physical_embryo_id}/{snip_id}.png            # and *_embryo.png
```
e.g. `.../20250612_30hpf_ctrl_atf6_A01/snips/..._A01_e01/..._A01_e01_BF_t0000.png`

Sample embryos at deciles along each cohort axis and lay them out in order.

**Caveat to state in the notebook:** these snips are the DEGRADED raster (smaller, more saturated) —
legacy snip images were not preserved. Fine for interpreting shape, but they are not the images the
latents were computed from.

### Deliverables

New companion script (e.g. `cohort_axes.py`) + notebook cells, following the existing pattern:
processing in scripts, notebook loads cached artifacts and plots. Keep everything inside this
sandbox directory.
