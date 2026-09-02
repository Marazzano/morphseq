# GENE14 Phase 3 — hooke DACT analysis

Goal: use Trapnell-lab tools (hooke / McClintock) to find transcriptional differences via
MorphSeq labels — differentially abundant cell types (DACTs) per perturbation and phenotype.

## Key fact: the McClintock/hooke run already exists
hooke was **already run** for GENE14 (v3.1.0):
`/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/mcclintock/GENE14/run_1`
- Per-perturbation contrasts: `hooke/perturb_{cep290-mut,b9d2-mut,ift88,sspo,foxj1a}/contrast_abundance_tbl.tsv`
- **DACTs = rows with `delta_q_value < 0.05`.**
- Projected cell-type-labeled CDS: `filter_embryos/embryo_filtered_cds` (has `cell_type`, `timepoint`, `embryo_ID`, ...).
- This run is PERTURBATION-level (cep290-mut vs sibling), NOT MorphSeq phenotype-split. The
  phenotype-split (cep290 HtL vs LtH, b9d2 CE vs HTA) is a SEPARATE McClintock run using the
  MorphSeq labels attached here.

## Scripts
- `scripts/0_reconcile_metadata_qc.R` — join the fitted-CDS colData + embryo metadata + QC list.
  Output: `output/gene14_embryos_reconciled.tsv` (520 embryos, perfect id concordance).
- `scripts/seq_imaging_crosswalk.py` — **the documented imaging<->sequencing bridge** (read its
  docstring: what maps cleanly vs the 5 tricky things — reformatted plates, hash_plate_num,
  plate03 skipped column, P18->P02 collision tie-break, 30to48 collection time).
- `scripts/1_attach_phenotype_labels.py` — thin consumer of the crosswalk. Attaches MorphSeq
  phenotype (CE/HTA/HtL/LtH), `pair` (breeding cross), and physical features
  (`total_length_um`, `baseline_deviation_normalized`) to the 520 DACT embryos.
  Output: `output/gene14_embryos_with_phenotype.tsv`. **169/169 phenotyped embryos resolved.**

## Status
- [x] zscapetools / hooke installed + tested in R (`software/zscapetools`; testthat passes)
- [x] metadata + QC reconciliation (520 embryos, 0 orphans)
- [x] imaging<->sequencing crosswalk (documented module)
- [x] phenotype + pair + length/curvature attached (169/169; 167 in the fitted CDS)

## Phase-3 Key Results (the analysis to produce next)
1. **Mutant phenotype vs control** — DACT list for each mutant phenotype vs non-mutant controls
   (e.g. `cep290_HtL` vs `cep290_wildtype_siblings`).
2. **Mutant phenotype vs mutant phenotype** — DACT list for one phenotype vs another
   (e.g. `cep290_HtL` vs `cep290_LtH`).
3. **Transcriptional similarity across all perturbation-phenotype labels** —
   (a) derive the shared DACT set across mutant phenotypes; (b) PCA plot where each datapoint is
   an embryo's DACT vector.

These require a phenotype-split hooke/McClintock run (or subspace refit) driven by the MorphSeq
phenotype labels attached in `output/gene14_embryos_with_phenotype.tsv` (the current v3.1.0 run is
perturbation-level only).

## Analysis scripts (2 -> 5) and status

**Quality filter (lab convention, applied before every DACT call):**
```
percent_max_abund >= 0.1  AND  log_abund_x >= -3
```
Then DACT = `delta_q_value < 0.05`.

- `scripts/2_baseline_perturbation_dacts.py` — **[DONE]** BASELINE for KR1. Loads the existing
  v3.1.0 perturbation-level `contrast_abundance_tbl.tsv` (mut vs negsib), applies the quality
  filter + `delta_q<0.05`. Output: `output/baseline_perturbation_dacts.tsv` (161 DACT rows),
  `..._dact_summary.tsv`. Headline: cep290 29 DACTs (peak at 24hpf), foxj1a 37, ift88 35,
  sspo 29, b9d2 only 1 (weak at whole-embryo perturbation level).
- `scripts/3_build_embryo_abundance.py` — **[DONE]** streams the 1GB per-cell coldata TSV,
  tallies per-(embryo,cell_type). Output: `embryo_celltype_counts.tsv`,
  `embryo_celltype_fraction.tsv` (520 embryos x 353 cell types, 1.66M cells).
- `scripts/4_embryo_abundance_umap.py` — **[DONE]** KR3. Per-embryo cell-type composition
  (CLR-transformed) embedded with **UMAP** (PCA PC1 carried too little variance -- stage-dominated).
  Each datapoint = one embryo; colored by phenotype label AND by timepoint. Views: all embryos;
  per-genotype (cep290 / b9d2 / crispants), each with a pooled view + a **within-timepoint** facet
  row (gene AND stage controlled); mutant-DACT-subset feature space. Figures `figures/kr3_embryo_umap_*`.
  Headline: **cep290 HtL vs LtH separate at 24hpf**; b9d2 CE vs HTA partial at 18hpf; crispants
  intermixed with controls at all stages. NOTE: "DACT is not per-embryo" -- it is a group-level
  contrast statistic; the per-embryo object is the composition vector, which is what this plots.
- `scripts/5_phenotype_split_hooke.R` + `scripts/5_submit_phenotype_split_hooke.sh` — **[DONE]**
  KR1 + KR2 at phenotype granularity. MUST run via qsub (login node = 8GB → CDS OOM). Submit
  wrapper requests 8×24G=192G and exports LD_LIBRARY_PATH to HDF5 1.14.3 (BPCells needs
  libhdf5.so.310). Fits one `platt:::fit_genotype_ccm` per TWO-GROUP contrast (case + ctrl both
  real `pheno_group` levels, subset to exactly those 2 groups). Uses `vhat_method="sandwich_var"`
  (analytical variance, ~6min/fit; bootstrap default was >90min/fit) and a constant `batch_one`
  column (hooke skips single-level batch → clean spline×knockout model, full rank). Whole run ~15min.
  Outputs `output/phenosplit_{contrasts,dacts}_<tag>.tsv` + `phenosplit_dact_summary.tsv`.

  **RESULTS (delta_q<0.05, `log_abund_x>=-3` quality filter; `get_perturbation_effects` output
  lacks `percent_max_abund` so that half of the filter is skipped):**

  **(CORRECTED labels — after the t01/t02 acquisition-priority fix; E01 HtL→LtH):**

  | KR | contrast | DACTs | cell types | up | down |
  |----|----------|-------|-----------|----|----|
  | KR1 | cep290 Low_to_High vs negsib | **142** | 90 | 21 | 121 |
  | KR1 | cep290 High_to_Low vs negsib | *skipped* | — | — | — |
  | KR1 | b9d2 CE vs negsib | 2 | 1 | 2 | 0 |
  | KR1 | b9d2 HTA vs negsib | 1 | 1 | 0 | 1 |
  | KR2 | cep290 High_to_Low vs Low_to_High | 17 | 17 | 3 | 14 |
  | KR2 | b9d2 CE vs HTA | 4 | 4 | 0 | 4 |

  (Pre-fix counts were cep290 LtH=99, HtL-vs-LtH=7; the crosswalk fix cleaned the LtH cohort and
  the cep290 signal strengthened. b9d2 unchanged — its plate02 already used t02.)

  Headline: **cep290 Low_to_High has a broad transcriptional signature** (99 DACTs, mostly a 24hpf
  depletion of differentiated types — enteric/retinal/oligodendrocyte/vascular — with tailbud NMPs
  and neural-crest head mesenchyme RETAINED: a developmental-delay/cilia signature). **b9d2 is
  nearly flat** at whole-embryo abundance (1-2 DACTs), consistent with the baseline (1 DACT) and
  KR3 (subtle b9d2 separation). KR2 cep290 HtL-vs-LtH cleanest DACT = heart-primordium mesoderm
  enriched in HtL at 24hpf. cep290 **High_to_Low vs ctrl was skipped**: HtL spans only tp 24-48
  (3 pts) → the spline(knot=36)×knockout basis is rank-deficient. To recover it, refit HtL with a
  reduced (no-interaction-knot) model.

## DACT definition (chosen threshold — loaf abundance filter)

A cell type counts as a **DACT** when it passes the loaf abundance quality filter AND is
significant:

```r
# loaf abundance thresholds (from loaf_example_pbx.Rmd) + significance
dact <- tbl %>%
  dplyr::filter(percent_max_abund >= 0.1,   # cell type is >=10% of its max abundance
                log_abund_x       >= -3) %>% # reference-side abundance not vanishingly rare
  dplyr::filter(delta_q_value < 0.05)        # significant abundance change
```

This is the threshold applied in `scripts/5_phenotype_split_hooke.R` (`apply_quality()` +
`Q_THRESH`), written to `output/phenosplit_dacts_*.tsv`. The bar plots (script 6) select the
cell types to show from this same DACT set — they are **not** re-thresholded downstream.

## Per-embryo cell-count bar plots (scripts 6) — rt_block-matched controls

`scripts/6_cell_count_barplots.R` (grid: `6_submit_cell_count_barplots.sh`) loads the CDS, merges
MorphSeq phenotype labels, and makes standard monocle-style bar plots of **raw cells per embryo**
for every significant-DACT cell type. x = condition (`POOLED, pheno1, pheno2, negsib, AB`), bar =
per-condition mean, **one dot per embryo overlaid**. Colors from `color_mapping_config.py`.

> **⚠ CONTROLS ARE MATCHED WITHIN `rt_block` — do NOT pool controls across experiments.**
> The GENE14 `rt_block`s are gene-segregated:
> - **b9d2**  → Bl6, Bl7, Bl8, Bl9  (`b9d2-mut` + `b9d2-negsib` + AB `reference`)
> - **cep290** → Bl2, Bl3, Bl4, Bl5  (`cep290-mut` + `cep290-negsib` + AB `reference`)
> - **Bl1** is a standalone AB-only block belonging to NEITHER gene's experiment.
>
> The AB `reference` cells admitted to a gene's panels are **only** those in the rt_blocks that
> host that gene's own mutants/negsibs. Pooling all AB across blocks (or reusing the other gene's
> AB / Bl1) mixes a different experiment's controls into the comparison. Script 6 enforces this in
> `per_gene_cells()` (`is_ab_matched <- is_ref & rt_block %in% gene_blocks`) and prints how many
> out-of-block AB cells it drops. `rt_block` is carried into `output/cell_counts_per_embryo.tsv`.

## Next steps
- [ ] (optional) recover cep290 High_to_Low vs ctrl with a reduced spline model (too few timepoints).
- [ ] Adopt the loaf abundance thresholds (per user) once the loaf scaffold is installed
      (`git@github.com:waltno/loaf.git` -> `software/loaf/`, `inst/examples/loaf_example_pbx.Rmd`).
- [ ] Regenerate phenotype predictions for the RESCUED reformatted-plate embryos (crispants +
      cep290 plate03) — rerun `20260607_sci_cilia_gene14_imaging_qc/2_predict_sequenced_embryos.py`.

Note: `output/*.tsv` and `figures/` are gitignored (regenerate by running the scripts).


use this filter:# Basic quality filter
hooke_data <- hooke_data %>%
  dplyr::filter(percent_max_abund >= 0.1, log_abund_x >= -3)
Also, for the goal, you should use the thresholds for abundance in the loaf script. 
