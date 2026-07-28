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

## Next steps
- [ ] **Download / install the `loaf` package** (`git@github.com:waltno/loaf.git`) — use
      `inst/examples/loaf_example_pbx.Rmd` as the scaffold for the DACT analysis. Put it in
      `software/loaf/` (same pattern as `software/zscapetools`).
- [ ] Phenotype-split McClintock/hooke run using the MorphSeq labels attached here, then produce
      Key Results 1-3 above (per-phenotype-vs-control and phenotype-vs-phenotype DACT lists +
      the per-embryo DACT-vector PCA).
- [ ] Load the existing perturbation-level contrasts (zscapetools `load_contrasts` on the run_1
      dir), extract DACT lists per perturbation (`delta_q_value < 0.05`) as the baseline.
- [ ] Regenerate phenotype predictions for the RESCUED reformatted-plate embryos (crispants +
      cep290 plate03) — they were skipped at prediction time as presumed QC-failures but actually
      passed; rerun `20260607_sci_cilia_gene14_imaging_qc/2_predict_sequenced_embryos.py`.

Note: `output/*.tsv` are gitignored (regenerate by running the scripts).
