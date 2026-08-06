# Cilia gene-module scores

This analysis colors the v3.1.0 reference atlas and the GENE14 McClintock UMAP
by interpretable cilia expression scores.

The main module is the union of the annotated 2013 SysCilia v1 genes and a
gold-standard motile-cilia list derived from CilioGenics. Smaller modules split
the SysCilia v1 genes by their curated protein localization, such as axoneme,
basal body, transition zone, and intraflagellar transport.

## Chosen modules for perturbation analysis

The module-correlation audit was used to choose five modules for downstream
embryo-level perturbation testing:

1. `v1_all`: the broad, annotated cilia program.
2. `gold_motile_cilia`: the focused motile-cilia program.
3. `v1_basal_body`: the structural and cilia-assembly program.
4. `v1_ciliary_tip`: a distinct, variable, and reproducible program.
5. `v1_transition_zone`: the targeted CEP290/B9D2 transition-zone hypothesis.

`combined_cilia` was not chosen because it ranks cell types almost identically
to `v1_all`, while using a larger gene list. Other localization modules remain
available for exploration but are not part of the approved primary analysis.

## Files

- `software/SysCilia/source/scgs.v1.tsv`: preserved annotated SysCilia v1
  source table.
- `cilia_gene_module_policy.R`: the short, shared localization and scoring
  rules.
- `cilia_gene_module.Rmd`: the readable analysis, from source lists through
  mapping, scoring, and figures.
- `compare_cilia_modules.Rmd`: compares module redundancy across cell types and
  checks whether their rankings agree between reference and GENE14.
- `build_embryo_module_scores.Rmd`: summarizes the five chosen scores by embryo
  and cell type, then audits possible minimum-cell requirements before testing.
- `differential_cilia_modules.Rmd`: fits the approved embryo-level module
  comparisons and applies BH correction within each perturbation contrast.
- `qsub_differential_cilia_modules.sh`: cluster entry point for the differential
  module analysis.
- `submit_differential_cilia_modules.sh`: submits every unfinished direct
  perturbation contrast from `contrast_plan.R` and skips completed results.
- `qsub_cilia_gene_module.sh`: cluster entry point for one notebook mode.
- `../0_shared/effect_concurrence.R`: conservative shared helper for asking
  where two comparable perturbation effects agree in direction and magnitude.
- `data/`: cleaned human modules and human-to-zebrafish mapping audits.
- `output/`: compact per-cell checkpoints and cell-type summaries.
- `figures/`: matching PDF and PNG versions of the UMAPs and cell-type module
  summaries.

## Notebook modes

The safe default is `prepare_modules`, which creates and displays the human
modules and zebrafish mapping without loading a large CDS.

- `prepare_modules`: build and audit the modules.
- `score_reference`: score the reference atlas.
- `score_gene14`: score the GENE14 CDS.
- `assemble`: combine completed score checkpoints and make figures.
- `all`: run every step sequentially in one large-memory job.

The scores are descriptive normalized-expression summaries. They are not
differential-expression tests and do not determine cell identity.
