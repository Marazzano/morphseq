# MVP — Load & Group GENE14 (loaf pbx as reference)

Reproducible path: load CDS → attach phenotype → group by condition (within rt_block) → DACTs → count bar plots.
Reference: `software/loaf/inst/examples/loaf_example_pbx.Rmd`.

## Inputs
- CDS (7.3G, BPCells): `/net/seahub_zfish/.../mcclintock/GENE14/run_1/filter_embryos/embryo_filtered_cds`
- coldata (fast): `.../embryo_filtered_cds_coldata.tsv`
- phenotype labels: `output/gene14_embryos_with_phenotype.tsv`
- DACTs (already fit): `output/phenosplit_contrasts_all.tsv`

CDS is already fit (hooke v3.1.0). Don't re-fit for plots.

## loaf → GENE14
- loaf `perturbation` → our `condition` (derived: genotype `perturbation` + imaging `morphseq_phenotype`)
- `cell_type`, `embryo_ID`, `timepoint` → same
- `contrast_abundance_tbl.tsv` → `phenosplit_contrasts_all.tsv`
- `new_cell_count_set()` → same

## 1. Load (loaf §2)
```r
cds <- load_monocle_objects(CDS_PATH)   # OOM-kills login node — run on grid
cd  <- as.data.frame(colData(cds))
```

## 2. Group — build `condition`
```r
pheno <- read_tsv(PHENO_TSV) %>% select(embryo_ID, morphseq_phenotype)
cd$morphseq_phenotype <- pheno$morphseq_phenotype[match(cd$embryo_ID, pheno$embryo_ID)]
```
condition: phenotype for mutants (`morphseq_phenotype`, exists only post-divergence); `negsib` if `perturbation=="<gene>-negsib"`; `AB` if `perturbation=="reference"`; **`POOLED` = ALL homozygous mutant embryos by genotype (`perturbation=="<gene>-mut"`), label-independent** — so POOLED is non-empty at every timepoint (incl. 18/24 hpf where no phenotype label exists yet).

Order: b9d2 = `POOLED, CE, HTA, negsib, AB`; cep290 = `POOLED, Low_to_High, High_to_Low, negsib, AB`.

### rt_block rule (controls matched within experiment)
b9d2 = Bl6–9, cep290 = Bl2–5, Bl1 = standalone AB. Admit AB only from a gene's own blocks:
```r
gene_blocks   <- unique(cd$rt_block[is_mut | is_ns])
is_ab_matched <- is_ref & cd$rt_block %in% gene_blocks
```

## 3a. DACTs (loaf §4.1) — chosen threshold
```r
dact <- read_tsv("output/phenosplit_contrasts_all.tsv") %>%
  filter(percent_max_abund >= 0.1, log_abund_x >= -3, delta_q_value < 0.05)
```

## 3b. Per-embryo counts (loaf §5.1)
```r
counts <- cd_g %>% count(gene, rt_block, embryo_ID, timepoint, cell_type, condition, name="n_cells")
# x=condition, y=n_cells, geom_col(mean) + geom_jitter(per-embryo dots)
```

## Scripts
- `1_attach_phenotype_labels.py` → phenotype TSV
- `5_phenotype_split_hooke.R` (+`5_submit_*.sh`, grid) → DACTs
- `6_cell_count_barplots.R` (+`6_submit_*.sh`, grid) → count bars

Run: 1 → 5 → 6. Outputs (gitignored): `cell_counts_per_embryo.tsv`, `count_bars_{b9d2,cep290}.png`.

## Deferred
- significance asterisks on count plots (loaf true `plot_cells_per_sample_with_significance`)
- volcano / heatmap / platt views (loaf §5.2–5.9)
- `cep290 High_to_Low vs ctrl` (skipped, too few timepoints)
- CDS snapshot carrying `condition`/`morphseq_phenotype` so step 2 isn't re-derived
