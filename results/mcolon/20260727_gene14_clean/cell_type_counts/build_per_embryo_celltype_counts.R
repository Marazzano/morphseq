#!/usr/bin/env Rscript
# Build a compact embryo-by-cell-type count table from McClintock.
#
# The source table has one row per cell. This script counts those cells and
# writes one row per embryo and cell type, including explicit zero counts.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
})

HERE <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260727_gene14_clean"
MCCLINTOCK_COLDATA <- paste0(
  "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/",
  "v3.1.0/mcclintock/GENE14/run_1/filter_embryos/embryo_filtered_cds_coldata.tsv"
)
OUTPUT_FILE <- file.path(HERE, "cell_type_counts", "per_embryo_celltype_counts.tsv")

message("Reading the McClintock cell metadata...")
cells <- read_tsv(
  MCCLINTOCK_COLDATA,
  col_select = c(embryo_ID, perturbation, rt_block, timepoint, cell_type),
  show_col_types = FALSE,
  progress = FALSE
) %>%
  filter(!is.na(timepoint), !is.na(cell_type))

# These fields describe the embryo, so each embryo must have only one combination.
embryo_attributes <- cells %>%
  distinct(embryo_ID, perturbation, rt_block, timepoint)

if (anyDuplicated(embryo_attributes$embryo_ID)) {
  stop("At least one embryo has inconsistent perturbation, RT block, or timepoint.")
}

all_cell_types <- sort(unique(cells$cell_type))

cells_per_embryo <- cells %>%
  count(embryo_ID, name = "total_cells")

# First count observed cells, then add zeroes for cell types absent from an embryo.
celltype_counts <- cells %>%
  count(embryo_ID, perturbation, rt_block, timepoint, cell_type, name = "n_cells") %>%
  complete(
    nesting(embryo_ID, perturbation, rt_block, timepoint),
    cell_type = all_cell_types,
    fill = list(n_cells = 0)
  ) %>%
  left_join(cells_per_embryo, by = "embryo_ID") %>%
  mutate(per_1000 = 1000 * n_cells / total_cells) %>%
  arrange(embryo_ID, cell_type)

write_tsv(celltype_counts, OUTPUT_FILE)

message(
  "Wrote ", OUTPUT_FILE, ": ",
  n_distinct(celltype_counts$embryo_ID), " embryos × ",
  length(all_cell_types), " cell types"
)
