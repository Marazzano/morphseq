#!/usr/bin/env Rscript
# 0_load_mcclintock.R  --  mcclintock in, small tables out
# =============================================================================
# The ONLY script that reads the big mcclintock artifacts. Everything else in
# this folder reads what this writes.
#
# Reads the per-CELL coldata TSV (1.04 GB) ONCE and emits two small tables:
#
#   embryo_table.tsv              THE SPINE. One row per sequencing embryo_ID.
#   per_embryo_celltype_counts.tsv  embryo x cell_type counts.
#   provenance.tsv                path / mtime / bytes / md5 of every input.
#
# It reads the coldata TSV, NOT the 19 GB CDS object, so it is cheap to re-run.
# Load the CDS itself only if you need expression.
#
# NOTHING IMAGING-RELATED HAPPENS HERE. The spine must exist before anything
# tries to attach labels to it -- that is script 3's job.
#
# WHY NO CDS SUBSET IS SAVED
#   The mcclintock CDS is already cilia-only: 520 embryos = cep290-mut 104,
#   reference 93, b9d2-mut 84, foxj1a/ift88/sspo 47 each, ctrl-inj 46,
#   b9d2-negsib 29, cep290-negsib 23. There is nothing else in it to filter out,
#   so a "cilia subset" would just duplicate 19 GB. provenance.tsv records the
#   CDS path for anything that needs gene-level access.
#
#   Rscript 0_shared/0_load_mcclintock.R
# =============================================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(tibble); library(tools)
})

HERE    <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260727_gene14_clean"
OUT_DIR <- file.path(HERE, "0_shared")

MCCLINTOCK <- "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/mcclintock/GENE14/run_1"
COLDATA    <- file.path(MCCLINTOCK, "filter_embryos", "embryo_filtered_cds_coldata.tsv")
CDS_PATH   <- file.path(MCCLINTOCK, "filter_embryos", "embryo_filtered_cds")   # recorded, not read
EMBRYO_META <- "/net/seahub_zfish/vol1/data/preprocessed/GENE14/GENE14_embryo_metadata.tsv"
QC_PASS     <- "/net/seahub_zfish/vol1/data/BBI_dmux_sci/GENE14/260625_GENE14_run2_novaseqx/GENE14/missing_embryos_list.csv" #generated for me by heidi 

log_ts <- function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)

# ---- 1. provenance, before anything is read ---------------------------------
log_ts("recording provenance ...")
prov <- tibble(
  role = c("mcclintock coldata (per cell)", "mcclintock CDS (not read here)",
           "embryo metadata", "sequencing QC pass list"),
  path = c(COLDATA, CDS_PATH, EMBRYO_META, QC_PASS)) %>%
  rowwise() %>%
  mutate(exists = file.exists(path),
         mtime  = if (exists) format(file.info(path)$mtime, "%Y-%m-%d %H:%M") else NA_character_,
         bytes  = if (exists && !dir.exists(path)) file.info(path)$size else NA_real_,
         # directories get no checksum; the CDS is a directory
         md5    = if (exists && !dir.exists(path)) unname(md5sum(path)) else NA_character_) %>%
  ungroup()
print(as.data.frame(prov %>% mutate(md5 = substr(md5, 1, 16))), row.names = FALSE)
stopifnot(all(prov$exists))
write_tsv(prov, file.path(OUT_DIR, "provenance.tsv"))

# ---- 2. read the per-cell coldata ONCE --------------------------------------
log_ts("reading coldata (1.04 GB, one row per cell) ...")
coldata <- read_tsv(COLDATA, show_col_types = FALSE, progress = FALSE)
log_ts("  ", nrow(coldata), " cells x ", ncol(coldata), " cols")

# ---- 3. the spine: one row per embryo --------------------------------------
# The coldata is one row per CELL. Collapse it to one row per EMBRYO.
#
# These columns describe the embryo, not the cell, so every cell of an embryo
# carries the same value. Taking the first one is therefore exact, not a guess.
embryo_cols <- c(
  "perturbation",      # the arm: foxj1a / cep290-mut / ctrl-inj / ...
  "target",            # the gene: foxj1a / cep290 / Control
  "pheno",             # sequencing-side phenotype label
  "timepoint",         # hpf
  "reference",         # is this an AB reference embryo
  "compare_against",   # which control mcclintock paired it with
  "collection_batch",
  "type", "allele", "strain",
  "hash_plate", "hash_well", "rt_block",   # the plate coords the crosswalk needs
  "expt", "sci_batch", "cross_batch"
)
# Guard against a coldata schema change silently dropping a column.
missing_cols <- setdiff(embryo_cols, colnames(coldata))
if (length(missing_cols)) {
  log_ts("NOTE these expected columns are absent: ",
         paste(missing_cols, collapse = ", "))
  embryo_cols <- intersect(embryo_cols, colnames(coldata))
}

seq_expr_spine <- coldata %>%
  group_by(embryo_ID) %>%
  summarise(
    n_cells = n(),                                   # cells sequenced per embryo
    across(all_of(embryo_cols), dplyr::first),       # assuming constant within embryo
    .groups = "drop"
  )

log_ts("spine: ", nrow(spine), " embryos, ", sum(spine$n_cells), " cells")

# ---- 4. join the metadata and the sequencing QC flag ------------------------
seq_expr_meta <- read_tsv(EMBRYO_META, show_col_types = FALSE)

qc <- read_csv(QC_PASS, show_col_types = FALSE)
qc <- select(qc, embryo_ID, pass)
qc$pass <- as.logical(qc$pass)

# Any CDS embryo with no metadata row is a hard mismatch (the redone 18hpf
# cep290 ids are the known case) -- surface it, never drop it silently.
orphans <- setdiff(seq_expr_spine$embryo_ID, seq_expr_meta$embryo_ID)
if (length(orphans)) {
  log_ts("WARNING ", length(orphans), " embryos have no metadata row: ",
         paste(head(orphans, 5), collapse = ", "))
}

# Take from the metadata only the columns the spine does NOT already have. The
# coldata versions win, because those are the values hooke actually fit on.
meta_extra_cols <- setdiff(colnames(meta), colnames(spine))
meta_extra <- select(meta, embryo_ID, all_of(meta_extra_cols))
log_ts("metadata adds ", length(meta_extra_cols), " columns the spine lacks")

embryo_table <- spine %>%
  left_join(qc, by = "embryo_ID") %>%
  left_join(meta_extra, by = "embryo_ID") %>%
  mutate(
    # absent from the QC list == did not pass
    pass = ifelse(is.na(pass), FALSE, pass),
    in_metadata = embryo_ID %in% meta$embryo_ID
  ) %>%
  arrange(perturbation, timepoint, embryo_ID)

# A left join must never add or drop an embryo.
stopifnot(nrow(embryo_table) == nrow(spine))

write_tsv(embryo_table, file.path(OUT_DIR, "embryo_table.tsv"))
log_ts("wrote embryo_table.tsv (", nrow(embryo_table), " rows, ",
       ncol(embryo_table), " cols)")

# The design table. Read the cep290-negsib row: it is 0 at 30hpf, which is the
# entire reason that timepoint has to borrow b9d2-negsib as its control.
design <- embryo_table %>%
  count(perturbation, timepoint) %>%
  pivot_wider(names_from = timepoint, values_from = n, values_fill = 0)
print(design)

# ---- 5. per-embryo x cell-type counts --------------------------------------
log_ts("building per-embryo x cell-type counts ...")

cd <- filter(coldata, !is.na(timepoint), !is.na(cell_type))
all_cell_types <- sort(unique(as.character(cd$cell_type)))
cells_per_embryo <- count(cd, embryo_ID, name = "total_cells")

# Count cells per (embryo, cell type). Only combinations that OCCUR appear here.
observed <- count(cd, embryo_ID, perturbation, rt_block, timepoint, cell_type,
                  name = "n_cells")

# Fill in the combinations that did NOT occur, as explicit zeros.
#
# This is the step that matters most in this script. Without it, "this embryo had
# zero cells of that type" and "that type was never measured here" are the same
# thing -- a missing row -- and every downstream ratio, CLR transform and gate
# would silently treat one as the other. nesting() keeps the embryo's own
# attributes together instead of crossing them combinatorially.
counts <- observed %>%
  complete(
    nesting(embryo_ID, perturbation, rt_block, timepoint),#Without nesting(), complete() might cross every embryo with every perturbation,
    cell_type = all_cell_types,
    fill = list(n_cells = 0)
  ) %>%
  left_join(cells_per_embryo, by = "embryo_ID") %>%
  mutate(per_1000 = 1000 * n_cells / total_cells)

write_tsv(counts, file.path(OUT_DIR, "per_embryo_celltype_counts.tsv"))
log_ts("wrote per_embryo_celltype_counts.tsv (", nrow(counts), " rows = ",
       n_distinct(counts$embryo_ID), " embryos x ",
       length(all_cell_types), " cell types)")

log_ts("done. next: 1_cell_type_lineage.R")

#why do we have so many smaller inidivual ouptus for this i feel like it oul be stelained a bit 