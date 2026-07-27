#!/usr/bin/env Rscript
# 0_reconcile_metadata_qc.R
# -----------------------------------------------------------------------------
# GENE14 Phase 3 (hooke / McClintock DACT) — metadata + QC reconciliation.
#
# hooke/McClintock has ALREADY been run for GENE14. This script does NOT run
# hooke. It does the deterministic bookkeeping *around* the finished run:
#   1. Reads the McClintock projected-CDS colData (the embryos hooke actually fit).
#   2. Joins the authoritative embryo metadata + the pass/fail sequencing-QC list.
#   3. Reconciles MorphSeq embryo_IDs against the updated sequencing embryo_IDs
#      (18hpf cep290 were redone; see 20260607_sci_cilia_gene14_imaging_qc/README).
#   4. Emits a single per-embryo reconciled table that downstream DACT/PCA
#      scripts consume, plus a QC report of any id mismatches.
#
# It reads the *coldata TSV* (fast, ~MBs), NOT the 7GB CDS object, so it is cheap
# to re-run. Load the CDS itself only in the actual hooke/PCA step.
#
# Run:
#   Rscript 0_reconcile_metadata_qc.R
# -----------------------------------------------------------------------------

suppressMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(tibble)
})

## ----------------------------- inputs ---------------------------------------
MCCLINTOCK_RUN <- "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/mcclintock/GENE14/run_1"

# Projected-CDS colData (one row per CELL) — the embryos McClintock kept.
CDS_COLDATA_TSV <- file.path(MCCLINTOCK_RUN, "filter_embryos", "embryo_filtered_cds_coldata.tsv")

# Authoritative embryo metadata (one row per embryo).
EMBRYO_METADATA_TSV <- "/net/seahub_zfish/vol1/data/preprocessed/GENE14/GENE14_embryo_metadata.tsv"

# Sequencing pass/fail list (TRUE = made it into the final cds). run2-aligned.
QC_PASS_CSV <- "/net/seahub_zfish/vol1/data/BBI_dmux_sci/GENE14/260625_GENE14_run2_novaseqx/GENE14/missing_embryos_list.csv"

OUT_DIR <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260715_gene14_hooke_dact/output"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ---------------------- 1. embryos hooke actually fit ------------------------
# The coldata is per-cell; collapse to one row per embryo. We only need the
# columns that identify the embryo + its grouping variables for hooke.
message("Reading projected-CDS colData: ", CDS_COLDATA_TSV)
coldata <- read_tsv(CDS_COLDATA_TSV, show_col_types = FALSE)

keep_cols <- intersect(
  c("embryo_ID", "perturbation", "target", "pheno", "timepoint",
    "reference", "compare_against", "collection_batch", "type", "allele"),
  colnames(coldata)
)

cds_embryos <- coldata %>%
  group_by(embryo_ID) %>%
  summarise(
    n_cells = n(),
    across(all_of(setdiff(keep_cols, "embryo_ID")), ~ dplyr::first(.x)),
    .groups = "drop"
  )
message("  embryos in fitted CDS: ", nrow(cds_embryos),
        "  (", sum(cds_embryos$n_cells), " cells)")

## ------------------------- 2. authoritative metadata ------------------------
message("Reading embryo metadata: ", EMBRYO_METADATA_TSV)
meta <- read_tsv(EMBRYO_METADATA_TSV, show_col_types = FALSE) %>%
  rename(embryo_ID = embryo_ID)   # explicit; header is already `embryo_ID`

## ------------------------------- 3. QC list ---------------------------------
message("Reading QC pass/fail list: ", QC_PASS_CSV)
qc <- read_csv(QC_PASS_CSV, show_col_types = FALSE) %>%
  select(embryo_ID, pass) %>%
  mutate(pass = as.logical(pass))

## --------------------- 4. reconcile / join / report -------------------------
# Set relationships between the three id universes.
ids_cds  <- unique(cds_embryos$embryo_ID)
ids_meta <- unique(meta$embryo_ID)
ids_qc   <- unique(qc$embryo_ID)

report <- tibble(
  set = c("in_cds", "in_metadata", "in_qc_list",
          "cds_not_in_metadata", "cds_not_in_qc",
          "qc_pass_not_in_cds", "metadata_not_in_qc"),
  n = c(
    length(ids_cds),
    length(ids_meta),
    length(ids_qc),
    length(setdiff(ids_cds, ids_meta)),
    length(setdiff(ids_cds, ids_qc)),
    length(setdiff(qc$embryo_ID[qc$pass %in% TRUE], ids_cds)),
    length(setdiff(ids_meta, ids_qc))
  )
)
message("\n=== id-set reconciliation ===")
print(report)

# Any CDS embryo missing from metadata is a hard mismatch (likely the redone
# 18hpf cep290 ids). Surface them explicitly rather than silently dropping.
cds_orphans <- setdiff(ids_cds, ids_meta)
if (length(cds_orphans)) {
  message("\nWARNING: ", length(cds_orphans),
          " CDS embryo_IDs have NO metadata row (candidate redone-id mismatches):")
  print(head(cds_orphans, 20))
}

# Build the reconciled per-embryo table. Left-join keeps every embryo hooke fit;
# metadata cols already present in the CDS coldata win (they are what hooke used).
reconciled <- cds_embryos %>%
  left_join(qc, by = "embryo_ID") %>%
  # metadata columns NOT already carried in the CDS coldata, for provenance
  left_join(
    meta %>% select(embryo_ID, any_of(setdiff(colnames(meta), keep_cols))),
    by = "embryo_ID"
  ) %>%
  mutate(
    pass = ifelse(is.na(pass), FALSE, pass),
    in_metadata = embryo_ID %in% ids_meta
  ) %>%
  arrange(perturbation, timepoint, embryo_ID)

## ------------------------------- 5. write -----------------------------------
out_tbl <- file.path(OUT_DIR, "gene14_embryos_reconciled.tsv")
out_rep <- file.path(OUT_DIR, "id_reconciliation_report.tsv")
out_orph <- file.path(OUT_DIR, "cds_embryos_missing_metadata.tsv")

write_tsv(reconciled, out_tbl)
write_tsv(report, out_rep)
write_tsv(tibble(embryo_ID = cds_orphans), out_orph)

# perturbation x timepoint x pass summary — sanity for the DACT contrasts.
summary_tbl <- reconciled %>%
  count(perturbation, timepoint, pass, name = "n_embryos") %>%
  arrange(perturbation, timepoint, pass)
write_tsv(summary_tbl, file.path(OUT_DIR, "perturbation_timepoint_pass_summary.tsv"))

message("\nWrote:")
message("  ", out_tbl,  "  (", nrow(reconciled), " embryos)")
message("  ", out_rep)
message("  ", out_orph, "  (", length(cds_orphans), " orphans)")
message("  ", file.path(OUT_DIR, "perturbation_timepoint_pass_summary.tsv"))
message("\n=== perturbation x timepoint x pass ===")
print(summary_tbl, n = Inf)
