#!/usr/bin/env Rscript
# Step 0 of the edgeR pipeline -- run ONCE.
#
# The only script in this pipeline that touches the 8.4 GB CDS. It reduces it to a
# cell-type x embryo count matrix (~226 x 553) and never needs to be run again; everything
# downstream reads that small matrix. Loading the CDS per contrast is what made the previous
# Hooke attempt cost hours.
#
# Only colData is needed -- cell type and embryo assignments -- so this reads cds_object.rds
# directly rather than going through load_monocle_objects(), which would additionally reconnect
# the 24 GB BPCells expression matrix for no reason. The expression values are irrelevant here:
# cell-type abundance is a property of the annotations, not the counts.
#
# Writes to data/edger/:
#   cell_counts.csv    cell types (rows) x embryo_ID (columns), integer counts
#   embryo_totals.csv  embryo_ID, total_cells  -- becomes the GLM offset
#   cell_type_index.csv  cell type, total cells, mean fraction -- abundance diagnostics
#
# Usage:
#   module load R/4.4.1 && Rscript build_count_table.R

suppressPackageStartupMessages({
  library(monocle3)
})

HERE <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
OUT <- file.path(HERE, "data", "edger")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

CDS_RDS <- paste0(
  "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/",
  "mcclintock/GENE7/run_1/filter_embryos/embryo_filtered_cds/cds_object.rds"
)
SAMPLE_GROUP <- "embryo_ID"
CELL_GROUP <- "cell_type"     # native annotations, not cell_type_broad

message("reading cds_object.rds (7.8 GB serialised -- expect several minutes) ...")
started <- Sys.time()
cds <- readRDS(CDS_RDS)
message(sprintf("  loaded in %.0f s", as.numeric(difftime(Sys.time(), started, units = "secs"))))

metadata <- as.data.frame(SummarizedExperiment::colData(cds))
message(sprintf("  colData: %d cells x %d fields", nrow(metadata), ncol(metadata)))
stopifnot(all(c(SAMPLE_GROUP, CELL_GROUP) %in% colnames(metadata)))

keep <- !is.na(metadata[[CELL_GROUP]]) & !is.na(metadata[[SAMPLE_GROUP]])
message(sprintf("  dropping %d cells with missing %s or %s",
                sum(!keep), CELL_GROUP, SAMPLE_GROUP))
metadata <- metadata[keep, , drop = FALSE]

counts <- table(
  factor(as.character(metadata[[CELL_GROUP]])),
  factor(as.character(metadata[[SAMPLE_GROUP]]))
)
counts <- as.matrix(counts)
message(sprintf("  count table: %d cell types x %d embryos", nrow(counts), ncol(counts)))

totals <- data.frame(
  embryo_ID = colnames(counts),
  total_cells = as.integer(colSums(counts)),
  stringsAsFactors = FALSE
)
fractions <- sweep(counts, 2, pmax(colSums(counts), 1), "/")
index <- data.frame(
  cell_type = rownames(counts),
  total_cells = as.integer(rowSums(counts)),
  mean_fraction = as.numeric(rowMeans(fractions)),
  max_fraction = as.numeric(apply(fractions, 1, max)),
  n_embryos_present = as.integer(rowSums(counts > 0)),
  stringsAsFactors = FALSE
)
index <- index[order(-index$mean_fraction), ]

write.csv(as.data.frame.matrix(counts), file.path(OUT, "cell_counts.csv"))
write.csv(totals, file.path(OUT, "embryo_totals.csv"), row.names = FALSE)
write.csv(index, file.path(OUT, "cell_type_index.csv"), row.names = FALSE)

# The compositional failure mode is "one abundant type moves and drags every other coefficient
# with it", so how concentrated the panel is decides how much to trust the per-type calls.
message("\n=== abundance concentration (the compositional risk check) ===")
message(sprintf("  cell types                      : %d", nrow(counts)))
message(sprintf("  embryos                         : %d", ncol(counts)))
message(sprintf("  total cells                     : %d", sum(counts)))
message(sprintf("  median cells per embryo         : %.0f", median(totals$total_cells)))
message(sprintf("  largest type's mean fraction    : %.3f  (%s)",
                index$mean_fraction[1], index$cell_type[1]))
message(sprintf("  top-5 types' summed fraction    : %.3f", sum(index$mean_fraction[1:5])))
message("  top 10 types:")
print(head(index, 10), row.names = FALSE)

message(sprintf("\nwrote -> %s", OUT))
