#!/usr/bin/env Rscript
# 1_cell_type_lineage.R  --  what IS this cell type?
# =============================================================================
# Writes cell_type_lineage.tsv: one row per cell type, carrying every grouping
# vocabulary available, so a figure can pick whichever axis it needs.
#
# Separate from 0_load_mcclintock.R on purpose: that script answers "how many
# cells were there", this one answers "what is this cell type". Only the heatmap
# needs this one.
#
# TWO SOURCES, because they are different vocabularies
#   1. the CDS coldata  -> germ_layer, projection_group, short_name_broad, CL/UBERON ids
#   2. combined_graph.rds -> sulston_group (~30 lineage groups)
#
# WHY sulston_group NEEDS A LIFT
#   It is an EDGE attribute on the state graph, not a node one. Take the majority
#   sulston_group over all edges touching a node. Old script 24 worked this out;
#   old 42 factored it out. Reading the graph RDS avoids loading the 19 GB CDS.
#
# WHY NOT projection_group FOR THE HEATMAP
#   Old script 24 found projection_group members scatter incoherently across the
#   state-graph layout, so hulls drawn around them sprawl. sulston_group IS the
#   graph's own lineage vocabulary, which is why the heatmap blocks rows by it.
#   Both are written here; the heatmap chooses.
#
#   Rscript 0_shared/1_cell_type_lineage.R
# =============================================================================

suppressPackageStartupMessages({
  library(igraph); library(readr); library(dplyr); library(tibble)
})

HERE    <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260727_gene14_clean"
OUT_DIR <- file.path(HERE, "0_shared")

MCCLINTOCK <- "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/mcclintock/GENE14/run_1"
COLDATA    <- file.path(MCCLINTOCK, "filter_embryos", "embryo_filtered_cds_coldata.tsv")
GRAPH_RDS  <- "/net/seahub_zfish/vol1/data/graphs/v3.1.0/combined_graph.rds"

log_ts <- function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)
stopifnot(file.exists(COLDATA), file.exists(GRAPH_RDS))

# ---- 1. the ontology columns, from coldata ----------------------------------
# Only the annotation columns are read, so this is far cheaper than the full load.
ONTOLOGY <- c("cell_type", "germ_layer", "projection_group", "projection_layer1",
              "projection_layer2", "short_name", "short_name_broad",
              "CL_name", "CL_ID", "UBERON_name", "ZFA_ID")
log_ts("reading ontology columns from coldata ...")
cd <- read_tsv(COLDATA, col_select = any_of(ONTOLOGY),
               show_col_types = FALSE, progress = FALSE)

onto <- cd %>%
  filter(!is.na(cell_type)) %>%
  group_by(cell_type) %>%
  summarise(across(everything(), ~ dplyr::first(.x)), .groups = "drop")
log_ts("  ", nrow(onto), " cell types, ", n_distinct(onto$germ_layer), " germ layers, ",
       n_distinct(onto$projection_group), " projection groups")

# ---- 2. sulston_group, lifted edge -> node ----------------------------------
g <- readRDS(GRAPH_RDS)
log_ts("graph: ", vcount(g), " nodes / ", ecount(g), " edges")
ed <- igraph::as_data_frame(g, what = "edges")
stopifnot("sulston_group" %in% names(ed))

sulston <- bind_rows(
    ed %>% transmute(cell_type = from, sulston_group),
    ed %>% transmute(cell_type = to,   sulston_group)) %>%
  filter(!is.na(sulston_group)) %>%
  count(cell_type, sulston_group) %>%
  group_by(cell_type) %>% slice_max(n, with_ties = FALSE) %>%
  ungroup() %>% select(cell_type, sulston_group)

# graph nodes touching no annotated edge still deserve a row
iso <- setdiff(V(g)$name, sulston$cell_type)
if (length(iso)) sulston <- bind_rows(sulston,
                    tibble(cell_type = iso, sulston_group = "Unknown"))
log_ts("  sulston_group: ", nrow(sulston), " cell types, ",
       n_distinct(sulston$sulston_group), " groups")

# ---- 3. join, keeping every cell type that appears in EITHER source ---------
# full_join, not left: a cell type in the data but not the graph must still get a
# row (labelled Unknown) or the heatmap would silently drop it.
lineage <- full_join(onto, sulston, by = "cell_type") %>%
  mutate(sulston_group = ifelse(is.na(sulston_group), "Unknown", sulston_group)) %>%
  arrange(sulston_group, cell_type)

write_tsv(lineage, file.path(OUT_DIR, "cell_type_lineage.tsv"))
log_ts("wrote cell_type_lineage.tsv (", nrow(lineage), " cell types)")

n_data_only <- sum(!lineage$cell_type %in% sulston$cell_type)
log_ts("  in data but not in the graph (-> Unknown): ", n_data_only)
print(lineage %>% count(sulston_group, sort = TRUE), n = 40)

log_ts("done. next: 2_seq_imaging_crosswalk.py / 3_attach_morphseq_labels.py")
