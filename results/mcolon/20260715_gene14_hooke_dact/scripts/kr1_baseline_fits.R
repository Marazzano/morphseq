#!/usr/bin/env Rscript
# kr1_baseline_fits.R
# -----------------------------------------------------------------------------
# KR1 baseline (grid version of section 5 of manual_walkthrough.Rmd):
# each perturbation group vs its own (pooled) control, one binary hooke fit each.
#   cep290-mut vs cep290-ctrl-pool (negsib + AB, fills 30hpf hole)
#   b9d2-mut   vs b9d2-ctrl-pool   (negsib + AB, consistency)
#   foxj1a / ift88 / sspo  vs ctrl-inj
#
# Saves to output/ so you can load results back into an interactive session:
#   kr1_raw_contrasts.tsv   — all 5 raw contrast tables, stacked (col `tag`)
#   kr1_dacts.tsv           — the DACT-gated subset
#   kr1_raw_contrasts.rds   — same raw table as an R object
#
# Run on the GRID (loads the 7.3G CDS; login node OOM-kills):
#   qsub results/mcolon/20260715_gene14_hooke_dact/scripts/kr1_submit.sh
# -----------------------------------------------------------------------------
suppressMessages({
  library(monocle3)
  library(hooke)
  library(platt)
  library(splines)   # ns() — hooke's timepoint spline
  library(dplyr)
  library(purrr)
  library(readr)
  library(tibble)
})

CDS_PATH <- "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/mcclintock/GENE14/run_1/filter_embryos/embryo_filtered_cds"
OUT_DIR  <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260715_gene14_hooke_dact/output"

# threads from the grid slot allocation (default 1 off-grid)
NUM_THREADS <- as.integer(Sys.getenv("NSLOTS", "1"))
if (is.na(NUM_THREADS) || NUM_THREADS < 1) NUM_THREADS <- 1
message("num_threads = ", NUM_THREADS)

## --------------------------- load ------------------------------------------
message("Loading CDS: ", CDS_PATH)
cds <- load_monocle_objects(CDS_PATH)
message("  cds: ", ncol(cds), " cells x ", nrow(cds), " genes")

## --------------------------- colData prep ----------------------------------
# genotype_block: the fence, straight from rt_block. We SUBSET the ccs by this
# before each fit — inside a block only that gene's cells remain (mut/negsib/own AB),
# so plain `perturbation` fits cleanly and `reference` can't leak across genes.
colData(cds)$genotype_block <- dplyr::case_when(
  colData(cds)$rt_block == "Bl1"                           ~ "crispant",
  colData(cds)$rt_block %in% c("Bl2", "Bl3", "Bl4", "Bl5") ~ "cep290",
  colData(cds)$rt_block %in% c("Bl6", "Bl7", "Bl8", "Bl9") ~ "b9d2",
  TRUE                                                     ~ NA_character_
)
# batch_one: constant no-op batch term (rt_block is rank-deficient here)
colData(cds)$batch_one <- "b1"
stopifnot(!anyNA(colData(cds)$genotype_block))

## --------------------------- ccs -------------------------------------------
ccs <- new_cell_count_set(cds, sample_group = "embryo_ID", cell_group = "cell_type")
stopifnot(all(c("perturbation", "genotype_block", "timepoint", "batch_one")
              %in% colnames(colData(ccs))))

## --------------------------- fits ------------------------------------------
# 11 fits. cep290 & b9d2 each vs 3 controls (negsib / reference / pool) + the
# sibling-vs-reference control-cleanliness check; 3 crispants vs ctrl-inj.
# ctrl is a LIST column so scalar strings and the pooled vector coexist.
contrasts_all <- tibble::tribble(
  ~block,    ~tag,                  ~genotype,     ~ctrl,
  "cep290",  "cep290_vs_negsib",    "cep290-mut",  "cep290-negsib",
  "cep290",  "cep290_vs_reference", "cep290-mut",  "reference",
  "cep290",  "cep290_vs_pool",      "cep290-mut",  list(c("cep290-negsib", "reference")),
  "b9d2",    "b9d2_vs_negsib",      "b9d2-mut",    "b9d2-negsib",
  "b9d2",    "b9d2_vs_reference",   "b9d2-mut",    "reference",
  "b9d2",    "b9d2_vs_pool",        "b9d2-mut",    list(c("b9d2-negsib", "reference")),
  "cep290",  "cep290_negsib_vs_ref","cep290-negsib","reference",
  "b9d2",    "b9d2_negsib_vs_ref",  "b9d2-negsib", "reference",
  "crispant","foxj1a_vs_ctrl",      "foxj1a",      "ctrl-inj",
  "crispant","ift88_vs_ctrl",       "ift88",       "ctrl-inj",
  "crispant","sspo_vs_ctrl",        "sspo",        "ctrl-inj",
)
contrasts_all$ctrl <- lapply(contrasts_all$ctrl, unlist)

fit_within_block <- function(block, genotype, ctrl_ids, tag) {
  ccs_sub <- ccs[, colData(ccs)$genotype_block == block]   # SUBSET = the fence
  message("\n=== ", tag, "  (", genotype, " vs ", paste(ctrl_ids, collapse = "+"),
          " | block=", block, ", ", ncol(ccs_sub), " embryos) ===")
  ccm <- platt:::fit_genotype_ccm(
    genotype         = genotype,
    ccs              = ccs_sub,
    perturbation_col = "perturbation",
    ctrl_ids         = ctrl_ids,
    interval_col     = "timepoint",
    batch_col        = "batch_one",
    vhat_method      = "sandwich_var",
    num_threads      = NUM_THREADS
  )
  # save the fitted MODEL (for plotting). LARGE (~6GB — bundles the ccs).
  saveRDS(ccm, file.path(OUT_DIR, paste0("dact_ccm_", tag, ".rds")))
  tbl <- platt:::get_perturbation_effects(ccm, interval_col = "timepoint")
  tbl$tag <- tag
  write_tsv(tbl, file.path(OUT_DIR, paste0("dact_contrast_", tag, ".tsv")))
  message("  -> ", nrow(tbl), " rows")
  tbl
}

dact_raw <- purrr::pmap_dfr(
  contrasts_all,
  function(block, tag, genotype, ctrl) fit_within_block(block, genotype, ctrl, tag)
)

## --------------------------- DACT gate + save ------------------------------
dact_hits <- dact_raw %>% filter(delta_q_value < 0.05, log_abund_x >= -3)

write_tsv(dact_raw,  file.path(OUT_DIR, "dact_raw_contrasts.tsv"))
saveRDS(dact_raw,    file.path(OUT_DIR, "dact_raw_contrasts.rds"))
write_tsv(dact_hits, file.path(OUT_DIR, "dact_hits.tsv"))

message("\n=== DACT counts per contrast ===")
print(dact_hits %>% count(tag, name = "n_dacts") %>% arrange(desc(n_dacts)))

message("\n=== DACT counts per contrast ===")
print(kr1_dacts %>% count(tag, name = "n_dacts") %>% arrange(desc(n_dacts)))
message("\nWrote: kr1_raw_contrasts.tsv / .rds, kr1_dacts.tsv (+ per-contrast tsvs)")
