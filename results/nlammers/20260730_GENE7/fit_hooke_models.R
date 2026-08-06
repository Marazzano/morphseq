#!/usr/bin/env Rscript
# Script 2 of 3 -- the only R in this pipeline.
#
#   Reads:  data/morph_covariates.csv   (export_morph_covariates.py)
#   Writes: data/hooke/cohort_coefficients.csv, cohort_status.csv, timings.csv
#
# ONE PLN REGRESSION PER COHORT.
#
#   cohort = target x temperature x timepoint  (48 cohorts, 9-12 embryos each)
#   formula = ~ cohort_PC1 + cohort_PC2
#
# The predictors are each cohort's OWN two principal axes, computed in Python inside a shared
# denoised subspace (the first 5 dims of a 10-component GENE7-native PCA), centered on that cohort's
# own mean. So within any one fit the predictor means one thing for every embryo in it. Cross-cohort
# comparison happens at the level of coefficients, never by assuming the axes are the same vector --
# they are not (measured subspace similarity is barely above the random-plane floor).
#
# Deliberately NOT done here:
#   * no pooled / cohort-interacted model
#   * no global PCs as predictors (they are not commensurable across cohorts)
#   * no latents saved -- per-cohort latents are conditioned on their own fit and are not comparable
#   * no covariance matrix saved -- spherical, not worth the file
#
# Depth is handled automatically: Hooke appends offset(log(Offset)) to every formula, with Offset
# from monocle3::size_factors(). Coefficients are on composition, not raw abundance.
#
# Usage:
#   module load R/4.4.1
#   Rscript fit_hooke_models.R                 # all 48 cohorts
#   Rscript fit_hooke_models.R --cohorts 1     # just the first -- runtime probe
#   Rscript fit_hooke_models.R --vhat variational_var   # cheaper SEs than the bootstrap default

suppressPackageStartupMessages({
  library(monocle3)
  library(BPCells)
  library(hooke)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) && length(args) > hit[1]) args[hit[1] + 1] else default
}

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
HERE <- normalizePath(dirname(script_path))
DATA_DIR <- file.path(HERE, "data")
OUT_ROOT <- file.path(DATA_DIR, "hooke")
COVARIATE_CSV <- file.path(DATA_DIR, "morph_covariates.csv")

# v3.1.0 embryo-filtered CDS -- the bead-milling generation that matches GENE7.
DEFAULT_CDS <- paste0(
  "/net/seahub_zfish/vol1/data/seahub_rna_processing/portal_inputs/v3.1.0/",
  "mcclintock/GENE7/run_1/filter_embryos/embryo_filtered_cds"
)
CDS_PATH  <- get_arg("--cds", DEFAULT_CDS)
N_COHORTS <- as.integer(get_arg("--cohorts", "0"))   # 0 = all
VHAT      <- get_arg("--vhat", "bootstrap")
# new_cell_count_model fits the REDUCED (~1) model with PLNmodels::PLNnetwork over an
# n_penalties-long sparse-precision path. covariance_type is never passed to that call, so it runs a
# full 226x226 penalised precision estimation regardless of the spherical setting on the full model
# -- and it dominated the runtime probe (>8 min for a 10-embryo cohort). The reduced fit is a
# required slot on the CCM class, so it cannot be skipped; shortening the path is the lever.
N_PENALTIES <- as.integer(get_arg("--penalties", "1"))
TEMP_PATH <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"

dir.create(TEMP_PATH, showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_ROOT, showWarnings = FALSE, recursive = TRUE)

SAMPLE_GROUP <- "embryo_ID"
CELL_GROUP   <- "cell_type"        # native annotations, not cell_type_broad
FORMULA_STR  <- "~ cohort_PC1 + cohort_PC2"

timings <- list()
stamp <- function(label, expr) {
  t0 <- Sys.time()
  value <- force(expr)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  timings[[label]] <<- elapsed
  message(sprintf("  [%s] %.1f s", label, elapsed))
  value
}

# ---- Covariates -------------------------------------------------------------
stopifnot(file.exists(COVARIATE_CSV))
covariates <- read.csv(COVARIATE_CSV, stringsAsFactors = FALSE)
stopifnot(all(c("sample", "cohort", "cohort_PC1", "cohort_PC2") %in% colnames(covariates)))
message(sprintf("covariates: %d embryos across %d cohorts",
                nrow(covariates), length(unique(covariates$cohort))))

# ---- Load CDS ---------------------------------------------------------------
message("loading CDS (8.4 GB -- expect this to dominate)...")
cds <- stamp("load_cds", {
  load_monocle_objects(
    CDS_PATH,
    matrix_control = list(matrix_class = "BPCells", matrix_path = TEMP_PATH)
  )
})
message(sprintf("  cds: %d genes x %d cells", nrow(cds), ncol(cds)))

keep <- !is.na(colData(cds)[[CELL_GROUP]])
message(sprintf("  dropping %d cells with NA %s", sum(!keep), CELL_GROUP))
cds <- cds[, keep]

cds_embryos <- unique(as.character(colData(cds)[[SAMPLE_GROUP]]))
shared <- intersect(cds_embryos, covariates$sample)
message(sprintf("  cds embryos %d | covariates %d | shared %d",
                length(cds_embryos), nrow(covariates), length(shared)))
if (length(shared) == 0) {
  stop("No overlap between CDS embryo_ID and covariate 'sample'. Check the RT-block suffix form.")
}

# ---- Per-cohort fits --------------------------------------------------------
cohorts <- sort(unique(covariates$cohort[covariates$sample %in% shared]))
if (N_COHORTS > 0) {
  cohorts <- head(cohorts, N_COHORTS)
  message(sprintf("\n*** RUNTIME PROBE: %d cohort(s) only ***", length(cohorts)))
}
message(sprintf("fitting %d cohort model(s): %s", length(cohorts), FORMULA_STR))
message(sprintf("  vhat_method=%s  pln_num_penalties=%d\n", VHAT, N_PENALTIES))

coefficient_rows <- list()
status_rows <- list()

for (cohort_name in cohorts) {
  message(sprintf("=== %s ===", cohort_name))
  members <- covariates[covariates$cohort == cohort_name & covariates$sample %in% shared, ]
  message(sprintf("  %d embryos", nrow(members)))

  if (nrow(members) < 5) {
    status_rows[[cohort_name]] <- data.frame(
      cohort = cohort_name, n_embryos = nrow(members),
      status = "skipped_too_few", message = NA_character_
    )
    next
  }

  cohort_cds <- cds[, as.character(colData(cds)[[SAMPLE_GROUP]]) %in% members$sample]
  # Hooke asserts nrow(sample_metadata) == n distinct sample_group values in the cds.
  members <- members[match(
    unique(as.character(colData(cohort_cds)[[SAMPLE_GROUP]])), members$sample
  ), , drop = FALSE]

  result <- tryCatch({
    ccs <- stamp(paste0("ccs_", cohort_name), {
      new_cell_count_set(
        cohort_cds,
        sample_group    = SAMPLE_GROUP,
        cell_group      = CELL_GROUP,
        sample_metadata = members,
        keep_cds        = FALSE
      )
    })
    message(sprintf("  ccs: %d cell groups x %d samples", nrow(ccs), ncol(ccs)))

    ccm <- stamp(paste0("fit_", cohort_name), {
      new_cell_count_model(
        ccs,
        main_model_formula_str = FORMULA_STR,
        covariance_type        = "spherical",
        vhat_method            = VHAT,
        pln_num_penalties      = N_PENALTIES,
        num_threads            = 4
      )
    })
    list(ccs = ccs, ccm = ccm)
  }, error = function(e) {
    message(sprintf("  FAILED: %s", conditionMessage(e)))
    status_rows[[cohort_name]] <<- data.frame(
      cohort = cohort_name, n_embryos = nrow(members),
      status = "failed", message = conditionMessage(e)
    )
    NULL
  })
  if (is.null(result)) next

  fit <- result$ccm@best_full_model

  # Coefficient matrix is TERMS x CELL TYPES (rows = (Intercept)/cohort_PC1/cohort_PC2).
  # Verified against a synthetic PLN fit -- do not assume the transpose.
  cf <- tryCatch(coef(fit, type = "main"), error = function(e) NULL)
  if (is.null(cf)) {
    status_rows[[cohort_name]] <- data.frame(
      cohort = cohort_name, n_embryos = nrow(members),
      status = "no_coefficients", message = NA_character_
    )
    next
  }
  B <- as.matrix(cf)

  # PLNmodels exposes per-coefficient variances as an ATTRIBUTE of coef(), element-wise aligned
  # with B -- there is no fit$standard_error and standard_error() returns NULL. The attribute name
  # tracks vhat_method: variance_bootstrap / variance_variational / etc.
  variance_attr <- paste0("variance_", sub("_var$", "", VHAT))
  V <- attributes(cf)[[variance_attr]]
  if (is.null(V)) {
    alternatives <- grep("^variance_", names(attributes(cf)), value = TRUE)
    if (length(alternatives)) V <- attributes(cf)[[alternatives[1]]]
  }

  long <- data.frame(
    cohort     = cohort_name,
    n_embryos  = nrow(members),
    term       = rep(rownames(B), times = ncol(B)),
    cell_type  = rep(colnames(B), each = nrow(B)),
    estimate   = as.vector(B),
    stringsAsFactors = FALSE
  )
  if (!is.null(V) && all(dim(as.matrix(V)) == dim(B))) {
    long$std_error <- sqrt(as.vector(as.matrix(V)))
    long$z_value   <- long$estimate / long$std_error
    long$p_value   <- 2 * pnorm(-abs(long$z_value))
  } else {
    message("  WARNING: no per-coefficient variances; writing point estimates only")
  }
  coefficient_rows[[cohort_name]] <- long

  status_rows[[cohort_name]] <- data.frame(
    cohort = cohort_name, n_embryos = nrow(members),
    status = "ok", message = NA_character_
  )
  message(sprintf("  wrote %d coefficient rows", nrow(long)))
}

# ---- Write ------------------------------------------------------------------
if (length(coefficient_rows)) {
  all_coefficients <- do.call(rbind, coefficient_rows)
  # Benjamini-Hochberg within each cohort: ~100 cell types x 2 terms per fit.
  if ("p_value" %in% colnames(all_coefficients)) {
    all_coefficients <- all_coefficients %>%
      group_by(cohort) %>%
      mutate(q_value = p.adjust(p_value, method = "BH")) %>%
      ungroup() %>%
      as.data.frame()
  }
  write.csv(all_coefficients, file.path(OUT_ROOT, "cohort_coefficients.csv"), row.names = FALSE)
  message(sprintf("\nwrote %d coefficient rows", nrow(all_coefficients)))
}

status <- do.call(rbind, status_rows)
write.csv(status, file.path(OUT_ROOT, "cohort_status.csv"), row.names = FALSE)
message("status:")
print(table(status$status))

timing_table <- data.frame(stage = names(timings), seconds = unlist(timings),
                           row.names = NULL, stringsAsFactors = FALSE)
write.csv(timing_table, file.path(OUT_ROOT, "timings.csv"), row.names = FALSE)
message("\n=== timings (s) ===")
print(timing_table)
message(sprintf("\noutputs -> %s", OUT_ROOT))
