#!/usr/bin/env Rscript
# Unsupervised counterpart to fit_edger_contrasts.R -- held deliberately separate.
#
#   Reads:  data/edger/cell_counts.csv, embryo_totals.csv
#           data/unsupervised/unsupervised_predictors.csv
#   Writes: data/unsupervised/{coefficients,global_stats,status}.csv
#
# TWO ARMS, neither of which uses a crispant/control label as a predictor:
#
#   pooled_pc1   PC1 of the pooled control + crispant cloud, regressed over ALL contrast members.
#                Asks: with no labels at all, does the dominant axis of morphological variation
#                recover the perturbation's transcriptional signature?
#
#   within_pc1   PC1 of the crispant embryos alone, regressed over CRISPANTS ONLY (~11 embryos).
#                The label-free counterpart of the supervised within-crispant dose slope.
#
# Machinery is identical to the supervised script -- edgeR quasi-likelihood, offset = log(total
# cells), BH within contrast, and the same permutation-calibrated global statistic -- so the two
# sets of results are directly comparable.
#
# WHERE A LABEL STILL LEAKS IN, deliberately and in one place only: cell-type filtering uses the
# binary design, exactly as the supervised run did. This fixes WHICH cell types are in play so the
# recovery denominators match between the two analyses; it does not touch the axis or the test. A
# strictly label-free filter would have shifted the denominator and made recovery fractions
# uninterpretable. Flagged rather than hidden.
#
# The `within_pc1` arm necessarily runs on a different embryo set (crispants only, ~11) from the
# binary reference (~23), so recovery fractions for that arm are not strictly comparable either.
#
# Usage:
#   module load R/4.4.1
#   Rscript fit_edger_unsupervised.R [--permutations 2000]

suppressPackageStartupMessages({ library(edgeR) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) && length(args) > hit[1]) args[hit[1] + 1] else default
}
HERE <- normalizePath(dirname(sub("^--file=", "",
                                  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
EDGER <- file.path(HERE, "data", "edger")
OUT <- file.path(HERE, "data", "unsupervised")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

N_PERMUTATIONS <- as.integer(get_arg("--permutations", "2000"))
SHRINKAGE <- as.numeric(get_arg("--shrinkage", "0.5"))
MIN_EMBRYOS <- 12L
MIN_CRISPANT <- 8L
set.seed(42)

# Same score-test machinery as the supervised script (duplicated rather than sourced, to keep the
# validated pipeline untouched by this exploratory one).
score_test <- function(residual, x_centered, n_permutations, shrinkage) {
  observed <- as.vector(residual %*% x_centered)
  null_u <- matrix(0, nrow = nrow(residual), ncol = n_permutations)
  for (b in seq_len(n_permutations)) null_u[, b] <- residual %*% sample(x_centered)
  null_sd <- apply(null_u, 1, sd); null_sd[null_sd <= 0] <- NA_real_
  z_observed <- observed / null_sd
  z_null <- sweep(null_u, 1, null_sd, "/")
  finite <- is.finite(z_observed) & apply(is.finite(z_null), 1, all)
  t_observed <- sum(z_observed[finite]^2)
  t_null <- colSums(z_null[finite, , drop = FALSE]^2)
  correlation <- cor(t(z_null[finite, , drop = FALSE]))
  regularised <- (1 - shrinkage) * correlation + shrinkage * diag(sum(finite))
  list(z = ifelse(finite, z_observed, 0), n_finite = sum(finite),
       t_std = (t_observed - mean(t_null)) / sd(t_null),
       p_global = (sum(t_null >= t_observed) + 1) / (n_permutations + 1),
       d_shrunk = sqrt(max(as.numeric(t(z_observed[finite]) %*%
                                        solve(regularised, z_observed[finite])), 0)))
}

message("reading inputs ...")
counts_all <- as.matrix(read.csv(file.path(EDGER, "cell_counts.csv"), row.names = 1,
                                 check.names = FALSE))
totals_all <- read.csv(file.path(EDGER, "embryo_totals.csv"), stringsAsFactors = FALSE)
rownames(totals_all) <- totals_all$embryo_ID
predictors <- read.csv(file.path(OUT, "unsupervised_predictors.csv"), stringsAsFactors = FALSE)
message(sprintf("  %d contrasts | %d cell types x %d embryos\n",
                length(unique(predictors$contrast)), nrow(counts_all), ncol(counts_all)))

coefficient_rows <- list(); global_rows <- list(); status_rows <- list()

for (contrast_name in sort(unique(predictors$contrast))) {
  members <- predictors[predictors$contrast == contrast_name, ]
  members <- members[members$sample %in% colnames(counts_all), ]
  if (nrow(members) < MIN_EMBRYOS) next

  library_size <- totals_all[members$sample, "total_cells"]
  dge <- DGEList(counts = counts_all[, members$sample, drop = FALSE], lib.size = library_size)
  dge <- scaleOffset(dge, matrix(log(library_size), nrow = nrow(dge), ncol = ncol(dge),
                                 byrow = TRUE))
  design_binary <- model.matrix(~ members$is_crispant)   # filtering only -- see header
  dge <- dge[filterByExpr(dge, design = design_binary), , keep.lib.sizes = TRUE]
  dge <- estimateDisp(dge, design_binary, robust = TRUE)

  emit <- function(arm, sub_dge, x, subset_note) {
    quasi <- glmQLFit(sub_dge, cbind(1, x), dispersion = dge$tagwise.dispersion[
      match(rownames(sub_dge), rownames(dge))], robust = TRUE)
    table <- topTags(glmQLFTest(quasi, coef = 2), n = Inf, sort.by = "none")$table
    null_fit <- glmFit(sub_dge, matrix(1, nrow = ncol(sub_dge), ncol = 1),
                       dispersion = dge$tagwise.dispersion[
                         match(rownames(sub_dge), rownames(dge))])
    result <- score_test(sub_dge$counts - null_fit$fitted.values, x - mean(x),
                         N_PERMUTATIONS, SHRINKAGE)
    coefficient_rows[[paste(contrast_name, arm)]] <<- data.frame(
      contrast = contrast_name, arm = arm, cell_type = rownames(sub_dge),
      logFC = table$logFC, logCPM = table$logCPM, F = table$F, p_value = table$PValue,
      q_value = p.adjust(table$PValue, method = "BH"), score_z = result$z,
      stringsAsFactors = FALSE)
    global_rows[[paste(contrast_name, arm)]] <<- data.frame(
      contrast = contrast_name, arm = arm, n_embryos = ncol(sub_dge),
      n_cell_types = result$n_finite, t_std = result$t_std, p_global = result$p_global,
      d_shrunk = result$d_shrunk,
      n_hits_q10 = sum(p.adjust(table$PValue, method = "BH") < 0.10, na.rm = TRUE),
      n_hits_q05 = sum(p.adjust(table$PValue, method = "BH") < 0.05, na.rm = TRUE),
      subset = subset_note, stringsAsFactors = FALSE)
  }

  # ---- arm 1: pooled PC1 over every embryo in the contrast ----
  if (sd(members$pooled_pc1_z) > 1e-9) {
    emit("pooled_pc1", dge, members$pooled_pc1_z, "all contrast members")
  }

  # ---- arm 2: within-crispant PC1, crispants only ----
  crispants <- which(members$is_crispant == 1 & is.finite(members$within_pc1_z))
  if (length(crispants) >= MIN_CRISPANT && sd(members$within_pc1_z[crispants]) > 1e-9) {
    emit("within_pc1", dge[, crispants, keep.lib.sizes = TRUE],
         members$within_pc1_z[crispants], "crispants only")
  }

  status_rows[[contrast_name]] <- data.frame(
    contrast = contrast_name, n_embryos = nrow(members),
    n_crispant = sum(members$is_crispant == 1), n_cell_types = nrow(dge),
    stringsAsFactors = FALSE)
  message(sprintf("  %-34s n=%2d  types=%3d", contrast_name, nrow(members), nrow(dge)))
}

write.csv(do.call(rbind, coefficient_rows), file.path(OUT, "coefficients.csv"), row.names = FALSE)
write.csv(do.call(rbind, global_rows), file.path(OUT, "global_stats.csv"), row.names = FALSE)
write.csv(do.call(rbind, status_rows), file.path(OUT, "status.csv"), row.names = FALSE)

global <- do.call(rbind, global_rows)
message("\n=== unsupervised arms ===")
for (arm in unique(global$arm)) {
  b <- global[global$arm == arm, ]
  message(sprintf("  %-12s %2d contrasts | median T_std %+.2f | p<0.05 in %2d | median hits %.0f | total hits %d",
                  arm, nrow(b), median(b$t_std), sum(b$p_global < 0.05),
                  median(b$n_hits_q10), sum(b$n_hits_q10)))
}
message(sprintf("\nwrote -> %s", OUT))
