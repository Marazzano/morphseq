#!/usr/bin/env Rscript
# Step 2 of the edgeR pipeline -- the only regression script.
#
#   Reads:  data/edger/cell_counts.csv, embryo_totals.csv, contrast_predictors.csv
#   Writes: data/edger/{coefficients,global_stats,arm_geometry,contrast_status}.csv
#
# THREE ARMS PER CONTRAST, one predictor each, identical degrees of freedom:
#   binary_z  the classical crispant/control indicator -- the baseline to beat
#   s_z       signed distance normal to the shrunken-LDA hyperplane
#   hinge_z   max(s - control mean, 0) -- controls collapsed, crispant gradient kept
#
# Same df means their deviances and global statistics are directly comparable with no penalty term.
#
# ---------------------------------------------------------------------------------------------
# WHY THE GLOBAL STATISTIC IS NOT A FULL MAHALANOBIS DISTANCE
# ---------------------------------------------------------------------------------------------
# The intent was D = ||Sigma_null^{-1/2} beta||, treating the coefficient vector as a point in
# ~370-D space and asking how far it sits from the null. That is degenerate here, and provably so.
#
# The score for cell type k is U_k = sum_i x_i (y_ki - mu0_ki), i.e. U = R x with R the K x n matrix
# of null-model residuals. U is LINEAR in the predictor. Permuting a z-scored x therefore gives
#   Cov(U) = R (I - 11'/n) R'
# whose rank is at most n - 1 = ~21, no matter how many cell types there are and no matter how many
# permutations are drawn -- the rank is set by the embryo count, not the sample size of the null.
# Inverting it (pseudo-inverse) yields D^2 = x'(I - 11'/n)x = n - 1 for EVERY z-scored predictor:
# identical for all three arms, carrying no information whatsoever.
#
# The honest statistic given n = 22 is the permutation-calibrated sum of squared standardised
# scores, T = sum_k z_k^2, compared against its own permutation distribution. The cross-type
# dependence is fully absorbed because the null is generated with that dependence intact -- it is
# handled by construction rather than by estimating and inverting a covariance that cannot be
# estimated. Reported as a standardised effect (T - mean_null) / sd_null, which is comparable
# across arms and contrasts, plus an exact permutation p-value.
#
# A shrinkage-regularised Mahalanobis is also written (`d_shrunk`) as a sensitivity arm: it
# interpolates between the diagonal (lambda = 1) and the degenerate full covariance (lambda = 0).
# It is a secondary number; T_std is the headline.
#
# Scale invariance: rescaling a predictor rescales U and its null SD together, so z, T and D are
# untouched. The comparison cannot be won by `s` merely having more range than a 0/1 indicator.
#
# Usage:
#   module load R/4.4.1
#   Rscript fit_edger_contrasts.R
#   Rscript fit_edger_contrasts.R --permutations 200   # quick pass

suppressPackageStartupMessages({
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) && length(args) > hit[1]) args[hit[1] + 1] else default
}

HERE <- normalizePath(dirname(sub("^--file=", "",
                                  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
DATA <- file.path(HERE, "data", "edger")

N_PERMUTATIONS <- as.integer(get_arg("--permutations", "2000"))
SHRINKAGE      <- as.numeric(get_arg("--shrinkage", "0.5"))
MIN_EMBRYOS    <- as.integer(get_arg("--min-embryos", "12"))
MIN_CELL       <- as.integer(get_arg("--min-cell", "4"))   # three-level factor level size
ARMS <- c("binary_z", "s_z", "hinge_z")
SEED <- 42

# ---------------------------------------------------------------------------------------------
# Shared score-test machinery
# ---------------------------------------------------------------------------------------------
# Every test below is "does predictor x shift the composition beyond its own null", and all of them
# reduce to the same object: the score U = R x, with R the residual matrix from whatever NULL model
# the test is conditioning on. Only two things change between tests -- which null model R comes
# from, and how x is permuted. Factoring it out keeps those two choices explicit instead of buried.
score_test <- function(residual, x_centered, permute, n_permutations, shrinkage) {
  observed <- as.vector(residual %*% x_centered)
  null_u <- matrix(0, nrow = nrow(residual), ncol = n_permutations)
  for (b in seq_len(n_permutations)) null_u[, b] <- residual %*% permute(x_centered)

  null_sd <- apply(null_u, 1, sd)
  null_sd[null_sd <= 0] <- NA_real_
  z_observed <- observed / null_sd
  z_null <- sweep(null_u, 1, null_sd, "/")
  finite <- is.finite(z_observed) & apply(is.finite(z_null), 1, all)

  t_observed <- sum(z_observed[finite]^2)
  t_null <- colSums(z_null[finite, , drop = FALSE]^2)

  correlation <- cor(t(z_null[finite, , drop = FALSE]))
  regularised <- (1 - shrinkage) * correlation + shrinkage * diag(sum(finite))
  d_shrunk <- sqrt(max(as.numeric(
    t(z_observed[finite]) %*% solve(regularised, z_observed[finite])
  ), 0))

  list(
    z = ifelse(finite, z_observed, 0), finite = finite,
    t_observed = t_observed, t_null_mean = mean(t_null), t_null_sd = sd(t_null),
    t_std = (t_observed - mean(t_null)) / sd(t_null),
    p_global = (sum(t_null >= t_observed) + 1) / (n_permutations + 1),
    d_shrunk = d_shrunk
  )
}

# Free reshuffle -- the null for a predictor whose assignment to embryos is arbitrary.
permute_free <- function(x) sample(x)

# Reshuffle WITHIN group -- the null for s_within. Permuting across groups would break the
# orthogonality to the binary indicator and test a different (already answered) hypothesis; the
# question here is only whether the ordering *inside* each group is arbitrary.
make_permute_within <- function(group) {
  levels_present <- unique(group)
  function(x) {
    out <- x
    for (level in levels_present) {
      index <- which(group == level)
      out[index] <- x[sample(index)]
    }
    out
  }
}

set.seed(SEED)

message("reading cached inputs ...")
counts_all <- as.matrix(read.csv(file.path(DATA, "cell_counts.csv"), row.names = 1,
                                 check.names = FALSE))
totals_all <- read.csv(file.path(DATA, "embryo_totals.csv"), stringsAsFactors = FALSE)
predictors <- read.csv(file.path(DATA, "contrast_predictors.csv"), stringsAsFactors = FALSE)
rownames(totals_all) <- totals_all$embryo_ID
message(sprintf("  %d cell types x %d embryos | %d contrasts",
                nrow(counts_all), ncol(counts_all), length(unique(predictors$contrast))))
message(sprintf("  %d permutations, shrinkage lambda = %.2f\n", N_PERMUTATIONS, SHRINKAGE))

coefficient_rows <- list()
global_rows <- list()
geometry_rows <- list()
status_rows <- list()

contrasts <- sort(unique(predictors$contrast))
started_all <- Sys.time()

for (contrast_name in contrasts) {
  members <- predictors[predictors$contrast == contrast_name, ]
  present <- members$sample %in% colnames(counts_all)
  dropped <- sum(!present)
  members <- members[present, ]

  if (nrow(members) < MIN_EMBRYOS || length(unique(members$is_crispant)) < 2) {
    status_rows[[contrast_name]] <- data.frame(
      contrast = contrast_name, n_embryos = nrow(members), n_dropped = dropped,
      n_cell_types = NA_integer_, n_control = NA_integer_, n_escaper = NA_integer_,
      n_severe = NA_integer_, three_level_estimable = NA,
      status = "skipped_too_few", stringsAsFactors = FALSE
    )
    message(sprintf("  [skip] %-34s only %d embryos in the CDS", contrast_name, nrow(members)))
    next
  }

  sub_counts <- counts_all[, members$sample, drop = FALSE]
  library_size <- totals_all[members$sample, "total_cells"]

  dge <- DGEList(counts = sub_counts, lib.size = library_size)
  # Total cells as the offset, NOT TMM. TMM assumes most features are unchanged, which is fine
  # across 20k genes but shaky across ~370 cell types where a phenotype can move many at once.
  # The explicit offset makes the compositional assumption visible instead of burying it in a
  # normalisation factor.
  #
  # Passed as a full K x n matrix: scaleOffset() given a bare vector stores a 1-D offset that then
  # fails to subset alongside the counts when cell types are filtered.
  dge <- scaleOffset(dge, matrix(log(library_size), nrow = nrow(dge), ncol = ncol(dge),
                                 byrow = TRUE))

  design_binary <- model.matrix(~ members$binary_z)
  keep <- filterByExpr(dge, design = design_binary)
  dge <- dge[keep, , keep.lib.sizes = TRUE]
  n_types <- nrow(dge)
  if (n_types < 10) {
    status_rows[[contrast_name]] <- data.frame(
      contrast = contrast_name, n_embryos = nrow(members), n_dropped = dropped,
      n_cell_types = n_types, n_control = NA_integer_, n_escaper = NA_integer_,
      n_severe = NA_integer_, three_level_estimable = NA,
      status = "skipped_no_types", stringsAsFactors = FALSE
    )
    next
  }

  # Dispersion is a nuisance parameter that does not change under label permutation, so it is
  # estimated ONCE per contrast and held fixed thereafter. Empirical-Bayes shrinkage across cell
  # types is the main reason edgeR has power at 11 vs 11.
  dge <- estimateDisp(dge, design_binary, robust = TRUE)

  # ---- null model, shared by all arms and by the permutation machinery --------------------
  null_fit <- glmFit(dge, matrix(1, nrow = ncol(dge), ncol = 1),
                     dispersion = dge$tagwise.dispersion)
  mu0 <- null_fit$fitted.values                       # K x n
  residual <- dge$counts - mu0                        # K x n
  weight <- mu0 / (1 + dge$tagwise.dispersion * mu0)  # NB working weights, K x n

  # ---- per-arm fits -----------------------------------------------------------------------
  z_by_arm <- list()
  for (arm in ARMS) {
    x <- members[[arm]]
    if (sd(x) < 1e-9) next
    x_centered <- x - mean(x)

    design <- model.matrix(~ x)
    quasi <- glmQLFit(dge, design, dispersion = dge$tagwise.dispersion, robust = TRUE)
    test <- glmQLFTest(quasi, coef = 2)
    table <- topTags(test, n = Inf, sort.by = "none")$table

    # Score statistic, standardised against its own permutation null. Linear in x, so the null is
    # generated by reshuffling x rather than by refitting -- exact, and effectively free.
    result <- score_test(residual, x_centered, permute_free, N_PERMUTATIONS, SHRINKAGE)
    z_by_arm[[arm]] <- result$z

    coefficient_rows[[paste(contrast_name, arm)]] <- data.frame(
      contrast = contrast_name, arm = arm, cell_type = rownames(dge),
      logFC = table$logFC, logCPM = table$logCPM, F = table$F,
      p_value = table$PValue,
      q_value = p.adjust(table$PValue, method = "BH"),   # BH WITHIN contrast, per the design
      score_z = result$z, stringsAsFactors = FALSE
    )
    global_rows[[paste(contrast_name, arm)]] <- data.frame(
      contrast = contrast_name, arm = arm, n_embryos = nrow(members),
      n_crispant = sum(members$is_crispant == 1), n_dropped = dropped,
      n_cell_types = sum(result$finite),
      t_observed = result$t_observed, t_null_mean = result$t_null_mean,
      t_null_sd = result$t_null_sd,
      t_std = result$t_std, p_global = result$p_global, d_shrunk = result$d_shrunk,
      n_hits_q10 = sum(p.adjust(table$PValue, method = "BH") < 0.10, na.rm = TRUE),
      n_hits_q05 = sum(p.adjust(table$PValue, method = "BH") < 0.05, na.rm = TRUE),
      max_abs_logfc = max(abs(table$logFC)), stringsAsFactors = FALSE
    )
  }

  # ---- arm 4: separate within-group slopes, conditional on the label --------------------------
  # ~ binary + s_within:group. s_within is orthogonal to binary by construction, so the binary
  # coefficient is unchanged and these slopes are pure within-group dose effects.
  #
  # The within-CONTROL slope is the built-in negative control and the reason for splitting the
  # slopes rather than pooling them. Control embryos also vary along s, and if that variation is
  # measurement noise the slope should be ~0. If instead BOTH slopes come out non-zero and similar,
  # that is the signature of a shared nuisance -- almost certainly developmental stage, which
  # morphology encodes strongly and which drives composition hard -- rather than a perturbation
  # dose. Pooling the slopes would silently average the two and hide exactly that.
  if ("s_within_z" %in% colnames(members) && sd(members$s_within_z) > 1e-9) {
    within_design <- cbind(
      `(Intercept)` = 1,
      binary = members$binary_z,
      s_within_control = members$s_within_z * (members$is_crispant == 0),
      s_within_crispant = members$s_within_z * (members$is_crispant == 1)
    )
    quasi <- glmQLFit(dge, within_design, dispersion = dge$tagwise.dispersion, robust = TRUE)

    # Null model is ~binary, so the residuals already have the group difference removed and the
    # score isolates the added within-group term.
    binary_null <- glmFit(dge, cbind(1, members$binary_z),
                          dispersion = dge$tagwise.dispersion)
    residual_binary <- dge$counts - binary_null$fitted.values
    permute_within <- make_permute_within(members$is_crispant)

    for (slope in c("s_within_control", "s_within_crispant")) {
      coef_index <- match(slope, colnames(within_design))
      test <- glmQLFTest(quasi, coef = coef_index)
      table <- topTags(test, n = Inf, sort.by = "none")$table
      x_slope <- within_design[, slope]
      result <- score_test(residual_binary, x_slope - mean(x_slope), permute_within,
                           N_PERMUTATIONS, SHRINKAGE)

      coefficient_rows[[paste(contrast_name, slope)]] <- data.frame(
        contrast = contrast_name, arm = slope, cell_type = rownames(dge),
        logFC = table$logFC, logCPM = table$logCPM, F = table$F, p_value = table$PValue,
        q_value = p.adjust(table$PValue, method = "BH"),
        score_z = result$z, stringsAsFactors = FALSE
      )
      global_rows[[paste(contrast_name, slope)]] <- data.frame(
        contrast = contrast_name, arm = slope, n_embryos = nrow(members),
        n_crispant = sum(members$is_crispant == 1), n_dropped = dropped,
        n_cell_types = sum(result$finite),
        t_observed = result$t_observed, t_null_mean = result$t_null_mean,
        t_null_sd = result$t_null_sd, t_std = result$t_std, p_global = result$p_global,
        d_shrunk = result$d_shrunk,
        n_hits_q10 = sum(p.adjust(table$PValue, method = "BH") < 0.10, na.rm = TRUE),
        n_hits_q05 = sum(p.adjust(table$PValue, method = "BH") < 0.05, na.rm = TRUE),
        max_abs_logfc = max(abs(table$logFC)), stringsAsFactors = FALSE
      )
    }
  }

  # ---- arms 5-6: the three-level fit, as two sub-contrasts against control -------------------
  # Escapers are crispants sitting inside the observed control range. Asking whether they differ
  # from controls is the direct test of why the graded score failed: if morphologically normal
  # crispants ARE transcriptionally perturbed, then morphological severity simply is not a proxy
  # for molecular severity. Each sub-contrast is a two-group comparison on a subset, so it reuses
  # the same machinery rather than needing a multi-df factor test.
  classes <- members$escaper_class
  for (level in c("escaper", "severe")) {
    subset_index <- which(classes %in% c("control", level))
    n_level <- sum(classes == level)
    if (n_level < MIN_CELL || sum(classes == "control") < MIN_CELL) next

    sub_dge <- dge[, subset_index, keep.lib.sizes = TRUE]
    indicator <- as.numeric(classes[subset_index] == level)
    sub_design <- cbind(1, indicator)
    quasi <- glmQLFit(sub_dge, sub_design, dispersion = dge$tagwise.dispersion, robust = TRUE)
    test <- glmQLFTest(quasi, coef = 2)
    table <- topTags(test, n = Inf, sort.by = "none")$table

    sub_null <- glmFit(sub_dge, matrix(1, nrow = length(subset_index), ncol = 1),
                       dispersion = dge$tagwise.dispersion)
    sub_residual <- sub_dge$counts - sub_null$fitted.values
    result <- score_test(sub_residual, indicator - mean(indicator), permute_free,
                         N_PERMUTATIONS, SHRINKAGE)

    arm <- paste0(level, "_vs_control")
    coefficient_rows[[paste(contrast_name, arm)]] <- data.frame(
      contrast = contrast_name, arm = arm, cell_type = rownames(sub_dge),
      logFC = table$logFC, logCPM = table$logCPM, F = table$F, p_value = table$PValue,
      q_value = p.adjust(table$PValue, method = "BH"),
      score_z = result$z, stringsAsFactors = FALSE
    )
    global_rows[[paste(contrast_name, arm)]] <- data.frame(
      contrast = contrast_name, arm = arm, n_embryos = length(subset_index),
      n_crispant = n_level, n_dropped = dropped, n_cell_types = sum(result$finite),
      t_observed = result$t_observed, t_null_mean = result$t_null_mean,
      t_null_sd = result$t_null_sd, t_std = result$t_std, p_global = result$p_global,
      d_shrunk = result$d_shrunk,
      n_hits_q10 = sum(p.adjust(table$PValue, method = "BH") < 0.10, na.rm = TRUE),
      n_hits_q05 = sum(p.adjust(table$PValue, method = "BH") < 0.05, na.rm = TRUE),
      max_abs_logfc = max(abs(table$logFC)), stringsAsFactors = FALSE
    )
  }

  # ---- geometry between arms, in the shared standardised metric ---------------------------
  # All arms are z-scored, so their score vectors are standardised by the same per-cell-type null
  # SD and live in one common metric. The decomposition asks whether `s` AMPLIFIED the binary
  # arm's vector or found a different one.
  if ("binary_z" %in% names(z_by_arm)) {
    reference <- z_by_arm[["binary_z"]]
    reference_norm <- sqrt(sum(reference^2))
    for (arm in names(z_by_arm)) {
      current <- z_by_arm[[arm]]
      cosine <- sum(current * reference) / (sqrt(sum(current^2)) * reference_norm)
      parallel <- sum(current * reference) / reference_norm
      geometry_rows[[paste(contrast_name, arm)]] <- data.frame(
        contrast = contrast_name, arm = arm,
        norm = sqrt(sum(current^2)), norm_binary = reference_norm,
        cosine_to_binary = cosine,
        parallel_component = parallel,
        orthogonal_component = sqrt(max(sum(current^2) - parallel^2, 0)),
        stringsAsFactors = FALSE
      )
    }
  }

  # ---- arm 7: severe vs escaper DIRECTLY, controls dropped ----------------------------------
  # Arms 5 and 6 each compare one crispant subgroup to the controls, so judging whether severity
  # stratifies meant comparing two test statistics from two models on two different embryo subsets
  # -- weak, and with no p-value of its own. This is the direct contrast.
  #
  # Both groups are injected, so injection efficiency, handling and batch cancel; escaper-vs-control
  # confounds "was it injected" with "does it look affected". The permutation null becomes exactly
  # "morphological class among crispants is arbitrary", which is the hypothesis of interest.
  #
  # It is also the binary counterpart of s_within_crispant on the same embryos, so the two together
  # measure what dichotomising the gradient costs.
  if (sum(classes == "escaper") >= MIN_CELL && sum(classes == "severe") >= MIN_CELL) {
    subset_index <- which(classes %in% c("escaper", "severe"))
    sub_dge <- dge[, subset_index, keep.lib.sizes = TRUE]
    indicator <- as.numeric(classes[subset_index] == "severe")

    quasi <- glmQLFit(sub_dge, cbind(1, indicator), dispersion = dge$tagwise.dispersion,
                      robust = TRUE)
    table <- topTags(glmQLFTest(quasi, coef = 2), n = Inf, sort.by = "none")$table
    sub_null <- glmFit(sub_dge, matrix(1, nrow = length(subset_index), ncol = 1),
                       dispersion = dge$tagwise.dispersion)
    result <- score_test(sub_dge$counts - sub_null$fitted.values,
                         indicator - mean(indicator), permute_free,
                         N_PERMUTATIONS, SHRINKAGE)

    coefficient_rows[[paste(contrast_name, "severe_vs_escaper")]] <- data.frame(
      contrast = contrast_name, arm = "severe_vs_escaper", cell_type = rownames(sub_dge),
      logFC = table$logFC, logCPM = table$logCPM, F = table$F, p_value = table$PValue,
      q_value = p.adjust(table$PValue, method = "BH"),
      score_z = result$z, stringsAsFactors = FALSE
    )
    global_rows[[paste(contrast_name, "severe_vs_escaper")]] <- data.frame(
      contrast = contrast_name, arm = "severe_vs_escaper", n_embryos = length(subset_index),
      n_crispant = sum(indicator), n_dropped = dropped, n_cell_types = sum(result$finite),
      t_observed = result$t_observed, t_null_mean = result$t_null_mean,
      t_null_sd = result$t_null_sd, t_std = result$t_std, p_global = result$p_global,
      d_shrunk = result$d_shrunk,
      n_hits_q10 = sum(p.adjust(table$PValue, method = "BH") < 0.10, na.rm = TRUE),
      n_hits_q05 = sum(p.adjust(table$PValue, method = "BH") < 0.05, na.rm = TRUE),
      max_abs_logfc = max(abs(table$logFC)), stringsAsFactors = FALSE
    )
  }

  status_rows[[contrast_name]] <- data.frame(
    contrast = contrast_name, n_embryos = nrow(members), n_dropped = dropped,
    n_cell_types = n_types,
    n_control = sum(classes == "control"),
    n_escaper = sum(classes == "escaper"),
    n_severe = sum(classes == "severe"),
    three_level_estimable = sum(classes == "escaper") >= MIN_CELL &&
      sum(classes == "severe") >= MIN_CELL,
    status = "ok", stringsAsFactors = FALSE
  )
  message(sprintf("  %-34s n=%2d (-%d)  types=%3d  T_std: binary %6.1f | s %6.1f | hinge %6.1f",
                  contrast_name, nrow(members), dropped, n_types,
                  global_rows[[paste(contrast_name, "binary_z")]]$t_std,
                  if (!is.null(global_rows[[paste(contrast_name, "s_z")]]))
                    global_rows[[paste(contrast_name, "s_z")]]$t_std else NA,
                  if (!is.null(global_rows[[paste(contrast_name, "hinge_z")]]))
                    global_rows[[paste(contrast_name, "hinge_z")]]$t_std else NA))
}

message(sprintf("\nfitted in %.0f s",
                as.numeric(difftime(Sys.time(), started_all, units = "secs"))))

write.csv(do.call(rbind, coefficient_rows), file.path(DATA, "coefficients.csv"), row.names = FALSE)
write.csv(do.call(rbind, global_rows), file.path(DATA, "global_stats.csv"), row.names = FALSE)
write.csv(do.call(rbind, geometry_rows), file.path(DATA, "arm_geometry.csv"), row.names = FALSE)
write.csv(do.call(rbind, status_rows), file.path(DATA, "contrast_status.csv"), row.names = FALSE)

global <- do.call(rbind, global_rows)
message("\n=== global statistic by arm (median across contrasts) ===")
counts_by_arm <- as.data.frame(table(global$arm), stringsAsFactors = FALSE)
colnames(counts_by_arm) <- c("arm", "n_contrasts")
print(merge(aggregate(cbind(t_std, d_shrunk, n_hits_q10) ~ arm, data = global, FUN = median),
            counts_by_arm, by = "arm"))

message("\n=== does the within-group gradient add anything beyond the label? ===")
for (slope in c("s_within_crispant", "s_within_control")) {
  block <- global[global$arm == slope, ]
  if (!nrow(block)) next
  message(sprintf("  %-18s median T_std %+.2f | p_global<0.05 in %2d/%2d | median hits %.0f",
                  slope, median(block$t_std), sum(block$p_global < 0.05), nrow(block),
                  median(block$n_hits_q10)))
}
message("  (the CONTROL slope is the negative control: it should be ~0 if control-side")
message("   variation along s is noise; a non-zero value points at a shared nuisance, most")
message("   likely developmental stage)")

message("\n=== are wildtype-looking crispants transcriptionally perturbed? ===")
for (level in c("escaper_vs_control", "severe_vs_control", "severe_vs_escaper")) {
  block <- global[global$arm == level, ]
  if (!nrow(block)) next
  message(sprintf("  %-20s median T_std %+.2f | p<0.05 in %2d/%2d contrasts | median hits %.0f",
                  level, median(block$t_std), sum(block$p_global < 0.05), nrow(block),
                  median(block$n_hits_q10)))
}

# Paired on the contrasts where BOTH sub-contrasts are estimable. Comparing the unpaired medians
# above would mix different contrast sets and overstate the gap.
paired <- merge(
  global[global$arm == "escaper_vs_control", c("contrast", "t_std")],
  global[global$arm == "severe_vs_control", c("contrast", "t_std")],
  by = "contrast", suffixes = c("_escaper", "_severe")
)
if (nrow(paired) > 5) {
  gap <- paired$t_std_severe - paired$t_std_escaper
  message(sprintf(
    "  PAIRED severe - escaper (vs control): median %+.2f in %d/%d contrasts, Wilcoxon p = %.4f",
    median(gap), sum(gap > 0), length(gap), wilcox.test(gap)$p.value))
}
direct <- global[global$arm == "severe_vs_escaper", ]
if (nrow(direct)) {
  message(sprintf(
    "  DIRECT severe vs escaper: %d/%d contrasts at p<0.05 (chance would give %.1f, binomial p = %.4f)",
    sum(direct$p_global < 0.05), nrow(direct), 0.05 * nrow(direct),
    binom.test(sum(direct$p_global < 0.05), nrow(direct), 0.05,
               alternative = "greater")$p.value))
  slope <- global[global$arm == "s_within_crispant" &
                    global$contrast %in% direct$contrast, ]
  message(sprintf("  ... versus the CONTINUOUS gradient on the same contrasts: %d/%d at p<0.05",
                  sum(slope$p_global < 0.05), nrow(slope)))
  message("  (the difference between those two is what dichotomising the gradient costs)")
}
estimable <- do.call(rbind, status_rows)
message(sprintf("  three-level fit estimable (both cells >= %d) in %d / %d contrasts",
                MIN_CELL, sum(estimable$three_level_estimable, na.rm = TRUE),
                sum(estimable$status == "ok")))

wide <- reshape(global[, c("contrast", "arm", "t_std")], idvar = "contrast",
                timevar = "arm", direction = "wide")
message("\n=== head-to-head: does the morphology axis beat the binary indicator? ===")
for (arm in c("t_std.s_z", "t_std.hinge_z")) {
  if (!arm %in% colnames(wide)) next
  delta <- wide[[arm]] - wide[["t_std.binary_z"]]
  delta <- delta[is.finite(delta)]
  message(sprintf("  %-14s wins %2d / %2d contrasts | median delta %+.2f | Wilcoxon p = %.4f",
                  sub("t_std.", "", arm), sum(delta > 0), length(delta), median(delta),
                  wilcox.test(delta)$p.value))
}

message(sprintf("\nwrote -> %s", DATA))
