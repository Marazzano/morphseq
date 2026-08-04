#!/usr/bin/env Rscript
# Audit companion to fit_edger_contrasts.R -- adds nothing to the pipeline, changes nothing in it.
#
#   Reads:  data/edger/cell_counts.csv, embryo_totals.csv, contrast_predictors.csv
#   Writes: data/edger/binary_coefficient_check.csv    (per contrast x cell type)
#           data/edger/binary_coefficient_summary.csv  (per contrast)
#
# WHY THIS EXISTS
# ---------------
# fit_edger_contrasts.R fits the binary indicator twice, in two different designs:
#
#   Model 1   ~ binary                                   -> coef 2 is the classical contrast
#   Model 2   ~ binary + s_within:group                  -> coef 3, 4 are the within-group slopes
#
# but only ever extracts coef 2 from Model 1 and coefs 3-4 from Model 2. The binary column of
# Model 2 is fitted and discarded. That leaves two questions unanswerable from the shipped
# coefficients.csv:
#
#   (a) Do the two extra columns COST the group contrast anything? The design comment claims not,
#       on the grounds that s_within is orthogonal to binary by construction. That orthogonality is
#       asserted in export_edger_inputs.py, which runs BEFORE the join to the CDS -- and ~22
#       embryos are dropped at that join for having no cells, which breaks the exact group-mean
#       centring the argument rests on.
#
#   (b) Is a "NEW" cell type (slope hit, not a Model 1 binary hit) new because of the GRADIENT, or
#       merely because it is being compared against a different fit? Scoring the slope against the
#       binary coefficient of its OWN design separates the two.
#
# Everything else matches fit_edger_contrasts.R exactly: same offset, same filterByExpr on the
# binary design so the tested cell-type set is identical, same estimateDisp held fixed across fits,
# same BH-within-contrast. No permutations -- this is purely the glmQLFTest side, which is what all
# the hit counts in the synthesis notebook use.
#
# Usage:
#   module load R/4.4.1
#   Rscript check_binary_coefficient.R

suppressPackageStartupMessages({ library(edgeR) })

HERE <- normalizePath(dirname(sub("^--file=", "",
                                  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
DATA <- file.path(HERE, "data", "edger")
MIN_EMBRYOS <- 12L
Q <- 0.10
set.seed(42)

counts_all <- as.matrix(read.csv(file.path(DATA, "cell_counts.csv"), row.names = 1,
                                 check.names = FALSE))
totals_all <- read.csv(file.path(DATA, "embryo_totals.csv"), stringsAsFactors = FALSE)
rownames(totals_all) <- totals_all$embryo_ID
predictors <- read.csv(file.path(DATA, "contrast_predictors.csv"), stringsAsFactors = FALSE)

per_type <- list()
per_contrast <- list()

for (contrast_name in sort(unique(predictors$contrast))) {
  members <- predictors[predictors$contrast == contrast_name, ]
  members <- members[members$sample %in% colnames(counts_all), ]
  if (nrow(members) < MIN_EMBRYOS || length(unique(members$is_crispant)) < 2) next
  if (!("s_within_z" %in% colnames(members)) || sd(members$s_within_z) < 1e-9) next

  library_size <- totals_all[members$sample, "total_cells"]
  dge <- DGEList(counts = counts_all[, members$sample, drop = FALSE], lib.size = library_size)
  dge <- scaleOffset(dge, matrix(log(library_size), nrow = nrow(dge), ncol = ncol(dge),
                                 byrow = TRUE))
  design_binary <- model.matrix(~ members$binary_z)
  dge <- dge[filterByExpr(dge, design = design_binary), , keep.lib.sizes = TRUE]
  if (nrow(dge) < 10) next
  dge <- estimateDisp(dge, design_binary, robust = TRUE)

  # ---- Model 1: the classical contrast, exactly as the pipeline fits it ----
  x <- members$binary_z
  m1 <- glmQLFit(dge, model.matrix(~ x), dispersion = dge$tagwise.dispersion, robust = TRUE)
  t1 <- topTags(glmQLFTest(m1, coef = 2), n = Inf, sort.by = "none")$table

  # ---- Model 2: the within-slope design; coef 2 is the column the pipeline discards ----
  within_design <- cbind(
    `(Intercept)`     = 1,
    binary            = members$binary_z,
    s_within_control  = members$s_within_z * (members$is_crispant == 0),
    s_within_crispant = members$s_within_z * (members$is_crispant == 1)
  )
  m2 <- glmQLFit(dge, within_design, dispersion = dge$tagwise.dispersion, robust = TRUE)
  t2 <- topTags(glmQLFTest(m2, coef = 2), n = Inf, sort.by = "none")$table
  t_slope <- topTags(glmQLFTest(m2, coef = 4), n = Inf, sort.by = "none")$table
  t_slope_ctrl <- topTags(glmQLFTest(m2, coef = 3), n = Inf, sort.by = "none")$table

  # ---- Models 3 and 4: the MATCHED CONTROL for "indirect" additions --------------------------
  # Cell types that are not significant under Model 1 but become significant on Model 2's binary
  # coefficient are gains from variance absorption, not from detection: the slope columns soak up
  # within-group scatter, the QL dispersion falls, and the group contrast sharpens. That mechanism
  # is generic to adding ANY informative covariate, so on its own it says nothing about morphology.
  #
  # These two designs add exactly ONE column each, so they cost identical degrees of freedom and
  # differ only in WHICH GROUP carries the morphological gradient -- the same logic as the
  # crispant-vs-control slope comparison, applied to the indirect channel:
  #
  #   Model 3   ~ binary + s_within_control      the negative control
  #   Model 4   ~ binary + s_within_crispant     the real thing
  #
  # If Model 4 yields many indirect additions and Model 3 yields ~none, the sharpening is being
  # driven by genuine within-crispant structure. If they are comparable, any covariate would have
  # done it and the indirect count carries no morphological claim at all.
  control_design <- within_design[, c("(Intercept)", "binary", "s_within_control")]
  crispant_design <- within_design[, c("(Intercept)", "binary", "s_within_crispant")]
  m3 <- glmQLFit(dge, control_design, dispersion = dge$tagwise.dispersion, robust = TRUE)
  m4 <- glmQLFit(dge, crispant_design, dispersion = dge$tagwise.dispersion, robust = TRUE)
  t3 <- topTags(glmQLFTest(m3, coef = 2), n = Inf, sort.by = "none")$table
  t4 <- topTags(glmQLFTest(m4, coef = 2), n = Inf, sort.by = "none")$table

  q1 <- p.adjust(t1$PValue, method = "BH")
  q2 <- p.adjust(t2$PValue, method = "BH")
  q3 <- p.adjust(t3$PValue, method = "BH")
  q4 <- p.adjust(t4$PValue, method = "BH")
  q_slope <- p.adjust(t_slope$PValue, method = "BH")
  q_slope_control <- p.adjust(t_slope_ctrl$PValue, method = "BH")

  per_type[[contrast_name]] <- data.frame(
    contrast = contrast_name, cell_type = rownames(dge),
    logfc_binary_m1 = t1$logFC, p_binary_m1 = t1$PValue, q_binary_m1 = q1,
    logfc_binary_m2 = t2$logFC, p_binary_m2 = t2$PValue, q_binary_m2 = q2,
    q_binary_m3_ctrlslope = q3,      # ~ binary + s_within_control   (matched negative control)
    q_binary_m4_crispslope = q4,     # ~ binary + s_within_crispant  (the real thing)
    logfc_slope = t_slope$logFC, q_slope = q_slope,
    q_slope_control = q_slope_control,   # the matched negative control for DIRECT gains
    stringsAsFactors = FALSE
  )

  hits1 <- rownames(dge)[q1 < Q]; hits2 <- rownames(dge)[q2 < Q]
  hits_slope <- rownames(dge)[q_slope < Q]
  per_contrast[[contrast_name]] <- data.frame(
    contrast = contrast_name, n_embryos = nrow(members), n_cell_types = nrow(dge),
    df_model1 = nrow(members) - 2L, df_model2 = nrow(members) - 4L,
    # design-level correlation of the added columns with the binary column, AFTER the CDS join
    max_abs_cor_to_binary = max(abs(c(
      cor(within_design[, "binary"], within_design[, "s_within_control"]),
      cor(within_design[, "binary"], within_design[, "s_within_crispant"])))),
    logfc_cor_m1_m2 = cor(t1$logFC, t2$logFC),
    median_abs_logfc_diff = median(abs(t1$logFC - t2$logFC)),
    max_abs_logfc_diff = max(abs(t1$logFC - t2$logFC)),
    hits_binary_m1 = length(hits1), hits_binary_m2 = length(hits2),
    hits_slope = length(hits_slope),
    shared_m1_m2 = length(intersect(hits1, hits2)),
    lost_in_m2 = length(setdiff(hits1, hits2)),
    gained_in_m2 = length(setdiff(hits2, hits1)),
    new_vs_m1 = length(setdiff(hits_slope, hits1)),
    new_vs_m2 = length(setdiff(hits_slope, hits2)),
    # indirect additions to the binary coefficient, per design (matched at one extra column each)
    indirect_m3_ctrlslope = length(setdiff(rownames(dge)[q3 < Q], hits1)),
    indirect_m4_crispslope = length(setdiff(rownames(dge)[q4 < Q], hits1)),
    lost_m3_ctrlslope = length(setdiff(hits1, rownames(dge)[q3 < Q])),
    lost_m4_crispslope = length(setdiff(hits1, rownames(dge)[q4 < Q])),
    stringsAsFactors = FALSE
  )
  message(sprintf("  %-34s n=%2d types=%3d | binary M1=%3d M2=%3d | slope=%3d (new vs M1=%2d, vs M2=%2d)",
                  contrast_name, nrow(members), nrow(dge), length(hits1), length(hits2),
                  length(hits_slope), length(setdiff(hits_slope, hits1)),
                  length(setdiff(hits_slope, hits2))))
}

type_frame <- do.call(rbind, per_type)
summary_frame <- do.call(rbind, per_contrast)
write.csv(type_frame, file.path(DATA, "binary_coefficient_check.csv"), row.names = FALSE)
write.csv(summary_frame, file.path(DATA, "binary_coefficient_summary.csv"), row.names = FALSE)

message("\n=========== binary coefficient: Model 1 vs Model 2 ===========")
message(sprintf("contrasts                          : %d", nrow(summary_frame)))
message(sprintf("max |cor| added cols vs binary     : %.3f  (asserted ~0 pre-CDS-join; %d of %d > 0.05)",
                max(summary_frame$max_abs_cor_to_binary),
                sum(summary_frame$max_abs_cor_to_binary > 0.05), nrow(summary_frame)))
message(sprintf("logFC correlation M1 vs M2         : median %.4f (min %.4f)",
                median(summary_frame$logfc_cor_m1_m2), min(summary_frame$logfc_cor_m1_m2)))
message(sprintf("binary hits  Model 1 / Model 2     : %d / %d  (shared %d, lost %d, gained %d)",
                sum(summary_frame$hits_binary_m1), sum(summary_frame$hits_binary_m2),
                sum(summary_frame$shared_m1_m2), sum(summary_frame$lost_in_m2),
                sum(summary_frame$gained_in_m2)))
message(sprintf("slope hits                         : %d", sum(summary_frame$hits_slope)))
message(sprintf("  NEW vs Model 1 binary            : %d", sum(summary_frame$new_vs_m1)))
message(sprintf("  NEW vs Model 2 binary (same fit) : %d", sum(summary_frame$new_vs_m2)))
message("")
message("INDIRECT additions to the binary coefficient, matched one-extra-column designs:")
message(sprintf("  ~ binary + s_within_CRISPANT : +%d  (-%d lost)   <- the real thing",
                sum(summary_frame$indirect_m4_crispslope), sum(summary_frame$lost_m4_crispslope)))
message(sprintf("  ~ binary + s_within_CONTROL  : +%d  (-%d lost)   <- matched negative control",
                sum(summary_frame$indirect_m3_ctrlslope), sum(summary_frame$lost_m3_ctrlslope)))
message(sprintf("\nwrote -> %s", DATA))
