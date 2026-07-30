# Reusable cell-type gate policy and calculations.
#
# These functions only transform tables. They do not read files, write files,
# choose contrasts, or discard rows.

CELLTYPE_GATE_POLICY <- list(
  minimum_control_detection = 0.5,
  minimum_control_log_abundance = -3
)


summarize_celltype_detection <- function(
    celltype_counts,
    group_column,
    perturbation_name,
    control_name,
    timepoint_hpf
) {
  if (!group_column %in% names(celltype_counts)) {
    stop("Cell-type counts are missing group column: ", group_column)
  }

  contrast_counts <- celltype_counts |>
    dplyr::filter(
      .data$timepoint == .env$timepoint_hpf,
      .data[[group_column]] %in%
        c(.env$perturbation_name, .env$control_name)
    ) |>
    dplyr::mutate(
      arm = dplyr::if_else(
        .data[[group_column]] == .env$perturbation_name,
        "perturbation",
        "control"
      )
    )

  duplicated_measurements <- contrast_counts |>
    dplyr::count(.data$embryo_ID, .data$cell_type) |>
    dplyr::filter(.data$n != 1)

  if (nrow(duplicated_measurements) > 0) {
    stop("Cell-type counts must contain one row per embryo and cell type.")
  }

  detection <- contrast_counts |>
    dplyr::group_by(.data$cell_type, .data$arm) |>
    dplyr::summarise(
      n_embryos = dplyr::n_distinct(.data$embryo_ID),
      n_zero = sum(.data$n_cells == 0),
      mean_cells = mean(.data$n_cells),
      .groups = "drop"
    ) |>
    tidyr::pivot_wider(
      names_from = "arm",
      values_from = c("n_embryos", "n_zero", "mean_cells"),
      names_glue = "{.value}_{arm}"
    ) |>
    dplyr::mutate(
      control_detection_fraction =
        1 - .data$n_zero_control / .data$n_embryos_control,
      perturbation_detection_fraction =
        1 - .data$n_zero_perturbation / .data$n_embryos_perturbation
    )

  required_arms <- c("n_embryos_control", "n_embryos_perturbation")
  missing_arms <- setdiff(required_arms, names(detection))
  if (length(missing_arms) > 0) {
    stop(
      "The contrast is missing an arm in the cell-type counts: ",
      paste(missing_arms, collapse = ", ")
    )
  }

  detection |>
    dplyr::select(
      cell_type,
      n_embryos_control,
      n_embryos_perturbation,
      mean_cells_control,
      mean_cells_perturbation,
      control_detection_fraction,
      perturbation_detection_fraction
    )
}


add_celltype_reliability <- function(
    model_results,
    detection_summary,
    gate_policy
) {
  required_policy_values <- c(
    "minimum_control_detection",
    "minimum_control_log_abundance"
  )
  missing_policy_values <- setdiff(
    required_policy_values,
    names(gate_policy)
  )
  if (length(missing_policy_values) > 0) {
    stop(
      "Gate policy is missing required values: ",
      paste(missing_policy_values, collapse = ", ")
    )
  }

  minimum_control_detection <- gate_policy$minimum_control_detection
  minimum_control_log_abundance <- gate_policy$minimum_control_log_abundance

  required_model_columns <- c(
    "cell_type",
    "log_abundance_control",
    "p_value"
  )
  missing_columns <- setdiff(required_model_columns, names(model_results))
  if (length(missing_columns) > 0) {
    stop(
      "Model results are missing required columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }

  if (anyDuplicated(detection_summary$cell_type)) {
    stop("Detection summary must contain one row per cell type.")
  }

  results <- model_results |>
    dplyr::left_join(detection_summary, by = "cell_type") |>
    dplyr::mutate(
      passes_control_detection = dplyr::if_else(
        is.na(.data$control_detection_fraction),
        NA,
        .data$control_detection_fraction >= .env$minimum_control_detection
      ),
      passes_control_abundance = dplyr::if_else(
        is.na(.data$log_abundance_control),
        NA,
        .data$log_abundance_control >= .env$minimum_control_log_abundance
      ),
      is_reliable =
        .data$passes_control_detection & .data$passes_control_abundance
    )

  reliable_rows <- which(
    results$is_reliable %in% TRUE & !is.na(results$p_value)
  )
  results$q_value_reliable <- NA_real_
  results$q_value_reliable[reliable_rows] <- stats::p.adjust(
    results$p_value[reliable_rows],
    method = "BH"
  )

  results |>
    dplyr::mutate(
      is_fdr_significant =
        .data$is_reliable %in% TRUE &
        !is.na(.data$q_value_reliable) &
        .data$q_value_reliable < 0.05,
      is_screening_hit =
        .data$is_reliable %in% TRUE &
        !is.na(.data$p_value) &
        .data$p_value < 0.10
    )
}
