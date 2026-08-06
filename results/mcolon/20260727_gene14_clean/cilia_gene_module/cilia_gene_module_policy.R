# Reusable cilia-module definitions and scoring calculation.
#
# These functions transform tables and a loaded Monocle CDS. They do not read
# source files, write outputs, choose datasets, or make figures.

MINIMUM_MAPPED_HUMAN_GENES <- 5L

# Approved after comparing redundancy and stability across cell types.
CHOSEN_CILIA_MODULES <- c(
  "v1_all",
  "gold_motile_cilia",
  "v1_basal_body",
  "v1_ciliary_tip",
  "v1_transition_zone"
)


# SysCilia v1 gives each protein a free-text localization such as
# "Basal Body, Cilium, IFT". These patterns turn those curator annotations
# into small, biologically named modules. A gene can belong to several modules.
V1_LOCATION_MODULES <- list(
  v1_basal_body = "basal body",
  v1_centrosome_centriole = paste(
    "centriole|centrosome|centriolar satellite|pericentriolar matrix"
  ),
  v1_axoneme = "axoneme|central pair",
  v1_transition_zone = "transition zone",
  v1_ift = "(^|, )ift($|, )",
  v1_ciliary_membrane = "ciliary membrane|cilia membrane",
  v1_ciliary_tip = "ciliary tip",
  v1_connecting_cilium = "connecting cilium",
  v1_generic_cilium = "(^|, )cilium($|, )|(^|, )cilia($|, )"
)


build_v1_location_membership <- function(scgs_v1) {
  required_columns <- c("current_human_symbol", "scgs_localisation")
  missing_columns <- setdiff(required_columns, names(scgs_v1))
  if (length(missing_columns) > 0L) {
    stop(
      "SysCilia v1 is missing required columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }

  # Standardize spelling and capitalization, but preserve the original text in
  # the audit table. We do not infer a location from curator notes.
  locations <- scgs_v1 |>
    dplyr::transmute(
      current_human_symbol = .data$current_human_symbol,
      location_for_matching = .data$scgs_localisation |>
        as.character() |>
        stringr::str_to_lower() |>
        stringr::str_replace_all("cilia membrane", "ciliary membrane") |>
        stringr::str_replace_all("\\s*,\\s*", ", ") |>
        stringr::str_squish()
    )

  module_rows <- vector("list", length(V1_LOCATION_MODULES))

  for (i in seq_along(V1_LOCATION_MODULES)) {
    module_name <- names(V1_LOCATION_MODULES)[[i]]
    location_pattern <- V1_LOCATION_MODULES[[i]]

    module_rows[[i]] <- locations |>
      dplyr::filter(
        !is.na(.data$location_for_matching),
        stringr::str_detect(.data$location_for_matching, location_pattern)
      ) |>
      dplyr::transmute(
        module_id = .env$module_name,
        current_human_symbol = .data$current_human_symbol,
        module_provenance = "SysCilia v1 curated protein localization"
      )
  }

  dplyr::bind_rows(module_rows) |>
    dplyr::distinct(.data$module_id, .data$current_human_symbol)
}


score_cilia_modules <- function(
    cds,
    human_to_zebrafish,
    human_module_membership
) {
  required_mapping_columns <- c(
    "current_human_symbol",
    "zebrafish_gene_id"
  )
  missing_mapping_columns <- setdiff(
    required_mapping_columns,
    names(human_to_zebrafish)
  )
  if (length(missing_mapping_columns) > 0L) {
    stop(
      "The orthology table is missing: ",
      paste(missing_mapping_columns, collapse = ", ")
    )
  }

  required_module_columns <- c("module_id", "current_human_symbol")
  missing_module_columns <- setdiff(
    required_module_columns,
    names(human_module_membership)
  )
  if (length(missing_module_columns) > 0L) {
    stop(
      "The module table is missing: ",
      paste(missing_module_columns, collapse = ", ")
    )
  }

  if (is.null(monocle3::size_factors(cds))) {
    stop("The CDS has no size factors; use the existing processed CDS.")
  }

  # Several zebrafish paralogs can map to one human cilia gene. Giving every
  # paralog its own module weight would over-weight duplicated families, so
  # Monocle first sums them into one pseudo-gene per human ortholog.
  ortholog_groups <- human_to_zebrafish |>
    dplyr::filter(.data$zebrafish_gene_id %in% rownames(cds)) |>
    dplyr::distinct(
      .data$zebrafish_gene_id,
      .data$current_human_symbol
    )

  if (nrow(ortholog_groups) == 0L) {
    stop("None of the mapped zebrafish cilia genes occur in this CDS.")
  }

  human_expression <- monocle3::aggregate_gene_expression(
    cds,
    gene_group_df = ortholog_groups,
    norm_method = "size_only",
    gene_agg_fun = "sum",
    scale_agg_values = FALSE
  )

  # log1p preserves zero expression while reducing domination by a few highly
  # expressed genes. Sparse zeros stay zero.
  if (inherits(human_expression, "sparseMatrix")) {
    human_expression@x <- log1p(human_expression@x)
  } else {
    human_expression <- log1p(human_expression)
  }

  score_membership <- human_module_membership |>
    dplyr::filter(.data$current_human_symbol %in% rownames(human_expression)) |>
    dplyr::distinct(.data$module_id, .data$current_human_symbol)

  mapped_module_sizes <- score_membership |>
    dplyr::count(.data$module_id, name = "n_mapped_human_genes")

  modules_to_score <- mapped_module_sizes |>
    dplyr::filter(
      .data$n_mapped_human_genes >= MINIMUM_MAPPED_HUMAN_GENES
    ) |>
    dplyr::pull("module_id")

  score_membership <- score_membership |>
    dplyr::filter(.data$module_id %in% .env$modules_to_score)

  if (nrow(score_membership) == 0L) {
    stop("No cilia module has enough mapped human genes to score.")
  }

  module_names <- unique(score_membership$module_id)
  human_genes <- rownames(human_expression)

  module_average <- Matrix::sparseMatrix(
    i = match(score_membership$module_id, module_names),
    j = match(score_membership$current_human_symbol, human_genes),
    x = 1,
    dims = c(length(module_names), length(human_genes))
  )
  module_average <- module_average / Matrix::rowSums(module_average)

  score_matrix <- module_average %*% human_expression
  rownames(score_matrix) <- module_names

  # There are only a few module columns, so this final cells-by-modules table
  # is small enough to store as an ordinary R matrix and data frame.
  scores <- Matrix::t(score_matrix) |>
    as.matrix() |>
    as.data.frame() |>
    tibble::rownames_to_column("cell") |>
    tibble::as_tibble()

  list(
    scores = scores,
    mapped_module_sizes = mapped_module_sizes
  )
}
