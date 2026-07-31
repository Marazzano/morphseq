# Every differential-abundance comparison we intend to run.
#
# `perturbation` is the group whose abundance is compared with `control`.
# The reported effect is perturbation minus control.
#
# `group_column` tells the notebook whether the groups come from the original
# sequencing perturbation or from a MorphSeq phenotype prediction.
#
# Contrasts in the same `fit_group` share one cluster job. That job loads the
# cell-type CDS once, then fits every contrast in the group.

MINIMUM_EMBRYOS_PER_ARM <- 3


# Original sequencing perturbations

perturbation_contrast_plan <- tibble::tribble(
  ~contrast_name,       ~fit_group,          ~group_column,  ~perturbation, ~control,        ~timepoint, ~expected_cross_rt_block, ~control_note,
  "foxj1a_18hpf",       "crispants_18hpf",   "perturbation", "foxj1a",      "ctrl-inj",              18, FALSE,                    "injection control",
  "foxj1a_24hpf",       "crispants_24hpf",   "perturbation", "foxj1a",      "ctrl-inj",              24, FALSE,                    "injection control",
  "foxj1a_30hpf",       "crispants_30hpf",   "perturbation", "foxj1a",      "ctrl-inj",              30, FALSE,                    "injection control",
  "foxj1a_48hpf",       "crispants_48hpf",   "perturbation", "foxj1a",      "ctrl-inj",              48, FALSE,                    "injection control",
  "ift88_18hpf",        "crispants_18hpf",   "perturbation", "ift88",       "ctrl-inj",              18, FALSE,                    "injection control",
  "ift88_24hpf",        "crispants_24hpf",   "perturbation", "ift88",       "ctrl-inj",              24, FALSE,                    "injection control",
  "ift88_30hpf",        "crispants_30hpf",   "perturbation", "ift88",       "ctrl-inj",              30, FALSE,                    "injection control",
  "ift88_48hpf",        "crispants_48hpf",   "perturbation", "ift88",       "ctrl-inj",              48, FALSE,                    "injection control",
  "sspo_18hpf",         "crispants_18hpf",   "perturbation", "sspo",        "ctrl-inj",              18, FALSE,                    "injection control",
  "sspo_24hpf",         "crispants_24hpf",   "perturbation", "sspo",        "ctrl-inj",              24, FALSE,                    "injection control",
  "sspo_30hpf",         "crispants_30hpf",   "perturbation", "sspo",        "ctrl-inj",              30, FALSE,                    "injection control",
  "sspo_48hpf",         "crispants_48hpf",   "perturbation", "sspo",        "ctrl-inj",              48, FALSE,                    "injection control",
  "cep290_mut_18hpf",   "cep290_18hpf",      "perturbation", "cep290-mut",  "cep290-negsib",         18, FALSE,                    "negative sibling",
  "cep290_mut_24hpf",   "cep290_24hpf",      "perturbation", "cep290-mut",  "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_mut_30hpf",   "cep290_30hpf",      "perturbation", "cep290-mut",  "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_mut_48hpf",   "cep290_48hpf",      "perturbation", "cep290-mut",  "cep290-negsib",         48, FALSE,                    "negative sibling",
  "b9d2_mut_14hpf",     "b9d2_14hpf",        "perturbation", "b9d2-mut",    "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_mut_18hpf",     "b9d2_18hpf",        "perturbation", "b9d2-mut",    "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_mut_30hpf",     "b9d2_30hpf",        "perturbation", "b9d2-mut",    "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_mut_48hpf",     "b9d2_48hpf",        "perturbation", "b9d2-mut",    "b9d2-negsib",           48, FALSE,                    "negative sibling"
)


# MorphSeq phenotype groups

phenotype_contrast_plan <- tibble::tribble(
  ~contrast_name,                              ~fit_group,        ~group_column,     ~perturbation, ~control,        ~timepoint, ~expected_cross_rt_block, ~control_note,
  "cep290_low_to_high_vs_negsib_18hpf",        "cep290_18hpf",    "phenotype_group", "Low_to_High", "cep290-negsib",         18, FALSE,                    "negative sibling",
  "cep290_high_to_low_vs_negsib_24hpf",        "cep290_24hpf",    "phenotype_group", "High_to_Low", "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_low_to_high_vs_negsib_24hpf",        "cep290_24hpf",    "phenotype_group", "Low_to_High", "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_high_to_low_vs_b9d2_negsib_30hpf",   "cep290_30hpf",    "phenotype_group", "High_to_Low", "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_low_to_high_vs_b9d2_negsib_30hpf",   "cep290_30hpf",    "phenotype_group", "Low_to_High", "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_high_to_low_vs_negsib_48hpf",        "cep290_48hpf",    "phenotype_group", "High_to_Low", "cep290-negsib",         48, FALSE,                    "negative sibling",
  "cep290_low_to_high_vs_negsib_48hpf",        "cep290_48hpf",    "phenotype_group", "Low_to_High", "cep290-negsib",         48, FALSE,                    "negative sibling",
  "cep290_high_to_low_vs_low_to_high_24hpf",   "cep290_24hpf",    "phenotype_group", "High_to_Low", "Low_to_High",           24, FALSE,                    "phenotype comparison",
  "cep290_high_to_low_vs_low_to_high_30hpf",   "cep290_30hpf",    "phenotype_group", "High_to_Low", "Low_to_High",           30, FALSE,                    "phenotype comparison",
  "cep290_high_to_low_vs_low_to_high_48hpf",   "cep290_48hpf",    "phenotype_group", "High_to_Low", "Low_to_High",           48, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_14hpf",                   "b9d2_14hpf",      "phenotype_group", "CE",           "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_14hpf",                  "b9d2_14hpf",      "phenotype_group", "HTA",          "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_14hpf",                      "b9d2_14hpf",      "phenotype_group", "CE",           "HTA",                    14, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_18hpf",                   "b9d2_18hpf",      "phenotype_group", "CE",           "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_18hpf",                  "b9d2_18hpf",      "phenotype_group", "HTA",          "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_18hpf",                      "b9d2_18hpf",      "phenotype_group", "CE",           "HTA",                    18, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_30hpf",                   "b9d2_30hpf",      "phenotype_group", "CE",           "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_30hpf",                  "b9d2_30hpf",      "phenotype_group", "HTA",          "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_30hpf",                      "b9d2_30hpf",      "phenotype_group", "CE",           "HTA",                    30, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_48hpf",                   "b9d2_48hpf",      "phenotype_group", "CE",           "b9d2-negsib",           48, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_48hpf",                  "b9d2_48hpf",      "phenotype_group", "HTA",          "b9d2-negsib",           48, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_48hpf",                      "b9d2_48hpf",      "phenotype_group", "CE",           "HTA",                    48, FALSE,                    "phenotype comparison"
)


contrast_plan <- dplyr::bind_rows(
  perturbation_contrast_plan,
  phenotype_contrast_plan
)

if (anyDuplicated(contrast_plan$contrast_name)) {
  stop("Every contrast_name must be unique.")
}

if (any(is.na(contrast_plan$fit_group) | contrast_plan$fit_group == "")) {
  stop("Every contrast must have a fit_group.")
}
