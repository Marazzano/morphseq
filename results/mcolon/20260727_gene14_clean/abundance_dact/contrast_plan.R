# Every differential-abundance comparison we intend to run.
#
# `perturbation` is the group whose abundance is compared with `control`.
# The reported effect is perturbation minus control.
#
# `group_column` tells the notebook whether the groups come from the original
# sequencing perturbation or from a MorphSeq phenotype prediction.

MINIMUM_EMBRYOS_PER_ARM <- 3


# Original sequencing perturbations

perturbation_contrast_plan <- tibble::tribble(
  ~contrast_name,       ~group_column,  ~perturbation, ~control,        ~timepoint, ~expected_cross_rt_block, ~control_note,
  "foxj1a_18hpf",       "perturbation", "foxj1a",      "ctrl-inj",              18, FALSE,                    "injection control",
  "foxj1a_24hpf",       "perturbation", "foxj1a",      "ctrl-inj",              24, FALSE,                    "injection control",
  "foxj1a_30hpf",       "perturbation", "foxj1a",      "ctrl-inj",              30, FALSE,                    "injection control",
  "foxj1a_48hpf",       "perturbation", "foxj1a",      "ctrl-inj",              48, FALSE,                    "injection control",
  "ift88_18hpf",        "perturbation", "ift88",       "ctrl-inj",              18, FALSE,                    "injection control",
  "ift88_24hpf",        "perturbation", "ift88",       "ctrl-inj",              24, FALSE,                    "injection control",
  "ift88_30hpf",        "perturbation", "ift88",       "ctrl-inj",              30, FALSE,                    "injection control",
  "ift88_48hpf",        "perturbation", "ift88",       "ctrl-inj",              48, FALSE,                    "injection control",
  "sspo_18hpf",         "perturbation", "sspo",        "ctrl-inj",              18, FALSE,                    "injection control",
  "sspo_24hpf",         "perturbation", "sspo",        "ctrl-inj",              24, FALSE,                    "injection control",
  "sspo_30hpf",         "perturbation", "sspo",        "ctrl-inj",              30, FALSE,                    "injection control",
  "sspo_48hpf",         "perturbation", "sspo",        "ctrl-inj",              48, FALSE,                    "injection control",
  "cep290_mut_18hpf",   "perturbation", "cep290-mut",  "cep290-negsib",         18, FALSE,                    "negative sibling",
  "cep290_mut_24hpf",   "perturbation", "cep290-mut",  "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_mut_30hpf",   "perturbation", "cep290-mut",  "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_mut_48hpf",   "perturbation", "cep290-mut",  "cep290-negsib",         48, FALSE,                    "negative sibling",
  "b9d2_mut_14hpf",     "perturbation", "b9d2-mut",    "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_mut_18hpf",     "perturbation", "b9d2-mut",    "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_mut_30hpf",     "perturbation", "b9d2-mut",    "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_mut_48hpf",     "perturbation", "b9d2-mut",    "b9d2-negsib",           48, FALSE,                    "negative sibling"
)


# MorphSeq phenotype groups

phenotype_contrast_plan <- tibble::tribble(
  ~contrast_name,                              ~group_column,     ~perturbation, ~control,        ~timepoint, ~expected_cross_rt_block, ~control_note,
  "cep290_low_to_high_vs_negsib_18hpf",        "phenotype_group", "Low_to_High", "cep290-negsib",         18, FALSE,                    "negative sibling",
  "cep290_high_to_low_vs_negsib_24hpf",        "phenotype_group", "High_to_Low", "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_low_to_high_vs_negsib_24hpf",        "phenotype_group", "Low_to_High", "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_high_to_low_vs_b9d2_negsib_30hpf",   "phenotype_group", "High_to_Low", "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_low_to_high_vs_b9d2_negsib_30hpf",   "phenotype_group", "Low_to_High", "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_high_to_low_vs_negsib_48hpf",        "phenotype_group", "High_to_Low", "cep290-negsib",         48, FALSE,                    "negative sibling",
  "cep290_low_to_high_vs_negsib_48hpf",        "phenotype_group", "Low_to_High", "cep290-negsib",         48, FALSE,                    "negative sibling",
  "cep290_high_to_low_vs_low_to_high_24hpf",   "phenotype_group", "High_to_Low", "Low_to_High",           24, FALSE,                    "phenotype comparison",
  "cep290_high_to_low_vs_low_to_high_30hpf",   "phenotype_group", "High_to_Low", "Low_to_High",           30, FALSE,                    "phenotype comparison",
  "cep290_high_to_low_vs_low_to_high_48hpf",   "phenotype_group", "High_to_Low", "Low_to_High",           48, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_14hpf",                   "phenotype_group", "CE",           "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_14hpf",                  "phenotype_group", "HTA",          "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_14hpf",                      "phenotype_group", "CE",           "HTA",                    14, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_18hpf",                   "phenotype_group", "CE",           "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_18hpf",                  "phenotype_group", "HTA",          "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_18hpf",                      "phenotype_group", "CE",           "HTA",                    18, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_30hpf",                   "phenotype_group", "CE",           "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_30hpf",                  "phenotype_group", "HTA",          "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_30hpf",                      "phenotype_group", "CE",           "HTA",                    30, FALSE,                    "phenotype comparison",
  "b9d2_ce_vs_negsib_48hpf",                   "phenotype_group", "CE",           "b9d2-negsib",           48, FALSE,                    "negative sibling",
  "b9d2_hta_vs_negsib_48hpf",                  "phenotype_group", "HTA",          "b9d2-negsib",           48, FALSE,                    "negative sibling",
  "b9d2_ce_vs_hta_48hpf",                      "phenotype_group", "CE",           "HTA",                    48, FALSE,                    "phenotype comparison"
)


contrast_plan <- dplyr::bind_rows(
  perturbation_contrast_plan,
  phenotype_contrast_plan
)

if (anyDuplicated(contrast_plan$contrast_name)) {
  stop("Every contrast_name must be unique.")
}
