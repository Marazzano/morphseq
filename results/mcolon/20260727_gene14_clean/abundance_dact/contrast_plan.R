# Every differential-abundance comparison we intend to run.
#
# `perturbation` and `control` are values in the sequencing `perturbation`
# column. We record whether a comparison is expected to cross RT blocks, then
# the runner derives the observed blocks and checks that they agree.

MINIMUM_EMBRYOS_PER_ARM <- 3

contrast_plan <- tibble::tribble(
  ~contrast_name,       ~perturbation, ~control,        ~timepoint, ~expected_cross_rt_block, ~control_note,
  "foxj1a_18hpf",       "foxj1a",      "ctrl-inj",              18, FALSE,                    "injection control",
  "foxj1a_24hpf",       "foxj1a",      "ctrl-inj",              24, FALSE,                    "injection control",
  "foxj1a_30hpf",       "foxj1a",      "ctrl-inj",              30, FALSE,                    "injection control",
  "foxj1a_48hpf",       "foxj1a",      "ctrl-inj",              48, FALSE,                    "injection control",
  "ift88_18hpf",        "ift88",       "ctrl-inj",              18, FALSE,                    "injection control",
  "ift88_24hpf",        "ift88",       "ctrl-inj",              24, FALSE,                    "injection control",
  "ift88_30hpf",        "ift88",       "ctrl-inj",              30, FALSE,                    "injection control",
  "ift88_48hpf",        "ift88",       "ctrl-inj",              48, FALSE,                    "injection control",
  "sspo_18hpf",         "sspo",        "ctrl-inj",              18, FALSE,                    "injection control",
  "sspo_24hpf",         "sspo",        "ctrl-inj",              24, FALSE,                    "injection control",
  "sspo_30hpf",         "sspo",        "ctrl-inj",              30, FALSE,                    "injection control",
  "sspo_48hpf",         "sspo",        "ctrl-inj",              48, FALSE,                    "injection control",
  "cep290_mut_18hpf",   "cep290-mut",  "cep290-negsib",         18, FALSE,                    "negative sibling",
  "cep290_mut_24hpf",   "cep290-mut",  "cep290-negsib",         24, FALSE,                    "negative sibling",
  "cep290_mut_30hpf",   "cep290-mut",  "b9d2-negsib",           30, TRUE,                     "borrowed control: cep290-negsib is absent",
  "cep290_mut_48hpf",   "cep290-mut",  "cep290-negsib",         48, FALSE,                    "negative sibling",
  "b9d2_mut_14hpf",     "b9d2-mut",    "b9d2-negsib",           14, FALSE,                    "negative sibling",
  "b9d2_mut_18hpf",     "b9d2-mut",    "b9d2-negsib",           18, FALSE,                    "negative sibling",
  "b9d2_mut_30hpf",     "b9d2-mut",    "b9d2-negsib",           30, FALSE,                    "negative sibling",
  "b9d2_mut_48hpf",     "b9d2-mut",    "b9d2-negsib",           48, FALSE,                    "negative sibling"
)

if (anyDuplicated(contrast_plan$contrast_name)) {
  stop("Every contrast_name must be unique.")
}
