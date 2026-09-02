# Conservative concurrence between two comparable effects
# =============================================================================
# Use this when two perturbations produce the SAME kind of effect estimate,
# such as two differential-abundance changes.
#
# If the effects agree in direction, keep the smaller absolute effect and its
# sign. If they disagree, return zero. Missing measurements remain missing.
#
# Examples:
#   -0.8 and -0.3  -> -0.3
#    0.8 and  0.3  ->  0.3
#   -0.8 and  0.3  ->  0.0
# =============================================================================

concurrent_effect <- function(effect_a, effect_b) {
  if (length(effect_a) != length(effect_b)) {
    stop("The two effect vectors must have the same length.")
  }

  both_measured <- !is.na(effect_a) & !is.na(effect_b)
  same_direction <- sign(effect_a) == sign(effect_b)

  concurrence <- rep(NA_real_, length(effect_a))
  concurrence[both_measured & !same_direction] <- 0
  concurrence[both_measured & same_direction] <-
    sign(effect_a[both_measured & same_direction]) *
    pmin(
      abs(effect_a[both_measured & same_direction]),
      abs(effect_b[both_measured & same_direction])
    )

  concurrence
}


# For two p-values, the larger one is the weaker evidence. A concurrence result
# should only look strong when BOTH comparisons are convincing.
concurrent_p_value <- function(p_value_a, p_value_b) {
  if (length(p_value_a) != length(p_value_b)) {
    stop("The two p-value vectors must have the same length.")
  }

  both_measured <- !is.na(p_value_a) & !is.na(p_value_b)

  weaker_p_value <- rep(NA_real_, length(p_value_a))
  weaker_p_value[both_measured] <- pmax(
    p_value_a[both_measured],
    p_value_b[both_measured]
  )

  weaker_p_value
}
