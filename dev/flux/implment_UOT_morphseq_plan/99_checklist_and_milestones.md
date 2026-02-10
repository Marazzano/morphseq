# Checklists & Milestones

## Milestone A — MVP Green
- [ ] `compute_minibatch_uot` (no bias)
- [ ] `time_weighted_spectral_clustering`
- [ ] `ClusteredMinibatchSampler`
- [ ] ODE training runs end‑to‑end
- [ ] Loss decreases on a 5k‑sample subset
- [ ] Seeds fixed; results reproducible

## Milestone B — `lower_bound` Biasing
- [ ] API extended with `bias_mode`, `bias_kwargs`
- [ ] Synthetic test shows increased self/near‑self mass
- [ ] Training completes with comparable wall clock
- [ ] Regression: `strength=0` == MVP numerically

## Milestone C — Additional Modes
- [ ] `z_hard` implemented + tested
- [ ] At least one prior‑based mode
- [ ] Ablations showing impact per mode