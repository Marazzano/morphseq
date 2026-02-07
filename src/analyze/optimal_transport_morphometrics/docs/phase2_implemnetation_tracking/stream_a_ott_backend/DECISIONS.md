# Stream A: OTT Backend — Decisions

## Parameter Mapping
- `config.epsilon` → ott-jax `epsilon` (direct pass-through)
- `config.marginal_relaxation` (reg_m) → `tau_a = tau_b = reg_m / (reg_m + epsilon)`
  - This is the standard KL→tau conversion from ott-jax docs
- `config.metric = "sqeuclidean"` → `ott.geometry.costs.SqEuclidean()`

## No cost-matrix scaling
POTBackend normalizes weights (a,b) not cost matrix. OTTBackend matches this exactly.

## No GPU assertion at import
OTTBackend works on CPU too. Device selection is runtime, not import-time.

## Sequential solve_batch
Uses loop, not vmap. Safer memory profile, avoids recompilation storms from shape bucketing.

## Concordance Tolerances (actual, tested)
| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Total transport cost | rtol=5% + atol=1e-3 | At epsilon=0.1 (raw coords), POT-OTT agree ~2% |
| Coupling marginals | rtol=35% + atol=5e-3 | Sinkhorn convergence path genuinely differs; relaxed from plan |
| Velocity direction | cosine sim > 0.9 | Flow agreement, not magnitude |
| Mass created/destroyed % | 5% absolute | Same creation/destruction pattern |

## Epsilon sensitivity for concordance
- At raw coord scale (~50), epsilon=0.1 gives <5% cost concordance
- At higher epsilon (1.0, 5.0, 10.0), tau conversion tau=reg_m/(reg_m+eps) diverges from reg_m, causing backends to solve effectively different problems
- For production use, epsilon=1e-5 with coord_scale=1/576 is known stable for POT CPU solver
- OTT concordance at production params (1e-5 + coord_scale) not yet tested — separate spike needed

## IMPORTANT: Concordance tests are NOT on canonical grid
- Tests use raw-coordinate synthetic circles (coords ~40-60), NOT canonical grid masks
- Production pipeline uses canonical grid (256x576 @ 10 um/px) with coord_scale=1/576
- Need a separate spike test to validate OTT at production epsilon=1e-5 on canonical grid
- The optimal epsilon range for OTT/GPU may differ from POT/CPU due to float32 vs float64
