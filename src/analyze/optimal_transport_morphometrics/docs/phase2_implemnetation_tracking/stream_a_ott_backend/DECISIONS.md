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

## Canonical Grid Spike Results (real embryos A05 vs E04 @ ~48 hpf)
Ran `canonical_grid_epsilon_spike.py` with reg_m=10.0, coord_scale=1/576 on canonical grid.

| Epsilon | POT Cost | OTT Cost | % Diff | OTT Converged |
|---------|----------|----------|--------|---------------|
| 1e-6 | ~0 | 18.27 | massive | False |
| 1e-5 | 0.00005 | 34.08 | massive | False |
| **1e-4** | **37.09** | **37.17** | **0.21%** | False |
| **1e-3** | **46.01** | **46.01** | **0.01%** | False |
| 1e-2 | 98.78 | 99.59 | 0.82% | False |
| 1e-1 | 300.32 | 326.85 | 8.84% | True |

### Interpretation
- **OTT sweet spot: eps=1e-4 to 1e-3** on canonical grid (<1% cost concordance with POT)
- At POT production epsilon (1e-5), OTT diverges completely. Likely cause: float32 Gibbs kernel
  exp(-C/eps) underflows when eps is too small relative to scaled squared-Euclidean costs.
  POT (float64) handles this; OTT (float32, ~7 decimal digits) cannot.
- OTT `converged=False` flag appears overly strict — costs agree well at eps=1e-4 to 1e-3
  despite the flag. May need to increase `max_iterations` or relax `threshold`.
- **Recommendation:** Use eps=1e-4 for OTT/GPU production on canonical grid. This gives
  excellent concordance (0.21%) while being biologically meaningful. POT continues at eps=1e-5.
- **Timing:** OTT consistently ~3s; POT varies 2-27s. OTT speedup most significant at small epsilon.

### Next steps
- Investigate whether increasing OTT `max_iterations` (currently 2000) or switching to
  float64 can push concordance to eps=1e-5
- Run velocity/coupling concordance at eps=1e-4 (not just cost)
- Test on GPU (current spike was CPU-only)
