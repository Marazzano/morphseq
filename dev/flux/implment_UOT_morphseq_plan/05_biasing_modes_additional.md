# Task 05 — Additional Biasing Modes

## `z_hard` (example)
- **Idea:** zero mass beyond a time/distance threshold.
- **Switch:** `bias_mode='z_hard'`, `bias_kwargs={'radius': r, 'time_tol': Δt}`.
- **Implementation:** mask `K[i,j]=0` if `||x_i-x_j||>r` or `|t_i-t_j|>Δt` before Sinkhorn.

## `prior_edges`
- **Idea:** supply a prior adjacency `A_prior` from heuristics; blend into `K` like lower_bound.

## Guardrails
- All modes must be strict no‑ops when disabled.
- Each lives in a small `if bias_mode == ...:` block to keep MVP readable.
- Prefer one new mode per PR; add per‑mode smoke tests.