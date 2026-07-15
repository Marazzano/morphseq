# Relative-Geometry Audit

## Verdict

The raw-latent baseline did **not** take the documented relative-geometry
configuration path. It copied the legacy absolute-coordinate settings from the
20260407 PBX generator.

## Documented contract

`src/analyze/trajectory_condensation/README.md` and `ALGORITHM.md` describe
`sigma` and `epsilon_r` as internal calibration-only quantities. The intended
public path is:

- attraction bandwidth: `attract_bandwidth_mult * s_global`
- temporal-coherence bandwidth: `temporal_cohere_bandwidth_mult * s_local`
- repulsion: derived from `s_local`
- elasticity: `elastic_strength` / `elastic_mix`, resolved from `s_step` and
  `s_bend`

## Actual raw-baseline configuration

`1_cluster_raw_latent.py` directly passes:

```python
sigma=0.5
epsilon_r=5e-4
```

and leaves both bandwidth multipliers as `None`.

In `condensation/engine/run.py`, this resolves to:

```python
sigma_att = config.sigma
sigma_coh = sigma_att
```

rather than a quantity derived from the raw run's geometry references. The
repulsion coefficient is likewise passed through directly to the force term.

## Consequence for this comparison

The raw and margin runs share absolute `sigma=0.5`, but their UMAP layouts have
different global spreads:

| arm | s_local | s_global | sigma / s_global |
| --- | ---: | ---: | ---: |
| raw z_mu_b | 0.389 | 1.026 | 0.487 |
| margin | 0.386 | 0.843 | 0.593 |

Thus the raw run's attraction/coherence kernel is effectively about 18% more
local relative to its own global layout. Its repulsion happens to be nearly
matched relative to `s_local^2` in this specific pair of layouts, but that is
accidental rather than enforced by the public API.

## Implementation gap

The code supports relative attraction and coherence bandwidths through
`*_bandwidth_mult`, but `CondensationConfig` still exposes `sigma` and
`epsilon_r`, and has no public `repulsion_strength` multiplier. This disagrees
with the README's statement that those fields are internal/not user-facing.

## Required next step

Do not interpret the current raw-vs-margin outcome as a clean input-space
comparison until the baseline is rerun through a configuration path that uses
the same dimensionless settings for both arms. That requires either adding the
missing public relative repulsion knob or explicitly documenting a temporary
compatibility conversion at the runner boundary.
