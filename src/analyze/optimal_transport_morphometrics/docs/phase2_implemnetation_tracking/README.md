# UOT Phase 2: Implementation Tracking

See the full plan in the conversation transcript. This directory tracks implementation progress.

## Commit Plan (atomic, in order)

1. `feat(ot): add ott backend + optional import wiring`
2. `test(ot): add pot-vs-ott concordance suite`
3. `feat(uot): pass backend through timeseries + loader plumbing`
4. `feat(viz): enforce support-mask/NaN contract`
5. `feat(ref): add reference field + deviation metrics`

## Workstream Directories

- `stream_a_ott_backend/` — OTT backend + concordance tests
- `stream_b_backend_plumbing/` — Backend parameter threading through timeseries
- `stream_c_viz_contract/` — Visualization NaN contract enforcement
- `stream_d_reference_embryo/` — Reference embryo + deviation metrics
- `stream_e_future/` — Future workstreams (feature-over-time, difference testing)
