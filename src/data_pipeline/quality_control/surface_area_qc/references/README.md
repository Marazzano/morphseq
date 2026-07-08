# surface_area_qc packaged reference

`surface_area_reference_v1.csv` is the curated, static reference curve that
`surface_area_qc` flags against. It is a **source asset**, versioned and reviewed
alongside the QC logic that consumes it — it is **not** a pipeline artifact and has
**no** row in `orchestration/paths.py`. The product loads it through
`reference.py::load_packaged_surface_area_reference`, never by a raw path.

## Columns (all five required, validated by `reference_contract.py`)

| column | meaning |
|---|---|
| `stage_hpf` | developmental stage (hours post fertilization) the bin is centered on |
| `p5` | 5th-percentile wildtype `area_um2` at that stage |
| `p50` | median wildtype `area_um2` (kept for review plots / retuning; not used by the flag) |
| `p95` | 95th-percentile wildtype `area_um2` at that stage |
| `n` | sample count behind the bin (provenance; not used by the flag) |

The flag math uses **only `p5` and `p95`** interpolated at each snip's
`predicted_stage_hpf`; `p50` and `n` are retained for review plots, future retuning,
and provenance (see `feature_world.md` surface_area_qc reference policy).

## Provenance

This `v1` curve was copied verbatim from `metadata/sa_reference_curves.csv`
(259 stage bins, 0.25–48.75 hpf). The build tooling that originally produced that
file lives at `quality_control/generate_references/build_sa_reference.py` — it is
provenance only (it carries hardcoded paths into the other repo) and is **not** a
pipeline step.

## Versioning

A new reference becomes `surface_area_reference_v2.csv` (etc.) and the product's
`reference_version` config knob selects it. Never edit a published version in place.
