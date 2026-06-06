# `shared/identifiers/` — the one place that names objects

This package is the **single sacred home** for the pipeline's identifier grammar.
Nothing else should mint, parse, or hand-roll an `image_id` / `embryo_id` /
`snip_id` / `well_id`. Import from here.

## The three kinds of helper

| Module | Verb | Functions |
|---|---|---|
| `constructors.py` | **mint** (assemble) | `build_well_id`, `build_image_id`, `build_embryo_id`, `build_snip_id`, `sanitize_experiment_id` |
| `parsers.py` | **decompose** (inverse of mint) | `normalize_embryo_local_track_id`, `split_well_id` |
| `validators.py` | **guard** (fail loudly) | `validate_well_id` |

`__init__.py` re-exports every public name, so
`from data_pipeline.shared.identifiers import build_image_id` keeps working.

## Constructors are dumb

They assemble a canonical string and nothing more — no biology inference, no name
mapping (`Brightfield -> BF` happens upstream in metadata ingest, not here).

## ⚠️ CURRENT vs TARGET `well_id` semantics

This package was split out of the flat `shared/identifiers.py` in **Scope 1** with
**signatures unchanged**. Under the **CURRENT** convention:

- `build_well_id(well_index) -> "A01"` — `well_id` is the plate-**LOCAL** label.
- `build_image_id(experiment_id, well_id, channel_id, time_int)` — experiment is a
  separate arg.

The **TARGET** refactor (**Scope 2**, see
`docs/refactors/streamline-snakemake/target/well_id_throughline_refactor_plan.md`)
makes `well_id` **GLOBAL**:

- `build_well_id(experiment_id, well) -> "{experiment_id}_{well}"` (sanitize moves up).
- `build_image_id`/`build_embryo_id` become **`well_id`-first**.
- `well_index` column is renamed to `well`.

`split_well_id` and `validate_well_id` describe **TARGET** (global-`well_id`)
behavior, so they currently raise `NotImplementedError` and **activate in Scope 2**.
They are scaffolded here so the package shape is final and the contract is visible.

See also `docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md`.
