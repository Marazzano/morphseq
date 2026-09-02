# `shared/identifiers/` — the one place that names objects

This package is the **single sacred home** for the pipeline's identifier grammar.
Nothing else should mint, parse, or hand-roll an `image_id` / `embryo_id` /
`snip_id` / `well_id`. Import from here.

> **Identifier strings are opaque outside `shared/identifiers`.** Code outside this
> package must use constructors and parsers — never string splitting, regex matching,
> or f-string minting. Tiny fence, giant moat.

## Canonical identifier grammar

```
experiment_id       = 20240418
well_index          = A01
well_id             = 20240418_A01                       global well id
channel_id          = BF                                 no underscores in channel_id
image_id            = {well_id}_{channel_id}_t{time_index:04d}
mask_id             = {image_id}_m{local_mask_index:04d}
no-mask mask_id     = {image_id}_mask_none
track_id            = {well_id}_track{track_index:04d}       zero-based track index
physical_embryo_id  = {well_id}_e{local_embryo_index:02d}   one-based (≥ 1)
embryo_id           = {physical_embryo_id}_{channel_id}
snip_id             = {embryo_id}_t{time_index:04d}
```

Tiny doctrine:
```
Physical embryo ID names the animal.
Embryo ID names the animal in a channel.
Snip ID names the animal-channel at a time.
```

## The three kinds of helper

| Module | Verb | Functions |
|---|---|---|
| `constructors.py` | **mint** (assemble) | `build_well_id`, `build_image_id`, `build_mask_id`, `build_no_mask_id`, `build_track_id`, `build_physical_embryo_id`, `build_embryo_id`, `build_snip_id`, `sanitize_experiment_id` |
| `parsers.py` | **decompose** (inverse of mint) + **transform** | `parse_image_id`, `parse_mask_id`, `parse_track_id`, `parse_physical_embryo_id`, `parse_embryo_id`, `parse_snip_id`, `parse_embryo_local_track_id`, `track_index_to_embryo_index`, `split_well_id` |
| `validators.py` | **guard** (fail loudly) | `validate_well_id` |

`__init__.py` re-exports every public name, so
`from data_pipeline.shared.identifiers import build_image_id` keeps working.

## Constructors are dumb

They assemble a canonical string and nothing more — no biology inference, no name
mapping (`Brightfield -> BF` happens upstream in metadata ingest, not here).
Constructors call parsers for any decomposition; they never split strings manually.

## Track-index to embryo-index conversion

Backend/SAM2 object IDs are zero-based. Embryo IDs are one-based (biologist-facing).
The conversion is explicit and named:

```python
raw_track_index    = parse_embryo_local_track_id(track_id)   # "embryo_0" -> 0
local_embryo_index = track_index_to_embryo_index(raw_track_index)  # 0 -> 1
physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
```

Never use raw `+ 1` arithmetic. The conversion site must be visible.

## ⚠️ CURRENT vs TARGET `well_id` semantics

This package was split out of the flat `shared/identifiers.py` in **Scope 1** with
**signatures unchanged**. Under the **CURRENT** convention `well_id` is GLOBAL:

- `build_well_id(experiment_id, well_index)` → `"{experiment_id}_{well_index}"`

The **TARGET** refactor (**Scope 2**, see
`docs/data_pipeline/specs/target/well_id_throughline_refactor_plan.md`)
makes `well_index` column renamed to `well`.

`split_well_id` and `validate_well_id` describe **TARGET** (global-`well_id`)
behavior. They are scaffolded here so the package shape is final and the contract is visible.

See also `docs/data_pipeline/specs/identifier_and_wildcard_contract.md`.
