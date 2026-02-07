# Stream D: Reference Embryo — Decisions

## Reference construction
Mean of multiple WT-like (Not Penetrant) embryo fields. Simple average is a good starting point.

## Deviation metrics
- RMSE: root mean squared error between test and reference velocity fields
- Cosine similarity: directional agreement per-pixel, then mean
- Residual field: test - reference, for visualization

## Dataclasses
- `ReferenceField`: single frame-pair reference (velocity, mass_created, mass_destroyed, support_mask, n_embryos)
- `ReferenceTimeseries`: dict mapping frame pairs to ReferenceField objects

## Yolk mask limitation
Missing yolk masks degrade canonical alignment but don't crash (fallback logic exists).
This is a known limitation documented here.
