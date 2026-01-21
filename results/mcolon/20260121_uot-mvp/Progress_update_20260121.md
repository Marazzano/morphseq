# Progress update (2026-01-21)

Summary
- Synthetic sanity checks now behave sensibly after scaling the coordinate cost (so Sinkhorn no longer collapses to all creation/destruction).
- The **uniform mass mode** produces the most interpretable outcomes so far, with clean velocity fields and reasonable creation/destruction for shape changes.
- Visualization overlays for creation/destruction are now aligned to the original mask space and use a consistent non-negative scale.

Key fixes made
- Added coordinate scaling so pixel distances are interpretable and avoid numerical underflow in the solver.
- Added synthetic test cases (including non-overlap) and multi–mass-mode runs for direct comparison.
- Added overlays of creation/destruction on source/target masks with correct shape alignment.
- Clamped creation/destruction plots to non-negative values so colorbars are comparable and not misleading.

Open questions
- The solver still sometimes creates/destroys small amounts of mass; we need to decide acceptable thresholds and whether to penalize this more aggressively.
- Current plotting auto-scales per image; we should standardize the scale so comparisons across runs are meaningful.
- We need an interpretable cost scale (ideally per-pixel). This may require choosing a canonical coordinate system and standardized units rather than ad hoc scaling.
- Velocity field visualization is sensitive to small changes; we should normalize the field so small motions do not visually compete with large motions.

Next steps
- Define a canonical grid for embryo masks and test it with synthetic cases.
- Standardize units so transport cost and velocity magnitudes are interpretable per pixel.
- Decide whether to increase epsilon or otherwise tune the cost scale to match physical intuition.
- Normalize velocity field plotting for consistent magnitude interpretation across runs.

Status: Success for a first pass.
