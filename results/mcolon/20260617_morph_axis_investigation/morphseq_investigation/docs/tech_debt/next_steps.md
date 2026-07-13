# Distribution engine follow-ups

These are tracked follow-ups from the TASK_E b9d2 integration. They are not
blockers for the current acceptance target.

## N-D peak finding

`engine.grid.evaluate_density` is already dimension-agnostic: a one-feature
`Grid` plus `evaluate_density` is a real 1-D KDE strip, and TASK_E uses that path
for the emergence figure.

`engine.labelers.label_peak_finding` is still 2-D only because it bridges to the
existing live `CanonicalGrid` peak detector. Future work should make
`peak_finding` N-D, starting with a simulated 1-D validation path before applying
it to real strip peaks.

For the current b9d2 example, this split is intentional: peak finding runs on the
2-D feature grid where the emergence signal is detected, while 1-D strips are a
projection for visualization.

## True KDE ridge plot over time

The current TASK_E figure is a faceted KDE strip panel, not a true ridge plot.
The desired follow-up is a KDE-over-time ridge plot: one vertical offset row per
time bin, with wildtype and b9d2/phenotype densities overlaid within each row.
That plot should show how the target distribution separates across time while
keeping the WT reference visible at the same timepoint.

Do not mark this complete until the densities are stacked/offset as a real
ridgeline/joyplot rather than displayed as independent faceted panels.

## Parallel-agent worktrees

The parallel task build had git races from multiple agents working on one branch.
The affected agents self-corrected and the resulting history is clean, but future
parallel runs should use isolated worktrees per task branch.
