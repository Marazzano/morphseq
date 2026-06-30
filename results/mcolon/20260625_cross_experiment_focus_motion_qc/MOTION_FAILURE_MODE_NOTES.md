# Motion QC Failure Mode Notes

## Anchors

Checked these 20251125 anchors:

- H05 t19: visible motion, not-dead, missed by current refined rule
- H05 t25: visible motion, not-dead, missed by current refined rule
- D06 t51: visible motion, not-dead, missed by current refined rule
- H05 t162: already caught by current rule, but not-dead is false

## Current Rule

The current posthoc refined rule is:

- `ncc_p05 < 0.85`
- AND `bad_pair_frac > 0.10`

`bad_pair_frac` is currently the fraction of adjacent z-slice pairs where the
mean NCC across all tiles is below 0.90.

Important terminology:

- "frame" in this motion-QC context really means an adjacent z-slice pair
  inside one z-stack, not a timelapse timepoint.
- A tile NCC is computed between the same spatial tile in z-slice `z` and
  z-slice `z + 1`.
- The current bad-pair definition is:
  `pair_mean_ncc_all_tiles < 0.90`.
- The current bad-stack summary is:
  `bad_pair_frac = n_bad_pairs / n_adjacent_z_pairs`.
- The current refined stack reject is:
  `ncc_p05 < 0.85 AND bad_pair_frac > 0.10`.

## Main Failure Mode

The missed visible-motion anchors are not invisible to NCC. They all have low
`ncc_p05`, low `ncc_min`, and elevated `ncc_bad_tile_frac`.

They are missed because `bad_pair_frac` is a whole-pair mean over all tiles.
Background and stable non-embryo tiles keep the pair mean above 0.90 even when
the embryo tiles are clearly unstable.

For example:

| anchor | table ncc_p05 | table bad_pair_frac | table bad_tile_frac | mask ncc_p05 | mask bad_pair_frac by mean | mask bad_tile_frac |
|---|---:|---:|---:|---:|---:|---:|
| H05 t19 | 0.768 | 0.000 | 0.120 | 0.361 | 1.000 | 0.643 |
| H05 t25 | 0.634 | 0.000 | 0.135 | -0.077 | 1.000 | 0.964 |
| D06 t51 | 0.815 | 0.000 | 0.100 | 0.310 | 0.250 | 0.263 |
| H05 t162 | 0.707 | 0.500 | 0.327 | 0.963 | 0.000 | 0.000 |

No-grid mask-pixel NCC was also tested on the same anchors. This computes one
NCC per adjacent z-pair using all pixels inside the embryo mask.

| anchor | mask-pixel ncc_p05 | mask-pixel bad_pair_frac | longest bad run |
|---|---:|---:|---:|
| H05 t19 | 0.669 | 1.000 | 8 |
| H05 t25 | 0.302 | 1.000 | 8 |
| D06 t51 | 0.277 | 0.375 | 3 |
| H05 t162 | 0.973 | 0.000 | 0 |

This simpler no-grid rule catches the three visible-motion not-dead anchors and
does not trigger H05 t162. The tradeoff is that it produces one pair-level NCC,
so it cannot expose a bad-tile fraction or spatially localize partial motion.

H05 t162 is a separate case: the all-tile metric catches something, but the
snip is not-dead false and the mask-overlapping NCC is clean. It should not be
used as evidence that the refined rule works on alive embryos.

## Cross-Experiment Scale

On not-dead rows, the current `ncc_p05`-only failures that are rescued by
`bad_pair_frac <= 0.10` are common:

| experiment | n not-dead rows | ncc-only current n | ncc-only current frac | reject-by-both current frac |
|---|---:|---:|---:|---:|
| 20250305 | 25803 | 3487 | 0.135 | 0.018 |
| 20251125 | 18158 | 7877 | 0.434 | 0.046 |
| 20260206 | 13126 | 6600 | 0.503 | 0.004 |

This means requiring `bad_pair_frac` as a confirmation is probably too
restrictive, especially for local embryo motion.

## Candidate Direction

Do not define "bad frame/pair" from the all-tile pair mean. Two viable
embryo-restricted options are now on the table.

Option A: no-grid mask-pixel NCC:

- restrict to all pixels inside the embryo mask
- compute one NCC per adjacent z-pair
- bad z-pair / bad frame:
  `mask_pixel_pair_ncc < 0.90`
- bad stack:
  `fraction(bad_z_pairs) > bad_stack_pair_fraction_threshold`

This is simpler and likely cheaper.

Option B: mask-tile local NCC:

- compute NCC grid as before
- restrict tiles to mask-overlapping embryo tiles
- for each z-pair, compute `pair_bad_tile_frac`
- call a pair bad when enough embryo-overlapping tiles are below NCC 0.90

Suggested explicit parameters for the next implementation:

- `tile_size_px = 128`
- `tile_stride_px = 128`
- `min_mask_tile_coverage = 0.10`
- `bad_tile_ncc_threshold = 0.90`
- `bad_pair_tile_fraction_threshold = 0.10`
- `bad_stack_pair_fraction_threshold = 0.10`

With those parameters:

- Bad tile:
  `tile_ncc < bad_tile_ncc_threshold`.
- Valid embryo tile:
  tile has at least `min_mask_tile_coverage` mask coverage.
- Bad z-pair / bad frame:
  among valid embryo tiles for that adjacent z-pair,
  `fraction(tile_ncc < bad_tile_ncc_threshold) > bad_pair_tile_fraction_threshold`.
- Bad stack:
  `fraction(bad_z_pairs) > bad_stack_pair_fraction_threshold`.

The existing `ncc_bad_tile_frac` already captures the all-tile version of this
idea and catches the missed anchors better than `bad_pair_frac`, but the anchor
diagnostic suggests the proper metric should be mask-aware.

Generated diagnostics:

- `tables/motion_anchor_failure_summary.csv`
- `tables/motion_anchor_pair_metrics.csv`
- `figures/motion_anchor_failure_modes/*_pair_profile.png`
- `figures/motion_anchor_failure_modes/*_worst_pair.png`
