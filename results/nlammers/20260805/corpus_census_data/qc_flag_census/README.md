# QC flag census

Generated: `2026-07-28T09:37:51.174764-07:00`

Snapshot definition: readable experiment-level merged `snip_qc` tables. Counts use BF rows when BF is identifiable.

## Strict current verdict

| Scope | Dataset Count | Timepoint Count | Timepoint Pass Rate | Well Count | Well Any Pass Rate | Zero Pass Wells | Physical Embryo Id Count | Embryo Any Pass Rate | Zero Pass Embryos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keyence | 29 | 28761 | 54.7% | 2414 | 89.3% | 258 | 2667 | 81.7% | 489 |
| YX1 | 26 | 422276 | 36.4% | 2389 | 93.2% | 163 | 3786 | 79.9% | 762 |
| All | 55 | 451037 | 37.6% | 4803 | 91.2% | 421 | 6453 | 80.6% | 1251 |

## Most frequent exclusion flags

### Keyence

| Flag | Triggered Timepoint Count | Trigger Rate All Timepoints | Exclusive Timepoint Count | Dataset Count Triggered |
| --- | --- | --- | --- | --- |
| sa_outlier_flag | 10025 | 34.9% | 2859 | 28 |
| focus_flag | 7000 | 24.3% | 835 | 24 |
| persistence_dead_flag | 5819 | 20.2% | 565 | 4 |
| viability_dead_flag | 5506 | 19.1% | 42 | 9 |
| edge_flag | 1486 | 5.2% | 159 | 15 |
| motion_blur_flag | 554 | 1.9% | 277 | 22 |
| overlapping_mask_flag | 436 | 1.5% | 134 | 22 |
| discontinuous_mask_flag | 122 | 0.4% | 19 | 6 |

### YX1

| Flag | Triggered Timepoint Count | Trigger Rate All Timepoints | Exclusive Timepoint Count | Dataset Count Triggered |
| --- | --- | --- | --- | --- |
| sa_outlier_flag | 194884 | 46.2% | 37166 | 26 |
| focus_flag | 149238 | 35.3% | 12541 | 26 |
| persistence_dead_flag | 109188 | 25.9% | 6835 | 25 |
| viability_dead_flag | 102941 | 24.4% | 1016 | 25 |
| motion_blur_flag | 54104 | 12.8% | 15867 | 25 |
| edge_flag | 40301 | 9.5% | 9590 | 22 |
| overlapping_mask_flag | 31152 | 7.4% | 5229 | 23 |
| discontinuous_mask_flag | 6819 | 1.6% | 652 | 26 |

### All

| Flag | Triggered Timepoint Count | Trigger Rate All Timepoints | Exclusive Timepoint Count | Dataset Count Triggered |
| --- | --- | --- | --- | --- |
| sa_outlier_flag | 204909 | 45.4% | 40025 | 54 |
| focus_flag | 156238 | 34.6% | 13376 | 50 |
| persistence_dead_flag | 115007 | 25.5% | 7400 | 29 |
| viability_dead_flag | 108447 | 24.0% | 1058 | 34 |
| motion_blur_flag | 54658 | 12.1% | 16144 | 47 |
| edge_flag | 41787 | 9.3% | 9749 | 37 |
| overlapping_mask_flag | 31588 | 7.0% | 5363 | 45 |
| discontinuous_mask_flag | 6941 | 1.5% | 671 | 32 |

## Policy sensitivity

| Scope | Policy | Timepoint Pass Rate | Well Any Pass Rate | Embryo Any Pass Rate | Zero Pass Well Count |
| --- | --- | --- | --- | --- | --- |
| Keyence | strict_current | 54.7% | 89.3% | 81.7% | 258 |
| Keyence | death_only | 77.9% | 99.8% | 99.4% | 5 |
| Keyence | death_and_mask_geometry | 72.7% | 97.5% | 94.8% | 61 |
| Keyence | mask_geometry_only | 93.0% | 97.7% | 95.4% | 56 |
| Keyence | ignore_surface_area | 64.6% | 95.7% | 89.6% | 103 |
| Keyence | ignore_focus_motion | 58.6% | 90.3% | 83.5% | 234 |
| YX1 | strict_current | 36.4% | 93.2% | 79.9% | 163 |
| YX1 | death_only | 72.0% | 98.8% | 97.9% | 29 |
| YX1 | death_and_mask_geometry | 59.5% | 97.2% | 90.7% | 68 |
| YX1 | mask_geometry_only | 82.5% | 98.6% | 92.9% | 34 |
| YX1 | ignore_surface_area | 45.2% | 95.4% | 86.4% | 110 |
| YX1 | ignore_focus_motion | 43.2% | 94.7% | 82.4% | 126 |
| All | strict_current | 37.6% | 91.2% | 80.6% | 421 |
| All | death_only | 72.4% | 99.3% | 98.5% | 34 |
| All | death_and_mask_geometry | 60.3% | 97.3% | 92.4% | 129 |
| All | mask_geometry_only | 83.1% | 98.1% | 93.9% | 90 |
| All | ignore_surface_area | 46.5% | 95.6% | 87.7% | 213 |
| All | ignore_focus_motion | 44.2% | 92.5% | 82.9% | 360 |

## Interpretation cautions

- Flag trigger counts overlap; a timepoint may contribute to several flags.
- A physical embryo ID is a tracking product, not necessarily one true embryo.
- A well with any passing timepoint is a better test of gross acquisition failure.
- Existing completed-QC datasets are a selected subset of the corpus.
- Historical QC artifacts may reflect different flag configurations.
- Policy sensitivities are diagnostic counterfactuals, not proposed training gates.
