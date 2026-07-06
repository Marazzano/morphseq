V0 modal benchmark
n = 80, reps = 3, seed = 7

Truth peak counts: all 8 distributions matched expected counts.
Pairwise checks passed: 5/8

Pairwise failures:
  - two_peaks_no_bridge vs two_peaks_low_bridge on valley_depth: win_rate=0.67, delta=0.057
  - two_peaks_low_bridge vs two_peaks_high_bridge on valley_depth: win_rate=0.67, delta=0.454
  - two_peaks_low_bridge vs two_peaks_high_bridge on fiedler: win_rate=0.67, delta=0.002
