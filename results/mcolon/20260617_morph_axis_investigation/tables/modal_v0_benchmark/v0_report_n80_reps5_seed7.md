V0 modal benchmark
n = 80, reps = 5, seed = 7

Truth peak counts: all 8 distributions matched expected counts.
Pairwise checks passed: 5/8

Pairwise failures:
  - two_peaks_no_bridge vs two_peaks_low_bridge on valley_depth: win_rate=0.40, delta=-0.010
  - two_peaks_low_bridge vs two_peaks_high_bridge on valley_depth: win_rate=0.60, delta=0.091
  - two_peaks_low_bridge vs two_peaks_high_bridge on mst_max_edge: win_rate=0.60, delta=0.895
