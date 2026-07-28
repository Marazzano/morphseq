# Modal V0 geometry-derived bandwidth calibration

Sample sizes: [80]
Seeds per sample size: 3
Rules: longest_non_outlier_MST_edge, q90_MST_edge_length, median_kNN_distance
Multipliers: [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

## Truth anchor check

      distribution_id  truth_peak_count  truth_global_valley_density_ratio  truth_bridge_pair_valley_density_ratio  truth_bridge_region_density_ratio  truth_bridge_region_mass_fraction truth_bridge_region_label
     one_peak_compact                 1                                NaN                                     NaN                                NaN                                NaN                          
     one_peak_diffuse                 1                                NaN                                     NaN                                NaN                                NaN                          
      one_peak_spiral                 1                                NaN                                     NaN                                NaN                                NaN                          
  two_peaks_no_bridge                 2                           0.798141                            6.235259e-11                           0.050659                           0.074982              unclassified
 two_peaks_low_bridge                 2                           0.788593                            1.433953e-02                           0.083990                           0.046163                low_bridge
two_peaks_high_bridge                 2                           0.688342                            1.330554e-01                           0.353657                           0.212338               high_bridge
  three_peaks_compact                 3                           0.697889                                     NaN                                NaN                                NaN                          

## Selected rule per sample size

 n               bandwidth_rule  bandwidth_multiplier  median_selected_bandwidth  median_abs_log_bridge_pair_valley_ratio_error  median_abs_bridge_pair_valley_depth_error  median_abs_log_global_valley_ratio_error  median_abs_global_valley_ratio_error  valley_ordering_recovery_rate  bridge_ordering_recovery_rate  false_split_rate  false_merge_rate  peak_count_sanity_rate
80 longest_non_outlier_MST_edge                  0.75                   0.310127                                       0.364536                                   0.007352                                  0.105770                              0.076382                            1.0                            1.0          0.333333               0.5                0.714286
80          median_kNN_distance                  0.85                   0.368383                                       0.669101                                   0.003864                                  0.091541                              0.069221                            1.0                            1.0          0.000000               0.5                0.857143
80          q90_MST_edge_length                  0.95                   0.311781                                       0.722713                                   0.006152                                  0.098882                              0.071608                            1.0                            1.0          0.222222               0.5                0.761905

## Notes

Pairwise bridge metrics use the first two mode components in V0.
Bridge-region metrics use explicit bridge components when present, and the no-bridge anchor keeps a synthetic corridor measurement.

- n=80 longest_non_outlier_MST_edge x 0.75: valley=1.00, bridge=1.00, peak=0.71
- n=80 median_kNN_distance x 0.85: valley=1.00, bridge=1.00, peak=0.86
- n=80 q90_MST_edge_length x 0.95: valley=1.00, bridge=1.00, peak=0.76
