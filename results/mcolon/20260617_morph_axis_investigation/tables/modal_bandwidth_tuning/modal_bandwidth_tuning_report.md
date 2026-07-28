# Modal V0 geometry-derived bandwidth calibration

Sample sizes: [80, 160]
Seeds per sample size: 5
Rules: median_kNN_distance, q90_kNN_distance, median_MST_edge_length, q90_MST_edge_length, longest_non_outlier_MST_edge, connectivity_90_radius, global_R50, global_R80
Multipliers: [0.75, 1.0, 1.25, 1.5]

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
 80 longest_non_outlier_MST_edge                  0.75                   0.304717                                       0.558910                                   0.007352                                  0.240695                              0.157538                            1.0                            1.0          0.266667               0.5                0.742857
160 longest_non_outlier_MST_edge                  1.00                   0.270759                                       0.779511                                   0.007763                                  0.116635                              0.081156                            1.0                            1.0          0.200000               0.5                0.771429
 80          q90_MST_edge_length                  1.00                   0.326181                                       0.785638                                   0.007803                                  0.221240                              0.145603                            1.0                            1.0          0.133333               0.5                0.800000
160          median_kNN_distance                  1.00                   0.300592                                       0.821824                                   0.008680                                  0.129145                              0.088317                            1.0                            1.0          0.000000               0.5                0.857143
160          q90_MST_edge_length                  1.25                   0.301265                                       0.879457                                   0.010473                                  0.135224                              0.093090                            1.0                            1.0          0.266667               0.5                0.742857
 80          median_kNN_distance                  0.75                   0.342903                                       0.892957                                   0.008780                                  0.249689                              0.162312                            1.0                            1.0          0.066667               0.5                0.828571
160             q90_kNN_distance                  0.75                   0.398045                                       1.119705                                   0.021617                                  0.145510                              0.107412                            1.0                            1.0          0.000000               0.5                0.857143
 80             q90_kNN_distance                  0.75                   0.557080                                       1.432511                                   0.043835                                  0.195415                              0.138442                            1.0                            1.0          0.000000               0.5                0.857143
160       connectivity_90_radius                  0.75                   0.572024                                       1.951363                                   0.102962                                  0.149352                              0.095477                            0.0                            0.0          0.066667               1.0                0.685714
160                   global_R50                  0.75                   1.154914                                       2.570630                                   0.222333                                  0.185730                              0.133668                            0.8                            0.8          0.000000               0.5                0.685714
 80       median_MST_edge_length                  1.50                   0.237577                                       2.590418                                   0.013264                                  0.310590                              0.195729                            1.0                            1.0          0.333333               0.5                0.714286
 80       connectivity_90_radius                  0.75                   0.613316                                       2.670791                                   0.245172                                  0.225677                              0.147990                            0.0                            0.0          0.000000               1.0                0.714286
 80                   global_R50                  0.75                   1.212773                                       2.753132                                   0.287655                                  0.282798                              0.193342                            0.8                            0.8          0.000000               0.5                0.714286
 80                   global_R80                  0.75                   1.872696                                       4.211560                                   0.949718                                  0.165037                              0.143216                            0.2                            0.2          0.000000               0.9                0.457143
160                   global_R80                  0.75                   1.875735                                       4.211984                                   0.952976                                  0.165037                              0.143216                            0.0                            0.8          0.000000               0.9                0.457143
160       median_MST_edge_length                  1.50                   0.164882                                       5.046157                                   0.014247                                  0.213606                              0.143216                            1.0                            1.0          0.333333               0.5                0.714286

## Notes

Pairwise bridge metrics use the first two mode components in V0.
Bridge-region metrics use explicit bridge components when present, and the no-bridge anchor keeps a synthetic corridor measurement.

- n=80 longest_non_outlier_MST_edge x 0.75: valley=1.00, bridge=1.00, peak=0.74
- n=160 longest_non_outlier_MST_edge x 1.00: valley=1.00, bridge=1.00, peak=0.77
- n=80 q90_MST_edge_length x 1.00: valley=1.00, bridge=1.00, peak=0.80
- n=160 median_kNN_distance x 1.00: valley=1.00, bridge=1.00, peak=0.86
- n=160 q90_MST_edge_length x 1.25: valley=1.00, bridge=1.00, peak=0.74
- n=80 median_kNN_distance x 0.75: valley=1.00, bridge=1.00, peak=0.83
- n=160 q90_kNN_distance x 0.75: valley=1.00, bridge=1.00, peak=0.86
- n=80 q90_kNN_distance x 0.75: valley=1.00, bridge=1.00, peak=0.86
- n=160 connectivity_90_radius x 0.75: valley=0.00, bridge=0.00, peak=0.69
- n=160 global_R50 x 0.75: valley=0.80, bridge=0.80, peak=0.69
- n=80 median_MST_edge_length x 1.50: valley=1.00, bridge=1.00, peak=0.71
- n=80 connectivity_90_radius x 0.75: valley=0.00, bridge=0.00, peak=0.71
- n=80 global_R50 x 0.75: valley=0.80, bridge=0.80, peak=0.71
- n=80 global_R80 x 0.75: valley=0.20, bridge=0.20, peak=0.46
- n=160 global_R80 x 0.75: valley=0.00, bridge=0.80, peak=0.46
- n=160 median_MST_edge_length x 1.50: valley=1.00, bridge=1.00, peak=0.71
