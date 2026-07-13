─────────────────────────────────────────────────────────────────────────────────────────────

• ### A. Current Architecture Map

  Truth path

  V0Distribution.density_spec
  → DensitySpec
  → compose_density_truth(spec)
  → ComposedDensityTruth
  → validate_composed_density_truth(...)
  → count_mass_significant_modes(...)
  → peak_count_detail(...)
  → detect_peaks(method="superlevel_cap_mass")
  → PeakDetectionResult, but validation keeps only resolved_peak_count

  Key evidence:

  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:96 defines DensitySpec.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:124 defines ComposedDensityTruth.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:744 builds ComposedDensityTruth.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:338 validates truth peak count via count_mass_significant_modes.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:906 defines count_mass_significant_modes.

  Truth currently has a composed canonical density grid and recipe/component provenance, but no durable resolved-truth peak object.

  Empirical path

  DensityRealization.points
  → propose_bandwidth_candidates(points)
  → precompute_squared_distances(grid_points, points)
  → evaluate_isotropic_gaussian_kde_from_dist2(...)
  → DensityGrid on canonical grid
  → detect_peaks(...)
  → PeakDetectionResult
  → PeakCandidateDetail tuple

  Key evidence:

  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:139 defines DensityRealization.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:781 samples grid density into points.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/bandwidth_tuning.py:553 proposes bandwidth candidates.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/bandwidth_tuning.py:609 evaluates isotropic Gaussian KDE.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/v0/plot_modal_v0_bandwidth_comparison.py:79 builds the empirical DensityGrid.
  - results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/v0/plot_modal_v0_bandwidth_comparison.py:96 runs all peak detectors.

  ### B. Existing Primitive Inventory

   primitive                             location                                       current role    quality                     reusable as-is?           notes
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
   CanonicalGrid                         results/                   canonical coordinates, cell area    good                        yes                       Required for comparable
                                         mcolon/20260617_                                                                                                     geometry.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         density_composit
                                         ion.py:25
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   DensityGrid                           results/                          evaluated density on grid    good                        yes                       Carries xx, yy, density,
                                         mcolon/20260617_                                                                                                     optional grid.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         density_composit
                                         ion.py:64
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   DensitySpec / DensityComponentSpec    results/                                  simulation recipe    good provenance             no as geometry            local_width is recipe
                                         mcolon/20260617_                                                                                                     scale, not resolved peak
                                         morph_axis_inves                                                                                                     radius.
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         density_composit
                                         ion.py:74
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   ComposedDensityTruth                  results/                            truth density container    good                        yes                       Needs builder to resolve
                                         mcolon/20260617_                                                                                                     peaks from
                                         morph_axis_inves                                                                                                     composed_grid.
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         density_composit
                                         ion.py:124
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   ComponentTruth                        results/                  component provenance and centroid    partial                     not as resolved peak      Component can be bridge
                                         mcolon/20260617_                                                                                                     or merged recipe
                                         morph_axis_inves                                                                                                     component, not resolved
                                         tigation/                                                                                                            peak.
                                         morphseq_investi
                                         gation/core/
                                         density_composit
                                         ion.py:105
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   PeakCandidateDetail                   results/                    detector evidence per candidate    mixed                       partially                 Too detector-specific to
                                         mcolon/20260617_                                                                                                     be shared geometry.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         peak_counting.py
                                         :29
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   PeakDetectionResult                   results/                                    detector output    good as detector result     yes, wrap                 Do not mutate into
                                         mcolon/20260617_                                                                                                     resolved distribution.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         peak_counting.py
                                         :53
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   _assign_points_to_peaks               results/                          nearest-center assignment    useful but incomplete       yes with caveat           Always assigns every
                                         mcolon/20260617_                                                                                                     point when peaks exist.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         peak_counting.py
                                         :253
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   _assign_cells_to_peaks                results/                 nearest-center Voronoi grid labels    useful but not watershed    yes with caveat           Not true density basin.
                                         mcolon/20260617_
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         peak_counting.py
                                         :237
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   _superlevel_split                     results/            first mass-significant superlevel split    good detector primitive     yes                       Provides labels only
                                         mcolon/20260617_                                                                                                     internally.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         peak_counting.py
                                         :273
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   _hdr_level_components                 results/                      HDR component count/mass/area    useful                      partially                 Does not preserve labels
                                         mcolon/20260617_                                                                                                     in public result.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         peak_counting.py
                                         :470
  ────────────────────────────────────  ──────────────────  ─────────────────────────────────────────  ──────────────────────────  ────────────────────────  ──────────────────────────
   bandwidth_geometry_scales             results/                                global sample scale    good                        no for per-peak radius    Has global R50 / R80,
                                         mcolon/20260617_                                                                                                     not peak-resolved.
                                         morph_axis_inves
                                         tigation/
                                         morphseq_investi
                                         gation/core/
                                         bandwidth_tuning
                                         .py:456

  ### C. Semantic Gap Table

   proposed field                        existing source                             semantic match    missing computation                      recommended definition
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   candidate_peak_id                     PeakCandidateDetail.candidate_peak_id              partial    stable provenance policy                 Use detector-local integer ID,
                                                                                                                                                document non-global stability.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   within_peak_total_support_fraction    empirical basin_sample_fraction;                   partial    accepted-only assignment and residual    Empirical V0: accepted assigned sample
                                         truth component_mass / superlevel                             support                                  fraction. Truth V0: density mass in
                                         mass                                                                                                   same resolved grid region.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   within_peak_density                   peak_height, basin_kde_mass,                          weak    explicit definition                      Prefer omit in V0 or define as support
                                         component area                                                                                         mass / basin area.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   within_peak_radius                    global R80, recipe _mode_r80_radius,               partial    per-peak distances from assignments      Empirical: R80 of accepted assigned
                                         sample distances                                                                                       sample distances to canonical peak
                                                                                                                                                center. Truth: weighted R80 of grid-
                                                                                                                                                cell distances inside resolved truth
                                                                                                                                                region.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   within_peak_cv_radius_from_center     none directly                                   computable    assignment vector or masks               std(distance) / mean(distance), with
                                                                                                                                                edge-case NaN.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   within_peak_center_coordinate         peak_x, peak_y, peak_locations           good but specific    canonical guarantee and accepted         Use local maximum coordinate on
                                                                                                       filtering                                canonical grid for MVP.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   number_of_peaks                       accepted_peak_count / n_modes                         good    accepted detail filtering                Use accepted resolved peaks count.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   across-peak summaries                 none                                               missing    summary functions                        Compute from ResolvedPeak list.
  ────────────────────────────────────  ───────────────────────────────────────  ───────────────────  ───────────────────────────────────────  ────────────────────────────────────────
   assigned_sample_fraction              sample fractions sum                               partial    unassigned support rule                  Currently always 1.0 if any peak and
                                                                                                                                                samples exist; needs rejection/
                                                                                                                                                unassigned semantics.

  ### PeakCandidateDetail Audit

  Fields are:

  candidate_peak_id
  peak_x
  peak_y
  peak_height
  nearest_saddle_or_merge_height
  prominence_ratio
  basin_sample_count
  basin_sample_fraction
  basin_kde_mass
  superlevel_cap_mass_at_split
  accepted
  reject_reason
  component_mass
  hdr_component_count
  component_area
  component_sample_count
  component_sample_fraction
  extra

  Evidence: results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:29.

  Population by method:

   method              populated fields                                                                    optional/method-specific
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   superlevel_cap_m    ID, peak coordinate, peak height, accepted, reject reason, component_mass           no sample assignment, no KDE basin, no saddle/prominence, no component area
   ass
  ──────────────────  ──────────────────────────────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────
   hdr_component_pe    ID, peak coordinate, peak height, component_mass, optional sample counts/           sample counts use nearest peak center, not HDR mask membership
   rsistence           fractions, extra["hdr_mass_level"]
  ──────────────────  ──────────────────────────────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────
   kde_peak_basins_    ID, peak coordinate, height, saddle, prominence, basin sample count/fraction,       nearest-center basin, not watershed or grid basin
   sample_support      basin KDE mass, acceptance, reject reason, extra support proxy

  Important findings:

  - candidate_peak_id is deterministic only within one detector result and current candidate ordering. It is not stable across detector methods, bandwidths, grids, or seeds.
  - Sample assignments are not preserved. Counts and fractions are preserved, but the assignment vector is local and discarded at results/mcolon/20260617_morph_axis_investigation/
    morphseq_investigation/core/peak_counting.py:719.

  - Peak centers are canonical coordinates when a DensityGrid is passed; otherwise they are grid indices. Evidence: results/mcolon/20260617_morph_axis_investigation/
    morphseq_investigation/core/peak_counting.py:153 and results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:167.

  - Accepted and rejected candidates are preserved together in candidate_details.
  - component_area and hdr_component_count are effectively unused in current construction.
  - PeakCandidateDetail is close as detector evidence, but not close enough as the shared per-peak primitive because it mixes peak height, cap mass, HDR mass, sample support, and
    rejection evidence.

  ### PeakDetectionResult Audit

  PeakDetectionResult guarantees a detector-level aggregate shape:

  method_name
  n_modes
  peak_density
  total_mass
  split_fraction
  split_level
  n_components_at_split
  component_masses
  candidate_peak_count
  accepted_peak_count
  rejected_peak_count
  peak_locations
  peak_heights
  basin_sample_counts
  basin_sample_fractions
  basin_kde_masses
  hdr_component_counts
  reject_reasons
  notes
  candidate_details

  Evidence: results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:53.

  Method-specific fields:

  - split_fraction means superlevel density fraction for superlevel_cap_mass, HDR mass level for hdr_component_persistence, and superlevel baseline for kde_peak_basins_sample_support.
  - component_masses means superlevel component KDE mass, HDR component KDE mass, or nearest-center basin KDE mass depending on method.
  - basin_sample_fractions only has primary meaning for kde_peak_basins_sample_support; HDR also fills it using nearest center assignment to HDR component peak centers.
  - hdr_component_counts only makes sense for HDR.

  Verdict:

  - Keep PeakDetectionResult as detector output.
  - Do not extend it into ResolvedPeakDistribution.
  - Build a wrapper/builder that consumes PeakDetectionResult, original sample_points, and DensityGrid.

  ### Sample Assignment Semantics

  For kde_peak_basins_sample_support:

  - Candidate peaks are local maxima after maximum_filter and deduplication by grid-cell separation. Evidence: results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/
    core/peak_counting.py:176 and results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:197.

  - Grid cells are assigned to nearest peak coordinate by Euclidean distance. Evidence: results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/
    peak_counting.py:237.

  - Samples are assigned to nearest peak coordinate by Euclidean distance. Evidence: results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:253.
  - There is no watershed, no gradient ascent basin, no confidence threshold, no density floor, no max distance, and no unassigned code path when peaks exist.
  - Every sample is assigned to some peak if sample_points and at least one peak exist.

  Therefore assigned_sample_fraction is currently uninformative for empirical KDE basin assignment: it will be 1.0 after filtering to accepted peaks only if all candidates are
  accepted; otherwise it can become the sum of accepted candidate fractions, but rejected samples were still assigned to rejected candidates, not truly unassigned.

  For assigned_sample_fraction to be informative, V0 needs an explicit rule such as accepted-candidate-only assignment, maximum assignment distance, density floor, watershed/basin
  membership, or assignment confidence. Existing code most strongly implies “accepted-candidate-only fraction” as the minimal V0 rule, because acceptance already exists.

  ### Meaning Of Support

   support notion                        where computed                             region                                empirical/density         usable as canonical support?
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   superlevel component KDE mass         _component_masses via _superlevel_split    cells above split level, connected    density-based             truth comparable, empirical
                                                                                    components                                                      detector-specific
  ────────────────────────────────────  ─────────────────────────────────────────  ────────────────────────────────────  ────────────────────────  ────────────────────────────────────
   HDR component mass                    _hdr_level_components                      connected HDR cells at selected       density-based             detector diagnostic, not canonical
                                                                                    mass level
  ────────────────────────────────────  ─────────────────────────────────────────  ────────────────────────────────────  ────────────────────────  ────────────────────────────────────
   KDE basin mass                        _detect_kde_peak_basins_sample_support     nearest-center Voronoi cells over     density-based             possible, but not sample preferred
                                                                                    all grid
  ────────────────────────────────────  ─────────────────────────────────────────  ────────────────────────────────────  ────────────────────────  ────────────────────────────────────
   sample count assigned to candidate    _assign_points_to_peaks                    nearest peak center                   empirical sample-based    best empirical V0 primitive
  ────────────────────────────────────  ─────────────────────────────────────────  ────────────────────────────────────  ────────────────────────  ────────────────────────────────────
   sample fraction assigned to           same                                       nearest peak center                   empirical sample-based    best empirical V0 support if
   candidate                                                                                                                                        accepted-only
  ────────────────────────────────────  ─────────────────────────────────────────  ────────────────────────────────────  ────────────────────────  ────────────────────────────────────
   truth component mass fraction         DensityComponentSpec.mass_fraction         recipe component                      truth recipe              not resolved peak support
  ────────────────────────────────────  ─────────────────────────────────────────  ────────────────────────────────────  ────────────────────────  ────────────────────────────────────
   resolved truth basin mass             not exposed                                should be resolved grid region        truth density-based       missing required primitive

  Recommendation:

  - Empirical V0 within_peak_total_support_fraction: sample fraction assigned to accepted resolved peaks, using the same assignment rule as the resolver.
  - Truth V0 analogue: density mass fraction integrated over resolved truth regions on the canonical grid.
  - Do not use recipe mass_fraction as resolved peak support; bridges and merged density make it semantically wrong.

  ### Meaning Of Density

  Candidate definitions:

   definition                                meaning                                  failure mode
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   peak height                               local max density                        duplicates “sharpness”, ignores support width
  ────────────────────────────────────────  ───────────────────────────────────────  ───────────────────────────────────────────────────────
   mean KDE density over assigned samples    average density experienced by points    bandwidth-sensitive, sample-count biased
  ────────────────────────────────────────  ───────────────────────────────────────  ───────────────────────────────────────────────────────
   mean KDE density over peak basin          basin mass / basin area                  depends on basin definition, but interpretable
  ────────────────────────────────────────  ───────────────────────────────────────  ───────────────────────────────────────────────────────
   sample count / basin area                 empirical occupancy density              duplicates support if area fixed poorly
  ────────────────────────────────────────  ───────────────────────────────────────  ───────────────────────────────────────────────────────
   KDE basin mass / basin area               density concentration in region          comparable for truth/empirical if same grid semantics

  Recommendation:

  - within_peak_density is too ambiguous for V0 unless defined as within_peak_total_support_fraction / within_peak_area.
  - If included, use density_mass_fraction / basin_area for both truth and empirical grid regions.
  - If the first implementation is sample-based only, omit within_peak_density from V0 rather than silently using peak height.

  ### Radius And CV

  Existing scale notions:

   scale                              applies to                         source
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   local_width                        recipe component                   results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:74
  ─────────────────────────────────  ─────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   approximate _mode_r80_radius       bridge endpoint recipe geometry    results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/density_composition.py:587
  ─────────────────────────────────  ─────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   global R50 / R80                   whole empirical cloud              results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/bandwidth_tuning.py:494
  ─────────────────────────────────  ─────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   component area                     HDR internal result                results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/core/peak_counting.py:509
  ─────────────────────────────────  ─────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   distance-to-center distribution    not exposed                        computable from assignments

  Recommendation:

  - Empirical within_peak_radius: R80 of assigned sample distances from canonical peak center.
  - Truth within_peak_radius: weighted R80 of grid-cell distances from canonical peak center, using density weights inside resolved truth region.
  - Do not use recipe local_width or _mode_r80_radius; those are construction hints, not resolved measurements.

  For within_peak_cv_radius_from_center:

  stdRecommended edge cases:

  - 0 assigned points:NaN- use `ddof=0 for radius.

   | center meaning |
  |---|--- |
  | `ComponentTruth.component_mass_centroid_*` | density-weighted recipe component centroid |
  .anchor` | recipe anchor |
  | proposed sample centroid | not implemented |
   basin centroid | notMVP Canonical center: local maximum coordinate on the canonical density grid.
  - Use it for radial distances, across-peak distances, and initial truth/empirical matching.
  - Preserve detector/source provenance separately if later needing component anchors or centroids.
  - IDs are detector-local only. Use `candidate_peak_id` as a local result key, not a stable biological identifier.

  ### Truth And Empirical Compatibility

  | field | truth source | empirical source | directly comparable? | caveat |
  |---|---|---:|---|---|
  | `center | local max of resolved region | local max of KDE candidate | mostly | KDE smoothing can shift empirical center |
  | `total_support_fraction` | density mass in truth resolved |
  | `density / resolved region area | mass / resolved region area | yes if grid-region based | not-only or peak height |
  | `radius` weighted80 grid | sample R80 distance | approximate grid |
  | `cv_radius_from_center` | sample distance CV | approximate | counts |

  Overall shared geometry is possible the builder/assignment semantics.

  ### Across-Peakaries

  Recommended conventions:

  | summary | 0 peaks | 1 peak | 2 peaks | 3+ peaks |
  ||---|---|---|---|
  | support mean | `NaN` | value | mean | mean |
  | support skew | `NaN` | `NaN` | `NaN` | skew |
  | density meanNa value | mean mean |
  | density skew | `NaN` | `NaN` | skew |
  | radius mean | `` | value | mean | mean |
  | radius skew | `Na` | `NaN` | skew |
  | CV radius mean | `NaN` | value ifpeak distance mean | `NaN` | `NaN pair distance | mean pairwise distance |

  Use `NaN`, not zero, because zero implies a real measured value.

  `across_peak_total_support_fraction_mean` isatically constrained by:

  sum(accepted support fractions) / number_of_peaks

  If assigned_sample_fraction is the sum of accepted support fractions, then:

  across_peak_total_support_fraction_mean = assigned_sample_fraction number_of_peaks
  `

  So it is redundant but useful table convenience.

  ### D. Class Design Verdict

  Smallest useful object model:

  python
  from dataclasses import dataclass
  import np

  @dataclass(f:
      candidate_peak: int
  _coordinate: tuple]
      total_support_fraction: float
      radius:    cv_radius_from    density: float None

  @dataclass(f=True)
  class ResolvedPeak:
  : PeakGeometry
      source    accepted Truerozen=True)
  olved:
  Peakproperty
      def numberGeometryulated;empirical", provenance=detector detail)` |
  |` | unnecessary wrapperPeak |
  | across- | derived### Eator API over a stateful.

  ```python
  def resolve_empirical_peak_distribution(
  : np,
      detection: PeakDetection method:kde_peak_basins_sample_support",
      accepted_onlyolvedPeakDistribution:
   ...

  _truth truth: ComposedDensityTruth    | None = None,
      = -> ResolvedPeakDistribution:
      ...
   yetwidth selection already exists as grid building already exists as pure functions Detectors already pure `detect_peaks(...)`.
   a resolver/b, not a state.

   Migration Map

  | existing output object mapping | |
  |---|---|---|
  |.nResolvedPeakDistribution.number_of_peaks` none if detector unchanged `ResolvedPeak.provenance["candidate_detail` | empirical `total_support_fraction` | low, but accepted-only
  behavior must be explicit |
  | `PeakCandidateDetail.peak_x/y` |_coordinate if grid requiredPeak.notes/reject_re` | unchanged legacy detector | none |
  | `count_mass_significant legacy none |
  | plotting rows | still consumeDetection` | none |

  Current legacy:

  - [modal](/net/trap/colon/proorphseq/results/mcolon/20260617_morph_axis_investigationorphseq_investigation/plot.py):355 formats `n_fraction`.
  - [modal_distribution_plotting.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/202.py):551 computes legacy truth/observ_distribution_plotting.py](/net/trapnell/
  home/md/morphseq/resultsorph_axis_investigation_investigation/ `split(/net/trap/mdj/m/20260617_morph_axis_investigation/mseq/bandwidth_tuning.py uses `peak_count tree.

  ### definition: empirical sample fraction requires unless defined as mass / not be silently used.
  - Radius: R80 from is computable empir truth requires density semantics: nearest-center assigns every sample when peaks exist.
   Truth-vs-: comparable only approximately unless both the same-` for fewer than 3 peaks.
  - Candidate ID stability: local only, not stable across method seed notcore/resolved_peak_distribution.py`.
  2. Define `PeakResolvedPeakPeak-center sample assignment and distance summaries.
  4. Implementirical(...)` wrapping `detect_pe(...)`.
  5. For empirical acceptedassigned_sample(accepted assigned sample.
  7. Implement truth `superlevel_cap_mass or an exposed split helper.
  8. Compute truth support from grid density mass radius distances.
  N` conventions.
  10. `PeakDetail`, and `count_mass` unchanged table/export adapter object is stable.

  Final judgments| question |---|---|
  |antically sufficient | no |
  | computationally sufficient | mostly, |
  | |

   are good detector and density primitives. They are not yet a resolved- distribution API packaging plus semantics, with primitive gap: preserved per-s/per resolved membership.