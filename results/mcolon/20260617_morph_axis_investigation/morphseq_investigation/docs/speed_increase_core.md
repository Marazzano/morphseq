

  Ranked Speed Wins

   Rank                                                     Change                  Impact        Ease    Notes
  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1                  Add a vote-only resolver for bootstrap draws                    High    Easy-Med    In _resolve_draw_for_vote, avoid full ResolvedPeakDistribution construction. The vote only needs
                                                                                                          (accepted_count, accepted_centers), so call density eval + detect_peaks() and extract accepted
                                                                                                          candidates. Skips sample reassignment, R80 geometry, dataclass validation on every draw.
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
   2             Use lower-resolution grid for bootstrap vote only               Very High      Medium    KDE eval is roughly grid_cells * n_points. A vote grid of 41 or 51 cells could be much cheaper than
                                                                                                          the final grid while preserving the final honest full-data resolve. Needs validation against count
                                                                                                          stability.
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
   3                                   Parallelize bootstrap draws    Very High wall-clock      Medium    Draws are independent. Keep deterministic chunk seeds. CPU total is unchanged, but wall time should
                                                                                                          scale well across cores. Best if this is run per distribution and not already parallelized outside.
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
   5       Cache reusable grid coordinates / flattened grid points                  Medium        Easy    canonical_grid.xx/yy are cached, but flattened grid_points and maybe DensityGrid scaffolding are
                                                                                                          rebuilt. Helpful especially for geometry-bandwidth paths using precompute_squared_distances().
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
   6         Special-case common stable one-peak / zero-peak cases                  Medium      Medium    If full-data detection has one strong accepted peak and high support, bootstrap may be overkill.
                                                                                                          This is a policy change, so I’d only do it with explicit tolerance for changed reliability
                                                                                                          semantics.
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
   7                              Optimize consensus seed matching                     Low        Easy    build_consensus_seed_set() copies distance matrices inside a tiny greedy loop. It is not likely the
                                                                                                          bottleneck unless peak counts get large.
  ──────  ─────────────────────────────────────────────────────────  ──────────────────────  ──────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────
   8                              Dataclass/readonly-array cleanup                     Low        Easy    Some overhead exists, but it is dwarfed by repeated KDE + detector work. Worth doing only after
                                                                                                          profiling confirms it matters.

  #: 1
  Optimization: Drop the redundant _graph_connectivity_radius pdist call inside bandwidth_geometry_scales — its output (connectivity_90_radius, etc.) is never read by the hot path, only
  median_kNN_distance/longest_non_outlier_MST_edge are
  Impact: ~15-20% off every single resolve (~11,000 of them in one full run)                  
  Effort: One-line: add a compute_connectivity=False flag
  Risk: Zero — no behavior change, dead computation
  ────────────────────────────────────────
  #: 2
  Optimization: Parallelize the draw loops with n_jobs — currently 100% sequential; resample.run() already supports joblib.Parallel, nobody passes n_jobs
  Impact: Roughly N-fold on an N-core machine, multiplicative with everything else, since draws are 100% independent
  Effort: Small: thread one parameter through _resample_adapters.py → call sites              
  Risk: Low — no math changes, just needs a quick check that thread vs process backend doesn't eat the gains at this draw size
  ────────────────────────────────────────
  #: 3  
  Optimization: Cache/reuse bandwidth-geometry across the 80 bootstrap draws instead of recomputing the O(n²) pdist+MST on every subsample                            
  Impact: The single biggest line item: ~450ms × 80 draws × 2 records × 20 comparisons ≈ tens of minutes
  Effort: Moderate: precompute once, thread a bandwidth override through resolve_points_with_analysis_spec
  Risk: Needs your sign-off — changes numeric behavior slightly (shared vs. per-draw bandwidth)
  ────────────────────────────────────────
  #: 4
  Optimization: Reduce sweep_steps (currently 50) for vote/null draws, which only need the peak count, not a precise split level
  Impact: ~10% of per-draw cost × ~11,000 resolves
  Effort: One-line config change, already a parameter
  Risk: Low — full-data resolve (used for rendering) stays at full precision
  ────────────────────────────────────────
  #: 5
  Optimization: Share the one "observed" full-data resolve between the bootstrap-vote path and the null-permutation path instead of both recomputing it independently
  Impact: Smallest win
  Effort: Most structural — needs a resolve cache threaded through render_gene
  Risk: Low risk but most rewor