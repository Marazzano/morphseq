# 20250912 analysis_ready QC gallery observations

Context: manual review of `data_pipeline_output/analysis_ready/20250912/analysis_ready/report/`
post-QC value-quartile galleries.

## Suspect snips

| snip_id | Gallery | Concern | Initial investigation target |
| --- | --- | --- | --- |
| `20250912_E07_e01_BF_t0003` | `20250912_post_qc_area_um2_gallery.png` | Looks like it is starting to die, but passed post-QC. | Check death persistence and lead-time broadcast. |
| `20250912_C12_e01_BF_t0082` | `20250912_post_qc_baseline_deviation_normalized_gallery.png` | Looks like it is moving; should it fail motion QC? | Check motion QC definition and `motion_blur_flag` value. |
| `20250912_D12_e01_BF_t0062` | `20250912_post_qc_baseline_deviation_normalized_gallery.png` | Looks like it is moving; should it fail motion QC? | Check motion QC definition and `motion_blur_flag` value. |
| `20250912_A08_e01_BF_t0080` | `20250912_post_qc_area_um2_gallery.png` | Looks like it should not pass motion QC. | Check motion QC definition and `motion_blur_flag` value. |
| `20250912_D08_e01_BF_t0098` | `20250912_post_qc_area_um2_gallery.png` | Tail is out of focus. | Check focus QC metric/flag and whether it sees local tail blur. |

## Early code-level notes

- Death detection defaults in `src/data_pipeline/quality_control/death_detection/config.py` currently use `lead_time_hr = 4.0`, `persistence_threshold = 0.80`, `dead_fraction_threshold = 0.90`, `decline_rate_threshold = 0.05`, `min_timepoints = 3`, and `smoothing_window = 5`.
- Death persistence is detected per `physical_embryo_id`; `broadcast_persistence_dead_flag()` flags rows with `time_index >= called_death_time_index`.
- Current `motion_blur_qc` is adjacent-z-plane NCC inside the mask for each snip, not temporal movement/displacement between successive timepoints. As of 2026-07-06, the pipeline default fails any nonzero bad adjacent-z-pair fraction (`bad_pair_frac_threshold = 0.0`).
- Need to confirm from the 20250912 tables whether each suspect snip was present in `analysis_ready` with `pass_qc == True`, and what raw `death_detection_qc`, `motion_blur_qc`, and `focus_qc` values were.

## Row-level check, 2026-07-06

All five suspect snips are present in `analysis_ready` with `use_snip = True` and empty
`qc_fail_reasons`.

| snip_id | use_snip | fraction_alive | motion_blur_flag | bad_z_pair_frac | focus_flag | interior_strong_edge_fraction | death flags |
| --- | --- | ---: | --- | ---: | --- | ---: | --- |
| `20250912_E07_e01_BF_t0003` | `True` | 0.993842 | `False` | 0.000000 | `False` | 0.626267 | viability `False`, persistence `False` |
| `20250912_C12_e01_BF_t0082` | `True` | 0.900349 | `False` | 0.071429 | `False` | 0.705450 | viability `False`, persistence `False` |
| `20250912_D12_e01_BF_t0062` | `True` | 1.000000 | `False` | 0.071429 | `False` | 0.705121 | viability `False`, persistence `False` |
| `20250912_A08_e01_BF_t0080` | `True` | 1.000000 | `False` | 0.071429 | `False` | 0.839162 | viability `False`, persistence `False` |
| `20250912_D08_e01_BF_t0098` | `True` | 0.951042 | `False` | 0.000000 | `False` | 0.554771 | viability `False`, persistence `False` |

Interpretation:

- `E07_e01` does get a death event later: `death_event_time_index = 26` and
  `death_event_stage_hpf = 34.078684`. The suspicious `t0003` frame is before the current
  lead-time-adjusted death call, so persistence death is active but does not reach that early.
- `C12_e01_BF_t0082`, `D12_e01_BF_t0062`, and `A08_e01_BF_t0080` each have exactly 1 bad adjacent
  z-plane pair out of 14 valid pairs (`bad_z_pair_frac = 0.071429`). They passed under the old
  `bad_pair_frac_threshold = 0.10` default, but should fail after the 2026-07-06 change to
  `bad_pair_frac_threshold = 0.0`. Separately, the joined
  `analysis_ready` table shows large temporal displacement/speed for the moving-looking examples:
  `C12_t0082` has `displacement_um = 1554.733552`, `speed_um_per_s = 0.528863`;
  `D12_t0062` has `displacement_um = 421.571932`, `speed_um_per_s = 0.142984`;
  `A08_t0080` has `displacement_um = 261.425774`, `speed_um_per_s = 0.088834`.
  That suggests the manual concern is temporal movement, while `motion_blur_qc` is currently a
  z-stack correlation QC.
- `D08_e01_BF_t0098` passes focus because `interior_strong_edge_fraction = 0.554771`, above the
  current fail threshold of `< 0.50`. This metric erodes the mask and measures whole-interior
  structure; a tail-specific focus problem can pass if the body interior still has enough edges.
