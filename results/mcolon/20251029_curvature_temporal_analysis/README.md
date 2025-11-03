# Curvature Temporal Analysis - 20201017 Dataset

**Date**: October 29, 2025
**Dataset**: 20201017_combined (93 embryos, 3,373 timepoints)

## Objective

Analyze curvature trends over developmental time for individual embryos, comparing three genotypes:
- `cep290_homozygous` (n=1,763 timepoints)
- `cep290_heterozygous` (n=501 timepoints)
- `cep290_wildtype` (n=502 timepoints)

## Key Questions

1. **Temporal Trajectories**: How does curvature change over development for each genotype?
2. **Temporal Correlation**: How does curvature at one timepoint correlate with other timepoints (horizon plots)?
3. **Morphology Space**: How well does distance in embedding space reflect curvature differences?
4. **Predictive Models**: Can embeddings predict curvature (and vice versa)?

## Curvature Metrics

**Primary metrics** (focusing on these two):
- `arc_length_ratio`: Normalized curvature (arc length / chord length), always ≥1.0
- `normalized_baseline_deviation`: Baseline deviation normalized by embryo length (to compare across embryos)

**Data source**:
- `morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_20251017_combined.csv`

**Embedding source**:
- `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv`

## Analysis Scripts

1. `load_data.py` - Load and merge curvature + embedding data, normalize metrics
2. `01_individual_trajectories.py` - Plot curvature over time for each embryo and aggregate by genotype
3. `02_horizon_plots.py` - Create timepoint × timepoint correlation heatmaps
4. `03_embedding_distance.py` - Analyze relationship between embedding distance and curvature differences
5. `04_predictive_models.py` - Build ML models: embeddings ↔ curvature

## Outputs

- `outputs/figures/` - All generated plots
- `outputs/tables/` - Statistical summaries and correlation matrices
- `outputs/models/` - Trained predictive models

## Notes

- Using `predicted_stage_hpf` for developmental time alignment (not absolute frame numbers)
- Baseline deviation normalized by `total_length_um` to make comparable across embryos
- Statistical comparisons use leave-one-embryo-out cross-validation to avoid data leakage
