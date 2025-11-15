# DLMA Integration for Zebrafish Morphometric Analysis

This module integrates the Deep Learning-Enabled Morphometric Analysis (DLMA) framework into morphseq for automated zebrafish organ and phenotype quantification.

## Quick Start

### 1. Install DLMA Dependencies

See [`DLMA_INSTALLATION.md`](./DLMA_INSTALLATION.md) for detailed installation instructions.

Quick install:
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Install supporting libraries
pip install opencv-python scikit-image pandas numpy scipy
```

### 2. Download Model Weights

Download pre-trained DLMA weights from the repository:
- Repository: https://github.com/gonggqing/DLMA
- Model weights link in README (Google Drive)

### 3. Run Inference

```bash
python dlma_inference_example.py \
    --image_dir /path/to/zebrafish/images \
    --output_dir /path/to/output \
    --model_weights /path/to/dlma/model_final.pth
```

### 4. Use in Your Pipeline

```python
from pathlib import Path
import pandas as pd
from src.data_pipeline.feature_extraction.zebrafish_morphometric_analysis import (
    extract_morphometric_features_batch,
    summarize_morphometric_features,
)

# Your tracking DataFrame
tracking_df = pd.read_csv("tracking.csv")

# Extract features
features = extract_morphometric_features_batch(
    tracking_df=tracking_df,
    dlma_predictions_dir=Path("dlma_predictions"),
    embryo_mask_dir=Path("masks"),  # Optional
)

# Save
features.to_csv("morphometric_features.csv", index=False)
```

## What This Module Provides

### Detected Classes (16 total)

**Organs (8):**
1. Eye
2. Heart
3. Yolk
4. Swim bladder
5. Otolith
6. Gut
7. Trunk
8. Tail

**Phenotypes (8):**
1. Pericardial edema
2. Yolk sac edema
3. Spinal curvature
4. Tail malformation
5. Craniofacial malformation
6. Reduced pigmentation
7. General edema
8. Hemorrhage

### Output Format

For each snip_id, the module computes:

```python
{
    'snip_id': 'zebrafish_001_t0',
    'eye_area_pct': 2.5,              # Eye is 2.5% of total embryo
    'heart_area_pct': 1.2,            # Heart is 1.2% of total embryo
    'yolk_area_pct': 15.3,            # Yolk is 15.3% of total embryo
    # ... (for all 16 classes)
    'total_embryo_area_px': 50000,    # Total area in pixels
    'total_embryo_area_um2': 125000,  # Total area in μm² (if pixel size provided)
}
```

## Module Components

### Core Functions

#### `compute_mask_area_percentages()`
Computes area percentage for each detected class.

```python
from zebrafish_morphometric_analysis import compute_mask_area_percentages

# instance_masks: (N, H, W) array of binary masks
# class_ids: (N,) array of class IDs
percentages = compute_mask_area_percentages(instance_masks, class_ids)
```

#### `extract_morphometric_features_batch()`
Batch processing function following morphseq pattern.

```python
features_df = extract_morphometric_features_batch(
    tracking_df=tracking_df,
    dlma_predictions_dir=Path("predictions"),
    embryo_mask_dir=Path("masks"),      # Optional
    prediction_format='detectron2',      # or 'numpy'
)
```

#### `summarize_morphometric_features()`
Aggregate features across groups.

```python
summary = summarize_morphometric_features(
    features_df,
    group_by=['treatment', 'timepoint']  # Optional grouping
)
```

### Helper Functions

```python
from zebrafish_morphometric_analysis import (
    get_organ_names,      # Returns list of organ classes
    get_phenotype_names,  # Returns list of phenotype classes
    get_all_class_names,  # Returns all class names
)
```

## Example Workflows

### Workflow 1: Basic Analysis

```python
import pandas as pd
from pathlib import Path
from zebrafish_morphometric_analysis import extract_morphometric_features_batch

# Load tracking data
tracking = pd.read_csv("tracking.csv")

# Extract morphometric features
features = extract_morphometric_features_batch(
    tracking_df=tracking,
    dlma_predictions_dir=Path("dlma_predictions"),
)

# Analyze results
print(features[['snip_id', 'eye_area_pct', 'heart_area_pct']].head())
```

### Workflow 2: Treatment Comparison

```python
from zebrafish_morphometric_analysis import (
    extract_morphometric_features_batch,
    summarize_morphometric_features,
)

# Extract features
features = extract_morphometric_features_batch(
    tracking_df=tracking,
    dlma_predictions_dir=Path("dlma_predictions"),
)

# Merge with treatment metadata
features = features.merge(metadata[['snip_id', 'treatment', 'dose']], on='snip_id')

# Summarize by treatment
summary = summarize_morphometric_features(
    features,
    group_by=['treatment', 'dose']
)

# Compare heart area across treatments
print(summary[['treatment', 'dose', 'heart_mean_pct', 'heart_std_pct']])
```

### Workflow 3: Phenotype Detection

```python
from zebrafish_morphometric_analysis import get_phenotype_names

# Get phenotype columns
phenotype_cols = [f"{name}_area_pct" for name in get_phenotype_names()]

# Find embryos with phenotypes
for phenotype_col in phenotype_cols:
    detected = features[features[phenotype_col] > 0]
    print(f"{phenotype_col}: {len(detected)} / {len(features)} embryos")

# Flag embryos with any abnormality
features['has_phenotype'] = features[phenotype_cols].gt(0).any(axis=1)
print(f"Embryos with abnormalities: {features['has_phenotype'].sum()}")
```

### Workflow 4: Temporal Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Merge with temporal data
features = features.merge(tracking[['snip_id', 'time_hpf']], on='snip_id')

# Plot organ development over time
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, organ in zip(axes.flat, ['eye', 'heart', 'yolk', 'swim_bladder']):
    col = f"{organ}_area_pct"

    # Plot with moving average
    sns.scatterplot(data=features, x='time_hpf', y=col, alpha=0.3, ax=ax)

    # Add trend line
    features_sorted = features.sort_values('time_hpf')
    ax.plot(
        features_sorted['time_hpf'],
        features_sorted[col].rolling(20, center=True).mean(),
        color='red',
        linewidth=2,
    )

    ax.set_title(f"{organ.title()} Development")
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Area (%)")

plt.tight_layout()
plt.savefig("organ_development_timeline.png", dpi=300)
```

## Integration with Existing morphseq Features

### Combining with Other Features

```python
from src.data_pipeline.feature_extraction.fraction_alive import extract_fraction_alive_batch
from src.data_pipeline.feature_extraction.mask_geometry_metrics import extract_geometry_metrics_batch
from zebrafish_morphometric_analysis import extract_morphometric_features_batch

# Extract all features
fraction_alive = extract_fraction_alive_batch(tracking, mask_dir, via_mask_dir)
geometry = extract_geometry_metrics_batch(tracking, mask_dir)
morphometric = extract_morphometric_features_batch(tracking, dlma_predictions_dir)

# Merge all features
all_features = tracking[['snip_id']].copy()
all_features = all_features.merge(fraction_alive, on='snip_id', how='left')
all_features = all_features.merge(geometry, on='snip_id', how='left')
all_features = all_features.merge(morphometric, on='snip_id', how='left')

# Save consolidated features
all_features.to_csv("consolidated_features.csv", index=False)
```

### Using in Quality Control

```python
# Flag embryos with abnormal organ sizes
features['abnormal_yolk'] = features['yolk_area_pct'] > 20  # Threshold
features['abnormal_heart'] = features['heart_area_pct'] < 0.5

# Combine with existing QC
from src.data_pipeline.quality_control.death_detection import compute_dead_flag2_persistence

features_with_qc = compute_dead_flag2_persistence(features)
features_with_qc['pass_qc'] = ~(
    features_with_qc['dead_flag2'] |
    features_with_qc['abnormal_yolk'] |
    features_with_qc['abnormal_heart']
)
```

## File Locations

```
src/data_pipeline/feature_extraction/
├── zebrafish_morphometric_analysis.py    # Main module
├── dlma_inference_example.py             # Inference script
├── DLMA_INSTALLATION.md                  # Installation guide
└── README_DLMA_INTEGRATION.md            # This file
```

## Performance Notes

- **Inference speed**: ~0.05-0.1s per image (512×512) on RTX 3090
- **Memory**: ~2-4GB GPU memory for batch inference
- **CPU mode**: Available but 10-20× slower

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'detectron2'`
- **Solution**: Install Detectron2 following DLMA_INSTALLATION.md

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use CPU mode

**Issue**: No detections / all NaN
- **Solution**: Check model weights path and score threshold

**Issue**: Class IDs don't match
- **Solution**: Verify model was trained with same class definitions

### Validation

Test the installation:

```python
# Test imports
from zebrafish_morphometric_analysis import (
    compute_mask_area_percentages,
    get_all_class_names,
)

# Test basic functionality
import numpy as np
masks = np.random.rand(3, 100, 100) > 0.7
classes = np.array([1, 2, 3])  # eye, heart, yolk
result = compute_mask_area_percentages(masks, classes)
print(result)
```

## References

1. **Paper**: Dong et al., "Deep Learning-Enabled Morphometric Analysis for Toxicity Screening Using Zebrafish Larvae", *Computers in Biology and Medicine*, 2024
2. **DLMA GitHub**: https://github.com/gonggqing/DLMA
3. **Detectron2**: https://github.com/facebookresearch/detectron2

## Support

For questions or issues:
- DLMA model: https://github.com/gonggqing/DLMA/issues
- morphseq integration: Contact morphseq team

---

**Version**: 1.0
**Last Updated**: 2025-11-15
**Author**: morphseq team
