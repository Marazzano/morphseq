# DLMA Deployment Workflow: Decoupled Inference and Analysis

## Problem

Running DLMA inference requires Detectron2 and PyTorch (large dependencies with CUDA requirements). Feature extraction only needs NumPy and scikit-image. Mixing these creates deployment issues:

- **Statistics node**: Can't load predictions without installing full ML stack
- **CPU-only machines**: Can't deserialize Detectron2 pickles
- **File sizes**: Pickle files are 5-10× larger than needed

## Solution: Two-Stage Workflow

**Stage 1: Inference** (GPU machine with Detectron2)
```bash
# Run DLMA model
python dlma_inference_example.py \
    --image_dir data/images \
    --output_dir predictions/ \
    --model_weights model_final.pth
```
Output: `predictions/*.pkl` (Detectron2 format)

**Stage 2: Convert** (Same machine or any machine with PyTorch)
```bash
# Convert to lightweight NumPy format
python convert_dlma_to_npz.py predictions/
```
Output: `predictions/*.npz` (NumPy format, ~10× smaller)

**Stage 3: Analysis** (Any machine, no Detectron2 needed)
```bash
# Extract features - only needs NumPy/pandas
python -c "
from pathlib import Path
import pandas as pd
from zebrafish_morphometric_analysis import extract_morphometric_features_batch

tracking_df = pd.read_csv('tracking.csv')
features = extract_morphometric_features_batch(
    tracking_df=tracking_df,
    dlma_predictions_dir=Path('predictions'),
    prediction_format='numpy',  # Uses .npz files
)
features.to_csv('morphometric_features.csv', index=False)
"
```

---

## Detailed Workflow

### 1. Inference (GPU Node)

**Requirements:**
- Detectron2 + PyTorch + CUDA
- DLMA model weights
- Input images

**Script:**
```python
from dlma_inference_example import run_dlma_batch_inference
from pathlib import Path

# Run inference
results = run_dlma_batch_inference(
    image_dir=Path("images/"),
    output_dir=Path("predictions/"),
    model_weights_path="models/model_final.pth",
    device='cuda',
)

# Output: predictions/zebrafish_001_dlma.pkl, zebrafish_002_dlma.pkl, ...
```

**Files created:**
```
predictions/
├── zebrafish_001_dlma.pkl  (2.5 MB)
├── zebrafish_002_dlma.pkl  (2.3 MB)
└── zebrafish_003_dlma.pkl  (2.4 MB)
```

---

### 2. Conversion (GPU Node or Transfer Node)

**Requirements:**
- PyTorch (Detectron2 optional)
- Input: `.pkl` files from Stage 1

**Command:**
```bash
# Convert entire directory
python convert_dlma_to_npz.py predictions/

# Or convert to separate directory
python convert_dlma_to_npz.py predictions/ --output-dir predictions_npz/
```

**Output:**
```
Converting predictions/zebrafish_001_dlma.pkl...
  Detections: 5
  Image size: 512×512
  File size: 2.50 MB → 0.25 MB (90.0% reduction)

Converting predictions/zebrafish_002_dlma.pkl...
  Detections: 6
  Image size: 512×512
  File size: 2.30 MB → 0.23 MB (90.0% reduction)

Successfully converted 3/3 files
```

**Files created:**
```
predictions/
├── zebrafish_001_dlma.pkl  (2.5 MB) ← Can delete after conversion
├── zebrafish_001_dlma.npz  (0.25 MB) ✓
├── zebrafish_002_dlma.pkl  (2.3 MB)
├── zebrafish_002_dlma.npz  (0.23 MB) ✓
└── ...
```

**What's in the .npz file:**
```python
import numpy as np
data = np.load('predictions/zebrafish_001_dlma.npz')

print(data.files)
# ['masks', 'classes', 'scores', 'boxes', 'image_height', 'image_width']

print(data['masks'].shape)   # (5, 512, 512) - binary masks
print(data['classes'])        # [1, 2, 3, 4, 7] - class IDs
print(data['scores'])         # [0.985, 0.923, ...] - confidences
```

---

### 3. Feature Extraction (Analysis Node)

**Requirements:**
- NumPy, pandas, scikit-image
- NO Detectron2/PyTorch needed!
- Input: `.npz` files from Stage 2

**Python:**
```python
from pathlib import Path
import pandas as pd
from zebrafish_morphometric_analysis import extract_morphometric_features_batch

# Load tracking data
tracking_df = pd.DataFrame({
    'snip_id': ['zebrafish_001', 'zebrafish_002', 'zebrafish_003'],
    'image_id': ['zebrafish_001', 'zebrafish_002', 'zebrafish_003'],
    'um_per_pixel': [2.5, 2.5, 2.5],  # Microscope calibration
})

# Extract features - uses .npz files, no Detectron2 needed
features = extract_morphometric_features_batch(
    tracking_df=tracking_df,
    dlma_predictions_dir=Path('predictions'),
    prediction_format='numpy',  # Default, uses .npz
)

# Save results
features.to_csv('morphometric_features.csv', index=False)
```

**Output:**
```csv
snip_id,eye_area_pct,heart_area_pct,yolk_area_pct,...,total_embryo_area_px,total_embryo_area_um2
zebrafish_001,2.50,1.78,11.20,...,50000,312500.0
zebrafish_002,2.60,1.85,11.50,...,51000,318750.0
zebrafish_003,2.45,1.72,11.00,...,49500,309375.0
```

---

## Deployment Patterns

### Pattern 1: Single Machine (Development)

```bash
# All stages on one GPU workstation
python dlma_inference_example.py --image_dir images/ --output_dir predictions/
python convert_dlma_to_npz.py predictions/
python extract_features.py --tracking tracking.csv --predictions predictions/
```

### Pattern 2: GPU + CPU Cluster

```bash
# GPU node: Inference only
sbatch --gres=gpu:1 run_inference.sh

# CPU node: Convert and analyze
sbatch convert_and_analyze.sh
```

**run_inference.sh:**
```bash
#!/bin/bash
#SBATCH --gres=gpu:1

module load cuda/11.8
source venv_ml/bin/activate

python dlma_inference_example.py \
    --image_dir /data/images \
    --output_dir /data/predictions
```

**convert_and_analyze.sh:**
```bash
#!/bin/bash

source venv_analysis/bin/activate  # Lightweight env, no CUDA

python convert_dlma_to_npz.py /data/predictions

python extract_features.py \
    --tracking /data/tracking.csv \
    --predictions /data/predictions \
    --output /data/features.csv
```

### Pattern 3: Cloud Pipeline

```bash
# Step 1: Inference on GPU instance (AWS p3.2xlarge)
docker run --gpus all dlma_inference \
    python run_inference.py

# Step 2: Convert on same instance (still has PyTorch)
docker run dlma_inference \
    python convert_dlma_to_npz.py /predictions

# Step 3: Transfer .npz files to storage
aws s3 sync /predictions s3://bucket/predictions/ --exclude "*.pkl"

# Step 4: Analysis on CPU instance (AWS t3.large)
docker run dlma_analysis \  # Lightweight image, no CUDA
    python extract_features.py
```

---

## File Format Comparison

| Format | Size | Load Time | Dependencies | Use Case |
|--------|------|-----------|--------------|----------|
| `.pkl` (Detectron2) | 2.5 MB | 0.5s | Detectron2 + PyTorch | Immediate post-inference |
| `.npz` (NumPy) | 0.25 MB | 0.05s | NumPy only | **Production/deployment** |
| Ratio | **10× larger** | **10× slower** | Heavy | Lightweight |

---

## Best Practices

### ✅ DO

1. **Convert immediately after inference**
   ```bash
   python dlma_inference_example.py ... && \
   python convert_dlma_to_npz.py predictions/
   ```

2. **Delete .pkl files after conversion** (save space)
   ```bash
   python convert_dlma_to_npz.py predictions/
   rm predictions/*.pkl  # Keep only .npz
   ```

3. **Use `um_per_pixel` from microscope calibration**
   ```python
   tracking_df['um_per_pixel'] = 2.5  # From microscope metadata
   ```

4. **Default to `prediction_format='numpy'`**
   ```python
   extract_morphometric_features_batch(
       ...,
       prediction_format='numpy',  # Default
   )
   ```

### ❌ DON'T

1. **Don't install Detectron2 on analysis nodes**
   - Use `.npz` format instead

2. **Don't use `.pkl` format in production**
   - 10× larger files
   - Requires full ML stack

3. **Don't hardcode `um_per_pixel=1.0`**
   - Results in meaningless µm² values
   - Use actual calibration from microscope

---

## Troubleshooting

### Error: "No module named 'detectron2'"

**Cause:** Trying to load `.pkl` files without Detectron2 installed

**Solution:**
```bash
# Convert predictions on machine with Detectron2
python convert_dlma_to_npz.py predictions/

# Then use on analysis machine
prediction_format='numpy'
```

### Error: "No .npz files found"

**Cause:** Forgot to convert after inference

**Solution:**
```bash
# Convert .pkl to .npz
python convert_dlma_to_npz.py predictions/

# Check files
ls predictions/*.npz
```

### Warning: "total_embryo_area_um2 equals total_embryo_area_px"

**Cause:** Using default `um_per_pixel=1.0` instead of real calibration

**Solution:**
```python
# Get calibration from microscope metadata
tracking_df['um_per_pixel'] = 2.5  # Example: 2.5 µm/pixel

# Or read from image EXIF
from PIL import Image
img = Image.open('image.tif')
um_per_pixel = img.tag[282][0]  # XResolution
```

---

## Migration Guide

### Old Way (Coupled)

```python
# Everything requires Detectron2
extract_morphometric_features_batch(
    tracking_df=df,
    dlma_predictions_dir=Path('predictions'),
    prediction_format='detectron2',  # ❌ Requires Detectron2
)
```

### New Way (Decoupled)

```bash
# Stage 1: Inference (GPU + Detectron2)
python dlma_inference_example.py --image_dir images/ --output_dir predictions/

# Stage 2: Convert (any machine with PyTorch)
python convert_dlma_to_npz.py predictions/

# Stage 3: Analyze (any machine, NumPy only)
```

```python
extract_morphometric_features_batch(
    tracking_df=df,
    dlma_predictions_dir=Path('predictions'),
    prediction_format='numpy',  # ✅ No Detectron2 needed
)
```

---

## Summary

**Problem:** Detectron2 dependency couples inference and analysis

**Solution:** Two-stage workflow with `.npz` conversion

**Benefits:**
- ✅ Decouple GPU inference from CPU analysis
- ✅ 10× smaller files
- ✅ 10× faster loading
- ✅ No Detectron2 on analysis nodes
- ✅ Works on CPU-only machines

**Key Changes:**
- Default format: `'detectron2'` → `'numpy'`
- Pixel size: `'micrometers_per_pixel'` → `'um_per_pixel'`
- New tool: `convert_dlma_to_npz.py`

---

**Version**: 1.0
**Last Updated**: 2025-11-15
**Author**: morphseq team
