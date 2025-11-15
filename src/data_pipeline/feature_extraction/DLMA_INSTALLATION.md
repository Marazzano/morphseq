# DLMA Integration Installation Guide

This guide explains how to install and integrate the Deep Learning-Enabled Morphometric Analysis (DLMA) framework into the morphseq pipeline for zebrafish organ and phenotype detection.

## Overview

**DLMA** is a Mask R-CNN-based instance segmentation model that detects:
- **8 organs**: eye, heart, yolk, swim bladder, otolith, gut, trunk, tail
- **8 abnormal phenotypes**: pericardial edema, yolk sac edema, spinal curvature, tail malformation, craniofacial malformation, reduced pigmentation, general edema, hemorrhage

**Reference**: Dong et al., "Deep Learning-Enabled Morphometric Analysis for Toxicity Screening Using Zebrafish Larvae" (Computers in Biology and Medicine, 2024)

## Installation

### Prerequisites

- Python 3.8+ (tested with 3.11)
- CUDA-capable GPU (recommended)
- Linux or macOS (Windows may require WSL)

### Step 1: Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 11.8 (check your CUDA version first with: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (not recommended for inference speed)
pip install torch torchvision
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Install Detectron2

Detectron2 v0.4.1 is required for DLMA compatibility.

**Option A: Pre-built wheels (recommended)**

```bash
# For Python 3.11 and CUDA 11.8
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# For other versions, see: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

**Option B: Build from source**

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout v0.4.1
pip install -e .
```

Verify installation:
```bash
python -c "import detectron2; print(detectron2.__version__)"
```

### Step 3: Install Supporting Libraries

```bash
pip install opencv-python scikit-image pandas numpy scipy
```

### Step 4: Clone DLMA Repository

```bash
cd /path/to/your/project
git clone https://github.com/gonggqing/DLMA.git
cd DLMA
```

### Step 5: Download Pre-trained Model Weights

The DLMA authors provide pre-trained model weights via Google Drive.

1. Download the model weights from the link in the DLMA repository README
2. Place the model file (e.g., `model_final.pth`) in a models directory:

```bash
mkdir -p models/dlma
# Download and move the .pth file to models/dlma/model_final.pth
```

### Step 6: Configure DLMA Paths

Update the DLMA config to point to your data:

```python
# In DLMA/config.py or inference script
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/path/to/models/dlma/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # 16 classes for DLMA
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
cfg.MODEL.DEVICE = "cuda"  # or "cpu"
```

## Integration with morphseq

### Directory Structure

Organize your morphseq pipeline to include DLMA predictions:

```
morphseq/
├── src/
│   └── data_pipeline/
│       └── feature_extraction/
│           ├── zebrafish_morphometric_analysis.py  # ← New module
│           ├── DLMA_INSTALLATION.md                # ← This guide
│           └── dlma_inference_example.py           # ← Example script
├── data/
│   ├── images/                   # Input zebrafish images
│   ├── masks/                    # Embryo segmentation masks
│   └── dlma_predictions/         # DLMA model outputs (to create)
└── DLMA/                         # Cloned DLMA repo
    ├── fishutil.py
    ├── fishclass.py
    └── inference.py
```

### Running DLMA Inference

See `dlma_inference_example.py` for a complete example. Basic workflow:

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2

# Setup config
cfg = get_cfg()
cfg.MODEL.WEIGHTS = "models/dlma/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
cfg.MODEL.DEVICE = "cuda"

# Create predictor
predictor = DefaultPredictor(cfg)

# Run inference
image = cv2.imread("data/images/zebrafish_001.png")
outputs = predictor(image)

# Save predictions
import pickle
with open("data/dlma_predictions/zebrafish_001_dlma.pkl", "wb") as f:
    pickle.dump(outputs, f)
```

### Using morphseq Integration

Once you have DLMA predictions, use the morphseq integration:

```python
import pandas as pd
from pathlib import Path
from src.data_pipeline.feature_extraction.zebrafish_morphometric_analysis import (
    extract_morphometric_features_batch
)

# Load your tracking DataFrame
tracking_df = pd.read_csv("tracking.csv")

# Extract morphometric features
features_df = extract_morphometric_features_batch(
    tracking_df=tracking_df,
    dlma_predictions_dir=Path("data/dlma_predictions"),
    embryo_mask_dir=Path("data/masks"),
    prediction_format='detectron2',
)

# Save results
features_df.to_csv("morphometric_features.csv", index=False)

# View area percentages
print(features_df[['snip_id', 'eye_area_pct', 'heart_area_pct', 'yolk_area_pct']].head())
```

## DLMA Output Format

### Instance Segmentation Format (Detectron2)

DLMA outputs Detectron2 `Instances` objects containing:

```python
outputs = {
    'instances': Instances(
        pred_boxes: Boxes,           # Bounding boxes (N, 4)
        scores: Tensor,              # Confidence scores (N,)
        pred_classes: Tensor,        # Class IDs (N,)
        pred_masks: Tensor,          # Binary masks (N, H, W)
    )
}
```

Where N is the number of detected instances.

### Class IDs

| ID | Class Name | Type |
|----|-----------|------|
| 1 | eye | organ |
| 2 | heart | organ |
| 3 | yolk | organ |
| 4 | swim_bladder | organ |
| 5 | otolith | organ |
| 6 | gut | organ |
| 7 | trunk | organ |
| 8 | tail | organ |
| 9 | pericardial_edema | phenotype |
| 10 | yolk_sac_edema | phenotype |
| 11 | spinal_curvature | phenotype |
| 12 | tail_malformation | phenotype |
| 13 | craniofacial_malformation | phenotype |
| 14 | reduced_pigmentation | phenotype |
| 15 | general_edema | phenotype |
| 16 | hemorrhage | phenotype |

### CSV Output (if using fishutil/fishclass)

DLMA's `fishutil.py` and `fishclass.py` can convert masks to quantitative parameters:

```csv
image_id,class_name,area_px,perimeter_px,centroid_x,centroid_y,confidence
zebrafish_001,eye,1250,142,256,180,0.95
zebrafish_001,heart,890,98,240,220,0.92
zebrafish_001,yolk,5600,280,260,300,0.98
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image resolution:

```python
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Default is 512
# Or resize images before inference
```

### Detectron2 Installation Issues

If pre-built wheels fail, build from source with specific CUDA version:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4.1'
```

### Model Loading Errors

Ensure model weights match the config architecture:

```python
# Check number of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # Must match training
```

### Missing fishutil/fishclass

These are custom DLMA utilities. Clone them from the repository:

```bash
cd DLMA
# fishutil.py and fishclass.py should be in the repo
# Import them as: from fishutil import *
```

## Performance Benchmarks

Typical inference times on a single NVIDIA RTX 3090:

- **Single image (512×512)**: ~0.05-0.1 seconds
- **Batch of 100 images**: ~5-8 seconds
- **CPU inference**: 10-20× slower

## Alternative: Using Pre-computed DLMA Outputs

If you have pre-computed DLMA predictions in CSV format:

```python
import pandas as pd

# Load DLMA CSV output
dlma_df = pd.read_csv("dlma_quantitative_output.csv")

# Convert to area percentages
# (implement custom converter based on CSV format)
```

## References

1. **Paper**: Dong et al., "Deep Learning-Enabled Morphometric Analysis for Toxicity Screening Using Zebrafish Larvae", Computers in Biology and Medicine, 2024
2. **DLMA GitHub**: https://github.com/gonggqing/DLMA
3. **Zebrafish Detection**: https://github.com/gonggqing/zebrafish_detection
4. **Detectron2**: https://github.com/facebookresearch/detectron2

## Support

For issues with:
- **DLMA model**: Open issue at https://github.com/gonggqing/DLMA/issues
- **morphseq integration**: Contact morphseq team
- **Detectron2**: See https://detectron2.readthedocs.io/

---

**Last Updated**: 2025-11-15
