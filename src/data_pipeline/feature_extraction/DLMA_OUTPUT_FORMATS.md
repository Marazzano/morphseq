# DLMA Model Output Formats - Detailed Guide

This document provides detailed information about the various output formats from the DLMA (Deep Learning-Enabled Morphometric Analysis) model.

## Overview

The DLMA model can produce outputs in several formats depending on how you run the inference and which utilities you use. This guide covers all major formats and how to work with them.

---

## Format 1: Detectron2 Raw Output (Primary Format)

This is the native output format from the Detectron2 Mask R-CNN model.

### Structure

When you run inference, DLMA returns a dictionary with an `Instances` object:

```python
outputs = predictor(image)

# outputs is a dict:
{
    'instances': Instances(
        num_instances=5,  # Number of detected objects
        image_height=512,
        image_width=512,
        fields={
            'pred_boxes': Boxes(...),      # Bounding boxes
            'scores': Tensor(...),         # Confidence scores
            'pred_classes': Tensor(...),   # Class IDs
            'pred_masks': Tensor(...)      # Binary segmentation masks
        }
    )
}
```

### Detailed Field Descriptions

#### 1. **pred_boxes** - Bounding Boxes
- **Type**: `detectron2.structures.Boxes`
- **Shape**: `(N, 4)` where N = number of detections
- **Format**: Each row is `[x1, y1, x2, y2]` in pixel coordinates
- **Description**: Axis-aligned bounding boxes around each detected object

```python
boxes = outputs['instances'].pred_boxes.tensor.numpy()
# Example output:
# array([[120.5, 180.3, 145.2, 210.8],  # Detection 1: eye
#        [115.0, 215.5, 135.0, 240.2],  # Detection 2: heart
#        [110.0, 250.0, 180.5, 320.8]]) # Detection 3: yolk
```

#### 2. **scores** - Confidence Scores
- **Type**: `torch.Tensor`
- **Shape**: `(N,)` where N = number of detections
- **Range**: 0.0 to 1.0
- **Description**: Model's confidence that the detection is correct

```python
scores = outputs['instances'].scores.numpy()
# Example output:
# array([0.985, 0.923, 0.978])
# Interpretation:
# - Detection 1: 98.5% confident
# - Detection 2: 92.3% confident
# - Detection 3: 97.8% confident
```

#### 3. **pred_classes** - Class IDs
- **Type**: `torch.Tensor`
- **Shape**: `(N,)` where N = number of detections
- **Values**: Integers 1-16 (or 0-15 depending on configuration)
- **Description**: Which organ/phenotype class each detection belongs to

```python
classes = outputs['instances'].pred_classes.numpy()
# Example output:
# array([1, 2, 3])
# Interpretation:
# - Detection 1: Class 1 = eye
# - Detection 2: Class 2 = heart
# - Detection 3: Class 3 = yolk
```

**Class ID Mapping**:
```
ID | Class Name              | Type
---+------------------------+----------
1  | eye                    | organ
2  | heart                  | organ
3  | yolk                   | organ
4  | swim_bladder           | organ
5  | otolith                | organ
6  | gut                    | organ
7  | trunk                  | organ
8  | tail                   | organ
9  | pericardial_edema      | phenotype
10 | yolk_sac_edema         | phenotype
11 | spinal_curvature       | phenotype
12 | tail_malformation      | phenotype
13 | craniofacial_malform   | phenotype
14 | reduced_pigmentation   | phenotype
15 | general_edema          | phenotype
16 | hemorrhage             | phenotype
```

#### 4. **pred_masks** - Segmentation Masks
- **Type**: `torch.Tensor` (boolean or uint8)
- **Shape**: `(N, H, W)` where:
  - N = number of detections
  - H = image height
  - W = image width
- **Values**: Binary (0 or 1)
- **Description**: Pixel-level segmentation for each detected object

```python
masks = outputs['instances'].pred_masks.numpy()
# Example output shape: (3, 512, 512)
# 3 detections, each with a 512x512 binary mask

# Mask for first detection (eye):
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 1, ..., 1, 0, 0],  # 1 = part of eye
#        ...
#        [0, 0, 0, ..., 0, 0, 0]])

# Calculate area:
eye_area_pixels = masks[0].sum()  # e.g., 1250 pixels
```

### Saving Detectron2 Format

**Option 1: Pickle (recommended for preserving full structure)**
```python
import pickle

with open('zebrafish_001_dlma.pkl', 'wb') as f:
    pickle.dump(outputs, f)

# Load later:
with open('zebrafish_001_dlma.pkl', 'rb') as f:
    outputs = pickle.load(f)
```

**Option 2: NumPy (lighter weight, loses some metadata)**
```python
import numpy as np

instances = outputs['instances'].to('cpu')
np.savez(
    'zebrafish_001_dlma.npz',
    masks=instances.pred_masks.numpy(),
    classes=instances.pred_classes.numpy(),
    scores=instances.scores.numpy(),
    boxes=instances.pred_boxes.tensor.numpy(),
)

# Load later:
data = np.load('zebrafish_001_dlma.npz')
masks = data['masks']
classes = data['classes']
```

### Example: Accessing All Data

```python
# Run inference
image = cv2.imread('zebrafish_001.png')
outputs = predictor(image)

# Extract to CPU and convert to numpy
instances = outputs['instances'].to('cpu')

# Get all fields
boxes = instances.pred_boxes.tensor.numpy()     # (N, 4)
scores = instances.scores.numpy()               # (N,)
classes = instances.pred_classes.numpy()        # (N,)
masks = instances.pred_masks.numpy()            # (N, H, W)

# Print summary
print(f"Found {len(instances)} detections")
for i in range(len(instances)):
    class_id = classes[i]
    class_name = DLMA_ALL_CLASSES[class_id]
    confidence = scores[i]
    bbox = boxes[i]
    area = masks[i].sum()

    print(f"Detection {i+1}:")
    print(f"  Class: {class_name} (ID: {class_id})")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    print(f"  Mask area: {area} pixels")
```

**Example Output**:
```
Found 5 detections
Detection 1:
  Class: eye (ID: 1)
  Confidence: 0.985
  Bounding box: [120.5, 180.3, 145.2, 210.8]
  Mask area: 1250 pixels
Detection 2:
  Class: heart (ID: 2)
  Confidence: 0.923
  Bounding box: [115.0, 215.5, 135.0, 240.2]
  Mask area: 890 pixels
Detection 3:
  Class: yolk (ID: 3)
  Confidence: 0.978
  Bounding box: [110.0, 250.0, 180.5, 320.8]
  Mask area: 5600 pixels
...
```

---

## Format 2: CSV Output (from fishutil/fishclass)

The DLMA repository includes custom utilities (`fishutil.py` and `fishclass.py`) that convert raw detections into quantitative morphometric parameters and save them as CSV files.

### Structure

The CSV format contains one row per detected object, with quantitative measurements:

```csv
image_id,detection_id,class_id,class_name,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,area_px,perimeter_px,centroid_x,centroid_y,length,width,aspect_ratio
zebrafish_001,0,1,eye,0.985,120.5,180.3,145.2,210.8,1250,142,132.8,195.5,24.7,20.5,1.20
zebrafish_001,1,2,heart,0.923,115.0,215.5,135.0,240.2,890,98,125.0,227.8,20.0,18.5,1.08
zebrafish_001,2,3,yolk,0.978,110.0,250.0,180.5,320.8,5600,280,145.2,285.4,70.5,65.2,1.08
zebrafish_001,3,9,pericardial_edema,0.876,112.0,212.0,138.0,242.0,1120,115,125.0,227.0,26.0,22.0,1.18
zebrafish_002,0,1,eye,0.991,118.3,175.8,143.9,206.3,1310,148,131.1,191.0,25.6,21.3,1.20
...
```

### Column Descriptions

| Column | Type | Description | Units | Example |
|--------|------|-------------|-------|---------|
| `image_id` | string | Image identifier | - | "zebrafish_001" |
| `detection_id` | int | Detection index within image | - | 0, 1, 2... |
| `class_id` | int | Numerical class identifier | - | 1-16 |
| `class_name` | string | Human-readable class name | - | "eye", "heart" |
| `confidence` | float | Model confidence score | 0-1 | 0.985 |
| `bbox_x1` | float | Bounding box min X | pixels | 120.5 |
| `bbox_y1` | float | Bounding box min Y | pixels | 180.3 |
| `bbox_x2` | float | Bounding box max X | pixels | 145.2 |
| `bbox_y2` | float | Bounding box max Y | pixels | 210.8 |
| `area_px` | int | Mask area | pixels² | 1250 |
| `perimeter_px` | float | Mask perimeter | pixels | 142.5 |
| `centroid_x` | float | Center of mass X | pixels | 132.8 |
| `centroid_y` | float | Center of mass Y | pixels | 195.5 |
| `length` | float | Major axis length | pixels | 24.7 |
| `width` | float | Minor axis length | pixels | 20.5 |
| `aspect_ratio` | float | Length/width ratio | - | 1.20 |

### Aggregate CSV Format

Some pipelines create aggregate summaries with one row per image:

```csv
image_id,n_detections,eye_area_px,heart_area_px,yolk_area_px,eye_area_pct,heart_area_pct,yolk_area_pct,has_pericardial_edema,has_yolk_edema,has_spine_curve
zebrafish_001,5,1250,890,5600,2.5,1.8,11.2,True,False,False
zebrafish_002,4,1310,920,5800,2.6,1.8,11.5,False,False,False
zebrafish_003,6,1180,850,5400,2.3,1.7,10.8,True,True,False
```

### Working with CSV Format

```python
import pandas as pd

# Load CSV
df = pd.read_csv('dlma_results.csv')

# Filter by image
image_data = df[df['image_id'] == 'zebrafish_001']

# Get all eyes detected
eyes = df[df['class_name'] == 'eye']

# Calculate mean eye area
mean_eye_area = eyes['area_px'].mean()

# Find images with abnormalities
abnormal = df[df['class_id'] >= 9]  # Phenotype classes
abnormal_images = abnormal['image_id'].unique()

# Summary statistics by class
summary = df.groupby('class_name').agg({
    'area_px': ['mean', 'std', 'count'],
    'confidence': 'mean'
})
```

---

## Format 3: morphseq Integration Output

The morphseq integration (`zebrafish_morphometric_analysis.py`) produces a different format optimized for morphseq workflows.

### Structure

One row per `snip_id` with area percentages for all classes:

```csv
snip_id,eye_area_pct,heart_area_pct,yolk_area_pct,swim_bladder_area_pct,otolith_area_pct,gut_area_pct,trunk_area_pct,tail_area_pct,pericardial_edema_area_pct,yolk_sac_edema_area_pct,spinal_curvature_area_pct,tail_malformation_area_pct,craniofacial_malformation_area_pct,reduced_pigmentation_area_pct,general_edema_area_pct,hemorrhage_area_pct,total_embryo_area_px,total_embryo_area_um2
zebrafish_001_t0,2.50,1.78,11.20,0.85,0.42,3.20,25.50,45.30,2.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,50000,125000.0
zebrafish_001_t1,2.55,1.80,11.00,0.88,0.45,3.25,25.80,45.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,51200,128000.0
zebrafish_002_t0,2.60,1.85,11.50,0.90,0.48,3.30,26.00,44.80,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,50500,126250.0
```

### Column Descriptions

- **`snip_id`**: Unique identifier for each image/timepoint
- **`{class_name}_area_pct`**: Percentage of total embryo area occupied by this class (0-100)
- **`total_embryo_area_px`**: Total embryo area in pixels
- **`total_embryo_area_um2`**: Total embryo area in square micrometers (if pixel size provided)

### Key Differences from CSV Format

1. **Aggregated by class**: Multiple instances of same class (e.g., two eyes) are summed
2. **Percentage-based**: Values are percentages (0-100) rather than absolute pixels
3. **All classes present**: Even undetected classes have 0.0 values
4. **Single row per snip**: One comprehensive row per image

### Working with morphseq Format

```python
import pandas as pd

# Load features
features = pd.read_csv('morphometric_features.csv')

# Calculate total organ area
organ_cols = ['eye_area_pct', 'heart_area_pct', 'yolk_area_pct',
              'swim_bladder_area_pct', 'otolith_area_pct', 'gut_area_pct']
features['total_organs_pct'] = features[organ_cols].sum(axis=1)

# Find abnormal embryos (any phenotype > 0)
phenotype_cols = ['pericardial_edema_area_pct', 'yolk_sac_edema_area_pct',
                  'spinal_curvature_area_pct', 'tail_malformation_area_pct']
features['has_phenotype'] = features[phenotype_cols].gt(0).any(axis=1)

# Convert percentages to absolute area
features['yolk_area_um2'] = (features['yolk_area_pct'] / 100.0) * features['total_embryo_area_um2']
```

---

## Format Comparison

| Feature | Detectron2 Pickle | CSV (fishutil) | morphseq CSV |
|---------|------------------|----------------|--------------|
| **Granularity** | Instance-level | Instance-level | Image-level |
| **File size** | Large (MB) | Medium (KB-MB) | Small (KB) |
| **Masks included** | ✓ Yes | ✗ No | ✗ No |
| **Area units** | Pixels | Pixels | Percentage + pixels |
| **Multiple instances** | Separate | Separate rows | Aggregated |
| **Undetected classes** | Absent | Absent | Present (0.0) |
| **Best for** | Visualization, detailed analysis | Quantitative analysis | Pipeline integration |

---

## Data Format Examples

### Example 1: Single Zebrafish with 5 Detections

**Detectron2 Format (Python)**:
```python
{
    'instances': Instances(
        pred_boxes: [[120, 180, 145, 211], [115, 215, 135, 240], ...],
        scores: [0.985, 0.923, 0.978, 0.856, 0.912],
        pred_classes: [1, 2, 3, 4, 7],  # eye, heart, yolk, swim_bladder, trunk
        pred_masks: (5, 512, 512)  # 5 binary masks
    )
}
```

**CSV Format**:
```csv
image_id,class_name,area_px,confidence
zebrafish_001,eye,1250,0.985
zebrafish_001,heart,890,0.923
zebrafish_001,yolk,5600,0.978
zebrafish_001,swim_bladder,425,0.856
zebrafish_001,trunk,12750,0.912
```

**morphseq Format**:
```csv
snip_id,eye_area_pct,heart_area_pct,yolk_area_pct,swim_bladder_area_pct,trunk_area_pct,total_embryo_area_px
zebrafish_001,2.50,1.78,11.20,0.85,25.50,50000
```

### Example 2: Abnormal Zebrafish with Phenotype

**Detectron2 Format**:
```python
{
    'instances': Instances(
        pred_classes: [1, 2, 3, 9],  # eye, heart, yolk, pericardial_edema
        scores: [0.990, 0.945, 0.982, 0.876]
    )
}
```

**CSV Format**:
```csv
image_id,class_name,area_px,confidence
zebrafish_005,eye,1280,0.990
zebrafish_005,heart,920,0.945
zebrafish_005,yolk,5700,0.982
zebrafish_005,pericardial_edema,1050,0.876
```

**morphseq Format**:
```csv
snip_id,eye_area_pct,heart_area_pct,yolk_area_pct,pericardial_edema_area_pct,total_embryo_area_px
zebrafish_005,2.56,1.84,11.40,2.10,50000
```

---

## Converting Between Formats

### Detectron2 → CSV

```python
import pandas as pd

def detectron2_to_csv(outputs, image_id, class_mapping):
    instances = outputs['instances'].to('cpu')

    data = []
    for i in range(len(instances)):
        row = {
            'image_id': image_id,
            'detection_id': i,
            'class_id': instances.pred_classes[i].item(),
            'class_name': class_mapping[instances.pred_classes[i].item()],
            'confidence': instances.scores[i].item(),
            'area_px': instances.pred_masks[i].sum().item(),
            'bbox_x1': instances.pred_boxes[i].tensor[0, 0].item(),
            'bbox_y1': instances.pred_boxes[i].tensor[0, 1].item(),
            'bbox_x2': instances.pred_boxes[i].tensor[0, 2].item(),
            'bbox_y2': instances.pred_boxes[i].tensor[0, 3].item(),
        }
        data.append(row)

    return pd.DataFrame(data)
```

### CSV → morphseq Format

```python
def csv_to_morphseq(csv_df, class_mapping):
    results = []

    for image_id in csv_df['image_id'].unique():
        image_data = csv_df[csv_df['image_id'] == image_id]
        total_area = image_data['area_px'].sum()

        row = {'snip_id': image_id}

        # Aggregate by class
        for class_id, class_name in class_mapping.items():
            if class_name == 'background':
                continue
            class_data = image_data[image_data['class_id'] == class_id]
            class_area = class_data['area_px'].sum()
            row[f'{class_name}_area_pct'] = (class_area / total_area * 100) if total_area > 0 else 0

        row['total_embryo_area_px'] = total_area
        results.append(row)

    return pd.DataFrame(results)
```

---

## Best Practices

1. **For Storage**: Save Detectron2 format as pickle for complete preservation
2. **For Analysis**: Convert to CSV for easier manipulation in pandas/Excel
3. **For Pipelines**: Use morphseq format for integration with existing workflows
4. **For Sharing**: CSV format is most portable and human-readable
5. **For Visualization**: Keep Detectron2 format to access full masks

---

## Summary

- **Detectron2 format**: Rich, complete, includes full masks and metadata
- **CSV format**: Tabular, quantitative parameters, easy to analyze
- **morphseq format**: Aggregated, percentage-based, pipeline-ready

Choose the format based on your needs:
- Research/exploration → Detectron2
- Quantitative analysis → CSV
- Pipeline integration → morphseq

---

**Version**: 1.0
**Last Updated**: 2025-11-15
**Author**: morphseq team
