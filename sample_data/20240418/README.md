# Sample SAM2 Data for Bridge Script Development

This directory contains minimal sample data extracted from the real 20240418 experiment for developing and testing the SAM2 metadata bridge script.

## Files

### `sample_grounded_sam_segmentations.json`
Minimal SAM2 annotation file containing:
- **2 wells**: A01 (2 embryos), A04 (1 embryo) 
- **5 frames total**: 3 from A01, 2 from A04
- **8 snips total**: 6 from A01, 2 from A04
- Real data structure with simplified RLE masks for some frames

### `masks/` directory
Contains exported mask PNG files corresponding to JSON data:
- `20240418_A01_ch00_t0000_masks_emnum_2.png` - Frame 0, 2 embryos
- `20240418_A01_ch00_t0001_masks_emnum_2.png` - Frame 1, 2 embryos  
- `20240418_A04_ch00_t0000_masks_emnum_1.png` - Frame 0, 1 embryo

## Data Structure

### Key Entities
- **Experiment**: 20240418
- **Videos**: 20240418_A01, 20240418_A04
- **Images**: 5 total (3 A01, 2 A04)
- **Embryos**: 3 total (2 A01, 1 A04)
- **Snips**: 8 total (6 A01, 2 A04)

### Expected CSV Output Schema
The bridge script should generate CSV with these columns:
```
image_id, embryo_id, snip_id, frame_index, area_px, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, mask_confidence, exported_mask_path, experiment_id, video_id, is_seed_frame
```

### Sample Expected Rows
```csv
20240418_A01_ch00_t0000,20240418_A01_e01,20240418_A01_e01_s0000,0,41642.0,0.576,0.512,0.674,0.629,0.85,/path/to/masks/20240418_A01_ch00_t0000_masks_emnum_2.png,20240418,20240418_A01,false
20240418_A01_ch00_t0000,20240418_A01_e02,20240418_A01_e02_s0000,0,41689.0,0.226,0.647,0.336,0.749,0.85,/path/to/masks/20240418_A01_ch00_t0000_masks_emnum_2.png,20240418,20240418_A01,false
```

## File Relationships

### JSON → Mask File Mapping
- Each `image_id` in JSON corresponds to one mask file
- Mask file naming: `{image_id}_masks_emnum_{num_embryos}.png`
- Pixel values in mask: embryo_id suffix number (e01→1, e02→2)

### ID Hierarchy
```
20240418 (experiment)
├── 20240418_A01 (video/well)
│   ├── 20240418_A01_ch00_t0000 (image)
│   │   ├── 20240418_A01_e01 (embryo)
│   │   │   └── 20240418_A01_e01_s0000 (snip)
│   │   └── 20240418_A01_e02 (embryo)
│   │       └── 20240418_A01_e02_s0000 (snip)
│   └── ...
└── 20240418_A04 (video/well)
    └── ...
```

## Usage

Test the bridge script with:
```bash
python export_sam2_metadata_to_csv.py \
  /path/to/sample_data/20240418/sample_grounded_sam_segmentations.json \
  -o sample_output.csv \
  --masks-dir /path/to/sample_data/20240418/masks
```

Expected output: 8 rows (one per snip) with proper CSV schema validation.