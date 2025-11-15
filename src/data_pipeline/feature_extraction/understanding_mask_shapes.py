#!/usr/bin/env python3
"""
Understanding Mask Shapes and Binary Masks

This script explains how (N, H, W) shape notation represents actual pixel masks
and demonstrates what the mask data really looks like.

Author: morphseq team
Date: 2025-11-15
"""

import numpy as np


print("=" * 80)
print("UNDERSTANDING MASK SHAPES: How (3, 512, 512) Becomes Actual Masks")
print("=" * 80)

# ============================================================================
# PART 1: Understanding Shape Notation
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: What Does (3, 512, 512) Mean?")
print("=" * 80)

print("""
(3, 512, 512) is NOT three numbers - it's the SHAPE/DIMENSIONS of an array!

Think of it like describing a building:
  "3 floors, each 512 feet wide, 512 feet deep"

For masks:
  (3, 512, 512) means:
    - 3 separate masks (one per detection)
    - Each mask is 512 pixels tall
    - Each mask is 512 pixels wide

Total number of values: 3 × 512 × 512 = 786,432 individual pixel values!
""")

# Create a small example to demonstrate
print("\nSIMPLE EXAMPLE: (2, 5, 5) - 2 masks, each 5×5 pixels")
print("-" * 80)

# Create 2 simple masks (5×5 each)
mask1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],  # Rectangle shape
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

mask2 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],  # Diamond shape
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
])

# Stack them into 3D array
masks = np.stack([mask1, mask2])

print(f"Shape: {masks.shape}")
print(f"This means: {masks.shape[0]} masks, each {masks.shape[1]}×{masks.shape[2]} pixels")
print(f"Total values: {masks.size} individual pixels")
print()

print("Mask 1 (eye - rectangle):")
print(mask1)
print(f"  Area: {mask1.sum()} pixels")
print()

print("Mask 2 (heart - diamond):")
print(mask2)
print(f"  Area: {mask2.sum()} pixels")
print()


# ============================================================================
# PART 2: Binary Masks Explained
# ============================================================================

print("=" * 80)
print("PART 2: How Binary Masks Work")
print("=" * 80)

print("""
Binary masks are 2D grids where:
  - 0 = NOT part of the object
  - 1 = IS part of the object

Each pixel in the image gets a 0 or 1 to show if it belongs to the detected organ.

Example: Eye detection at position (120, 180) in a 512×512 image
""")

# Create a realistic example (smaller for display)
H, W = 20, 30  # Small image for demonstration
eye_mask = np.zeros((H, W), dtype=int)

# Add an ellipse shape for the eye
center_y, center_x = 10, 15
for y in range(H):
    for x in range(W):
        # Simple ellipse equation
        if ((x - center_x)**2 / 25 + (y - center_y)**2 / 16) < 1:
            eye_mask[y, x] = 1

print(f"\nExample: Eye mask in a {H}×{W} image region:")
print("-" * 80)
print("0 = background, 1 = eye pixels")
print()
print(eye_mask)
print()
print(f"Eye area: {eye_mask.sum()} pixels")
print(f"Background: {(eye_mask == 0).sum()} pixels")
print(f"Total pixels: {eye_mask.size}")


# ============================================================================
# PART 3: Multiple Detections = 3D Array
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: Multiple Detections Stack Into 3D Array")
print("=" * 80)

print("""
When DLMA detects multiple organs, each gets its own 2D mask.
These are stacked into a 3D array:

  Detection 1 (eye):    512×512 binary array
  Detection 2 (heart):  512×512 binary array  } Stack these
  Detection 3 (yolk):   512×512 binary array

  Result: (3, 512, 512) 3D array
""")

# Demonstrate with small example
H, W = 10, 15

# Create 3 different organ masks
eye_mask = np.zeros((H, W), dtype=int)
eye_mask[2:5, 3:7] = 1  # Small rectangle

heart_mask = np.zeros((H, W), dtype=int)
heart_mask[6:9, 8:12] = 1  # Another rectangle

yolk_mask = np.zeros((H, W), dtype=int)
yolk_mask[1:8, 10:14] = 1  # Larger rectangle

# Stack into 3D array
all_masks = np.stack([eye_mask, heart_mask, yolk_mask])

print(f"\nExample with 3 detections:")
print(f"Shape: {all_masks.shape}")
print()

print("Detection 0 - Eye mask:")
print(eye_mask)
print(f"Area: {eye_mask.sum()} pixels")
print()

print("Detection 1 - Heart mask:")
print(heart_mask)
print(f"Area: {heart_mask.sum()} pixels")
print()

print("Detection 2 - Yolk mask:")
print(yolk_mask)
print(f"Area: {yolk_mask.sum()} pixels")
print()

print("All three masks stacked (shape: {})".format(all_masks.shape))
print("First mask (eye):")
print(all_masks[0])
print()


# ============================================================================
# PART 4: Accessing Individual Masks
# ============================================================================

print("=" * 80)
print("PART 4: How to Access Individual Masks")
print("=" * 80)

print("""
Given: masks with shape (3, 512, 512)

Access individual masks:
  masks[0]     → First mask (512, 512)   - the eye
  masks[1]     → Second mask (512, 512)  - the heart
  masks[2]     → Third mask (512, 512)   - the yolk

Access individual pixels:
  masks[0, 100, 200]  → Pixel at row 100, col 200 in first mask
  masks[1, 250, 300]  → Pixel at row 250, col 300 in second mask
""")

# Demonstrate
print("\nDemonstration:")
print(f"all_masks.shape = {all_masks.shape}")
print(f"all_masks[0].shape = {all_masks[0].shape}  (first mask - eye)")
print(f"all_masks[1].shape = {all_masks[1].shape}  (second mask - heart)")
print(f"all_masks[2].shape = {all_masks[2].shape}  (third mask - yolk)")
print()

print("Accessing specific pixels:")
print(f"all_masks[0, 3, 4] = {all_masks[0, 3, 4]}  (eye mask at position [3,4])")
print(f"all_masks[1, 7, 9] = {all_masks[1, 7, 9]}  (heart mask at position [7,9])")
print()


# ============================================================================
# PART 5: Real DLMA Example
# ============================================================================

print("=" * 80)
print("PART 5: Real DLMA Output Example")
print("=" * 80)

print("""
When you run DLMA inference on a 512×512 zebrafish image:

outputs = predictor(image)
masks = outputs['instances'].pred_masks.numpy()

You might get: masks.shape = (5, 512, 512)

This means:
  - 5 organs/structures detected
  - Each has a 512×512 binary mask
  - Total data: 5 × 512 × 512 = 1,310,720 values (each 0 or 1)

Example breakdown:
""")

# Simulate realistic mask areas
simulated_detections = [
    ("eye", 1250),
    ("heart", 890),
    ("yolk", 5600),
    ("swim_bladder", 425),
    ("trunk", 12750),
]

total_image_pixels = 512 * 512

print(f"\nImage size: 512×512 = {total_image_pixels:,} total pixels")
print()

for i, (organ, area) in enumerate(simulated_detections):
    percentage = (area / total_image_pixels) * 100
    print(f"masks[{i}] - {organ:15s}: {area:5d} pixels are 1, "
          f"{total_image_pixels - area:6d} pixels are 0  "
          f"({percentage:.2f}% of image)")

print()


# ============================================================================
# PART 6: Visualizing How Complicated Shapes Work
# ============================================================================

print("=" * 80)
print("PART 6: How Complicated Shapes Are Represented")
print("=" * 80)

print("""
Binary masks can represent ANY shape - even very complex ones!
The shape is defined by which pixels are 1 vs 0.

Example: Irregular yolk sac shape
""")

# Create a more realistic irregular shape
H, W = 20, 25
irregular_mask = np.zeros((H, W), dtype=int)

# Create an irregular blob
for y in range(H):
    for x in range(W):
        # Irregular equation (not a perfect circle)
        distance = ((x - 12)**2 / 30 + (y - 10)**2 / 25)
        noise = np.sin(x * 0.5) * np.cos(y * 0.5) * 0.3
        if distance + noise < 1:
            irregular_mask[y, x] = 1

print("\nIrregular organ shape (like a real yolk sac):")
print("█ = organ pixels (1), · = background (0)")
print()

for row in irregular_mask:
    line = ""
    for pixel in row:
        line += "█" if pixel == 1 else "·"
    print(line)

print()
print(f"Shape area: {irregular_mask.sum()} pixels")
print(f"Perimeter: ~{np.sum(np.abs(np.diff(irregular_mask, axis=0))) + np.sum(np.abs(np.diff(irregular_mask, axis=1)))} pixels")
print()

print("""
Key insight: No matter how complicated the shape, it's just a grid of 0s and 1s!
  - Perfect circles: 0s and 1s
  - Irregular blobs: 0s and 1s
  - Zigzag boundaries: 0s and 1s
  - ANY shape: 0s and 1s

The complexity is in WHICH pixels are 1, not in the data structure.
""")


# ============================================================================
# PART 7: Summary
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
1. SHAPE NOTATION:
   (3, 512, 512) is NOT three values - it describes array dimensions
   - Dimension 0: 3 masks
   - Dimension 1: 512 pixels tall
   - Dimension 2: 512 pixels wide
   - Total values: 786,432 pixels!

2. BINARY MASKS:
   Each mask is a 2D grid of 0s (background) and 1s (object)
   Can represent ANY shape, no matter how complex

3. 3D STACKING:
   Multiple detections are stacked:
   [mask1, mask2, mask3] → (3, H, W)

4. ACTUAL DATA SIZE:
   For (5, 512, 512):
   - NOT 5 numbers
   - 1,310,720 individual pixel values (0 or 1)
   - That's why files can be large!

5. HOW TO THINK ABOUT IT:
   (N, H, W) is like a stack of N photographs
   Each photograph is H × W pixels
   Each pixel is either black (0) or white (1)
""")

print("\n" + "=" * 80)
print("For more details, see DLMA_OUTPUT_FORMATS.md")
print("=" * 80)


if __name__ == "__main__":
    pass
