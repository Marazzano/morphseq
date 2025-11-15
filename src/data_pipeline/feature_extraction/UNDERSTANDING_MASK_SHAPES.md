# Understanding Mask Shapes: How (3, 512, 512) Represents Complicated Shapes

## The Question

**"How does (3, 512, 512) actually get turned into a mask? Masks can be complicated shapes but this is only three numbers."**

## The Answer

**(3, 512, 512) is NOT three numbers** - it's a **shape description** of a 3-dimensional array containing **786,432 individual values**!

Think of it like describing a building: "3 floors, 512 feet wide, 512 feet deep"

---

## Part 1: Understanding Array Shapes

### What (3, 512, 512) Really Means

```
(3, 512, 512)
 │   │    │
 │   │    └─── Width: 512 pixels
 │   └──────── Height: 512 pixels
 └──────────── Number of masks: 3

Total values: 3 × 512 × 512 = 786,432 individual pixels!
```

### Simple Example: (2, 5, 5)

Let's use a tiny example: **2 masks, each 5×5 pixels** = 50 total values

```
Mask 1 (eye - rectangular shape):
[
  [0, 0, 0, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 1, 0],    ← Each number is ONE pixel
  [0, 1, 1, 1, 0],    ← 0 = background, 1 = eye
  [0, 0, 0, 0, 0]
]
Area: 9 pixels

Mask 2 (heart - diamond shape):
[
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],    ← Notice the different shape!
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0]
]
Area: 13 pixels

Stack them together → Shape: (2, 5, 5)
Total values: 2 × 5 × 5 = 50 individual pixels
```

---

## Part 2: Binary Masks Explained

### What Are Binary Masks?

A **binary mask** is a 2D grid where each pixel is either:
- **0** = NOT part of the object (background)
- **1** = IS part of the object (the organ/structure)

### Visual Example: Eye Detection

Here's what an eye mask actually looks like (visualized as a grid):

```
Image Region (20×30 pixels):

· · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
· · · · · · · · · · 1 1 1 1 1 1 1 · · · · · · · · · · · · ·
· · · · · · · · 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · · · ·
· · · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · · ·
· · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · ·  ← Eye!
· · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · ·
· · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · ·
· · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · ·
· · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · ·
· · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · ·
· · · · · · · 1 1 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · · ·
· · · · · · · · 1 1 1 1 1 1 1 1 1 1 1 · · · · · · · · · · ·
· · · · · · · · · · 1 1 1 1 1 1 1 · · · · · · · · · · · · ·
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

Legend:
  · = 0 (background)
  1 = 1 (eye pixel)

Array representation:
mask[0, 0] = 0   (background)
mask[6, 15] = 1  (part of eye)
mask[10, 15] = 1 (part of eye)
```

**The shape is defined by which pixels are 1 vs 0!**

---

## Part 3: How Complicated Shapes Work

### ANY Shape Can Be Represented

Binary masks can represent **any shape** - no matter how irregular or complex:

#### Simple Circle:
```
· · · 1 1 1 · · ·
· · 1 1 1 1 1 · ·
· 1 1 1 1 1 1 1 ·
1 1 1 1 1 1 1 1 1
· 1 1 1 1 1 1 1 ·
· · 1 1 1 1 1 · ·
· · · 1 1 1 · · ·
```

#### Irregular Blob (realistic yolk sac):
```
· · · · 1 1 1 1 · · · · ·
· · 1 1 1 1 1 1 1 1 · · ·
· 1 1 1 1 1 1 1 1 1 1 1 ·
1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 ·
1 1 1 1 1 1 1 1 1 1 1 · ·
· 1 1 1 1 1 1 1 1 1 · · ·
· · 1 1 1 1 1 1 1 · · · ·
· · · 1 1 1 1 · · · · · ·
```

#### Complex Shape with Holes:
```
· · 1 1 1 1 1 1 1 1 · ·
· 1 1 1 1 1 1 1 1 1 1 ·
1 1 1 1 · · · · 1 1 1 1    ← Hole in the middle!
1 1 1 · · · · · · 1 1 1
1 1 1 1 · · · · 1 1 1 1
· 1 1 1 1 1 1 1 1 1 1 ·
· · 1 1 1 1 1 1 1 1 · ·
```

**Key Insight**: The complexity is in **which pixels are 1**, not in the data structure!

---

## Part 4: Multiple Detections → 3D Array

### Stacking Multiple Masks

When DLMA detects 3 organs in a 512×512 image:

```
Detection 0 (Eye):           Detection 1 (Heart):        Detection 2 (Yolk):
┌─────────────┐             ┌─────────────┐             ┌─────────────┐
│ · · · · · · │             │ · · · · · · │             │ · · · · · · │
│ · 1 1 1 · · │             │ · · · · · · │             │ · · · · · · │
│ · 1 1 1 · · │  512×512    │ · · 1 1 · · │  512×512    │ · 1 1 1 1 · │  512×512
│ · 1 1 1 · · │             │ · · 1 1 · · │             │ 1 1 1 1 1 1 │
│ · · · · · · │             │ · · 1 1 · · │             │ 1 1 1 1 1 1 │
│ · · · · · · │             │ · · · · · · │             │ · 1 1 1 1 · │
└─────────────┘             └─────────────┘             └─────────────┘
   1,250 pixels                890 pixels                 5,600 pixels
    are 1                       are 1                      are 1

                                    ↓
                              Stack them!
                                    ↓

                          Shape: (3, 512, 512)
                         Total: 786,432 values
                        (262,144 per mask)
```

### Accessing Individual Masks

```python
masks.shape = (3, 512, 512)

masks[0]          → First mask (512, 512) - the eye
masks[1]          → Second mask (512, 512) - the heart
masks[2]          → Third mask (512, 512) - the yolk

masks[0, 200, 150]  → Pixel at row 200, col 150 in eye mask (0 or 1)
masks[1, 250, 180]  → Pixel at row 250, col 180 in heart mask (0 or 1)
```

---

## Part 5: Real DLMA Example

### Actual Data from a Zebrafish Image

```
Image: 512×512 pixels
Detections: 5 organs found

outputs['instances'].pred_masks.shape = (5, 512, 512)

Breaking it down:
  Dimension 0: 5 detections
  Dimension 1: 512 pixels tall
  Dimension 2: 512 pixels wide
  Total values: 5 × 512 × 512 = 1,310,720 individual pixels!

masks[0] = Eye mask (512×512 grid)
  - 1,250 pixels are 1 (eye)
  - 260,894 pixels are 0 (background)
  - Total: 262,144 pixels

masks[1] = Heart mask (512×512 grid)
  - 890 pixels are 1 (heart)
  - 261,254 pixels are 0 (background)
  - Total: 262,144 pixels

masks[2] = Yolk mask (512×512 grid)
  - 5,600 pixels are 1 (yolk)
  - 256,544 pixels are 0 (background)
  - Total: 262,144 pixels

masks[3] = Swim bladder mask (512×512 grid)
  - 425 pixels are 1 (swim bladder)
  - 261,719 pixels are 0 (background)
  - Total: 262,144 pixels

masks[4] = Trunk mask (512×512 grid)
  - 12,750 pixels are 1 (trunk)
  - 249,394 pixels are 0 (background)
  - Total: 262,144 pixels
```

### Size Comparison

```
What (3, 512, 512) actually contains:

NOT this:
  [3, 512, 512]  ← Only 3 numbers

BUT this:
  786,432 individual pixel values!
  Each one is 0 or 1
  Stored in a 3-dimensional array structure

File size:
  - As pickle: ~1-5 MB (includes metadata)
  - As PNG (per mask): ~10-50 KB
  - As numpy .npz: ~100 KB - 1 MB
```

---

## Part 6: Visual Summary

### The Complete Picture

```
┌─────────────────────────────────────────────────────────┐
│ DLMA runs on a 512×512 zebrafish image                 │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Detects 3 organs: eye, heart, yolk                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Creates 3 separate binary masks                        │
│                                                         │
│   Mask 0 (eye):   512×512 = 262,144 pixels (0s and 1s) │
│   Mask 1 (heart): 512×512 = 262,144 pixels (0s and 1s) │
│   Mask 2 (yolk):  512×512 = 262,144 pixels (0s and 1s) │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Stacks them into 3D array                              │
│                                                         │
│   Shape: (3, 512, 512)                                 │
│   Total: 786,432 individual pixel values               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Each mask traces the EXACT shape of the organ          │
│ by marking which pixels belong to it                   │
│                                                         │
│ No matter how complicated the shape,                   │
│ it's just a pattern of 0s and 1s!                      │
└─────────────────────────────────────────────────────────┘
```

---

## Part 7: Common Misconceptions

### ❌ WRONG: "There are only 3 numbers"
```
(3, 512, 512) contains just 3 values
```

### ✅ CORRECT: "Shape notation describes array dimensions"
```
(3, 512, 512) describes the dimensions:
  - 3 separate masks
  - Each mask is 512 × 512 pixels
  - Total: 786,432 individual values
```

---

### ❌ WRONG: "Simple shapes need less data"
```
A circle needs less data than an irregular blob
```

### ✅ CORRECT: "All shapes in same-size image use same data"
```
Both shapes need 512×512 values (if image is 512×512)
- Circle: 262,144 values (mostly 0s, some 1s)
- Blob:   262,144 values (mostly 0s, some 1s)

Difference is in WHICH pixels are 1, not HOW MUCH data
```

---

### ❌ WRONG: "Masks are coordinates or equations"
```
Mask stores: [(x1, y1), (x2, y2), (x3, y3), ...]
```

### ✅ CORRECT: "Masks are pixel grids"
```
Mask is a complete grid:
Every pixel in the image gets a value (0 or 1)
No compression, no equations - just raw pixel values
```

---

## Key Takeaways

1. **(N, H, W) is shape notation, not values**
   - Describes dimensions of 3D array
   - Actual data: N × H × W individual pixels

2. **Binary masks are 2D grids of 0s and 1s**
   - 0 = background
   - 1 = object
   - Can represent ANY shape complexity

3. **Complicated shapes = complicated patterns of 0s and 1s**
   - Same data structure
   - Complexity is in which pixels are 1

4. **Multiple detections stack into 3D array**
   - Each detection gets its own 2D mask
   - Stacked along first dimension

5. **Data size is fixed by image dimensions**
   - 512×512 image → 262,144 pixels per mask
   - Whether simple circle or complex blob

---

## Analogy: Coloring Book

Think of a binary mask like a coloring book page:

```
Binary Mask = Coloring Book Page

- The page has a GRID of tiny squares (pixels)
- Each square is either:
  * Colored in (1) = part of the picture
  * Left blank (0) = background

- Complex pictures have more colored squares
- But SAME TOTAL number of squares on the page
- The picture's shape comes from WHICH squares are colored

Same with binary masks:
- Fixed grid size (512×512 squares)
- Each square is 0 or 1
- The organ's shape comes from which squares are 1
```

---

## Further Reading

- **DLMA_OUTPUT_FORMATS.md** - Detailed format specifications
- **dlma_output_example.py** - Runnable code examples
- **zebrafish_morphometric_analysis.py** - How we compute area percentages

---

**Version**: 1.0
**Last Updated**: 2025-11-15
**Author**: morphseq team
