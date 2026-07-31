# image_geometry

Coordinates and transforms on image grids.

## Why this is a root-level package

Image geometry belongs to neither the analysis stack nor the data pipeline. Both are clients:

```text
analyze ─────┐
             ├── image_geometry
data_pipeline┘
```

These primitives previously lived in `analyze.utils.coord`, which meant the data pipeline could only
reach them by importing the analysis layer — inverting the real dependency and risking a cycle
(`analyze` already imports `data_pipeline`). Moving them under `data_pipeline/shared/` would have
fixed the import direction while asserting an equally false ownership claim, and a directory named
`shared` describes *who uses* something rather than *what it is*. Names like `shared`, `common`, and
`utils` become attics; a semantic name gives the package a border.

## What this package owns

- pixel-coordinate boxes (`BoxYX`) — half-open, yx order
- image-grid transforms and their composition (`GridTransform`, `TransformChain`)
- affine conventions — OpenCV xy matrices over yx-ordered metadata
- the interpolation vocabulary (`Interp`)
- shape and coordinate provenance

## What this package does NOT own

Segmentation, visualization, file loading, morphology features, biological registration models.

Concretely, these stayed in `analyze.utils.coord` because they encode analysis-specific products:
the canonical embryo grid, the yolk-aware aligner, `Frame` (which carries a yolk mask), and the
`Canonical*` / `RegisterResult` containers. A test asserts they never appear here
(`tests/test_promotion_contract.py::test_analysis_specific_types_did_NOT_move`).

## Two parallel truths

A `TransformChain` records both, and they are not the same thing:

- **Coordinate truth** — where a source location lands in the output. Composes into one matrix;
  `composite_affine()` derives it. Use it to map *points*.
- **Raster truth** — how the pixel *values* were produced. Does **not** compose into a matrix.

The distinction is load-bearing: *correct geometric transform ≠ correct image resampling*. An affine
matrix describes the coordinate mapping perfectly while saying nothing about the frequency content
that must be removed before sampling onto a coarser grid.

Downsampling and rotation solve different problems. Downsampling changes the sampling rate and needs
a low-pass prefilter first, or high frequencies fold into false structure. Rotation remaps
coordinates — it needs interpolation, but not area integration. So they are separate steps.

**Measured, not assumed:** `cv2.warpAffine` *silently ignores* `INTER_AREA` — passing it yields
output byte-identical to `INTER_LINEAR`. A fused affine containing a large downscale therefore
cannot anti-alias at all. On a checkerboard at 0.414×, a correct anti-aliased resize collapses it to
flat grey (std ≈ 1–2) while `warpAffine` leaves std ≈ 33 of pure aliasing artifact. That is why
`resize` is its own step kind and not a scale folded into the affine.

## Step kinds

`GridTransform.name` selects the raster operation:

| kind | what it does | interpolation |
|---|---|---|
| `resize` | anti-aliased scale change; lands the physical pixel size | `INTER_AREA` on downscale, linear on upscale |
| `crop_pad` | pure indexing — no resampling, no new pixel values | none |
| `flip_x` | horizontal mirror | none |
| *(other)* | general affine: rotation, translation, shear | linear (nearest for masks) |

Build chains from the step constructors rather than raw `GridTransform`s, so each step's raster
semantics are explicit:

```python
from image_geometry import TransformChain, resize_step, affine_step, crop_pad_step

chain = TransformChain([
    resize_step(in_shape_yx=src.shape, out_shape_yx=(h, w)),   # sampling rate
    affine_step(affine_2x3=rot, in_shape_yx=(h, w), out_shape_yx=(h, w)),  # orientation + placement
    crop_pad_step(in_shape_yx=(h, w), y0=y0, x0=x0, out_shape_yx=(576, 256)),  # exact extent
])
```

Keep large scale changes out of the affine: `warpAffine`'s filter footprint is a small local
neighborhood, far too narrow when shrinking substantially, and it cannot be widened by a flag.

## Conventions

- Coordinate **values** in metadata are **yx**-ordered.
- OpenCV affine matrices are **xy**-ordered internally (`params["affine_convention"] = "opencv_xy"`).
- Boxes are **half-open**: `arr[box.to_slices()]` selects exactly the content, and two boxes that
  touch edge-to-edge do *not* intersect.
- Interpolation is chosen by content, not by caller preference: **masks nearest, images linear**.
  `TransformChain.apply_to_mask` forces `INTER_NEAREST` regardless of the transform's `interp`
  field, so a binary mask cannot be silently blurred into fractional values.

## Usage

```python
from image_geometry import BoxYX, GridTransform, TransformChain

box = BoxYX.from_mask(mask)              # None when the mask is empty
search = box.pad(radius, radius).clamp(h, w)
if search.intersects(other_box):
    ...

snip = chain.apply_to_image(image)       # linear
mask_snip = chain.apply_to_mask(mask)    # nearest, always
```

## Compatibility

The former import paths still resolve and yield the *same* objects:

```python
from analyze.utils.coord import BoxYX, TransformChain          # re-export
from analyze.utils.coord.types import BoxYX                    # re-export
from analyze.utils.coord.transforms import TransformChain      # shim module
```

Prefer `from image_geometry import ...` in new code.
