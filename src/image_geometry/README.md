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
