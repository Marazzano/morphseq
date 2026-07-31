"""Coordinates and transforms on image grids.

Image geometry belongs to neither the analysis stack nor the data pipeline. Both are clients, so
these primitives live at the root rather than inside either one — putting them under
``data_pipeline/shared`` would have fixed the import direction while asserting a false ownership
("analysis depends on the pipeline"), and a directory named ``shared`` describes who uses something
rather than what it is.

This package owns:

- pixel-coordinate boxes (half-open, yx order)
- image-grid transforms and their composition
- affine conventions (OpenCV xy matrices over yx-ordered metadata)
- the interpolation vocabulary
- shape and coordinate provenance

This package does NOT own: segmentation, visualization, file loading, morphology features, or
biological registration models. Canonical-embryo alignment, the yolk-aware aligner, and the
Canonical*/Frame result containers deliberately stayed in ``analyze.utils.coord``, which re-exports
the names below so existing importers keep working.

Conventions:
- Coordinate VALUES in metadata are yx-ordered.
- OpenCV affine matrices are xy-ordered internally (``affine_convention: "opencv_xy"``).
- Boxes are half-open: ``arr[box.to_slices()]`` selects exactly the content.
"""

from .boxes import BoxYX, CoordConvention
from .transforms import (
    CROP_PAD,
    FLIP_X,
    RESIZE,
    GridTransform,
    Interp,
    TransformChain,
    affine_step,
    crop_pad_step,
    resize_step,
)

__all__ = [
    "BoxYX",
    "CoordConvention",
    "GridTransform",
    "Interp",
    "TransformChain",
    # Step-kind constructors — build a chain from these rather than raw GridTransforms, so the
    # raster semantics of each step are explicit.
    "affine_step",
    "crop_pad_step",
    "resize_step",
    # Step-kind tokens, for dispatch and assertions.
    "CROP_PAD",
    "FLIP_X",
    "RESIZE",
]
