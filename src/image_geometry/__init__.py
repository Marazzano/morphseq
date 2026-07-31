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
from .candidates import (
    CANDIDATE_KEYS,
    CANDIDATE_KEYS_NO_FLIP,
    PlacedCandidate,
    candidate_keys,
    centered_placement_affine,
    enumerate_orientation_candidates,
    pca_major_axis_angle_deg,
    vertical_flip_partner,
)
from .transforms import (
    AFFINE,
    CROP_PAD,
    FLIP_X,
    RESIZE,
    GridTransform,
    Interp,
    StepKind,
    TransformChain,
    affine_step,
    crop_pad_step,
    flip_x_step,
    resize_step,
    support_mask_for,
)

__all__ = [
    "BoxYX",
    "CoordConvention",
    "GridTransform",
    "Interp",
    "StepKind",
    "TransformChain",
    # Step-kind constructors — build a chain from these rather than raw GridTransforms, so the
    # raster semantics of each step are explicit.
    "affine_step",
    "crop_pad_step",
    "flip_x_step",
    "resize_step",
    # Distinguishes real measurements from synthesized zero padding — required before any
    # quantitative measurement that could sample outside the source raster.
    "support_mask_for",
    # Step-kind tokens, for dispatch and assertions.
    "AFFINE",
    "CROP_PAD",
    "FLIP_X",
    "RESIZE",
    # Orientation-candidate MECHANICS. A PCA axis fixes an elongated object's pose only up
    # to a 180-degree rotation and a mirror; these enumerate that closed four-way choice.
    # CHOOSING among them is domain policy and lives in embryo_geometry, not here.
    "CANDIDATE_KEYS",
    "CANDIDATE_KEYS_NO_FLIP",
    "PlacedCandidate",
    "candidate_keys",
    "centered_placement_affine",
    "enumerate_orientation_candidates",
    "pca_major_axis_angle_deg",
    "vertical_flip_partner",
]
