"""Tests for the promotion of the generic coordinate primitives to ``image_geometry``.

Two things are pinned here:

1. **The compatibility contract.** The primitives moved out of ``analyze.utils.coord``, and roughly
   a dozen existing consumers import them from the old paths. Those paths must keep resolving, and
   must resolve to the SAME objects — a shim that returns a copy would silently break ``isinstance``
   checks and dataclass equality across the two import routes.

2. **``BoxYX.intersects``**, which was added during the promotion and therefore has no coverage in
   the inherited ``analyze/utils/coord/tests/test_boyx.py``.

The pre-existing BoxYX behavior (pad/clamp/union/validate/from_mask) is already covered by that
inherited suite, which imports through ``analyze.utils.coord.types`` and so doubles as a live test
of the re-export path. It is deliberately not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from image_geometry import BoxYX, GridTransform, Interp, TransformChain


class TestCompatibilityContract:
    """The old import paths must keep working, and yield identical objects."""

    def test_deep_transforms_import_still_resolves(self):
        from analyze.utils.coord.transforms import GridTransform as GT
        from analyze.utils.coord.transforms import TransformChain as TC

        assert TC is TransformChain
        assert GT is GridTransform

    def test_types_reexport_still_resolves(self):
        from analyze.utils.coord.types import BoxYX as B

        assert B is BoxYX

    def test_package_root_reexport_still_resolves(self):
        from analyze.utils.coord import BoxYX as B
        from analyze.utils.coord import TransformChain as TC

        assert B is BoxYX
        assert TC is TransformChain

    def test_analysis_specific_types_did_NOT_move(self):
        # The split is the point: generic primitives promoted, analysis products stayed.
        # If these ever appear in image_geometry, the package boundary has eroded.
        import image_geometry

        for analysis_name in (
            "Frame", "CanonicalGrid", "CanonicalMaskResult",
            "CanonicalImageResult", "CanonicalFrameResult", "RegisterResult",
        ):
            assert not hasattr(image_geometry, analysis_name), (
                f"{analysis_name} is analysis-specific and must stay in analyze.utils.coord; "
                "image_geometry owns only generic image-grid primitives."
            )

    def test_image_geometry_does_not_import_its_clients(self):
        # The whole reason for a root package: neither analyze nor data_pipeline owns this, so it
        # must not import either. A cycle here would defeat the promotion.
        import image_geometry.boxes
        import image_geometry.transforms

        for module in (image_geometry.boxes, image_geometry.transforms):
            source = module.__file__
            with open(source) as handle:
                text = handle.read()
            assert "import analyze" not in text and "from analyze" not in text, (
                f"{source} imports analyze; image_geometry must not depend on its clients."
            )
            assert "data_pipeline" not in text, (
                f"{source} references data_pipeline; image_geometry must not depend on its clients."
            )


class TestIntersects:
    """Added during promotion for the snip neighbor-exclusion prefilter."""

    def test_overlapping_boxes_intersect(self):
        assert BoxYX(10, 20, 10, 20).intersects(BoxYX(15, 25, 15, 25))

    def test_disjoint_boxes_do_not_intersect(self):
        assert not BoxYX(0, 10, 0, 10).intersects(BoxYX(50, 60, 50, 60))

    def test_edge_touching_does_not_intersect(self):
        # Half-open semantics: [0,10) and [10,20) share no pixel.
        assert not BoxYX(0, 10, 0, 10).intersects(BoxYX(10, 20, 0, 10))

    def test_overlap_required_on_BOTH_axes(self):
        # Overlapping rows but disjoint columns is NOT an intersection — the bug a naive
        # single-axis test would introduce in the neighbor prefilter.
        assert not BoxYX(0, 10, 0, 10).intersects(BoxYX(5, 15, 50, 60))

    def test_containment_intersects(self):
        outer, inner = BoxYX(0, 100, 0, 100), BoxYX(10, 20, 10, 20)
        assert outer.intersects(inner) and inner.intersects(outer)

    def test_is_symmetric(self):
        a, b = BoxYX(0, 10, 0, 10), BoxYX(5, 15, 5, 15)
        assert a.intersects(b) == b.intersects(a)

    def test_annulus_prefilter_shape(self):
        # The actual snip use: expand the target box by the annulus radius, then test neighbors.
        # A neighbor outside the tight box but inside the annulus MUST be caught — missing it is
        # precisely the contamination case the exclusion mask exists for.
        target = BoxYX.from_mask(_blob(cy=50, cx=50, half=5))
        neighbor = BoxYX.from_mask(_blob(cy=50, cx=68, half=5))
        assert not target.intersects(neighbor), "fixture must place the neighbor outside the box"
        assert target.pad(15, 15).intersects(neighbor), "annulus-expanded box must catch it"


def _blob(*, cy: int, cx: int, half: int, shape=(100, 100)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[cy - half:cy + half, cx - half:cx + half] = 1
    return mask
