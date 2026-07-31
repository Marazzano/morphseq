"""The resize coordinate convention — pinned before any code depends on it.

WHY THIS FILE EXISTS. Removing the integer-centroid latch from the snip pipeline means computing a
crop center in source coordinates and mapping it through the resize analytically. Get that mapping
wrong and you trade a one-pixel latch for a *systematic sub-pixel bias* — which is strictly worse,
because it is invisible: every snip shifts by the same amount, so pixel diffs, SSIM, and mean-error
all stay clean while every embryo moves relative to the historical embedding space.

The naive rule ``x_out = sf * x_src`` is wrong by exactly ``(sf-1)/2``: -0.375 px at sf=0.25,
-0.25 px at sf=0.5. The correct rule treats coordinates as PIXEL CENTERS:

    x_out = sf * (x_src + 0.5) - 0.5          (forward)
    x_src = (x_out + 0.5) / sf - 0.5          (inverse)

Verified exact to float64 for cv2 INTER_AREA, cv2 INTER_LINEAR, and skimage.rescale. The convention
is COMMON TO BOTH ENGINES, so switching kernels changes pixel values but never coordinate semantics
— which is what makes the kernel migration safe to reason about separately.

HOW IT IS MEASURED. A linear ramp ``v(x) = x``. Linear interpolation of a linear function is exact,
so an output pixel's VALUE is literally the source coordinate it sampled: ``out[j] == s(j)``. That
reads the mapping directly. (A single bright pixel does NOT work — at strong downscale it lands
inside one output pixel and its measured position quantizes to that pixel's integer index, which
cannot resolve sub-pixel offsets at all.)
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from skimage.transform import rescale

N = 64


def _ramp(n: int = N) -> np.ndarray:
    """v(x) = x, so a resized pixel's value IS the source coordinate it sampled."""
    return np.tile(np.arange(n, dtype=np.float32), (n, 1))


def _sampled_source_coords(engine: str, out_n: int, n: int = N) -> np.ndarray:
    ramp = _ramp(n)
    if engine == "area":
        out = cv2.resize(ramp, (out_n, out_n), interpolation=cv2.INTER_AREA)
    elif engine == "linear":
        out = cv2.resize(ramp, (out_n, out_n), interpolation=cv2.INTER_LINEAR)
    elif engine == "skimage":
        out = rescale(ramp, (out_n / n, out_n / n), order=1, preserve_range=True,
                      anti_aliasing=False)
    else:  # pragma: no cover
        raise ValueError(engine)
    return out[0, :]


ENGINES = ["area", "linear", "skimage"]


def forward(x_src: float, sf: float) -> float:
    """Source coordinate -> output coordinate, pixel-center convention."""
    return sf * (x_src + 0.5) - 0.5


def inverse(x_out: float, sf: float) -> float:
    """Output coordinate -> source coordinate, pixel-center convention."""
    return (x_out + 0.5) / sf - 0.5


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("out_n", [16, 32])  # sf = 0.25, 0.5 — exact ratios
def test_half_pixel_rule_is_exact(engine, out_n):
    sf = out_n / N
    sampled = _sampled_source_coords(engine, out_n)
    for j in (0, 1, 2, out_n - 1):
        assert sampled[j] == pytest.approx(inverse(j, sf), abs=1e-4), (
            f"{engine} at sf={sf}: output pixel {j} sampled source {sampled[j]}, "
            f"expected {inverse(j, sf)} under the pixel-center convention"
        )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("out_n", [16, 32])
def test_naive_rule_is_wrong_by_half_a_pixel(engine, out_n):
    # Pinned as a REGRESSION GUARD: if someone "simplifies" the mapping to sf*x, this fires.
    sf = out_n / N
    sampled = _sampled_source_coords(engine, out_n)
    naive_inverse = 1 / sf  # what (j)/sf would give for j=1
    assert sampled[1] != pytest.approx(naive_inverse, abs=1e-3)
    # The error is exactly (1-sf)/(2*sf) in source units — a systematic bias, not noise.
    assert sampled[1] - naive_inverse == pytest.approx((1 - sf) / (2 * sf), abs=1e-4)


@pytest.mark.parametrize("out_n", [16, 32])
def test_all_engines_agree_on_the_convention(out_n):
    # THE migration-safety property: switching kernels changes pixel VALUES, never coordinate
    # SEMANTICS. If this ever fails, the kernel swap stops being separable from the geometry work.
    coords = [_sampled_source_coords(e, out_n) for e in ENGINES]
    for other in coords[1:]:
        assert np.allclose(coords[0], other, atol=1e-4)


def test_forward_and_inverse_round_trip():
    for sf in (0.25, 0.5, 0.4142, 2.0):
        for x in (0.0, 7.0, 31.5, 63.0):
            assert inverse(forward(x, sf), sf) == pytest.approx(x, abs=1e-9)


class TestRealizedScaleFactor:
    """The effective scale is out_n/in_n — NOT the requested factor. This bites at real scales."""

    def test_requested_scale_is_not_the_realized_scale(self):
        # Production: 3.2308 -> 7.8 um/px asks for sf=0.4142, but round(64*0.4142)=27, so the
        # realized ratio is 27/64 = 0.4219. Mapping coordinates with the REQUESTED factor
        # reintroduces exactly the class of systematic bias this file exists to prevent.
        requested = 0.4142
        out_n = int(round(N * requested))
        realized = out_n / N
        assert realized != pytest.approx(requested, abs=1e-3)

        sampled = _sampled_source_coords("area", out_n)
        # The engine follows the REALIZED ratio.
        assert sampled[1] == pytest.approx(inverse(1, realized), abs=0.05)
        # ...and disagrees with the requested one by a visible margin.
        assert abs(sampled[1] - inverse(1, requested)) > 0.05

    def test_resize_step_derives_scale_from_shapes(self):
        # image_geometry.resize_step computes scale from in/out SHAPES rather than storing a
        # requested factor, so it is correct by construction here. Pinned so a refactor that
        # "helpfully" accepts a scale argument cannot silently reintroduce the bias.
        from image_geometry import resize_step

        step = resize_step(in_shape_yx=(N, N), out_shape_yx=(27, 27))
        assert step.params["scale_y"] == pytest.approx(27 / N)
        assert step.params["scale_x"] == pytest.approx(27 / N)
