"""Synthetic embryo masks for orientation tests.

Real embryo masks are not checked in, and a test that needs a fixture file cannot assert
much about the geometry it is testing. These builders make the anatomy explicit: an
elongated body at a known angle, with an optional yolk blob at a known end and a dorsal
bulge on a known side. A test can therefore state what the right answer IS, not merely
that two implementations agree.
"""

from __future__ import annotations

import numpy as np


def _ellipse(shape_yx, center_yx, radii_yx, angle_deg=0.0):
    h, w = shape_yx
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dy = ys - float(center_yx[0])
    dx = xs - float(center_yx[1])
    t = np.deg2rad(float(angle_deg))
    # Rotate the sample grid into the ellipse's own frame.
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    ry, rx = float(radii_yx[0]), float(radii_yx[1])
    return ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0


def make_embryo(
    shape_yx=(200, 400),
    center_yx=(100.0, 200.0),
    body_radii_yx=(22.0, 110.0),
    angle_deg=0.0,
    *,
    yolk_at="left",
    yolk_radius=26.0,
    yolk_offset=70.0,
    dorsal_side="up",
    dorsal_bulge=True,
):
    """Build (embryo_mask, yolk_mask) with a known head end and a known dorsal side.

    ``yolk_at`` places the yolk toward the ``"left"`` or ``"right"`` end of the body axis
    BEFORE rotation; ``dorsal_side`` puts a bulge ``"up"`` or ``"down"`` from the yolk, so
    the back point is unambiguous. ``yolk_at=None`` returns an all-zero yolk mask, which is
    the empty-yolk production case on experiment 20250416.
    """
    body = _ellipse(shape_yx, center_yx, body_radii_yx, angle_deg)

    t = np.deg2rad(float(angle_deg))
    axis = np.array([np.cos(t), np.sin(t)])  # xy, along the body
    normal = np.array([-np.sin(t), np.cos(t)])  # xy, across the body

    sign = {"left": -1.0, "right": +1.0, None: 0.0}[yolk_at]
    cx, cy = float(center_yx[1]), float(center_yx[0])
    yx = cx + sign * float(yolk_offset) * axis[0]
    yy = cy + sign * float(yolk_offset) * axis[1]

    mask = body.copy()
    if dorsal_bulge and yolk_at is not None:
        d = {"up": -1.0, "down": +1.0}[dorsal_side]
        # Normal points toward +y in image coords when angle=0, so "up" is -normal.
        bx = yx + d * 18.0 * normal[0]
        by = yy + d * 18.0 * normal[1]
        mask = mask | _ellipse(shape_yx, (by, bx), (26.0, 30.0), angle_deg)

    if yolk_at is None:
        yolk = np.zeros(shape_yx, dtype=np.uint8)
    else:
        yolk = (_ellipse(shape_yx, (yy, yx), (yolk_radius, yolk_radius), angle_deg) & mask).astype(np.uint8)

    return mask.astype(np.uint8), yolk


def orientation_cases():
    """A spread of (name, embryo, yolk) covering both yolk ends, both dorsal sides, and
    several body angles -- plus the no-yolk case."""
    cases = []
    for angle in (0.0, 12.0, -25.0, 40.0, 88.0, 155.0):
        for yolk_at in ("left", "right"):
            for dorsal in ("up", "down"):
                m, y = make_embryo(angle_deg=angle, yolk_at=yolk_at, dorsal_side=dorsal)
                cases.append((f"a{angle:g}_yolk{yolk_at}_dorsal{dorsal}", m, y))
    for angle in (0.0, 30.0, -60.0):
        m, y = make_embryo(angle_deg=angle, yolk_at=None)
        cases.append((f"a{angle:g}_noyolk", m, y))
    return cases
