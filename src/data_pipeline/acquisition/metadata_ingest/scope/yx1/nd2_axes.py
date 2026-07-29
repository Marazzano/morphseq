"""ND2 AXES — the ONE place ND2 dimensions are read, always BY NAME.

===========================================================================================
THE ND2 MENTAL MODEL — what each index encodes. Read this before touching ND2 code.
===========================================================================================

An ND2 has **two different kinds of axis**, and conflating them is the bug this module exists to
prevent.

**1. SEQUENCE axes — what ``frame_metadata(i)`` / ``read_frame(i)`` address.**
``nd.experiment`` lists the loops the microscope actually RAN, outermost first::

    TimeLoop    -> T   the time axis          (ABSENT for a snapshot)
    XYPosLoop   -> P   stage positions        (the plate scan)
    ZStackLoop  -> Z   focal planes

``frameCount == product of the loop counts``. A flat frame index ranges over THESE AXES ONLY.

**2. WITHIN-FRAME axes — C, Y, X.**
``C`` is **NOT** a sequence axis: one frame carries ALL of its channels
(``frame_metadata(0).channels`` returns them together). ``Y``/``X`` are the pixel grid. These
describe a frame's CONTENTS, never its address.

Verified on three real files (note ``frameCount`` excludes C every time)::

    pbx pilot        loops=[XYPos 96, ZStack 9]            frameCount=864     sizes={P:96, Z:9, C:2,  Y, X}
    wikAB timecourse loops=[Time 64, XYPos 71, ZStack 21]  frameCount=95424   sizes={T:64, P:71, Z:21, Y, X}
    lmx1b timeseries loops=[Time 59, XYPos 89, ZStack 27]  frameCount=141777  sizes={T:59, P:89, Z:27, Y, X}

So the two questions have two DIFFERENT sources::

    "how many metadata rows?"  -> sizes  (P x Z x C — C included: one row per channel)
    "which frame do I read?"   -> experiment loops  (P x Z — C excluded)

===========================================================================================
THE TWO BUGS THIS PREVENTS
===========================================================================================

**(a) Positional unpacking.** ``nd.shape`` is a bare tuple whose axes vary per acquisition. The pbx
pilot is ``(P, Z, C, Y, X)`` — no T at all — so reading ``shape[:3]`` as ``(T, W, Z)`` gave::

    P=96 (positions)    read as T  ->  96 wells became 96 timepoints
    Z=9  (focal planes) read as W  ->  9 planes became 9 positions
    C=2  (channels)     read as Z  ->  2 channels became 2 planes

...and 1728 plausible-looking rows, which is why it went unnoticed. ``nd.sizes`` is a NAME->size
mapping, so it cannot be misread that way.

**(b) Hand-rolled frame arithmetic.** ``idx = position * n_z * n_c`` bakes in both an axis ORDER and
the false assumption that C is addressable. ``frame_index`` strides over the SEQUENCE axes in the
file's real loop order, so a missing, reordered, or within-frame axis cannot corrupt a lookup.

Scope boundary: this is ND2-specific and therefore lives under ``scope/yx1/``. Keyence never uses it
— its dimensions come from per-plane TIFF path tokens (``XY##``/``T####``/``Z###``), which are already
name-anchored. Per the pipeline's per-scope backend doctrine, the two scopes share no dimension code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

# ND2 axis labels this reader understands. P is the position/series axis; W is its LEGACY alias
# (older files, and what some nd2 versions report) — both mean "position".
_POSITION_KEYS = ("P", "W")
_SPATIAL_KEYS = ("Y", "X")
# Every non-spatial axis that can appear in `sizes`. These set the metadata ROW grain.
_GRAIN_KEYS = ("T", "P", "W", "Z", "C")
# The SEQUENCE axes — the only ones a flat frame index ranges over. C is deliberately absent: one
# frame carries all of its channels, so C describes frame CONTENTS, not a frame address.
_SEQUENCE_KEYS = ("T", "P", "W", "Z")

# nd.experiment loop class name -> the axis it drives. This is the AUTHORITATIVE sequence structure
# (frameCount == product of the loop counts), so frame addressing is derived from it, not from sizes.
_LOOP_TYPE_TO_AXIS = {
    "TimeLoop": "T",
    "NETimeLoop": "T",  # "non-equidistant" time loop — still the time axis
    "XYPosLoop": "P",
    "ZStackLoop": "Z",
}


@dataclass(frozen=True)
class Nd2Axes:
    """The named dimensions of one ND2, plus order-aware flat frame indexing.

    ``n_t``/``n_p``/``n_z``/``n_c`` are 1 when the corresponding axis is ABSENT — an ND2 with no T
    axis is a single-timepoint (snapshot) acquisition, not an error. ``frame_order`` is the real
    order of the non-spatial axes as the file reports them, and is what makes ``frame_index`` safe.
    """

    n_t: int
    n_p: int
    n_z: int
    n_c: int
    height_px: int
    width_px: int
    # The SEQUENCE axes in loop order (outermost first) — from nd.experiment when available, else
    # inferred from `sizes`. C is never here. This is what frame addressing strides over.
    sequence_order: tuple[str, ...]

    @property
    def is_snapshot(self) -> bool:
        """True when the file carries a single timepoint (no T axis, or T of length 1)."""
        return self.n_t <= 1

    @property
    def n_rows(self) -> int:
        """Metadata ROW count: one row per (position, timepoint, z, channel). C IS included."""
        return self.n_t * self.n_p * self.n_z * self.n_c

    @property
    def n_frames(self) -> int:
        """Addressable FRAME count — the product of the SEQUENCE axes. C is excluded.

        Equals the ND2's own ``metadata.contents.frameCount``; a mismatch means the axis model is
        wrong for that file.
        """
        count = 1
        for axis in self.sequence_order:
            count *= self.size_of(axis)
        return count

    def size_of(self, axis: str) -> int:
        return {
            "T": self.n_t,
            "P": self.n_p,
            "W": self.n_p,
            "Z": self.n_z,
            "C": self.n_c,
        }[axis]

    def frame_index(self, *, position: int = 0, time: int = 0, z: int = 0) -> int:
        """Flat frame index for one SEQUENCE coordinate, in the file's real loop order.

        Takes no ``channel``: a frame carries all of its channels at once, so C is not part of a
        frame address. Ask for the frame, then pick the channel out of it.

        Strides come from ``sequence_order`` (outermost axis varies slowest), so this is correct for
        ``(P, Z)``, ``(T, P, Z)``, or any other order the file declares — unlike
        ``position * n_z * n_c``, which is wrong on both order and the inclusion of C.

        Coordinates for axes the file does not have must be 0 (there is nothing to index).
        """
        requested = {"T": time, "P": position, "W": position, "Z": z}
        for axis, value in requested.items():
            limit = self.size_of(axis)
            if not (0 <= value < limit):
                raise IndexError(
                    f"Nd2Axes.frame_index: {axis}={value} is out of range for this ND2 "
                    f"({axis} has size {limit}; sequence order {self.sequence_order}). An absent "
                    "axis has size 1, so its only valid coordinate is 0."
                )

        index = 0
        for axis in self.sequence_order:
            index = index * self.size_of(axis) + requested[axis]
        return index


def read_nd2_axes(
    sizes: Mapping[str, int], loop_type_names: Sequence[str] | None = None
) -> Nd2Axes:
    """Build ``Nd2Axes`` from an ND2's ``sizes`` mapping, and its ``experiment`` loop order.

    Takes plain data (a mapping + a list of loop class names) rather than the open file, so this is a
    pure function — unit-testable against every real axis layout without an ND2 on disk.

    Args:
        sizes: ``nd.sizes``, e.g. ``{"P": 96, "Z": 9, "C": 2, "Y": 2304, "X": 2304}``. Sets the
            metadata ROW grain (C included).
        loop_type_names: the class names of ``nd.experiment``, outermost first, e.g.
            ``["XYPosLoop", "ZStackLoop"]``. This is the AUTHORITATIVE sequence order for frame
            addressing. When omitted, the sequence order is inferred from ``sizes`` (C excluded),
            which is correct for the canonical T/P/Z ordering but cannot detect an unusual one —
            always pass it when an open file is available.

    Raises:
        ValueError: if Y/X are missing (not an image), an axis label is unrecognized, both P and W
            are present, or a declared loop names an axis that ``sizes`` does not have.
    """
    missing_spatial = [key for key in _SPATIAL_KEYS if key not in sizes]
    if missing_spatial:
        raise ValueError(
            f"read_nd2_axes: ND2 sizes {dict(sizes)} is missing spatial axis/axes "
            f"{missing_spatial}. Every ND2 frame must declare Y and X."
        )

    unknown = [key for key in sizes if key not in _GRAIN_KEYS + _SPATIAL_KEYS]
    if unknown:
        raise ValueError(
            f"read_nd2_axes: unrecognized ND2 axis label(s) {unknown} in sizes {dict(sizes)}. "
            f"Known axes: {_GRAIN_KEYS + _SPATIAL_KEYS}. Add the axis to this reader rather than "
            "ignoring it — an unmapped axis means frames would be indexed wrongly."
        )

    if "P" in sizes and "W" in sizes:
        raise ValueError(
            f"read_nd2_axes: ND2 sizes {dict(sizes)} declares BOTH 'P' and 'W' position axes. They "
            "are aliases for the same axis; a file carrying both is ambiguous."
        )

    if loop_type_names is None:
        # Fall back to the order `sizes` reports, minus C (never addressable).
        sequence_order = tuple(key for key in sizes if key in _SEQUENCE_KEYS)
    else:
        sequence_order = ()
        for loop_name in loop_type_names:
            axis = _LOOP_TYPE_TO_AXIS.get(str(loop_name))
            if axis is None:
                raise ValueError(
                    f"read_nd2_axes: unrecognized ND2 experiment loop {loop_name!r}. Known loops: "
                    f"{sorted(_LOOP_TYPE_TO_AXIS)}. The loops define frame addressing, so an "
                    "unmapped loop must be added here rather than skipped."
                )
            # A loop for an axis `sizes` does not report would make frame_index stride over nothing.
            if axis == "P":
                has_axis = any(key in sizes for key in _POSITION_KEYS)
            else:
                has_axis = axis in sizes
            if not has_axis:
                raise ValueError(
                    f"read_nd2_axes: experiment declares {loop_name!r} (axis {axis!r}) but sizes "
                    f"{dict(sizes)} has no such axis. The loop structure and the array dimensions "
                    "disagree; refusing to guess a frame layout."
                )
            sequence_order += (axis,)

    n_position = next((int(sizes[key]) for key in _POSITION_KEYS if key in sizes), 1)
    # An absent axis has extent 1: a file with no T is a single-timepoint acquisition (a snapshot),
    # which is normal for these plate scans — not an error.
    return Nd2Axes(
        n_t=int(sizes.get("T", 1)),
        n_p=n_position,
        n_z=int(sizes.get("Z", 1)),
        n_c=int(sizes.get("C", 1)),
        height_px=int(sizes["Y"]),
        width_px=int(sizes["X"]),
        sequence_order=sequence_order,
    )


def select_zyx_stack(
    array,
    axes: Nd2Axes,
    array_axis_order: Sequence[str],
    *,
    position: int,
    time: int = 0,
    channel: int = 0,
):
    """Slice a (Z, Y, X) stack for ONE (position, time, channel) out of an ND2 array.

    ``nd.to_dask()``/``nd.asarray()`` return an array whose axis order is the file's own — e.g.
    ``(P, Z, C, Y, X)`` for a snapshot with channels, or ``(T, P, Z, C, Y, X)`` for a timelapse.
    Indexing it positionally (``array[t, w, :, :, :]``) is only correct for ONE of those layouts. On
    the pbx pilot ``(P, Z, C, Y, X)`` that expression reads position as T, z-plane as the position,
    and leaves the 2-channel axis standing in for the Z stack — so focus-stacking would run over
    ``[BF, tdTomato]`` as if they were focal planes. Silent pixel corruption, not a crash.

    This builds the slice from axis NAMES, so any layout works and a missing axis is simply absent.

    Args:
        array: the ND2 array (dask or numpy) as returned by ``to_dask()``/``asarray()``.
        axes: the file's named axes.
        array_axis_order: the array's axis labels in order, e.g. ``("P", "Z", "C", "Y", "X")``.
        position/time/channel: the coordinate to extract.

    Returns:
        The ``(Z, Y, X)`` stack — single-channel, with Z first, whatever the file's layout.
    """
    coordinate = {"T": time, "P": position, "W": position, "C": channel}
    order = tuple(str(axis) for axis in array_axis_order)

    for spatial in _SPATIAL_KEYS:
        if spatial not in order:
            raise ValueError(
                f"select_zyx_stack: array axis order {order} has no {spatial!r} axis; expected the "
                "ND2 array's own axis labels (e.g. ('P','Z','C','Y','X'))."
            )
    if "Z" not in order:
        raise ValueError(
            f"select_zyx_stack: array axis order {order} has no 'Z' axis, so there is no focal stack "
            "to slice. A single-plane acquisition must be handled by the caller."
        )

    selector = []
    for axis in order:
        if axis in ("Y", "X", "Z"):
            selector.append(slice(None))  # keep the stack + pixel grid
        elif axis in coordinate:
            selector.append(coordinate[axis])
        else:
            raise ValueError(
                f"select_zyx_stack: unrecognized array axis {axis!r} in order {order}."
            )

    stack = array[tuple(selector)]
    # After integer-indexing every non-(Z,Y,X) axis, exactly (Z, Y, X) remains.
    if stack.ndim != 3:
        raise ValueError(
            f"select_zyx_stack: expected a 3-D (Z, Y, X) stack after selection but got ndim="
            f"{stack.ndim} from array order {order}. The axis model does not fit this array."
        )
    return stack


def array_axis_order_of(nd) -> tuple[str, ...]:
    """The axis labels of ``nd.to_dask()``/``nd.asarray()``, in order — from ``nd.sizes``.

    ``nd.sizes`` is an ordered mapping matching the array's own axis order, so its keys ARE the
    array's axis labels. Use this with ``select_zyx_stack`` rather than assuming a layout.
    """
    return tuple(str(key) for key in nd.sizes)


def axes_of(nd) -> Nd2Axes:
    """Read ``Nd2Axes`` from an OPEN ``nd2.ND2File`` — the entry point callers should use.

    Passes the experiment loop order through (so frame addressing follows the file's real loop
    structure) and cross-checks the derived frame count against the ND2's own ``frameCount``. That
    check is the tripwire: if the two disagree, the axis model is wrong for this file and every frame
    lookup would be silently off — exactly the failure that motivated this module.
    """
    axes = read_nd2_axes(nd.sizes, [type(loop).__name__ for loop in nd.experiment])

    declared = getattr(getattr(nd, "metadata", None), "contents", None)
    declared_frames = getattr(declared, "frameCount", None)
    if declared_frames is not None and int(declared_frames) != axes.n_frames:
        raise ValueError(
            f"axes_of: derived frame count {axes.n_frames} (sequence axes {axes.sequence_order}) "
            f"does not match the ND2's declared frameCount {int(declared_frames)}. sizes="
            f"{dict(nd.sizes)}. The axis model does not fit this file; frame lookups would be wrong."
        )
    return axes
