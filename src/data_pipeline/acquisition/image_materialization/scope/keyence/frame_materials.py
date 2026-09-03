"""FrameMaterials — the product-independent half of Keyence materialization.

One Keyence well is materialized into several image products (``BF__projection__focus_stack``,
``BF__z_stack``, ...). Every one of those products is built from the SAME acquisition-derived facts:
the raw per-tile Z stacks, the focus reduction over them, and the tile transform that places the
tiles on a canvas. Deriving those facts once and handing them to every product is what this module
exists for.

This is NOT a separate pipeline phase. There is one operation — materialization — with a shared
portion (here) and product-specific emission (``materialize_well_keyence``). Only ``materialize``
names a pipeline operation; ``prepare`` and ``emit`` describe mechanics underneath it.

    materialize_keyence_products_for_well()
        ├── prepare_keyence_frame()  ->  FrameMaterials     (this module: read, focus, geometry)
        ├── emit_projection_product(materials, ...)
        └── emit_z_stack_product(materials, ...)

Import rules: imports the shared image primitives (``focus_stack_group``, ``frame_tiler``) and the
Keyence plane reader. It MUST NOT import write policies, materialized paths, the frame-inventory
contract, orchestration, or tasks — it knows nothing about products.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.acquisition.image_building.shared.focus_stack_group import (
    FocusStackConfig,
    FocusStackGroupResult,
    focus_stack_group,
)
from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    PreComputeStitchParams,
    TileSpec,
    TileTransform,
    UnstitchableFrameError,
    stitch_frame_tiles,
)
from data_pipeline.acquisition.image_materialization.scope.keyence.keyence_plane_io import (
    read_keyence_plane,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameMaterials:
    """Everything the products of ONE ``(channel_id, time_index)`` frame are built from.

    Held in memory and consumed by several emitters; deliberately not named "frame", because it is
    not one stitched ``(Y, X)`` image — it is the per-tile stacks plus the facts derived from them.

    Attributes:
        channel_id: the channel these materials belong to. Materials are NEVER shared across
            channels: pixel reductions and write policies are per-channel concerns.
        time_index: the timepoint these materials belong to.
        tile_ids: tile order. ``tile_stacks`` is parallel to this, and ``tile_transforms`` is keyed
            by these ids.
        tile_stacks: one ``(Z, Y, X)`` uint16 stack per tile, in ``tile_ids`` order. Every stack
            shares one ``(Z, Y, X)`` — asserted at construction, see ``z_indices``.
        z_indices: maps a position in the Z axis of every ``tile_stacks`` entry back to its real
            ``z_index``. ONE array, shared by all tiles: see ``_assert_uniform_z`` for why that is a
            requirement rather than a simplification.
        focus: the shared focus reduction (``projection_u8`` + per-tile ``focus_index_map``), or
            ``None`` when no product needed it (``compute_focus=False``).
        tile_transforms: the resolved per-tile placement, ALWAYS populated. Emitters consume this
            and never learn whether it was solved from this frame (``auto``) or loaded from the
            experiment-grain master map (``prior_only``) — that provenance is a geometry concern,
            and letting it leak would re-open the seam this module exists to draw.
    """

    channel_id: str
    time_index: int
    tile_ids: tuple[str, ...]
    tile_stacks: tuple[np.ndarray, ...]
    z_indices: np.ndarray
    focus: FocusStackGroupResult | None
    tile_transforms: dict[str, TileTransform]

    def stack_for(self, tile_id: str) -> np.ndarray:
        """Return the raw ``(Z, Y, X)`` stack for one tile."""
        return self.tile_stacks[self.tile_ids.index(str(tile_id))]

    def plane_for(self, tile_id: str, z_index: int) -> np.ndarray:
        """Return the raw ``(Y, X)`` plane for one tile at one REAL ``z_index``.

        Indexes through ``z_indices`` rather than positionally, so a stack whose planes were
        filtered cannot silently return the wrong depth.
        """
        positions = np.flatnonzero(self.z_indices == int(z_index))
        if positions.size != 1:
            raise ValueError(
                f"z_index={z_index!r} appears {positions.size} times in this frame's z_indices "
                f"{self.z_indices.tolist()} (channel={self.channel_id}, time_index="
                f"{self.time_index}); expected exactly one."
            )
        return self.stack_for(tile_id)[int(positions[0])]


def prepare_keyence_frame(
    *,
    frame_rows: pd.DataFrame,
    channel_id: str,
    time_index: int,
    well_id: str,
    tiling_config: FrameTilingConfig,
    fallback: PreComputeStitchParams,
    compute_focus: bool,
    device: str = "cpu",
    resolve_tiff=lambda p: Path(p),
    focus_config: FocusStackConfig | None = None,
) -> FrameMaterials:
    """Read and derive everything ONE frame's products share.

    Reads each required source plane exactly once, computes at most one focus reduction, and
    resolves exactly one tile transform — regardless of how many products will consume the result.

    Args:
        frame_rows: the acquisition-inventory rows for this ``(channel_id, time_index)`` only.
        compute_focus: compute the shared focus reduction. Says WHAT to do, not why: the caller
            derives it from the product plan (any projection product needs it; so does ``auto``
            geometry, which aligns on sharp composites).
        tiling_config: carries ``mode``, which is the SINGLE source of truth for how geometry is
            obtained — ``auto``/``align_only`` solve it from this frame, ``prior_only`` reads the
            experiment-grain master map. There is deliberately no separate ``determine_geometry``
            flag: two knobs over one decision could disagree, and the stitcher already branches on
            ``mode`` internally. Transforms are resolved either way.
        resolve_tiff: maps a stored inventory path to a readable path (re-anchoring onto the
            input root). Injected so this module does not import path-root helpers.

    Returns:
        ``FrameMaterials`` with ``tile_transforms`` always populated.

    Raises:
        ValueError: on an empty frame, a tile whose planes are all unreadable, or tiles that
            survive plane filtering at differing depths.
        UnstitchableFrameError: when geometry cannot be trusted.
    """
    if frame_rows.empty:
        raise ValueError(
            f"prepare_keyence_frame: no rows for well={well_id!r} channel={channel_id!r} "
            f"time_index={time_index!r}."
        )

    tile_ids, tile_stacks, per_tile_z = _read_tile_stacks(
        frame_rows, well_id=well_id, time_index=time_index, resolve_tiff=resolve_tiff
    )
    z_indices = _assert_uniform_z(
        tile_ids, per_tile_z, well_id=well_id, channel_id=channel_id, time_index=time_index
    )

    focus: FocusStackGroupResult | None = None
    if compute_focus:
        focus = focus_stack_group(
            list(tile_stacks), config=focus_config or FocusStackConfig(), device=device
        )

    tile_transforms = _resolve_tile_transforms(
        tile_ids=tile_ids,
        tile_stacks=tile_stacks,
        focus=focus,
        tiling_config=tiling_config,
        fallback=fallback,
        well_id=well_id,
        time_index=time_index,
    )

    return FrameMaterials(
        channel_id=str(channel_id),
        time_index=int(time_index),
        tile_ids=tile_ids,
        tile_stacks=tile_stacks,
        z_indices=z_indices,
        focus=focus,
        tile_transforms=tile_transforms,
    )


def _read_tile_stacks(
    frame_rows: pd.DataFrame,
    *,
    well_id: str,
    time_index: int,
    resolve_tiff,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], dict[str, np.ndarray]]:
    """Read every tile's Z stack ONCE, filtering planes and their z_index TOGETHER.

    Planes and their ``z_index`` are filtered together because ``focus_index_map`` indexes positions
    in the returned stack while ``z_indices`` maps those positions back to real ``z_index`` values.
    Dropping a plane without dropping its ``z_index`` desyncs the two and attributes every focus
    index after the gap to the wrong Z.
    """
    tile_ids: list[str] = []
    tile_stacks: list[np.ndarray] = []
    per_tile_z: dict[str, np.ndarray] = {}

    for tile_id, tile_rows in frame_rows.groupby("tile_id", sort=True):
        sorted_rows = tile_rows.sort_values("z_index")
        kept_planes: list[np.ndarray] = []
        kept_z: list[int] = []
        for row in sorted_rows.itertuples(index=False):
            # missing_ok=True: one corrupt plane must not abort an otherwise-good experiment. The
            # resulting raggedness is caught by _assert_uniform_z, which names the tile.
            image = read_keyence_plane(resolve_tiff(row.source_tiff_path), missing_ok=True)
            if image is None:
                continue
            kept_planes.append(image)
            kept_z.append(int(row.z_index))

        if not kept_planes:
            raise ValueError(
                f"All z-plane TIFFs unreadable for well={well_id} time_index={time_index} "
                f"tile_id={tile_id}."
            )

        tile_ids.append(str(tile_id))
        tile_stacks.append(np.stack(kept_planes, axis=0))
        per_tile_z[str(tile_id)] = np.asarray(kept_z, dtype=np.int32)

    return tuple(tile_ids), tuple(tile_stacks), per_tile_z


def _assert_uniform_z(
    tile_ids: tuple[str, ...],
    per_tile_z: dict[str, np.ndarray],
    *,
    well_id: str,
    channel_id: str,
    time_index: int,
) -> np.ndarray:
    """Assert every tile survived plane filtering with the IDENTICAL z_index sequence.

    Why this is a hard requirement and not a convenience: ``focus_stack_group`` composes the group
    into one ``(N, Z, Y, X)`` batch and rejects a non-uniform ``(Z, Y, X)``, so ragged tiles could
    never be focus-stacked anyway. Historically that downstream shape check was the ONLY thing
    standing between a corrupt plane and a mis-attributed focus index — it fired with a message
    about batch composition, naming neither the tile nor the dropped plane, and it did not fire at
    all on the ``prior_only`` path, which skips focus entirely.

    Asserting here makes the invariant explicit, names the offending tile, and covers every path.
    """
    reference_tile = tile_ids[0]
    reference_z = per_tile_z[reference_tile]

    for tile_id in tile_ids[1:]:
        z_indices = per_tile_z[tile_id]
        if z_indices.shape != reference_z.shape or not np.array_equal(z_indices, reference_z):
            missing = sorted(set(reference_z.tolist()) - set(z_indices.tolist()))
            extra = sorted(set(z_indices.tolist()) - set(reference_z.tolist()))
            raise ValueError(
                f"Ragged Keyence z planes for well={well_id} channel={channel_id} "
                f"time_index={time_index}: tile {tile_id!r} survived with z_indices "
                f"{z_indices.tolist()} but tile {reference_tile!r} has {reference_z.tolist()}"
                + (f"; unreadable in {tile_id!r}: {missing}" if missing else "")
                + (f"; unexpected in {tile_id!r}: {extra}" if extra else "")
                + ". Every tile of a frame must contribute the same Z planes — the tiles are "
                "composed into one batch and share one canvas."
            )

    return reference_z


def _resolve_tile_transforms(
    *,
    tile_ids: tuple[str, ...],
    tile_stacks: tuple[np.ndarray, ...],
    focus: FocusStackGroupResult | None,
    tiling_config: FrameTilingConfig,
    fallback: PreComputeStitchParams,
    well_id: str,
    time_index: int,
) -> dict[str, TileTransform]:
    """Resolve the frame's tile placement — solved or loaded, but ALWAYS returned.

    A single tile has nothing to align, so it gets an explicit identity transform rather than the
    ``None`` the stitcher short-circuits to: emitters index this mapping unconditionally, and a
    field whose type varies with tile count would push that branch onto every consumer.

    For multi-tile frames the transform comes from ``stitch_frame_tiles``, which returns
    ``tile_transforms`` under every mode — solving them under ``auto``/``align_only`` and reading
    them from the experiment-grain master map under ``prior_only``. Geometry is a property of the
    FRAME (the stage does not translate as it steps through focus), so it is derived once here from
    the sharpest images available and applied verbatim to every plane and every product.
    """
    if len(tile_ids) == 1:
        return {tile_ids[0]: TileTransform(tile_ids[0], 0.0, 0.0, source="identity")}

    if focus is not None:
        # Align on the focus composites: sharp and feature-rich. Out-of-focus planes carry too few
        # descriptors to align reliably, and a plane yielding <2 aborts OpenCV's FLANN matcher.
        images = [focus.tiles[i].projection_u8 for i in range(len(tile_ids))]
    elif tiling_config.mode == "prior_only":
        # Coords come from the master map (run_align=False), so the pixels only need to be
        # correctly shaped — no feature matching happens on them. First plane is the cheapest.
        images = [stack[0] for stack in tile_stacks]
    else:
        raise ValueError(
            f"Cannot determine geometry for well={well_id} time_index={time_index}: tiling mode "
            f"{tiling_config.mode!r} aligns from image content, but no focus reduction was "
            "computed. Either request focus (compute_focus=True) or use mode='prior_only'. "
            "Aligning raw out-of-focus planes is exactly the failure this refusal prevents."
        )

    specs = [TileSpec(tile_id=tid, image=images[i]) for i, tid in enumerate(tile_ids)]
    result = stitch_frame_tiles(specs, tiling_config, fallback)
    if not result.qc.passed:
        raise UnstitchableFrameError(
            f"Frame-geometry stitch failed for well={well_id} time_index={time_index}: "
            f"reasons={result.qc.reasons} fallback_used={result.fallback_used}. Refusing to "
            f"materialize any product against untrustworthy geometry."
        )
    return result.tile_transforms
