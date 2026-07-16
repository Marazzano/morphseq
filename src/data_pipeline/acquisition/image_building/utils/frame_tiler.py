"""Pure utility for stitching Keyence frame tiles with deterministic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import skimage.util as skutil


Orientation = Literal["vertical", "horizontal"]
TilingMode = Literal["auto", "prior_only", "align_only"]
FallbackStep = Literal["per_frame", "master", "concat"]
FallbackUsed = Literal["none", "per_frame", "master", "concat"]


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    image: np.ndarray


@dataclass(frozen=True)
class TileTransform:
    tile_id: str
    dx_px: float
    dy_px: float
    source: Literal["identity", "align", "fallback"]
    confidence: float | None = None


@dataclass(frozen=True)
class TilingQC:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    suggested_action: Literal["ok", "use_master", "concat", "fail"]


@dataclass(frozen=True)
class FrameTilingConfig:
    orientation: Orientation
    mode: TilingMode = "auto"
    fallback_policy: tuple[FallbackStep, ...] = ("per_frame", "master", "concat")
    max_abs_shift_px: float = 500.0
    enable_alignment: bool = True
    transpose_after_stitch: bool = True
    use_legacy_canvas: bool = True
    compat_postprocess: bool = True
    # DEPRECATED / INERT: display-polarity inversion is no longer done in the stitcher. It moved to
    # the shared image-math layer (apply_display_polarity), applied once post-composition by the
    # materializer so every microscope shares one polarity. Field kept only for back-compat of
    # existing FrameTilingConfig(...) call sites; it has no effect on stitch output.
    invert_intensity: bool = True


@dataclass(frozen=True)
class PreComputeStitchParams:
    """Pre-computed stitch coordinates consumed per-well (experiment-grain, generated once).

    ``master_params_path``: path to the experiment-grain ``master_params.json`` produced by
    ``build_keyence_stitch_map``. When set, tile coords are read from this file and alignment
    is skipped (``run_align=False``). ``None`` = align every frame from scratch via stitch2d.
    ``per_frame_params_path``: optional per-frame override coords (rarely used; kept for legacy compat).
    """
    master_params_path: Path | None = None
    per_frame_params_path: Path | None = None


# Legacy alias — remove once materialize_stitched_images.py is strangled.
FallbackParams = PreComputeStitchParams


@dataclass(frozen=True)
class FrameTileResult:
    stitched: np.ndarray
    tile_transforms: dict[str, TileTransform]
    canvas_shape: tuple[int, int]
    qc: TilingQC
    fallback_used: FallbackUsed


def legacy_canvas_shape(
    n_tiles: int,
    orientation: Orientation,
    tile_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if n_tiles <= 1:
        return (0, 0)
    shape_map = {
        2: np.array([800, 630]),
        3: np.array([1140, 630]) if orientation == "vertical" else np.array([1140, 480]),
    }
    if n_tiles not in shape_map:
        return (0, 0)
    target = shape_map[n_tiles].astype(float)
    if tile_shape is not None:
        tile_width = float(tile_shape[1])
        size_factor = tile_width / 640.0 if tile_width > 0 else 1.0
        if np.isfinite(size_factor) and size_factor > 0:
            target = target * size_factor
    return tuple(np.round(target).astype(int))


def stitch_frame_tiles(
    tile_specs: Sequence[TileSpec],
    config: FrameTilingConfig,
    fallback: FallbackParams | None = None,
) -> FrameTileResult:
    tiles = _prepare_tile_specs(tile_specs)
    if not tiles:
        raise ValueError("No tile specs provided")

    if len(tiles) == 1:
        stitched = _finalize_image(
            tiles[0].image,
            config=config,
            n_tiles=1,
            tile_shape=tiles[0].image.shape[:2],
        )
        qc = TilingQC(
            passed=True,
            reasons=tuple(),
            metrics={"tile_count": 1.0, "max_abs_shift_px": 0.0},
            suggested_action="ok",
        )
        tr = {tiles[0].tile_id: TileTransform(tiles[0].tile_id, 0.0, 0.0, source="identity")}
        return FrameTileResult(
            stitched=stitched,
            tile_transforms=tr,
            canvas_shape=stitched.shape[:2],
            qc=qc,
            fallback_used="none",
        )

    fallback = fallback or FallbackParams()

    # Load master coords (if available) up front — used both as a fallback stitch source and as
    # the plausibility reference (ground truth) for QC on ANY per-frame align result. Comparing
    # against master, not an absolute-canvas-position threshold, is the fix for the legitimate
    # dy≈tile-pitch stacking offset of a correctly-aligned multi-tile strip being misflagged by
    # ``max_abs_shift_px``.
    master_coords: dict[int, list[float]] | None = None
    if fallback.master_params_path is not None and fallback.master_params_path.exists():
        try:
            master_coords = _load_master_coords(fallback.master_params_path)
        except Exception:
            master_coords = None

    if config.mode != "prior_only" and config.enable_alignment:
        try:
            stitched_align, tr_align = _stitch_with_stitch2d(
                tiles=tiles,
                orientation=config.orientation,
                load_params_path=None,
                run_align=True,
            )
            qc_align = _run_tiling_qc(tr_align, config, master_coords=master_coords, tiles=tiles)
            if qc_align.passed or config.mode == "align_only":
                layout_orientation = _infer_layout_orientation(
                    tr_align,
                    fallback_orientation=config.orientation,
                )
                stitched = _finalize_image(
                    stitched_align,
                    config=config,
                    n_tiles=len(tiles),
                    tile_shape=tiles[0].image.shape[:2],
                    layout_orientation=layout_orientation,
                )
                return FrameTileResult(
                    stitched=stitched,
                    tile_transforms=tr_align,
                    canvas_shape=stitched.shape[:2],
                    qc=qc_align,
                    fallback_used="none",
                )
            # Alignment ran and placed all tiles but the result is implausible vs. master —
            # fall through to the master/fail path below.
        except IncompleteTileAlignmentError:
            # Legitimate stitch2d behavior: a weak pairwise feature-match can drop a tile even
            # though the raster layout fully determines its position. NOT a hard failure — fall
            # through to the master fallback (or loud failure if no master).
            pass

    for step in config.fallback_policy:
        if step == "per_frame" and fallback.per_frame_params_path is not None and fallback.per_frame_params_path.exists():
            try:
                stitched_pf, tr_pf = _stitch_with_stitch2d(
                    tiles=tiles,
                    orientation=config.orientation,
                    load_params_path=fallback.per_frame_params_path,
                    run_align=False,
                )
                qc_pf = _run_tiling_qc(tr_pf, config, master_coords=master_coords, tiles=tiles)
                layout_orientation = _infer_layout_orientation(
                    tr_pf,
                    fallback_orientation=config.orientation,
                )
                stitched = _finalize_image(
                    stitched_pf,
                    config=config,
                    n_tiles=len(tiles),
                    tile_shape=tiles[0].image.shape[:2],
                    layout_orientation=layout_orientation,
                )
                return FrameTileResult(
                    stitched=stitched,
                    tile_transforms=tr_pf,
                    canvas_shape=stitched.shape[:2],
                    qc=qc_pf,
                    fallback_used="per_frame",
                )
            except Exception:
                pass

        if step == "master" and fallback.master_params_path is not None and fallback.master_params_path.exists():
            stitched_master, tr_master = _stitch_with_stitch2d(
                tiles=tiles,
                orientation=config.orientation,
                load_params_path=fallback.master_params_path,
                run_align=False,
            )
            qc_master = _run_tiling_qc(tr_master, config, master_coords=master_coords, tiles=tiles)
            layout_orientation = _infer_layout_orientation(
                tr_master,
                fallback_orientation=config.orientation,
            )
            stitched = _finalize_image(
                stitched_master,
                config=config,
                n_tiles=len(tiles),
                tile_shape=tiles[0].image.shape[:2],
                layout_orientation=layout_orientation,
            )
            return FrameTileResult(
                stitched=stitched,
                tile_transforms=tr_master,
                canvas_shape=stitched.shape[:2],
                qc=qc_master,
                fallback_used="master",
            )

        if step == "concat":
            # Concat is diagnostics-only from here on — see the loud-failure raise below.
            # Never returned as the normal-flow result: an unstitchable frame with no master
            # must fail loud, not silently write a dumb stack that passes downstream QC.
            continue

    raise UnstitchableFrameError(
        f"Could not stitch {len(tiles)} tiles: per-frame alignment was incomplete/implausible "
        f"and no usable master_params_path fallback was available "
        f"(master_params_path={fallback.master_params_path})."
    )


def _prepare_tile_specs(tile_specs: Sequence[TileSpec]) -> list[TileSpec]:
    out: list[TileSpec] = []
    for spec in tile_specs:
        if not isinstance(spec.image, np.ndarray):
            raise TypeError(f"Tile '{spec.tile_id}' image must be numpy.ndarray")
        if spec.image.ndim not in (2, 3):
            raise ValueError(f"Tile '{spec.tile_id}' image must be 2D or 3D")
        image = spec.image if spec.image.dtype == np.uint8 else skutil.img_as_ubyte(spec.image)
        out.append(TileSpec(tile_id=str(spec.tile_id), image=image))
    out.sort(key=lambda item: item.tile_id)
    return out


def _init_transforms_from_priors(tiles: Sequence[TileSpec]) -> dict[str, TileTransform]:
    return {
        tile.tile_id: TileTransform(tile_id=tile.tile_id, dx_px=0.0, dy_px=0.0, source="identity")
        for tile in tiles
    }


def _coords_to_transforms(tiles: Sequence[TileSpec], coords: dict) -> dict[str, TileTransform]:
    out: dict[str, TileTransform] = {}
    for idx, tile in enumerate(tiles):
        value = coords.get(idx, coords.get(str(idx), (0.0, 0.0)))
        # stitch2d stores coords as [y, x]. TileTransform exposes dx/dy in image axes.
        y_val = float(value[0]) if len(value) > 0 else 0.0
        x_val = float(value[1]) if len(value) > 1 else 0.0
        out[tile.tile_id] = TileTransform(
            tile_id=tile.tile_id,
            dx_px=x_val,
            dy_px=y_val,
            source="align",
        )
    return out


class IncompleteTileAlignmentError(RuntimeError):
    """Raised internally by ``_stitch_with_stitch2d`` when stitch2d's ``align()`` (or
    ``load_params``) placed fewer tiles than exist. Caught by ``stitch_frame_tiles`` and treated
    as a signal to fall back to master coords — NOT propagated to callers directly."""


class UnstitchableFrameError(RuntimeError):
    """Raised by ``stitch_frame_tiles`` when a frame cannot be stitched correctly: per-frame
    alignment was incomplete or implausible AND no master fallback was available/usable. The
    caller must NOT materialize anything for this frame — silently concatenating tiles produces
    a wrong-but-passing image, which is the bug this error exists to prevent."""


def _load_master_coords(master_params_path: Path) -> dict[int, list[float]]:
    """Read ``{"coords": {...}}`` from ``master_params_path`` with integer tile-index keys."""
    import json

    raw = json.loads(Path(master_params_path).read_text())
    coords_raw = raw.get("coords", {})
    return {int(k): [float(v[0]), float(v[1])] for k, v in coords_raw.items()}


def raw_stitch2d_align(
    tiles: Sequence[TileSpec],
    orientation: Orientation,
) -> dict[int, list[float]]:
    """Run stitch2d ``align()`` on ``tiles`` and return the RAW ``coords`` dict (tile index ->
    ``[y, x]``, stitch2d's own convention), WITHOUT raising when alignment is incomplete.

    This is the primitive the experiment-grain master-builder needs: it must be able to inspect
    ``len(coords)`` itself and skip partial samples (mirrors legacy
    ``build01A_compile_keyence_images.py`` lines ~495-514), rather than have incompleteness
    turned into an exception it can't distinguish from a real failure.
    """
    from stitch2d import StructuredMosaic
    from stitch2d.tile import OpenCVTile

    tile_images = [tile.image for tile in tiles]
    mosaic = StructuredMosaic(
        [OpenCVTile(img) for img in tile_images],
        dim=len(tile_images),
        origin="upper left",
        direction=orientation,
        pattern="raster",
    )
    mosaic.align()
    return mosaic.params.get("coords", {})


def _feather_composite(
    tiles: Sequence[TileSpec],
    transforms: dict[str, TileTransform],
    orientation: Orientation,
) -> np.ndarray:
    """Composite tiles at their transform offsets with per-tile intensity gain + linear feathering.

    Replaces stitch2d's ``smooth_seams()`` (per-tile gamma match) + ``stitch()`` (hard last-wins
    placement) with:
      - **tile-wide gain**: scale each tile so its whole-tile median matches the CENTER tile's — the
        intensity-match that made seams agree in the illumination-correction experiments.
      - **linear feather**: in each overlap band, cross-fade tiles by distance from their own edge so
        one fades out as the next fades in — the universal microscopy-stitch blend (Elements / Fiji /
        ASHLAR). Removes the hard seam cut regardless of any residual offset.

    Geometry is taken from ``transforms`` (dx/dy in px). Feathering is applied along the STITCH axis
    only (x for horizontal strips, y for vertical). NOTE: currently assumes a 1-D strip (the Keyence
    case); a 2-D grid would need feathering on both axes.
    """
    imgs = [np.asarray(t.image, dtype=np.float64) for t in tiles]
    ids = [t.tile_id for t in tiles]

    # tile-wide gain -> match each tile's median to the center tile's.
    meds = [np.median(im) for im in imgs]
    center = len(imgs) // 2
    gains = [(meds[center] / m if m > 0 else 1.0) for m in meds]

    # placement offsets (round to int px)
    stitch_axis = 1 if orientation != "vertical" else 0  # 1 = x (columns), 0 = y (rows)
    xs = [int(round(transforms[i].dx_px)) for i in ids]
    ys = [int(round(transforms[i].dy_px)) for i in ids]
    xs = [x - min(xs) for x in xs]
    ys = [y - min(ys) for y in ys]

    H = max(y + im.shape[0] for y, im in zip(ys, imgs))
    W = max(x + im.shape[1] for x, im in zip(xs, imgs))
    acc = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)

    # order tiles along the stitch axis so "previous overlaps" are well defined
    order = sorted(range(len(imgs)), key=lambda k: (xs[k] if stitch_axis == 1 else ys[k]))
    placed_extent: list[tuple[int, int]] = []  # (start, end) along stitch axis of already-placed tiles
    for k in order:
        im = imgs[k] * gains[k]
        h, w = im.shape
        y0, x0 = ys[k], xs[k]
        # per-column (or per-row) feather weight for THIS tile along the stitch axis
        n = w if stitch_axis == 1 else h
        weight = np.ones(n, dtype=np.float64)
        start = x0 if stitch_axis == 1 else y0
        end = start + n
        # left/upper overlap with any already-placed tile -> ramp 0->1 over the overlapping span
        for (ps, pe) in placed_extent:
            ov = min(end, pe) - max(start, ps)
            if ov > 0 and max(start, ps) == start:      # overlap is on THIS tile's leading edge
                weight[:ov] = np.linspace(0.0, 1.0, ov)
            if ov > 0 and min(end, pe) == end:          # overlap on trailing edge (next tile handles it)
                weight[-ov:] = np.linspace(1.0, 0.0, ov)
        w2d = np.broadcast_to(weight, (h, w)) if stitch_axis == 1 else np.broadcast_to(weight[:, None], (h, w))
        acc[y0:y0+h, x0:x0+w] += im * w2d
        wsum[y0:y0+h, x0:x0+w] += w2d
        placed_extent.append((start, end))

    wsum[wsum == 0] = 1.0
    out = acc / wsum
    # clip to the source dtype range and restore integer type of the inputs
    src_dtype = tiles[0].image.dtype
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(src_dtype)


def _stitch_with_stitch2d(
    tiles: Sequence[TileSpec],
    orientation: Orientation,
    load_params_path: Path | None,
    run_align: bool,
) -> tuple[np.ndarray, dict[str, TileTransform]]:
    import json
    import tempfile

    from stitch2d import StructuredMosaic
    from stitch2d.tile import OpenCVTile, Tile

    tile_images = [tile.image for tile in tiles]
    if load_params_path is not None:
        mosaic = StructuredMosaic(
            [Tile(img) for img in tile_images],
            dim=len(tile_images),
            origin="upper left",
            direction=orientation,
            pattern="raster",
        )
        try:
            mosaic.load_params(str(load_params_path))
        except ValueError as exc:
            if "JSON param 'shape' does not match this mosaic" not in str(exc):
                raise
            raw = json.loads(Path(load_params_path).read_text())
            raw.setdefault("metadata", {})["shape"] = (
                [len(tile_images), 1] if orientation == "vertical" else [1, len(tile_images)]
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=True,
            ) as tmp:
                json.dump(raw, tmp)
                tmp.flush()
                mosaic.load_params(tmp.name)
    else:
        mosaic = StructuredMosaic(
            [OpenCVTile(img) for img in tile_images],
            dim=len(tile_images),
            origin="upper left",
            direction=orientation,
            pattern="raster",
        )
        if run_align:
            try:
                mosaic.align()
            except RuntimeError as exc:
                if "Could not align tiles" in str(exc):
                    raise IncompleteTileAlignmentError(
                        f"stitch2d could not align tiles "
                        f"(load_params_path={load_params_path})."
                    ) from exc
                raise

    coords = mosaic.params.get("coords", {})
    if len(coords) != len(tile_images):
        raise IncompleteTileAlignmentError(
            f"stitch2d placed {len(coords)}/{len(tile_images)} tiles "
            f"(load_params_path={load_params_path})."
        )
    transforms = _coords_to_transforms(tiles, coords)

    # Composite with per-tile gain + linear feather instead of stitch2d's smooth_seams()/stitch()
    # (per-tile gamma match + hard last-wins placement). This is the microscopy-standard blend and
    # removes the hard seam cut; geometry still comes entirely from the stitch2d transforms above.
    stitched = _feather_composite(tiles, transforms, orientation)
    return stitched, transforms


def _concat_tiles(
    tiles: Sequence[TileSpec],
    orientation: Orientation,
) -> tuple[np.ndarray, dict[str, TileTransform]]:
    concat_axis = 0 if orientation == "vertical" else 1
    stitched = np.concatenate([tile.image for tile in tiles], axis=concat_axis)

    transforms: dict[str, TileTransform] = {}
    cursor_x = 0.0
    cursor_y = 0.0
    for tile in tiles:
        transforms[tile.tile_id] = TileTransform(
            tile_id=tile.tile_id,
            dx_px=cursor_x,
            dy_px=cursor_y,
            source="fallback",
        )
        if concat_axis == 1:
            cursor_x += float(tile.image.shape[1])
        else:
            cursor_y += float(tile.image.shape[0])
    return stitched, transforms


_MASTER_PLAUSIBILITY_TOL_PX: float = 50.0
"""Max allowed per-tile deviation (px) between a per-frame align result and the master
(ground-truth empirical) coords before the frame is considered implausible. Generous relative to
the ~1-2px legacy cross-axis check, but tight relative to a full tile pitch (~677px here) — it
exists to catch alignments that placed tiles in a genuinely different arrangement than the
raster layout demands, not to flag normal jitter."""

_LEGACY_CROSS_AXIS_SHIFT_TOL_PX: dict[int, dict[Orientation, float]] = {
    2: {"vertical": 1.0, "horizontal": 1.0},
    3: {"vertical": 2.0, "horizontal": 2.0},
}
"""Legacy stitch QC: for a vertical strip, x wobble should be tiny; for a horizontal strip,
y wobble should be tiny. The along-strip tile pitch can be hundreds/thousands of pixels and is
not evidence of a bad alignment."""


def _run_tiling_qc(
    transforms: dict[str, TileTransform],
    config: FrameTilingConfig,
    allow_zero_shift: bool = True,
    master_coords: dict[int, list[float]] | None = None,
    tiles: Sequence[TileSpec] | None = None,
) -> TilingQC:
    max_abs_shift = 0.0
    max_cross_axis_shift = 0.0
    all_zero = True
    for tr in transforms.values():
        local_max = max(abs(float(tr.dx_px)), abs(float(tr.dy_px)))
        max_abs_shift = max(max_abs_shift, local_max)
        cross_axis = float(tr.dx_px) if config.orientation == "vertical" else float(tr.dy_px)
        max_cross_axis_shift = max(max_cross_axis_shift, abs(cross_axis))
        if local_max > 0:
            all_zero = False

    reasons: list[str] = []
    max_master_dev: float | None = None

    if master_coords is not None and tiles is not None:
        # Plausibility vs. ground truth (the master grid), NOT vs. an absolute-canvas-position
        # threshold — max_abs_shift_px legitimately reaches ~tile-pitch (e.g. ~1440px for a
        # correct 3-tile vertical strip) and must not be flagged.
        max_master_dev = 0.0
        for idx, tile in enumerate(tiles):
            tr = transforms.get(tile.tile_id)
            master_yx = master_coords.get(idx)
            if tr is None or master_yx is None:
                continue
            dev = max(abs(float(tr.dx_px) - master_yx[1]), abs(float(tr.dy_px) - master_yx[0]))
            max_master_dev = max(max_master_dev, dev)
        if max_master_dev > _MASTER_PLAUSIBILITY_TOL_PX:
            reasons.append("deviates_from_master")
    else:
        n_tiles = len(transforms)
        tol = _LEGACY_CROSS_AXIS_SHIFT_TOL_PX.get(
            n_tiles,
            {"vertical": float(config.max_abs_shift_px), "horizontal": float(config.max_abs_shift_px)},
        )[config.orientation]
        if max_cross_axis_shift > tol:
            reasons.append("cross_axis_shift_exceeds_threshold")

    if not allow_zero_shift and all_zero:
        reasons.append("unresolved_transforms")

    passed = len(reasons) == 0
    suggested_action: Literal["ok", "use_master", "concat", "fail"] = "ok" if passed else "use_master"
    metrics = {
        "max_abs_shift_px": float(max_abs_shift),
        "max_cross_axis_shift_px": float(max_cross_axis_shift),
        "tile_count": float(len(transforms)),
    }
    if max_master_dev is not None:
        metrics["max_master_deviation_px"] = float(max_master_dev)
    return TilingQC(
        passed=passed,
        reasons=tuple(reasons),
        metrics=metrics,
        suggested_action=suggested_action,
    )


def trim_to_shape(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Center-crop (or center-pad) ``image`` to ``target`` (Y, X). The canonical trim primitive —
    use this instead of the retired ``src.build.export_utils.trim_to_shape``."""
    target_y, target_x = target
    image_y, image_x = image.shape[:2]

    pad_y = max(0, target_y - image_y)
    pad_x = max(0, target_x - image_x)
    if pad_y or pad_x:
        image = np.pad(
            image,
            (
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2),
            ),
            mode="constant",
        )

    start_y = (image.shape[0] - target_y) // 2
    start_x = (image.shape[1] - target_x) // 2
    return image[start_y : start_y + target_y, start_x : start_x + target_x]


def _infer_layout_orientation(
    transforms: dict[str, TileTransform],
    *,
    fallback_orientation: Orientation,
) -> Orientation:
    """Infer the physical tile-strip direction from stitched coordinates."""
    if len(transforms) <= 1:
        return fallback_orientation

    dx_values = [float(tr.dx_px) for tr in transforms.values()]
    dy_values = [float(tr.dy_px) for tr in transforms.values()]
    dx_span = max(dx_values) - min(dx_values)
    dy_span = max(dy_values) - min(dy_values)
    if dx_span == 0 and dy_span == 0:
        return fallback_orientation
    return "horizontal" if dx_span > dy_span else "vertical"


def _finalize_image(
    image: np.ndarray,
    config: FrameTilingConfig,
    n_tiles: int,
    tile_shape: tuple[int, int],
    layout_orientation: Orientation | None = None,
) -> np.ndarray:
    orientation = layout_orientation or config.orientation
    out = image
    if n_tiles > 1 and orientation == "horizontal" and config.transpose_after_stitch:
        out = out.T

    if config.use_legacy_canvas:
        target = legacy_canvas_shape(
            n_tiles=n_tiles,
            orientation=orientation,
            tile_shape=tile_shape,
        )
        if target != (0, 0):
            out = trim_to_shape(out, target)

    # NOTE: display polarity (bright-embryo/dark-background inversion) is intentionally NOT done
    # here. Stitching composes tiles and returns them in the SAME polarity it received. Inversion
    # is owned by the shared image-math layer (image_building/shared: apply_display_polarity) and
    # applied ONCE by the materializer after composition, so YX1 (no stitch) and Keyence (stitch)
    # share one polarity. The old hidden `max - out` here was Keyence-only and is exactly what let
    # YX1 drift to the opposite polarity.
    return out
