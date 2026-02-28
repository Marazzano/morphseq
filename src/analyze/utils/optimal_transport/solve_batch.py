"""Batch solvers for WorkGridBatch / PairPack / StarPack.

Sequential Python loops for now. Data model is [B, H, W] arrays + index
arrays — ready for future OTT-JAX vmap without restructuring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from analyze.utils.optimal_transport.backends.base import UOTBackend
from analyze.utils.optimal_transport.config import UOTConfig
from analyze.utils.optimal_transport.multiscale_sampling import build_support
from analyze.utils.optimal_transport.results import UOTResultWork
from analyze.utils.optimal_transport.work_grid_batch import (
    PairPack,
    StarPack,
    CropPolicy,
)


def _solve_one_pair(
    src_density: np.ndarray,
    tgt_density: np.ndarray,
    uot_cfg: UOTConfig,
    backend: UOTBackend,
) -> UOTResultWork:
    """Solve UOT for a single (src, tgt) density pair on the work grid."""
    from analyze.utils.optimal_transport.solve import run_uot_on_working_grid
    from analyze.utils.optimal_transport.working_grid import WorkingGridPair
    from analyze.utils.optimal_transport.config import PairFrameGeometry
    from analyze.utils.coord.types import BoxYX

    h, w = src_density.shape
    # Minimal WorkingGridPair wrapper for the solver
    pair = WorkingGridPair(
        coord_frame_id="canonical_grid",
        coord_frame_version=1,
        canonical_um_per_px=1.0,
        work_um_per_px=1.0,
        pair_frame=PairFrameGeometry(
            canon_shape_hw=(h, w),
            pair_crop_box_yx=BoxYX(0, h, 0, w),
            crop_pad_hw=(0, 0),
            downsample_factor=1,
            work_shape_hw=(h, w),
            px_size_um=1.0,
        ),
        src_canon_mask=np.zeros((h, w), dtype=np.uint8),
        tgt_canon_mask=np.zeros((h, w), dtype=np.uint8),
        src_work_density=src_density,
        tgt_work_density=tgt_density,
        meta={},
    )
    return run_uot_on_working_grid(pair, config=uot_cfg, backend=backend)


def solve_pairs(
    pair_pack: PairPack,
    uot_cfg: UOTConfig,
    backend: UOTBackend,
) -> list[UOTResultWork]:
    """Solve UOT for each pair in a PairPack. Sequential iteration."""
    results: list[UOTResultWork] = []
    batch = pair_pack.batch
    for k in range(len(pair_pack.src_indices)):
        si = int(pair_pack.src_indices[k])
        ti = int(pair_pack.tgt_indices[k])

        src_d = batch.densities_full[si]
        tgt_d = batch.densities_full[ti]

        # Apply per-pair crop if available
        if pair_pack.crop_boxes_work is not None:
            crop = pair_pack.crop_boxes_work[k]
            sl = crop.to_slices()
            src_d = np.ascontiguousarray(src_d[sl])
            tgt_d = np.ascontiguousarray(tgt_d[sl])

        result = _solve_one_pair(src_d, tgt_d, uot_cfg, backend)
        results.append(result)

    return results


def solve_star(
    star_pack: StarPack,
    uot_cfg: UOTConfig,
    backend: UOTBackend,
) -> dict[tuple[str, str], UOTResultWork]:
    """Solve UOT for star topology (ref × src). Chunked iteration over sources."""
    results: dict[tuple[str, str], UOTResultWork] = {}
    batch = star_pack.batch

    n_src = len(star_pack.src_indices)
    chunk_size = star_pack.chunk_size

    for chunk_start in range(0, n_src, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_src)
        chunk_src_indices = star_pack.src_indices[chunk_start:chunk_end]

        # Gather src densities for this chunk
        src_chunk = batch.densities_full[chunk_src_indices]

        for ri, ref_idx in enumerate(star_pack.ref_indices):
            ref_d = star_pack.ref_densities[ri]

            for ci in range(len(chunk_src_indices)):
                src_d = src_chunk[ci]

                # Apply global crop if available
                if star_pack.crop_box_work is not None:
                    sl = star_pack.crop_box_work.to_slices()
                    ref_d_cropped = np.ascontiguousarray(ref_d[sl])
                    src_d_cropped = np.ascontiguousarray(src_d[sl])
                else:
                    ref_d_cropped = ref_d
                    src_d_cropped = src_d

                result = _solve_one_pair(ref_d_cropped, src_d_cropped, uot_cfg, backend)

                ref_id = star_pack.ref_ids[ri]
                src_id = star_pack.src_ids[chunk_start + ci]
                results[(ref_id, src_id)] = result

    return results


def solve_pairs_gpu(pair_pack: PairPack, uot_cfg: UOTConfig, backend: UOTBackend):
    """OTT-JAX batch solver (not yet implemented)."""
    raise NotImplementedError("OTT-JAX batch solver not yet implemented")


def solve_star_gpu(star_pack: StarPack, uot_cfg: UOTConfig, backend: UOTBackend):
    """OTT-JAX star solver (not yet implemented)."""
    raise NotImplementedError("OTT-JAX star solver not yet implemented")
