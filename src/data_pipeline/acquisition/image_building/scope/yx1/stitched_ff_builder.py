"""
YX1 Image Building: Focus Stacking and Stitched FF Image Generation

This module handles YX1 microscope data processing:
- Reads ND2 files
- Focus stacking using LoG (Laplacian of Gaussian) method
- Writes stitched FF images to built_image_data/{exp}/stitched_ff_images/{well}/{channel}/

MVP Requirements:
- Read ND2 file
- Focus stack (LoG method)
- Write stitched TIFFs
- GPU support with proper device parameter
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from tqdm import tqdm
import nd2
import skimage
import skimage.io as skio

# Import shared LoG utilities used across pipeline image builders.
from data_pipeline.acquisition.image_building.shared.log_focus import LoG_focus_stacker, im_rescale
from data_pipeline.utils.cuda_diagnostics import resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def _read_nd2(path: Path) -> nd2.ND2File:
    """Read ND2 file from directory."""
    nd2_files = list(path.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No nd2 in {path}")
    if len(nd2_files) > 1:
        raise RuntimeError(f"Multiple nd2 files in {path}")
    return nd2.ND2File(nd2_files[0])


def _get_stack(
    dask_arr,
    t: int,
    w: int,
    n_z_keep: int | None = None,
    *,
    axes=None,
    array_axis_order: tuple[str, ...] | None = None,
    channel: int = 0,
) -> np.ndarray:
    """Return Z×Y×X BF stack (no channels).

    ``axes`` + ``array_axis_order`` select by axis NAME (see
    ``metadata_ingest/scope/yx1/nd2_axes.py``). Pass them whenever the ND2 is open: positional
    indexing assumes ``(T, W, Z, C, Y, X)``, which is wrong for any other layout — on a
    ``(P, Z, C, Y, X)`` snapshot it reads the position as T, a z-plane as the position, and leaves
    the channel axis standing in for the Z stack, so focus-stacking would silently run over
    ``[BF, fluorescence]`` as if they were focal planes.

    Omitting them falls back to the legacy positional slice, which is correct ONLY for a
    6-D ``(T, W, Z, C, Y, X)`` or 5-D ``(T, W, Z, Y, X)`` array.
    """
    if axes is not None and array_axis_order is not None:
        from data_pipeline.acquisition.metadata_ingest.scope.yx1.nd2_axes import select_zyx_stack

        stack = select_zyx_stack(
            dask_arr, axes, array_axis_order, position=w, time=t, channel=channel
        )
        nz = stack.shape[0]
        buf = max((nz - n_z_keep) // 2, 0) if n_z_keep else 0
        if buf:
            stack = stack[buf : nz - buf, :, :]
        return stack.compute() if hasattr(stack, "compute") else np.asarray(stack)

    nz = dask_arr.shape[2]
    buf = max((nz - n_z_keep) // 2, 0) if n_z_keep else 0
    return (
        dask_arr[t, w, buf : nz - buf, :, :].compute()
        if buf or n_z_keep
        else dask_arr[t, w, :, :, :].compute()
    )


def _focus_stack(
    stack_zyx: np.ndarray,
    device: str,
    filter_size: int = 3
) -> np.ndarray:
    """Apply LoG focus stacking to Z-stack."""
    # Normalize and convert to tensor
    norm, _, _ = im_rescale(stack_zyx)
    norm = norm.astype(np.float32)
    device = resolve_device(device)
    tensor = torch.from_numpy(norm).to(device)

    # Apply focus stacking
    ff_t, _ = LoG_focus_stacker(tensor, filter_size, device)
    arr = ff_t.cpu().numpy()
    arr_clipped = np.clip(arr, 0, 65535)
    ff_i = arr_clipped.astype(np.uint16)

    # Convert to 8 bit
    ff_8 = skimage.util.img_as_ubyte(ff_i)

    return ff_8


def _write_stitched_ff(
    output_dir: Path,
    well_name: str,
    channel_name: str,
    time_index: int,
    image: np.ndarray,
    overwrite: bool = False
):
    """
    Write stitched FF image to standardized location.

    Output structure:
    built_image_data/{exp}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_index:04d}.tif
    """
    well_dir = output_dir / well_name / channel_name
    well_dir.mkdir(parents=True, exist_ok=True)

    output_path = well_dir / f"{well_name}_{channel_name}_t{time_index:04d}.tif"

    if output_path.exists() and not overwrite:
        return

    skio.imsave(output_path, image, check_contrast=False)


def compile_yx1_data(
    raw_data_root: Path,
    output_root: Path,
    exp_name: str,
    well_series_mapping: dict[str, int],  # well_name -> series_number (1-based)
    overwrite: bool = False,
    device: str = "cuda",
    n_workers: int = 1,
    z_buffer: bool = False,
):
    """
    Compile YX1 ND2 data into stitched FF images.

    Args:
        raw_data_root: Path to raw_image_data/YX1/
        output_root: Path to built_image_data/
        exp_name: Experiment name
        well_series_mapping: Dict mapping well names (e.g., 'A01') to ND2 series numbers (1-based)
        overwrite: Whether to overwrite existing files
        device: PyTorch device for focus stacking ('cuda' or 'cpu')
        n_workers: Number of workers (not used in MVP, kept for compatibility)
        z_buffer: Whether to trim Z-stack (specific to exp 20231206)

    Output structure:
        built_image_data/{exp_name}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_index:04d}.tif
    """

    exp_path = raw_data_root / exp_name
    output_dir = output_root / exp_name / "stitched_ff_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)
    log.info("Processing YX1 data: %s (device=%s)", exp_name, device)

    if device == "cpu":
        log.warning("Using CPU. This may be slow. GPU recommended.")

    # Read ND2 file
    nd = _read_nd2(exp_path)
    # Axes BY NAME (see metadata_ingest/scope/yx1/nd2_axes.py) — `nd.shape` is positional and its
    # layout varies per acquisition, so unpacking it as (T, W, Z) mis-reads any other layout.
    axes = axes_of(nd)
    array_axis_order = array_axis_order_of(nd)
    n_t, n_w, n_z = axes.n_t, axes.n_p, axes.n_z
    log.info(
        "ND2 axes: T=%d P=%d Z=%d C=%d (array order %s)",
        n_t, n_w, n_z, axes.n_c, array_axis_order,
    )

    dask_arr = nd.to_dask()
    channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]

    # This legacy compile path has no acquisition inventory to read `channel_index` from, and the
    # pipeline's live YX1 materializer (materialize_well_yx1) is the supported route — it resolves the
    # index from the minted triple. Rather than re-introduce name-matching here, translate the raw
    # names through the scope's ONE canonical map and require an unambiguous brightfield channel.
    channel_ids = [YX1_CHANNEL_MAP.to_canonical(name) for name in channel_names]
    bf_positions = [i for i, cid in enumerate(channel_ids) if cid == "BF"]
    if len(bf_positions) != 1:
        raise ValueError(
            f"stitched_ff_builder: expected exactly ONE brightfield channel, found "
            f"{len(bf_positions)} in {list(zip(channel_names, channel_ids))}. Fix the scope's "
            "channel_map.py (the one canonical raw->channel_id generator); this legacy path does "
            "not guess. The supported route is materialize_well_yx1, which reads channel_index from "
            "the acquisition inventory."
        )
    bf_idx = bf_positions[0]
    log.info("Brightfield channel_index=%d (raw %r)", bf_idx, channel_names[bf_idx])

    # Build lookup of ND2 well index -> well name
    well_name_lookup = {int(series)-1: name for name, series in well_series_mapping.items()}

    log.info("Processing %d wells, %d timepoints", len(well_name_lookup), n_t)

    # Z-stack buffer for specific experiment
    n_z_keep = 12 if z_buffer else None

    # Process each well and timepoint
    total_frames = len(well_name_lookup) * n_t
    processed = 0
    skipped = 0

    for nd2_idx, well_name in sorted(well_name_lookup.items()):
        for t in range(n_t):
            # Check if already exists
            output_path = output_dir / well_name / "BF" / f"{well_name}_BF_t{t:04d}.tif"
            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            try:
                # Get Z-stack for this well and timepoint
                stack = _get_stack(dask_arr, t, nd2_idx, n_z_keep=n_z_keep)

                # Apply focus stacking
                ff = _focus_stack(stack, device, filter_size=3)

                # Write output
                _write_stitched_ff(output_dir, well_name, "BF", t, ff, overwrite)

                processed += 1

                if processed % 50 == 0:
                    log.info("Processed %d/%d frames", processed, total_frames - skipped)

            except Exception as e:
                log.error("Failed processing well=%s, t=%d: %s", well_name, t, e)
                continue

    nd.close()

    log.info("YX1 processing complete: processed=%d, skipped=%d, total=%d",
             processed, skipped, total_frames)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# REMOVED: _determine_bf_channel (+ the YX1_BF_CHANNEL_INDEX env override).
#
# It was a SECOND channel vocabulary, competing with the scope's own `channel_map.py`. It matched
# "BF" / "EYES - Dia" / "Empty" by name and knew nothing of e.g. "BF-no bin", so on a real
# fluorescence plate it matched nothing, was not single-channel, and RAISED — telling the caller to
# set an env var (a haunted global that silently reassigns channel identity process-wide).
#
# `channel_index` is a FACT OF THE FILE. The scope adapter mints the
# channel_index / raw_channel_name / channel_id triple once into the acquisition inventory (1:1:1,
# guarded by assert_channel_mapping_consistent).
#
# The rule, stated precisely: SUPPORTED pipeline consumers must use the recorded inventory mapping
# (`scope/shared/acquisition_channels.resolve_channel_index`) — see materialize_well_yx1. A LEGACY
# reader with no inventory (like `compile_yx1_data` below) may derive an index only by translating raw
# names through the scope's ONE canonical channel_map.py, and must fail loud if that is ambiguous.
# What is never acceptable is a private alias table or an environment override.
# ─────────────────────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Example usage (not executed in pipeline)
    from pathlib import Path

    data_root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
    raw_root = data_root / "raw_image_data" / "YX1"
    built_root = data_root / "built_image_data"

    # Example well mapping (normally comes from series_well_mapper)
    example_mapping = {
        "A01": 1,
        "A02": 2,
        "B01": 3,
        "B02": 4,
    }

    compile_yx1_data(
        raw_data_root=raw_root,
        output_root=built_root,
        exp_name="20240314",
        well_series_mapping=example_mapping,
        overwrite=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
