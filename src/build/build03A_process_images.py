from pathlib import Path
import sys

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

import os
import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import skimage
import cv2
import pandas as pd
from src.functions.utilities import path_leaf
from src.functions.image_utils import crop_embryo_image, get_embryo_angle, process_masks
from skimage.morphology import disk, binary_closing, remove_small_objects
import warnings
import scipy
# from parfor import pmap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import skimage
from scipy.stats import truncnorm
import numpy as np
import skimage.io as io
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map 
from skimage.transform import rescale, resize
from src.build.export_utils import trim_to_shape
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
from typing import Sequence, List
from itertools import chain
from segmentation_sandbox.scripts.utils.parsing_utils import build_snip_id

# Suppress the specific warning from skimage
warnings.filterwarnings("ignore", message="Only one label was provided to `remove_small_objects`")

def estimate_image_background(root, embryo_metadata_df, bkg_seed=309, n_bkg_samples=100):

    np.random.seed(bkg_seed)
    bkg_sample_indices = np.random.choice(range(embryo_metadata_df.shape[0]), n_bkg_samples, replace=True)
    bkg_pixel_list = []

    for r in tqdm(range(len(bkg_sample_indices)), "Estimating background..."):
        sample_i = bkg_sample_indices[r]
        row = embryo_metadata_df.iloc[sample_i].copy()

        # set path to segmentation data
        ff_image_path = os.path.join(root, 'built_image_data', 'stitched_FF_images', '')
        segmentation_path = os.path.join(root, 'segmentation', '')
        segmentation_model_path = os.path.join(root, 'segmentation', 'segmentation_models', '')

         # get list of up-to-date models
        seg_mdl_list_raw = glob.glob(segmentation_model_path + "*")
        seg_mdl_list = [s for s in seg_mdl_list_raw if ~os.path.isdir(s)]
        emb_mdl_name = [path_leaf(m) for m in seg_mdl_list if "mask" in m][0]
        via_mdl_name = [path_leaf(m) for m in seg_mdl_list if "via" in m][0]

        # generate path and image name
        seg_dirs_raw = glob.glob(segmentation_path + "*")
        seg_dirs = [s for s in seg_dirs_raw if os.path.isdir(s)]

        emb_path = [m for m in seg_dirs if emb_mdl_name in m][0]
        via_path = [m for m in seg_dirs if via_mdl_name in m][0]

        well = row["well"]
        time_int = row["time_int"]
        date = str(row["experiment_date"])

        ############
        # Load masks from segmentation
        ############
        stub_name = well + f"_t{time_int:04}*"
        im_emb_path = glob.glob(os.path.join(emb_path, date, stub_name))[0]
        im_via_path = glob.glob(os.path.join(via_path, date, stub_name))[0]

        # load main embryo mask
        # im_emb_path = os.path.join(emb_path, date, lb_name)
        im_mask = io.imread(im_emb_path)
        im_mask = np.round(im_mask / 255 * 2 - 1).astype(int)

        im_via = io.imread(im_via_path)
        im_via = np.round(im_via / 255 * 2 - 1).astype(int)

        im_bkg = np.ones(im_mask.shape, dtype="uint8")
        im_bkg[np.where(im_mask == 1)] = 0
        im_bkg[np.where(im_via == 1)] = 0

        # load image
        im_ff_path = glob.glob(os.path.join(ff_image_path, date, stub_name))[0] #os.path.join(ff_image_path, date, im_name)
        im_ff = io.imread(im_ff_path)
        if im_ff.shape[0] < im_ff.shape[1]:
            im_ff = im_ff.transpose(1, 0)
        if im_ff.dtype != "uint8":
            im_ff = skimage.util.img_as_ubyte(im_ff)

        # extract background pixels
        bkg_pixels = im_ff[np.where(im_bkg == 1)].tolist()

        bkg_pixel_list += bkg_pixels

    # get mean and standard deviation
    px_mean = np.mean(bkg_pixel_list)
    px_std = np.std(bkg_pixel_list)

    return px_mean, px_std


def export_embryo_snips(r: int,
                        root: str | Path,
                        stats_df: pd.DataFrame,
                        dl_rad_um: int,
                        outscale: float,
                        outshape: List,
                        px_mean: float, px_std: float):
    """
    Refactored function to use SAM2 metadata and disable unavailable QC checks.
    
    This function is refactored to use pre-computed SAM2 metadata. It loads
    the embryo mask from the path specified in the SAM2 CSV. Since yolk masks
    are not available in the SAM2 pipeline output, a dummy yolk mask is used.
    """
    
    root = Path(root)
    row = stats_df.iloc[r].copy()

    # --- Load embryo mask from SAM2 CSV path ---
    masks_base_dir = root / "segmentation_sandbox" / "data" / "exported_masks"
    date = str(row["experiment_date"])
    mask_filename = row["exported_mask_path"]
    mask_path = masks_base_dir / date / "masks" / mask_filename

    if not mask_path.exists():
        warnings.warn(f"Mask file not found, skipping snip export: {mask_path}", stacklevel=2)
        return True # Return True for out_of_frame_flag to indicate failure

    im_mask_multi_embryo = io.imread(mask_path)
    lbi = row["region_label"]
    im_mask = (im_mask_multi_embryo == lbi).astype(int)
    
    # Create a dummy yolk mask since it's not available from SAM2 pipeline
    im_yolk = np.zeros_like(im_mask)

    # --- Load Full-Frame Image ---
    ff_image_path = root / 'segmentation_sandbox' / 'data' / 'raw_data_organized'
    im_stub = f"{row['image_id']}*"
    
    ff_image_paths = sorted((ff_image_path / date / "images" / row['video_id']).glob(im_stub))
    if not ff_image_paths:
        warnings.warn(f"FF image not found for {im_stub} in {ff_image_path / date}", stacklevel=2)
        return True

    im_ff = io.imread(ff_image_paths[0])

    # --- Continue with legacy processing ---
    if 'Height (um)' in row and 'Height (px)' in row and row['Height (px)'] > 0:
        px_dim_raw = row["Height (um)"] / row["Height (px)"]
    else:
        px_dim_raw = 1.0

    im_mask_ft, im_yolk = process_masks(im_mask, im_yolk, row)

    if im_ff.shape[0] < im_ff.shape[1]:
        im_ff = im_ff.transpose(1,0)

    if im_ff.dtype != "uint8":
        im_ff_scaled = skimage.exposure.rescale_intensity(im_ff, in_range='image', out_range=(0, 255))
        im_ff = im_ff_scaled.astype(np.uint8)

    im_ff_rs = rescale(im_ff, (px_dim_raw / outscale, px_dim_raw / outscale), order=1, preserve_range=True)
    mask_emb_rs = resize(im_mask_ft.astype(float), im_ff_rs.shape, order=1)
    mask_yolk_rs = resize(im_yolk.astype(float), im_ff_rs.shape, order=1)

    angle_to_use = get_embryo_angle((mask_emb_rs > 0.5).astype(np.uint8),(mask_yolk_rs > 0.5).astype(np.uint8))

    im_ff_rotated = rotate_image(im_ff_rs, np.rad2deg(angle_to_use))
    emb_mask_rotated = rotate_image(mask_emb_rs, np.rad2deg(angle_to_use))
    im_yolk_rotated = rotate_image(mask_yolk_rs, np.rad2deg(angle_to_use))

    im_cropped, emb_mask_cropped, yolk_mask_cropped = crop_embryo_image(im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape)
    
    emb_mask_cropped2 = scipy.ndimage.binary_fill_holes(emb_mask_cropped > 0.5).astype(np.uint8)
    yolk_mask_cropped = scipy.ndimage.binary_fill_holes(yolk_mask_cropped > 0.5).astype(np.uint8)

    noise_array_raw = np.reshape(truncnorm.rvs(-px_mean/px_std, 4, size=outshape[0]*outshape[1]), outshape)
    noise_array = noise_array_raw*px_std + px_mean
    noise_array[np.where(noise_array < 0)] = 0

    im_cropped = skimage.exposure.equalize_adapthist(im_cropped)*255
    mask_cropped_gauss = skimage.filters.gaussian(emb_mask_cropped2.astype(float), sigma=dl_rad_um / outscale)
    im_cropped_gauss = np.multiply(im_cropped.astype(float), mask_cropped_gauss) + np.multiply(noise_array, 1-mask_cropped_gauss)

    out_of_frame_flag = (np.sum(emb_mask_cropped == 1) / np.sum(emb_mask_rotated == 1)) < 0.98 if np.sum(emb_mask_rotated) > 0 else True

    im_name = row["snip_id"]
    exp_date = str(row["experiment_date"])
    im_snip_dir = root / 'training_data' / 'bf_embryo_snips'
    mask_snip_dir = root / 'training_data' / 'bf_embryo_masks'
    
    ff_dir = im_snip_dir / exp_date
    ff_save_path = ff_dir / f"{im_name}.jpg"
    if not ff_dir.is_dir():
        ff_dir.mkdir(parents=True, exist_ok=True)

    ff_dir_uc = (im_snip_dir.parent / (im_snip_dir.name + "_uncropped")) / exp_date
    ff_save_path_uc = ff_dir_uc / f"{im_name}.jpg"
    if not ff_dir_uc.is_dir():
        ff_dir_uc.mkdir(parents=True, exist_ok=True)

    io.imsave(ff_save_path, im_cropped_gauss.astype(np.uint8), check_contrast=False)
    io.imsave(ff_save_path_uc, im_cropped.astype(np.uint8), check_contrast=False)
    io.imsave(mask_snip_dir / f"emb_{im_name}.jpg", emb_mask_cropped2, check_contrast=False)
    io.imsave(mask_snip_dir / f"yolk_{im_name}.jpg", yolk_mask_cropped, check_contrast=False)

    return out_of_frame_flag



def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def make_well_names():
    row_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    well_name_list = []
    for r in row_list:
        well_names = [r + f'{c+1:02}' for c in range(12)]
        well_name_list += well_names

    return well_name_list


def get_unprocessed_metadata(
    master_df: pd.DataFrame,
    existing_meta_path: str | Path,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Return only the rows in master_df whose (well, experiment_date, time_int)
    are not in the existing_meta_path CSV (if present).  Drops duplicates on
    the three key columns before doing the comparison.
    """
    key_cols = ["well", "experiment_date", "time_int"]
    # 1) dedupe the master list
    df = master_df.drop_duplicates(subset=key_cols).copy()
    # 2) if overwriting or no existing file, we're done
    existing = Path(existing_meta_path)
    if overwrite or not existing.exists():
        return df

    # 3) read and dedupe the done list
    done = (
        pd.read_csv(existing, usecols=key_cols)
          .drop_duplicates(subset=key_cols)
    )
    # 4) merge with indicator
    merged = df.merge(
        done, on=key_cols, how="left", indicator=True
    )
    # 5) keep only those not in 'done'
    new_only = merged.loc[merged["_merge"] == "left_only", df.columns]
    return new_only.reset_index(drop=True)


# ─── helper: sample one JPG per date, warn & drop missing ────────────────────
def sample_experiment_shapes(exp_dir: Path):
    # shapes, missing = {}, []
    jpg = next(exp_dir.glob("*.jpg"), None)
    if jpg is None:
        raise Exception(f"FF images not found for {exp_dir.name}")
    else:
        shape = io.imread(jpg).shape[:2]  # (H, W)
        
    return shape

def process_mask_images(image_path):

    # load label image
    im = io.imread(image_path)
    im_mask = (np.round(im / 255 * 2) - 1).astype(np.uint8)

    # load viability image
    seg_path = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
    date_dir = os.path.basename(os.path.dirname(image_path))
    im_stub = os.path.basename(image_path)[:9]
    via_dir = glob.glob(os.path.join(seg_path, "via_*"))[0]
    via_path = glob.glob(os.path.join(via_dir, date_dir, im_stub + "*"))[0]
    im_via = io.imread(via_path)
    im_via = (np.round(im_via / 255 * 2) - 1).astype(np.uint8)
    
    # make a combined mask
    cb_mask = np.ones_like(im_mask)
    cb_mask[np.where(im_mask == 1)] = 1  # alive
    cb_mask[np.where((im_mask == 1) & (im_via == 1))] = 2  # dead

    return im_mask, cb_mask

# ─── helper: find mask paths & valid row indices ─────────────────────────────
# def get_mask_paths_from_diff_old(df_diff, emb_dir, strict=False):
#     emb_dir = Path(emb_dir)
#     paths, idxs = [], []
#     for i, row in df_diff.iterrows():
#         folder = emb_dir / str(row["experiment_date"])
#         stub   = f"{row['well']}_t{int(row['time_int']):04}"
#         files  = list(folder.glob(f"{stub}*"))
#         if not files:
#             msg = f"segment_wells: no mask for {stub} in {folder}"
#             if strict:
#                 raise FileNotFoundError(msg)
#             warnings.warn(msg, stacklevel=2)
#             continue
#         paths.append(str(files[0]))
#         idxs.append(i)
#     return paths, idxs


def get_mask_paths_from_diff(df_diff, emb_dir, strict=False):
    # unprocessed_date_index = np.unique(df_diff["experiment_date"]).astype(str)

    well_id_list = df_diff["well"].values
    well_date_list = df_diff["experiment_date"].values.astype(str)
    well_time_list = df_diff["time_int"].values

    dates_u = np.unique(well_date_list)
    experiment_list = [emb_dir / d for d in dates_u]

    images_to_process = []
    valid_indices = []
    for _, experiment_path in enumerate(experiment_list):
        ename = path_leaf(experiment_path)
        date_indices = np.where(well_date_list == ename)[0]

        # get channel info
        image_list_full = sorted(glob.glob(os.path.join(experiment_path, "*.jpg")))

        if len(image_list_full) > 0:
            im_name_test = path_leaf(image_list_full[0])
            image_suffix = im_name_test[9:]

            # get list of image prefixes
            im_names = [os.path.join(experiment_path, well_id_list[i] + f"_t{well_time_list[i]:04}" + image_suffix) for i in date_indices]
            
            full_set = set(image_list_full)
            # find the (df_diff) indices whose im_names actually exist on disk
            valid = [(idx, name) 
                    for idx, name in zip(date_indices, im_names) 
                    if name in full_set]
            # unpack into two parallel lists
            vi, itp = map(list, zip(*valid)) if valid else ([], [])

            valid_indices += vi
            images_to_process += itp

    return images_to_process, np.asarray(valid_indices)


# DELETED: count_embryo_regions() function
# This legacy function has been replaced by SAM2 metadata bridge approach.
# SAM2 provides pre-computed embryo areas, centroids, and bounding boxes,
# eliminating the need for regionprops calculations.
# 
# Original function removed as part of refactor-003 Phase 2 implementation.
# See: docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md
    pass  # Function body removed - replaced by SAM2 bridge approach



# DELETED: do_embryo_tracking() function  
# This legacy function has been replaced by SAM2 metadata bridge approach.
# SAM2 provides inherent temporal tracking with stable embryo IDs across frames,
# eliminating the need for Hungarian algorithm tracking.
# 
# Original function removed as part of refactor-003 Phase 2 implementation.
# See: docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md
def do_embryo_tracking_DELETED(
    well_id: str,
    df_update: pd.DataFrame
) -> pd.DataFrame:
    """DELETED - Function replaced by SAM2 bridge approach."""
    # slice out just this well’s rows, in time‐order
    sub = df_update[df_update["well_id"] == well_id].copy().reset_index(drop=True)
    n_obs = sub["n_embryos_observed"].astype(int).values
    max_n = n_obs.max()

    # if zero embryos ever seen, nothing to emit
    if max_n == 0:
        return pd.DataFrame([], columns=list(sub.columns) + 
                            ["xpos","ypos","fraction_alive","region_label","embryo_id"])

    # the columns we carry forward
    temp_cols = list(sub.columns)

    # simple‐case: exactly one embryo in every frame => no assignment needed
    if max_n == 1:
        out = sub[temp_cols].copy()
        out["xpos"]           = sub["e0_x"].values
        out["ypos"]           = sub["e0_y"].values
        out["fraction_alive"] = sub["e0_frac_alive"].values
        out["region_label"]   = sub["e0_label"].values
        out["embryo_id"]      = well_id + "_e00"
        return out

    # --- otherwise multiple embryos may appear/disappear, we do a linear‐assignment over time ---
    T = len(sub)
    # preallocate assignment array: (T × max_n)
    ids = np.full((T, max_n), np.nan)
    # initialize at first time‐point that saw any embryos
    first = np.argmax(n_obs > 0)
    ids[first, : n_obs[first]] = np.arange(n_obs[first], dtype=float)

    # keep track of last positions so we can compute cost‐matrix
    last_pos = np.zeros((max_n, 2), dtype=float)
    for e in range(n_obs[first]):
        last_pos[e, 0] = sub.loc[first, f"e{e}_x"]
        last_pos[e, 1] = sub.loc[first, f"e{e}_y"]

    # step through subsequent frames
    for t in range(first + 1, T):
        k = n_obs[t]
        if k == 0:
            # carry forward previous ids but no new detections
            continue
        # build current positions
        curr = np.vstack([
            [sub.loc[t, f"e{j}_x"], sub.loc[t, f"e{j}_y"]]
            for j in range(k)
        ])
        # cost‐matrix between last_pos and curr
        D = pairwise_distances(last_pos, curr)
        # assign old→new
        row_ind, col_ind = linear_sum_assignment(D)
        # record assignments
        ids[t, row_ind] = col_ind
        # update last_pos for those that moved
        for r, c in zip(row_ind, col_ind):
            last_pos[r, :] = curr[c]

    # build output rows by walking each track separately
    out_rows: List[pd.DataFrame] = []
    for track in range(max_n):
        # select time‐points where track was assigned
        times, cols = np.where(~np.isnan(ids[:, track][:, None]))
        if len(times) == 0:
            continue
        subtrk = sub.iloc[times].copy().reset_index(drop=True)
        subtrk["xpos"]           = [subtrk.loc[i, f"e{int(ids[times[i], track])}_x"] for i in range(len(times))]
        subtrk["ypos"]           = [subtrk.loc[i, f"e{int(ids[times[i], track])}_y"] for i in range(len(times))]
        subtrk["fraction_alive"] = [subtrk.loc[i, f"e{int(ids[times[i], track])}_frac_alive"] for i in range(len(times))]
        subtrk["region_label"]   = [subtrk.loc[i, f"e{int(ids[times[i], track])}_label"] for i in range(len(times))]
        subtrk["embryo_id"]      = well_id + f"_e{track:02}"
        out_rows.append(subtrk[temp_cols + 
            ["xpos","ypos","fraction_alive","region_label","embryo_id"]
        ])

    if not out_rows:
        # nothing survived—return empty but typed
        return pd.DataFrame([], columns=temp_cols + 
                            ["xpos","ypos","fraction_alive","region_label","embryo_id"])
    
    return pd.concat(out_rows, ignore_index=True)


def get_embryo_stats(index: int,
                     root: str | Path,
                     embryo_metadata_df: pd.DataFrame,
                     qc_scale_um: int,
                     ld_rat_thresh: float):
    """
    Refactored function to use SAM2 metadata and disable unavailable QC checks.
    
    This function is refactored as per refactor-003-prd to use pre-computed
    SAM2 metadata. It loads the embryo mask from the path specified in the
    SAM2 CSV. QC checks depending on bubble, focus, and yolk masks are
    disabled as these masks are not available in the SAM2 pipeline output.
    """
    
    root = Path(root)
    row = embryo_metadata_df.loc[index].copy()

    # --- Load embryo mask from SAM2 CSV path ---
    masks_base_dir = root / "segmentation_sandbox" / "data" / "exported_masks"
    
    date = str(row["experiment_date"])
    mask_filename = row["exported_mask_path"]
    
    mask_path = masks_base_dir / date / "masks" / mask_filename
    
    if not mask_path.exists():
        warnings.warn(f"Mask file not found: {mask_path}", stacklevel=2)
        row.loc["dead_flag"] = False
        row.loc["no_yolk_flag"] = False
        row.loc["frame_flag"] = False
        row.loc["focus_flag"] = False
        row.loc["bubble_flag"] = False
        return pd.DataFrame(row).transpose()

    im_mask_multi_embryo = io.imread(mask_path)
    
    lbi = row["region_label"]
    im_mask_lb = (im_mask_multi_embryo == lbi).astype(int)

    # --- Perform calculations and QC checks ---
    # Calculations depending on legacy metadata columns that are not in the SAM2 CSV
    # are disabled or use placeholder values.
    
    px_dim = 1.0 
    qc_scale_px = int(np.ceil(qc_scale_um / px_dim))

    row.loc["surface_area_um"] = row["area_px"] * (px_dim ** 2)
    
    yy, xx = np.indices(im_mask_lb.shape)
    mask_coords = np.c_[xx[im_mask_lb==1], yy[im_mask_lb==1]]
    if mask_coords.shape[0] > 1:
        pca = PCA(n_components=2)
        mask_coords_rot = pca.fit_transform(mask_coords)
        row.loc["length_um"], row.loc["width_um"] = (np.max(mask_coords_rot, axis=0) - np.min(mask_coords_rot, axis=0)) * px_dim
    else:
        row.loc["length_um"], row.loc["width_um"] = 0.0, 0.0

    row.loc["speed"] = np.nan

    ######
    # now do QC checks
    ######

    row.loc["dead_flag"] = row["fraction_alive"] < ld_rat_thresh
    row.loc["no_yolk_flag"] = False
    im_trunc = im_mask_lb[qc_scale_px:-qc_scale_px, qc_scale_px:-qc_scale_px]
    if np.sum(im_mask_lb) > 0:
        row.loc["frame_flag"] = np.sum(im_trunc) <= 0.98 * np.sum(im_mask_lb)
    else:
        row.loc["frame_flag"] = False
    row.loc["focus_flag"] = False
    row.loc["bubble_flag"] = False

    row_out = pd.DataFrame(row).transpose()
    
    return row_out

####################
# Main process function 2
####################

def segment_wells_sam2_csv(
    root: str | Path,
    exp_name: str,
    sam2_csv_path: str | Path = None,
    min_sa_um: float = 250_000,
    max_sa_um: float = 2_000_000,
    par_flag: bool = False,
    overwrite_well_stats: bool = False,
    force_retrack: bool = False,
):
    """
    SAM2-based well segmentation using pre-computed metadata bridge CSV.
    
    This function replaces the legacy segment_wells() function that used
    regionprops calculations and Hungarian algorithm tracking. Instead,
    it loads pre-computed embryo metadata from SAM2 bridge CSV.
    
    Args:
        root: Project root directory
        exp_name: Experiment name (e.g., "20240418")
        sam2_csv_path: Path to SAM2 metadata CSV (if None, looks for sam2_metadata_{exp_name}.csv)
        min_sa_um: Minimum surface area filter (deprecated - handled by SAM2)
        max_sa_um: Maximum surface area filter (deprecated - handled by SAM2)
        par_flag: Parallel processing flag (deprecated)
        overwrite_well_stats: Overwrite existing stats flag
        force_retrack: Rebuild snip IDs even if present in CSV
        
    Returns:
        DataFrame with embryo tracking data compatible with legacy format
    """
    
    root = Path(root)
    
    # Determine CSV path
    if sam2_csv_path is None:
        sam2_csv_path = root / f"sam2_metadata_{exp_name}.csv"
    else:
        sam2_csv_path = Path(sam2_csv_path)
    
    if not sam2_csv_path.exists():
        raise FileNotFoundError(f"SAM2 metadata CSV not found: {sam2_csv_path}")
    
    print(f"🔄 Loading SAM2 metadata from {sam2_csv_path}")
    
    # Load SAM2 CSV with pre-computed metadata
    sam2_df = pd.read_csv(sam2_csv_path, dtype={'experiment_id': str})
    
    # Filter by experiment if needed
    if exp_name not in sam2_df['experiment_id'].values:
        raise ValueError(f"Experiment {exp_name} not found in SAM2 CSV")

    exp_df = sam2_df[sam2_df['experiment_id'] == exp_name].copy()

    if force_retrack or "snip_id" not in exp_df.columns:
        exp_df["snip_id"] = exp_df.apply(
            lambda r: build_snip_id(r["embryo_id"], r["frame_index"]),
            axis=1
        )
    
    print(f"📊 SAM2 data loaded: {len(exp_df)} snips from experiment {exp_name}")
    
    # Transform SAM2 CSV format to legacy format expected by compile_embryo_stats
    # Key transformation: CSV has one row per snip, legacy expects one row per embryo per well per time
    
    # Extract position from bounding box center
    exp_df['xpos'] = (exp_df['bbox_x_min'] + exp_df['bbox_x_max']) / 2
    exp_df['ypos'] = (exp_df['bbox_y_min'] + exp_df['bbox_y_max']) / 2
    
    # Add required columns for legacy compatibility
    exp_df['fraction_alive'] = 1.0  # Placeholder - SAM2 doesn't compute this
    exp_df['region_label'] = exp_df['embryo_id'].str.extract(r'_e(\d+)$')[0].astype(int)
    
    # Add experiment metadata columns that legacy system expects
    exp_df['experiment_date'] = exp_name
    exp_df['well_id'] = exp_df['video_id'].str.extract(r'_([A-H]\d{2})$')[0]
    exp_df['time_int'] = exp_df['frame_index']
    
    print(f"✅ SAM2 data transformed to legacy format: {len(exp_df)} rows ready")
    
    return exp_df

def segment_wells(
    root: str | Path,
    exp_name: str, 
    min_sa_um: float = 250_000,
    max_sa_um: float = 2_000_000,
    par_flag: bool = False,
    overwrite_well_stats: bool = False,
):
    """
    LEGACY FUNCTION - Replaced by SAM2 bridge approach.
    
    This function has been replaced by segment_wells_sam2_csv() which uses
    pre-computed SAM2 metadata instead of image processing.
    
    For new workflows, use the SAM2 bridge CSV approach instead.
    """
    print("⚠️ WARNING: Using legacy segment_wells function")
    print("   Consider migrating to segment_wells_sam2_csv() with SAM2 bridge CSV")
    print("   See: docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md")
    
    # Keep original function for backwards compatibility during transition
    
    n_workers = np.ceil(os.cpu_count()/4).astype(int)

    root     = Path(root)
    meta_dir = root / "metadata" / "built_metadata_files"
    out_dir     = meta_dir / "embryo_metadata_df_tracked.csv"

    # 1) load + diff
    meta_df     = pd.read_csv(meta_dir / f"{exp_name}_metadata.csv", low_memory=False)
    df_to_process = get_unprocessed_metadata(meta_df, out_dir, overwrite_well_stats)
    df_to_process["experiment_date"] = df_to_process["experiment_date"].astype(str)
    
    # 2) sample shapes, drop missing dates
    # dates   = df_to_process["experiment_date"].unique().tolist()
    ff_dir = root/"built_image_data"/"stitched_FF_images"/exp_name
    shape = sample_experiment_shapes(ff_dir)

    # if missing_dates:
    #     df_to_process = df_to_process.loc[
    #         ~df_to_process["experiment_date"].isin(missing_dates)
    #     ].reset_index(drop=True)

    # 3) build shape_arr (N×2)
    # image_shape_array = np.vstack([
    #     shapes[dt] for dt in df_to_process["experiment_date"]
    # ])

    # 4) find mask folder & collect mask paths, drop missing
    seg_root     = root / "segmentation"
    mask_root    = next(p for p in seg_root.iterdir() if p.is_dir() and "mask" in p.name)
    images_to_process, valid_idx = get_mask_paths_from_diff(df_to_process, mask_root, strict=False)

    if len(valid_idx) < len(df_to_process):
        missing = set(df_to_process.index) - set(valid_idx)
        warnings.warn(
            f"segment_wells: skipping {len(missing)} rows with no mask (indices {sorted(missing)})",
            stacklevel=2
        )
        df_to_process = df_to_process.loc[valid_idx].reset_index(drop=True)
        # image_shape_array     = image_shape_array[valid_idx]

    if df_to_process.empty:
        print("✅ segment_wells: nothing new to process.")
        return {}

    else:
        # initialize empty columns to store embryo information
        df_to_process = df_to_process.copy()
        # remove nuisance columns 
        drop_cols = [col for col in df_to_process.columns if "Unnamed" in col]
        df_to_process = df_to_process.drop(labels=drop_cols, axis=1)

        df_to_process["n_embryos_observed"] = np.nan
        df_to_process["FOV_size_px"] = shape[0] * shape[1]
        df_to_process["FOV_height_px"] = shape[0]
        df_to_process["FOV_width_px"] = shape[1]
        for n in range(4):  # allow for a maximum of 4 embryos per well
            for suffix in ("x", "y", "label", "frac_alive"):
                col_name = f"e{n}_{suffix}"
                if suffix in ("x", "y", "frac_alive"):
                    df_to_process[col_name] = pd.Series([pd.NA] * len(df_to_process), dtype="Float32")
                else:
                    df_to_process[col_name] = pd.Series([pd.NA] * len(df_to_process), dtype="Int8")


        ##########################
        # extract position and live/dead status of each embryo in each well
        ##########################

        meta_lookup   = { (row.well, row.experiment_date, row.time_int): idx
                  for idx, row in df_to_process.iterrows() }
        
        # initialize function
        run_count_embryos = partial(count_embryo_regions, meta_lookup=meta_lookup, image_list=images_to_process, master_df_update=df_to_process,
                                              max_sa_um=max_sa_um, min_sa_um=min_sa_um)
        
        if par_flag:
            raw = process_map(run_count_embryos, range(len(images_to_process)), max_workers=n_workers, chunksize=4)
        else:
            raw = [
                run_count_embryos(i)
                for i in tqdm(range(len(images_to_process)), desc="Counting embryo regions…")
            ]

        # 7) scatter results back
        local_idxs, rows = zip(*raw)
        result_df = pd.concat(rows, ignore_index=True)
        df_to_process.loc[list(local_idxs), result_df.columns] = result_df.values

        # Next, iterate through the extracted positions and use rudimentary tracking to assign embryo instances to stable
        # embryo_id that persists over time

        # get list of unique well instances
        if np.any(np.isnan(df_to_process["n_embryos_observed"].values.astype(float))):
            raise Exception("Missing rows found in metadata df")

        well_id_list = df_to_process["well_id"].unique()
        track_df_list = []
        # print("Performing embryo tracking...")
        for well_id in tqdm(well_id_list, "Doing embryo tracking..."):
            track_df_list.append(do_embryo_tracking(well_id, df_to_process))

        track_df_list = [df for df in track_df_list if isinstance(df, pd.DataFrame)]
        tracked = pd.concat(track_df_list, ignore_index=True)
        # embryo_metadata_df = embryo_metadata_df.iloc[:, 1:]

        # if track_ckpt.exists() and not overwrite_well_stats:
        #     prev = pd.read_csv(track_ckpt, low_memory=False)
        #     tracked  = pd.concat([prev, tracked], ignore_index=True)
        
        # tracked.to_csv(track_ckpt, index=False)

        # print(f"✔️  wrote {track_ckpt}")

        tracked = tracked.dropna(subset=["region_label"])

        return tracked


def compile_embryo_stats(root: str, 
                         tracked_df: pd.DataFrame, 
                         overwrite_flag: bool=False,
                         ld_rat_thresh: float=0.9, 
                         qc_scale_um: int=150, 
                         n_workers: int=1):

    par_flag = n_workers > 1

    # meta_root = os.path.join(root, 'metadata', "built_metadata_files", '')
    # segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

    # track_path = (os.path.join(meta_root, "embryo_metadata_df_tracked.csv"))
    # embryo_metadata_df = pd.read_csv(track_path, index_col=0)

    ######
    # Add key embryo characteristics and flag QC issues
    ######
    # initialize new variables
    new_cols = ["surface_area_um", "surface_area_um", "length_um", "width_um", "bubble_flag",
                "focus_flag", "frame_flag", "dead_flag", "no_yolk_flag"]
    tracked_df["surface_area_um"] = pd.Series([pd.NA] * len(tracked_df), dtype="Float32")
    tracked_df["length_um"] = pd.Series([pd.NA] * len(tracked_df), dtype="Float32")
    tracked_df["width_um"] = pd.Series([pd.NA] * len(tracked_df), dtype="Float32")
    tracked_df["bubble_flag"] = False
    tracked_df["focus_flag"] = False
    tracked_df["frame_flag"] = False
    tracked_df["dead_flag"] = False
    tracked_df["no_yolk_flag"] = False

    # make stable embryo ID
    tracked_df["snip_id"] = tracked_df["embryo_id"] + "_t" + tracked_df["time_int"].astype(str).str.zfill(4)

    # check for existing embryo metadata
    # if os.path.isfile(os.path.join(meta_root, "embryo_metadata_df.csv")) and not overwrite_flag:
    #     embryo_metadata_df_prev = pd.read_csv(os.path.join(meta_root, "embryo_metadata_df.csv"))
    #     # First, check to see if there are new embryo-well-time entries (rows)
    #     merge_skel = embryo_metadata_df.loc[:, ["embryo_id", "time_int"]]
    #     df_all = merge_skel.merge(embryo_metadata_df_prev.drop_duplicates(subset=["embryo_id", "time_int"]), on=["embryo_id", "time_int"],
    #                              how='left', indicator=True)
    #     diff_indices = np.where(df_all['_merge'].values == 'left_only')[0].tolist()

    #     # second, check to see if some or all of the stat columns are already filled in prev table
    #     for nc in new_cols:
    #         embryo_metadata_df[nc] = df_all.loc[:, nc]
    #         sa_nan = np.isnan(embryo_metadata_df["surface_area_um"].values.astype(float))
    #         bb_nan = embryo_metadata_df["bubble_flag"].astype(str) == "nan"
    #         ff_nan = embryo_metadata_df["focus_flag"].astype(str) == "nan"

    #     nan_indices = np.where(sa_nan | ff_nan | bb_nan)[0].tolist()
    #     indices_to_process = np.unique(diff_indices + nan_indices).tolist()
    # else:
    indices_to_process = range(tracked_df.shape[0])

    tracked_df.reset_index(inplace=True, drop=True)

    run_embryo_stats = partial(get_embryo_stats, root=root, embryo_metadata_df=tracked_df,
                                            qc_scale_um=qc_scale_um, ld_rat_thresh=ld_rat_thresh)
    if par_flag:
        emb_df_list = process_map(run_embryo_stats, indices_to_process, max_workers=n_workers, chunksize=4)
        
    else:
        emb_df_list = []
        for index in tqdm(indices_to_process, "Extracting embryo stats..."):
            df_temp = run_embryo_stats(index) #, root, embryo_metadata_df, qc_scale_um, ld_rat_thresh)
            emb_df_list.append(df_temp)

    # update metadata
    emb_df = pd.concat(emb_df_list, axis=0, ignore_index=True)
    emb_df = emb_df.loc[:, ["snip_id"] + new_cols]

    snip1 = emb_df["snip_id"].to_numpy()
    snip2 = tracked_df.loc[indices_to_process, "snip_id"].to_numpy()
    assert np.all(snip1==snip2)
    tracked_df.loc[indices_to_process, new_cols] = emb_df.loc[:, new_cols].values

    # make master flag
    tracked_df["use_embryo_flag"] = ~(
            tracked_df["bubble_flag"].values.astype(bool) | tracked_df["focus_flag"].values.astype(bool) |
            tracked_df["frame_flag"].values.astype(bool) | tracked_df["dead_flag"].values.astype(bool) |
            tracked_df["no_yolk_flag"].values.astype(bool)
            ) #| (tracked_df["well_qc_flag"].values==1).astype(bool))

    # tracked_df.to_csv(os.path.join(meta_root, "embryo_metadata_df.csv"))

    return tracked_df


def extract_embryo_snips(root: str | Path, 
                         stats_df: pd.DataFrame, 
                         outscale: float=6.5, 
                         n_workers: int=1,
                         overwrite_flag: bool=False, outshape=None, dl_rad_um=75):

    
    par_flag = n_workers > 1
    root = Path(root)

    if outshape == None:
        outshape = [576, 256]

    dates = stats_df["experiment_date"].unique()
    if len(dates) > 1:
        raise Exception(f"Detected multiple dates in input dataset: {dates}")
    
    # read in metadata
    meta_root = root / 'metadata' / "embryo_metadata_files"
    os.makedirs(meta_root, exist_ok=True)
    stats_df = stats_df.drop_duplicates(subset=["snip_id"])

    # make directory for embryo snips
    im_snip_dir = root / 'training_data' / 'bf_embryo_snips'
    mask_snip_dir = root / 'training_data' / 'bf_embryo_masks'
    mask_snip_dir.mkdir(parents=True, exist_ok=True)
    #embryo_metadata_df["embryo_id"] + "_" + embryo_metadata_df["time_int"].astype(str)

    if not os.path.isdir(im_snip_dir) | overwrite_flag:
        os.makedirs(im_snip_dir)
        export_indices = range(stats_df.shape[0])
        stats_df["out_of_frame_flag"] = False
        stats_df["snip_um_per_pixel"] = outscale
    else: 
        # get list of exported images
        dates = stats_df["experiment_date"].unique()
        extant_images = list(chain.from_iterable(
                                sorted(im_snip_dir.rglob(f"{d}*.jpg")) for d in dates
                            ))
        extant_snip_array = np.asarray([e.name.replace(".jpg", "") for e in extant_images])
        curr_snip_array = stats_df["snip_id"].unique()
        export_indices = np.where(~np.isin(curr_snip_array, extant_snip_array))[0]
        # embryo_metadata_df = embryo_metadata_df.drop(labels=["_merge"], axis=1)

        # # transfer info from previous version of df01
        # embryo_metadata_df01 = pd.read_csv(os.path.join(meta_root, "embryo_metadata_df01.csv"))
        # embryo_metadata_df01["snip_um_per_pixel"] = outscale

        # embryo_df_new.update(embryo_metadata_df, overwrite=False)
        # embryo_metadata_df = embryo_metadata_df.merge(embryo_metadata_df01.loc[:, ["snip_id", "out_of_frame_flag", "snip_um_per_pixel"]], how="left", on="snip_id", indicator=True)
        # export_indices_df = np.where(embryo_metadata_df["_merge"]=="left_only")[0]
        # embryo_metadata_df = embryo_metadata_df.drop(labels=["_merge"], axis=1)

        # embryo_metadata_df = embryo_df_new.copy()
        # export_indices = np.union1d(export_indices_im, export_indices_df)
        stats_df.loc[export_indices, "out_of_frame_flag"] = False
        stats_df.loc[export_indices, "snip_um_per_pixel"] = outscale


    stats_df["time_int"] = stats_df["time_int"].astype(int)

    # draw random sample to estimate background
    # print("Estimating background...")
    px_mean, px_std = 10, 5 #estimate_image_background(root, stats_df, bkg_seed=309, n_bkg_samples=100)

    # extract snips
    out_of_frame_flags = []

    run_export_snips = partial(export_embryo_snips,root=root, stats_df=stats_df,
                                      dl_rad_um=dl_rad_um, outscale=outscale, outshape=outshape,
                                      px_mean=px_mean, px_std=px_std)
    if par_flag:
        out_of_frame_flags = process_map(run_export_snips, export_indices, max_workers=n_workers, chunksize=4)

    else:
        for r in tqdm(export_indices, "Exporting snips..."):
            oof = run_export_snips(r)
            out_of_frame_flags.append(oof)
        
    out_of_frame_flags = np.asarray(out_of_frame_flags)

    # add oof flag
    stats_df.loc[export_indices, "out_of_frame_flag"] = out_of_frame_flags
    stats_df.loc[export_indices, "use_embryo_flag_orig"] = stats_df.loc[export_indices, "use_embryo_flag"].copy()
    stats_df.loc[export_indices, "use_embryo_flag"] = stats_df.loc[export_indices, "use_embryo_flag"] & ~stats_df.loc[export_indices, "out_of_frame_flag"]

    # save
    exp_name = dates[0]
    stats_df.to_csv(meta_root / f"{exp_name}_embryo_metadata.csv", index=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Process images with optional re-tracking")
    parser.add_argument("--force-retrack", action="store_true", help="Rebuild snip IDs even if present")
    args = parser.parse_args()

    root = REPO_ROOT

    # SAM2 Pipeline Integration - Phase 2 Implementation
    # Using bridge CSV instead of legacy image processing
    exp_name = "20240418"  # Use experiment with SAM2 data available

    # The path to the CSV is now constructed relative to the project root.
    # The PRD specifies the CSV is in the project root.
    sam2_csv_path = root / f"sam2_metadata_{exp_name}.csv"

    # Option 1: Use new SAM2 CSV-based function (recommended)
    tracked_df = segment_wells_sam2_csv(
        root,
        exp_name=exp_name,
        sam2_csv_path=sam2_csv_path,
        force_retrack=args.force_retrack,
    )

    # Using a smaller sample of 100 rows for testing.
    print("Using a smaller sample of 100 rows for testing.")
    tracked_df = tracked_df.head(100)

    # Option 2: Use legacy function (fallback during transition)
    # tracked_df = segment_wells(root, exp_name=exp_name)

    stats_df = compile_embryo_stats(root, tracked_df)

    # print('Extracting embryo snips...')
    extract_embryo_snips(root, stats_df=stats_df, outscale=6.5, dl_rad_um=50, overwrite_flag=False)