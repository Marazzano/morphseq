from __future__ import annotations

import numpy as np

from analyze.utils.optimal_transport import UOTConfig
from analyze.optimal_transport_morphometrics.uot_masks.preprocess import preprocess_pair


def test_preprocess_pair_sets_work_grid_coord_frame():
    cfg = UOTConfig()
    cfg.align_mode = "none"
    cfg.padding_px = 2
    cfg.downsample_divisor = 0

    src = np.zeros((20, 30), dtype=np.uint8)
    tgt = np.zeros((20, 30), dtype=np.uint8)
    src[5:10, 5:10] = 1
    tgt[6:11, 7:12] = 1

    _src2, _tgt2, meta = preprocess_pair(src, tgt, cfg)
    assert meta["coord_frame_id"] == "work_grid"
    assert meta["coord_frame_version"] == 1
    assert meta["coord_convention"] == "yx"
    assert meta["inputs_coord_frame_id"] == {"src": "unknown", "tgt": "unknown"}
    assert meta["inputs_coord_frame_version"] == {"src": None, "tgt": None}


def test_uot_grid_shim_still_imports():
    from analyze.optimal_transport_morphometrics.uot_masks.uot_grid import (  # noqa: F401
        CanonicalAligner,
        CanonicalGridConfig,
    )


def test_register_to_fixed_identity_on_empty_masks():
    from analyze.coord.register import register_to_fixed

    moving = np.zeros((32, 32), dtype=np.uint8)
    fixed = np.zeros((32, 32), dtype=np.uint8)
    reg = register_to_fixed(moving=moving, fixed=fixed, apply=False)
    assert reg.applied is False
    assert len(reg.transform.transforms) >= 1
    assert reg.moving_in_fixed is None

