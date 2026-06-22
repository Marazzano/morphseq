"""SAM2 loader wrapper.

## Why this is finicky — read before editing

SAM2 uses Hydra for config resolution. Hydra searches `pkg://sam2` (the installed
package's config directory) when looking up config names. In production we do NOT
pip-install sam2; instead we rely on a repo checkout at `<models_root>/sam2/`.

The only reliable way to make Hydra find the configs without a pip install is:
  1. `sys.path` must include the *parent* of the `sam2/` package dir (so `import sam2`
     works at all).
  2. The *working directory must be the `sam2/` package dir itself* when
     `build_sam2_video_predictor` is called — Hydra's `pkg://sam2` search resolves
     relative to the package found via that cwd/sys.path combination.
  3. The config name passed to `build_sam2_video_predictor` must be a path *relative
     to the package dir*, e.g. `"configs/sam2.1/sam2.1_hiera_s.yaml"` — NOT an
     absolute path and NOT just a bare name like `"sam2.1_hiera_s"`.

## What the caller must pass

  - `sam2_models_root`: the directory that *contains* the `sam2/` package subdir.
    In the pipeline this is `MODELS_DIR / "sam2"` from `env.yaml` — which on disk is
    typically a symlink to the checkout. Pass the symlink as-is; do not resolve it
    before passing (resolved absolute paths do not change Hydra behaviour either way,
    but the symlink path is what you'll have from the config).
  - `config_path`: pass as a *relative* path like
    `Path("configs/sam2.1/sam2.1_hiera_s.yaml")`. This loader resolves it to
    `sam2_pkg_dir / config_path` for the exists-check, then strips back to the relative
    form for Hydra.
  - `checkpoint_path`: pass as a relative path like
    `Path("checkpoints/sam2.1_hiera_small.pt")`. Resolved against `sam2_models_root`.

## Verified working pattern (Session C smoke, 2026-06-22)

    predictor = load_sam2_video_predictor(
        sam2_models_root=MODELS_DIR / "sam2",          # symlink ok
        config_path=Path("configs/sam2.1/sam2.1_hiera_s.yaml"),   # relative
        checkpoint_path=Path("checkpoints/sam2.1_hiera_small.pt"), # relative
        device="cuda",
    )
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from data_pipeline.utils.cuda_diagnostics import resolve_device


def _resolve_sam2_pkg_dir(sam2_models_root: Path) -> Path:
    sam2_models_root = Path(sam2_models_root)
    if not sam2_models_root.exists():
        raise FileNotFoundError(f"SAM2 models root not found: {sam2_models_root}")

    # Typical layout: <sam2_models_root>/sam2/<python package>
    candidate = sam2_models_root / "sam2"
    if candidate.is_dir():
        return candidate

    # Alternate: the configured root might already be the package dir.
    if (sam2_models_root / "build_sam.py").exists():
        return sam2_models_root

    raise FileNotFoundError(
        "Could not locate SAM2 package directory. "
        f"Expected either {candidate} or a package-like directory at {sam2_models_root}."
    )


def load_sam2_video_predictor(
    *,
    sam2_models_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
):
    device = resolve_device(device)
    sam2_models_root = Path(sam2_models_root)
    sam2_pkg_dir = _resolve_sam2_pkg_dir(sam2_models_root)
    sam2_models_root = sam2_models_root.resolve()
    sam2_pkg_dir = sam2_pkg_dir.resolve()

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    # Resolve relative paths:
    # - config is typically under sam2_pkg_dir (e.g. sam2/configs/*.yaml)
    # - checkpoint is typically under sam2_models_root (e.g. checkpoints/*.pt)
    if not config_path.is_absolute():
        config_path = sam2_pkg_dir / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = sam2_models_root / checkpoint_path

    if not config_path.exists():
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    # Make `import sam2...` work:
    # - if pkg dir is <root>/sam2 then sys.path should include <root>
    # - if pkg dir is <root> itself, sys.path should include parent(<root>)
    if sam2_pkg_dir.name == "sam2":
        sys_path_root = sam2_pkg_dir.parent
    else:
        sys_path_root = sam2_pkg_dir.parent

    root_str = str(sys_path_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    original_cwd = os.getcwd()
    try:
        os.chdir(str(sam2_pkg_dir))
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import SAM2. "
                f"sam2_models_root={sam2_models_root}, sam2_pkg_dir={sam2_pkg_dir}."
            ) from e
        # Hydra expects config_name relative to the sam2 package config search path.
        # Pass a relative path like "configs/sam2/sam2_hiera_l.yaml" (NOT an absolute path).
        cfg_name = str(config_path.relative_to(sam2_pkg_dir))
        return build_sam2_video_predictor(cfg_name, str(checkpoint_path), device=device)
    finally:
        os.chdir(original_cwd)
