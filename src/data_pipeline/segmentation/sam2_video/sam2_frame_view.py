"""Temporary SAM2 frame views.

SAM2 video prediction expects a directory containing sequentially named image files
(`00000.jpg`, `00001.jpg`, ...). This module builds that backend-local view from a
validated frame/model view and carries the mapping back to pipeline frame identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterator

import pandas as pd


REQUIRED_SAM2_FRAME_VIEW_COLUMNS: tuple[str, ...] = (
    "image_id",
    "time_index",
    "source_image_path",
    "image_width_px",
    "image_height_px",
)


@dataclass(frozen=True)
class Sam2FrameView:
    """A temporary SAM2 frame directory plus index-to-frame identity mapping."""

    path: Path
    index: pd.DataFrame

    @property
    def by_sam2_index(self) -> dict[int, dict[str, object]]:
        return {
            int(row["sam2_frame_index"]): row.to_dict()
            for _, row in self.index.iterrows()
        }


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def _resolve_source_path(value: object, image_root: Path | None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    return path


@contextmanager
def build_sam2_frame_view(
    model_frame_view: pd.DataFrame,
    *,
    image_root: Path | None = None,
    temp_root: Path | None = None,
    suffix: str = ".jpg",
) -> Iterator[Sam2FrameView]:
    """Create a temporary sequential symlink directory for SAM2.

    Args:
        model_frame_view: Frame rows sorted by the pipeline only through `time_index`.
        image_root: Optional root used to resolve relative `source_image_path` values.
        temp_root: Optional temp directory parent.
        suffix: Filename suffix for SAM2-local links. SAM2 commonly expects `.jpg`.

    Yields:
        `Sam2FrameView` with `path` and a mapping table containing `sam2_frame_index`.
    """
    _require_columns(model_frame_view, REQUIRED_SAM2_FRAME_VIEW_COLUMNS, "model_frame_view")
    if model_frame_view.empty:
        raise ValueError("model_frame_view cannot be empty")

    ordered = (
        model_frame_view.copy()
        .sort_values(["time_index", "image_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    temp_parent = Path(temp_root) if temp_root is not None else None
    with tempfile.TemporaryDirectory(prefix="sam2_frames_", dir=temp_parent) as tmp:
        frame_dir = Path(tmp)
        rows: list[dict[str, object]] = []

        for sam2_idx, row in ordered.iterrows():
            src = _resolve_source_path(row["source_image_path"], image_root)
            if not src.exists():
                raise FileNotFoundError(f"Source frame not found: {src}")

            link_path = frame_dir / f"{sam2_idx:05d}{suffix}"
            link_path.symlink_to(src.resolve())

            mapped = row.to_dict()
            mapped["sam2_frame_index"] = int(sam2_idx)
            mapped["sam2_frame_path"] = str(link_path)
            mapped["source_image_path_resolved"] = str(src.resolve())
            rows.append(mapped)

        index = pd.DataFrame(rows)
        yield Sam2FrameView(path=frame_dir, index=index)
