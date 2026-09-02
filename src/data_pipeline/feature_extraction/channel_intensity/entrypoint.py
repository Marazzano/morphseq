"""Pool the well-level background null and emit corrected intensity.

THE SEAM THIS SITS ON. ``object_extraction/channel_intensity`` emits RAW poolable evidence;
this module chooses an estimator and applies it. That split is why the null can be re-estimated
later without re-reading a single pixel -- the histograms already on disk are sufficient.

WHY IT RUNS AT EXPERIMENT GRAIN, not per well. A well's null needs every embryo-time in that well,
so it cannot be computed while measuring the first embryo. Reading the merged raw table once is
simpler than a second per-well fanout, and the pooling is arithmetic on histograms -- cheap enough
that the fanout would cost more than it saved.

TWO OUTPUTS, NOT ONE. The null table is emitted separately from the corrected rows because it is a
different grain (well x source product vs embryo-time x source product) and because it is the
artifact you inspect when a corrected number looks wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_pipeline.feature_extraction.channel_intensity.correction import correct_row
from data_pipeline.feature_extraction.channel_intensity.pooling import estimate_well_null

# The columns pooling needs back as real integer arrays, not the JSON strings a CSV round-trip
# leaves behind.
_HISTOGRAM_COLUMNS = ("annulus_hist_counts", "embryo_hist_counts")


def _load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in _HISTOGRAM_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: json.loads(value) if isinstance(value, str) else value
            )
    return frame


def run_channel_intensity_null(
    *,
    channel_intensity_csv: Path,
    output_null_csv: Path,
    output_corrected_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate each (well, source product) null, then correct every row against its own well."""
    raw = _load_raw(Path(channel_intensity_csv))
    if raw.empty:
        raise ValueError(
            f"{channel_intensity_csv} has no rows. A background null cannot be estimated from an "
            "empty measurement table; emit no null rather than a fabricated one."
        )

    group_keys = ["experiment_id", "well_id", "source_image_product_key"]
    missing = [key for key in group_keys if key not in raw.columns]
    if missing:
        raise ValueError(
            f"{channel_intensity_csv} lacks {missing}. The null is keyed by well AND source product "
            "-- intensity off a CLAHE'd raster must never be pooled with intensity off a "
            "quantitative one."
        )

    null_rows: list[dict] = []
    corrected_rows: list[dict] = []
    for keys, group in raw.groupby(group_keys, sort=True):
        identity = dict(zip(group_keys, keys))
        null = estimate_well_null(group.to_dict("records"))
        # by_time is a dict; JSON so it survives a CSV round-trip as one cell rather than being
        # stringified in whatever way pandas happens to choose this version.
        null_row = dict(identity)
        null_row.update(null)
        null_row["null_mode_dn_by_time"] = json.dumps(null["null_mode_dn_by_time"])
        null_rows.append(null_row)

        for row in group.to_dict("records"):
            corrected = dict(row)
            # Histograms stay OUT of the corrected table: they are bulky, already persisted in the
            # raw artifact, and nothing downstream of correction reads them.
            for column in _HISTOGRAM_COLUMNS:
                corrected.pop(column, None)
            corrected.update(correct_row(row, null))
            corrected_rows.append(corrected)

    null_frame = pd.DataFrame(null_rows)
    corrected_frame = pd.DataFrame(corrected_rows)
    _write(null_frame, Path(output_null_csv))
    _write(corrected_frame, Path(output_corrected_csv))
    return null_frame, corrected_frame


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)
