from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from data_pipeline.viz.reporting import (
    plot_metric_histogram,
    plot_metric_vs_reference,
    render_quartile_gallery,
    render_quartile_gallery_vs_band,
    select_quartile_bands,
    select_quartile_bands_vs_band,
)


def _make_synthetic_df(tmp_path: Path, n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        metric = float(i)  # 0..n-1, spans clearly-fail to clearly-pass around cutoff=10
        image_path = tmp_path / f"snip_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 10 % 255, 0, 0)).save(image_path)
        rows.append({"snip_id": f"snip_{i}", "metric": metric, "image_path": str(image_path)})
    return pd.DataFrame(rows)


def test_select_quartile_bands_orders_by_distance_below(tmp_path: Path) -> None:
    df = _make_synthetic_df(tmp_path)
    bands = select_quartile_bands(df, "metric", 10.0, fail_direction="below", n_per_band=3)

    assert list(bands["worst_of_worst"]["metric"]) == [0.0, 1.0, 2.0]
    assert list(bands["borderline_fail"]["metric"]) == [9.0, 8.0, 7.0]
    assert list(bands["borderline_pass"]["metric"]) == [10.0, 11.0, 12.0]
    assert list(bands["clear_pass"]["metric"]) == [19.0, 18.0, 17.0]


def test_select_quartile_bands_above_direction_flips_fail_side(tmp_path: Path) -> None:
    df = _make_synthetic_df(tmp_path)
    bands = select_quartile_bands(df, "metric", 10.0, fail_direction="above", n_per_band=3)

    assert list(bands["worst_of_worst"]["metric"]) == [19.0, 18.0, 17.0]
    assert list(bands["clear_pass"]["metric"]) == [0.0, 1.0, 2.0]


def test_select_quartile_bands_handles_no_fail_rows(tmp_path: Path) -> None:
    df = _make_synthetic_df(tmp_path)
    bands = select_quartile_bands(df, "metric", -1.0, fail_direction="below", n_per_band=3)

    assert bands["worst_of_worst"].empty
    assert bands["borderline_fail"].empty
    assert len(bands["borderline_pass"]) == 3


def test_plot_metric_histogram_writes_file(tmp_path: Path) -> None:
    metric = pd.Series([0.5, 0.7, 0.85, 0.92, 0.99], name="fraction_alive")
    output_path = tmp_path / "hist.png"

    result = plot_metric_histogram(
        metric, 0.90, fail_direction="below", title="synthetic", output_path=output_path
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_quartile_gallery_writes_file_and_handles_missing_image(tmp_path: Path) -> None:
    df = _make_synthetic_df(tmp_path, n=10)
    # simulate a missing image to exercise the fallback panel
    df.loc[0, "image_path"] = str(tmp_path / "does_not_exist.png")
    output_path = tmp_path / "gallery.png"

    result = render_quartile_gallery(
        df,
        "metric",
        5.0,
        fail_direction="below",
        image_path_col="image_path",
        label_col="snip_id",
        title="synthetic gallery",
        output_path=output_path,
        n_per_band=3,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _make_band_df(tmp_path: Path, n: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n):
        metric = 10.0 + i  # 10..19
        stage = float(i)
        image_path = tmp_path / f"band_snip_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 10 % 255, 0, 0)).save(image_path)
        rows.append({
            "snip_id": f"band_snip_{i}",
            "metric": metric,
            "stage": stage,
            "lower": 12.0,
            "upper": 18.0,
            "image_path": str(image_path),
            "fail_flag": not (12.0 <= metric <= 18.0),
        })
    return pd.DataFrame(rows)


def test_select_quartile_bands_vs_band_orders_by_distance_to_nearest_edge(tmp_path: Path) -> None:
    df = _make_band_df(tmp_path)
    bands = select_quartile_bands_vs_band(df, "metric", "lower", "upper", n_per_band=2)

    # metric=10,11 are below lower=12 (fail); metric=19 is above upper=18 (fail).
    assert set(bands["worst_of_worst"]["metric"]) | set(bands["borderline_fail"]["metric"]) == {10.0, 11.0, 19.0}
    # metric in [12, 18] passes; closest to an edge is 12 or 18, farthest is 15 (band center).
    assert 15.0 in set(bands["clear_pass"]["metric"])


def test_plot_metric_vs_reference_writes_file(tmp_path: Path) -> None:
    df = _make_band_df(tmp_path)
    reference_df = pd.DataFrame({
        "ref_stage": [0.0, 20.0],
        "ref_lower": [12.0, 12.0],
        "ref_upper": [18.0, 18.0],
    })
    output_path = tmp_path / "vs_reference.png"

    result = plot_metric_vs_reference(
        df,
        "stage",
        "metric",
        fail_col="fail_flag",
        reference_df=reference_df,
        reference_x_col="ref_stage",
        reference_lower_col="ref_lower",
        reference_upper_col="ref_upper",
        title="synthetic vs reference",
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_quartile_gallery_vs_band_writes_file(tmp_path: Path) -> None:
    df = _make_band_df(tmp_path)
    output_path = tmp_path / "band_gallery.png"

    result = render_quartile_gallery_vs_band(
        df,
        "metric",
        "lower",
        "upper",
        image_path_col="image_path",
        label_col="snip_id",
        title="synthetic band gallery",
        output_path=output_path,
        n_per_band=2,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
