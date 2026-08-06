"""Plot helpers for GENE7 morphology PCA against the reference trajectory.

Encodes the visual grammar the notebook uses, so the notebook stays declarative:

    colour   = rearing temperature (continuous, RdBu_r — cold blue, WHITE AT 28C, hot red)
    marker   = perturbation target (symbol per crispant target)
    black curve = the wildtype reference spline

Temperature is the continuous variable of interest, so it gets the colour channel; target is
categorical and gets symbols. That is the opposite of the usual default and is deliberate.

The colour range is anchored at 28C rather than at the data's own min/max — see
``gene7_config.temperature_limits``. White therefore means standard rearing temperature in every
panel of every notebook, instead of drifting with whichever cohorts are in the current frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PCA_COLUMNS: tuple[str, ...] = tuple(f"PCA_{p:02}_bio" for p in range(5))

# Plotly symbol vocabularies differ between 2D and 3D traces, so they are listed separately.
SYMBOLS_2D: tuple[str, ...] = ("circle", "square", "diamond", "cross", "triangle-up", "x")
SYMBOLS_3D: tuple[str, ...] = ("circle", "square", "diamond", "cross", "x")

from gene7_config import REFERENCE_TEMPERATURE, TEMPERATURE_SCALE, temperature_limits

SPLINE_COLOUR = "black"

FONT = dict(family="Arial, sans-serif", size=15, color="black")


def axis_label(column: str) -> str:
    """``PCA_00_bio`` -> ``morph PC 1``."""
    return f"morph PC {int(str(column).split('_')[1]) + 1}"


def base_target(target: object) -> str:
    """Strip the temperature-arm suffix from a sequencing target label.

    GENE7's ``target`` encodes both the crispant and the thermal arm (``atf6,hot``,
    ``Control,cold``). Symbols should distinguish the genetic perturbation only — temperature is
    already carried by colour — so the arm suffix is removed.
    """
    text = str(target)
    parts = [part for part in text.split(",") if part not in ("hot", "cold")]
    return ",".join(parts) if parts else text


def add_perturbation_column(frame: pd.DataFrame, *, source_column: str = "target") -> pd.DataFrame:
    """Add ``perturbation_group``: the crispant target with the thermal arm stripped."""
    out = frame.copy()
    out["perturbation_group"] = [base_target(value) for value in out[source_column]]
    return out


def _symbol_map(groups: "list[str]", symbols: "tuple[str, ...]") -> dict[str, str]:
    """Stable group -> symbol assignment. Controls first so they read as the baseline."""
    ordered = sorted(groups, key=lambda g: (not str(g).lower().startswith("control"), str(g)))
    return {group: symbols[index % len(symbols)] for index, group in enumerate(ordered)}


def _spline_trace_2d(spline: pd.DataFrame, x: str, y: str, *, name: str, dash: str | None = None):
    return go.Scatter(
        x=spline[x],
        y=spline[y],
        mode="lines",
        line=dict(color=SPLINE_COLOUR, width=3, dash=dash),
        name=name,
    )


def _spline_trace_3d(spline: pd.DataFrame, x: str, y: str, z: str, *, name: str, dash=None):
    return go.Scatter3d(
        x=spline[x],
        y=spline[y],
        z=spline[z],
        mode="lines",
        line=dict(color=SPLINE_COLOUR, width=5, dash=dash),
        name=name,
    )


def plot_pca_2d(
    gene7: pd.DataFrame,
    splines: "dict[str, pd.DataFrame]",
    *,
    dims: "tuple[int, int]" = (0, 1),
    reference: pd.DataFrame | None = None,
    title: str = "GENE7 morphology PCA vs wildtype reference",
    width: int = 950,
    height: int = 750,
) -> go.Figure:
    """2D PCA scatter: colour = temperature, symbol = perturbation, black curve = reference spline.

    ``reference`` is optional; when given it is drawn as a faint grey backdrop so the reference
    population's extent is visible without competing with the GENE7 points.
    """
    x, y = PCA_COLUMNS[dims[0]], PCA_COLUMNS[dims[1]]
    frame = add_perturbation_column(gene7)
    groups = sorted(frame["perturbation_group"].unique())

    figure = go.Figure()

    if reference is not None:
        figure.add_trace(
            go.Scatter(
                x=reference[x],
                y=reference[y],
                mode="markers",
                marker=dict(size=3, color="lightgrey", opacity=0.35),
                name="reference population",
                hoverinfo="skip",
            )
        )

    symbols = _symbol_map(groups, SYMBOLS_2D)
    temperatures = pd.to_numeric(frame["temperature"], errors="coerce")
    temp_min, temp_max = temperature_limits(temperatures)
    for group in groups:
        subset = frame.loc[frame["perturbation_group"] == group]
        figure.add_trace(
            go.Scatter(
                x=subset[x],
                y=subset[y],
                mode="markers",
                name=group,
                marker=dict(
                    size=9,
                    symbol=symbols[group],
                    color=pd.to_numeric(subset["temperature"], errors="coerce"),
                    colorscale=TEMPERATURE_SCALE,
                    cmin=temp_min,
                    cmax=temp_max,
                    line=dict(color="black", width=0.6),
                    colorbar=dict(title=f"temp (C)<br>white={REFERENCE_TEMPERATURE:.0f}C", len=0.55, y=0.5),
                    showscale=(group == groups[0]),
                ),
                customdata=np.stack(
                    [subset["temperature"], subset["target"], subset["well_id"]], axis=-1
                ),
                hovertemplate=(
                    f"{axis_label(x)}: %{{x:.2f}}<br>{axis_label(y)}: %{{y:.2f}}"
                    "<br>temp: %{customdata[0]}C<br>target: %{customdata[1]}"
                    "<br>%{customdata[2]}<extra></extra>"
                ),
            )
        )

    for name, spline in splines.items():
        figure.add_trace(
            _spline_trace_2d(
                spline,
                x,
                y,
                name=f"reference spline ({name})",
                dash="dash" if name == "unweighted" else None,
            )
        )

    figure.update_layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(title=axis_label(x)),
        yaxis=dict(title=axis_label(y)),
        font=FONT,
        legend=dict(title="perturbation", itemsizing="constant"),
        plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    return figure


def plot_pca_3d(
    gene7: pd.DataFrame,
    splines: "dict[str, pd.DataFrame]",
    *,
    dims: "tuple[int, int, int]" = (0, 1, 2),
    reference: pd.DataFrame | None = None,
    title: str = "GENE7 morphology PCA vs wildtype reference (3D)",
    width: int = 1100,
    height: int = 900,
) -> go.Figure:
    """3D PCA scatter with the same encoding as ``plot_pca_2d``."""
    x, y, z = (PCA_COLUMNS[dims[0]], PCA_COLUMNS[dims[1]], PCA_COLUMNS[dims[2]])
    frame = add_perturbation_column(gene7)
    groups = sorted(frame["perturbation_group"].unique())
    symbols = _symbol_map(groups, SYMBOLS_3D)
    temperatures = pd.to_numeric(frame["temperature"], errors="coerce")
    temp_min, temp_max = temperature_limits(temperatures)

    figure = go.Figure()

    if reference is not None:
        figure.add_trace(
            go.Scatter3d(
                x=reference[x],
                y=reference[y],
                z=reference[z],
                mode="markers",
                marker=dict(size=1.5, color="lightgrey", opacity=0.25),
                name="reference population",
                hoverinfo="skip",
            )
        )

    for group in groups:
        subset = frame.loc[frame["perturbation_group"] == group]
        figure.add_trace(
            go.Scatter3d(
                x=subset[x],
                y=subset[y],
                z=subset[z],
                mode="markers",
                name=group,
                marker=dict(
                    size=4.5,
                    symbol=symbols[group],
                    color=pd.to_numeric(subset["temperature"], errors="coerce"),
                    colorscale=TEMPERATURE_SCALE,
                    cmin=temp_min,
                    cmax=temp_max,
                    line=dict(color="black", width=0.5),
                    colorbar=dict(title=f"temp (C)<br>white={REFERENCE_TEMPERATURE:.0f}C", len=0.5, y=0.5, x=1.02),
                    showscale=(group == groups[0]),
                ),
                customdata=np.stack(
                    [subset["temperature"], subset["target"], subset["well_id"]], axis=-1
                ),
                hovertemplate=(
                    "temp: %{customdata[0]}C<br>target: %{customdata[1]}"
                    "<br>%{customdata[2]}<extra></extra>"
                ),
            )
        )

    for name, spline in splines.items():
        figure.add_trace(
            _spline_trace_3d(
                spline, x, y, z, name=f"reference spline ({name})",
                dash="dash" if name == "unweighted" else None,
            )
        )

    figure.update_layout(
        width=width,
        height=height,
        title=title,
        scene=dict(
            xaxis=dict(title=axis_label(x)),
            yaxis=dict(title=axis_label(y)),
            zaxis=dict(title=axis_label(z)),
        ),
        font=FONT,
        legend=dict(title="perturbation", itemsizing="constant"),
    )
    return figure


def distance_to_spline(frame: pd.DataFrame, spline: pd.DataFrame, *, n_components: int = 5,
                       stride: int = 10) -> np.ndarray:
    """Euclidean distance from each row to its nearest point on the spline, in PCA space.

    ``stride`` subsamples the 2500 spline points; at the curve's sampling density this changes the
    result negligibly and keeps the pairwise computation small.
    """
    columns = list(PCA_COLUMNS[:n_components])
    points = frame.loc[:, columns].to_numpy(float)
    curve = spline.loc[:, columns].to_numpy(float)[::stride]
    return np.sqrt(((points[:, None, :] - curve[None, :, :]) ** 2).sum(-1)).min(1)


def save_figure(figure: go.Figure, path) -> None:
    """Write HTML always, PNG only when a static-image engine is available.

    ``kaleido`` is not installed in every environment on this cluster, and a missing static exporter
    should not abort a notebook run — the interactive HTML is the artifact that matters for 3D.
    """
    from pathlib import Path

    target = Path(path)
    figure.write_html(target.with_suffix(".html"))
    try:
        figure.write_image(target.with_suffix(".png"), scale=2)
    except Exception as exc:  # kaleido/orca absent or misconfigured
        print(f"  [note] PNG export skipped for {target.name}: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Cohort-axis image strips
# ---------------------------------------------------------------------------


def image_strip_figure(strip: pd.DataFrame, *, coord_column: str, title: str,
                       tile_px: int = 130, cmap: str = "gray"):
    """Render one cohort axis as a row of embryo snips, ordered low to high along the axis.

    Uses matplotlib rather than plotly: this is a raster contact sheet, and embedding a dozen PNGs
    per row as plotly images would bloat the notebook for no interactive benefit.
    """
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    present = strip.loc[strip["image_exists"]].reset_index(drop=True)
    if present.empty:
        raise ValueError(f"no resolvable images for {title!r}")

    n = len(present)
    figure, axes = plt.subplots(1, n, figsize=(n * tile_px / 100 * 1.4, tile_px / 100 * 1.9))
    axes = np.atleast_1d(axes)
    for position, row in present.iterrows():
        axis = axes[position]
        axis.imshow(mpimg.imread(row["image_path"]), cmap=cmap)
        axis.set_title(f"{row[coord_column]:+.2f}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("0.7")
    figure.suptitle(title, fontsize=11, y=1.04)
    figure.tight_layout()
    return figure
