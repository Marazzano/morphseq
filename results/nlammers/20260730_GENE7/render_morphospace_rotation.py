"""Render a rotating 3D PCA turntable of GENE7 co-embedded with the WT reference.

Frames only — compile them yourself, e.g.

    ffmpeg -framerate 30 -i figures/morphospace_rotation/frame_%04d.png \\
           -c:v libx264 -pix_fmt yuv420p -crf 18 gene7_morphospace_rotation.mp4

The co-embedding is the *registered* one from ``gene7_morphospace_registration.ipynb``: GENE7 is
rigidly registered onto the WT atlas trajectory using control landmarks. Rather than reimplement
that fit (and risk it drifting from the notebook), this script executes the notebook's own code
cells 1/3/5/7/9 — everything up to and including the registration — and reads ``gene7_registered``,
``atlas_lines`` and ``atlas_reference`` straight out of the resulting namespace. Cells that render or
save figures (13, 19, 25, 26, 28, 30, 33) are never executed, so nothing existing is overwritten.

Two things are deliberately fixed across every frame: the axis limits and the colour normalisation.
Letting either float per frame makes the cloud appear to breathe and the colours to shift while the
camera moves, which reads as data changing when only the viewpoint is.

Matplotlib rather than plotly because there is no static-image engine (kaleido) in this environment,
and a turntable needs a few hundred deterministic PNGs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

REGISTRATION_NOTEBOOK = HERE / "gene7_morphospace_registration.ipynb"
# Setup, load, atlas curve, control landmarks, registration fit. Nothing that draws or writes.
REGISTRATION_CELLS = (1, 3, 5, 7, 9)

DEFAULT_FRAMES = 180
DEFAULT_ELEV = 18.0
DEFAULT_DPI = 200

REFERENCE_COLOR = "#9AA3AF"
CURVE_COLOR = "#111827"


def load_registered_state(*, verbose: bool = True) -> dict:
    """Execute the registration notebook's fitting cells and return its namespace."""
    notebook = json.loads(REGISTRATION_NOTEBOOK.read_text())
    sources = {
        index: "".join(cell["source"])
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    }
    missing = [c for c in REGISTRATION_CELLS if c not in sources]
    if missing:
        raise RuntimeError(f"registration notebook has no code cell(s) {missing}; it was restructured")

    # `display` is an IPython builtin the cells call; stub it so they run headless.
    namespace: dict = {"__name__": "__main__", "display": lambda *a, **k: None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            for index in REGISTRATION_CELLS:
                exec(compile(sources[index], f"<registration cell {index}>", "exec"), namespace)

    for required in ("gene7_registered", "atlas_lines", "atlas_reference", "SELECTED_MODEL"):
        if required not in namespace:
            raise RuntimeError(f"registration cells did not define {required!r}")
    if verbose:
        print(f"registration model : {namespace['SELECTED_MODEL']}")
        print(f"GENE7 registered   : {len(namespace['gene7_registered'])} wells")
        print(f"WT reference       : {len(namespace['atlas_reference'])} snips")
        print(f"atlas trajectory   : {len(next(iter(namespace['atlas_lines'].values())))} points")
    return namespace


def build_figure(state: dict, *, dims=(0, 1, 2), elev: float = DEFAULT_ELEV,
                 reference_alpha: float = 0.35, point_size: float = 27.0,
                 figsize=(9.6, 7.2), zoom_pad: float = 0.18):
    """Build the turntable figure once. Frames then only change the camera azimuth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import gene7_config as cfg
    import plotting as pl

    columns = [pl.PCA_COLUMNS[d] for d in dims]
    gene7 = pl.add_perturbation_column(state["gene7_registered"])
    reference = state["atlas_reference"]
    curve = next(iter(state["atlas_lines"].values()))
    curve_name = next(iter(state["atlas_lines"].keys()))

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111, projection="3d")

    # Faint reference backdrop. Drawn first and with depthshade off so it stays uniformly faint
    # instead of darkening on the near side as the camera swings.
    ax.scatter(*[reference[c].to_numpy(float) for c in columns],
               s=2.6, c=REFERENCE_COLOR, alpha=reference_alpha, linewidths=0,
               depthshade=False)

    ax.plot(*[curve[c].to_numpy(float) for c in columns],
            color=CURVE_COLOR, linewidth=2.6, zorder=5, label=curve_name)

    temperatures = gene7["temperature"].to_numpy(float)
    vmin, vmax = cfg.temperature_limits(temperatures)
    handle = ax.scatter(*[gene7[c].to_numpy(float) for c in columns],
                        c=temperatures, cmap=cfg.TEMPERATURE_SCALE, vmin=vmin, vmax=vmax,
                        s=point_size, edgecolors="black", linewidths=0.4,
                        depthshade=False, zorder=6)

    colorbar = figure.colorbar(handle, ax=ax, shrink=0.55, pad=0.02)
    colorbar.set_label(f"temperature (C)   white = {cfg.REFERENCE_TEMPERATURE:.0f}C", fontsize=9)

    # Fixed limits, from the reference's 0.5-99.5 percentile unioned with everything drawn, so a few
    # extreme reference snips cannot shrink the region of interest and no frame clips a GENE7 point.
    # Framed on the subject (GENE7 + atlas curve), not on the backdrop: the reference cloud has
    # long tails, and letting them set the limits shrinks everything worth looking at. Reference
    # points outside the box simply clip, which is the correct trade for a backdrop.
    for axis, column in zip("xyz", columns):
        low = min(gene7[column].min(), curve[column].min())
        high = max(gene7[column].max(), curve[column].max())
        pad = zoom_pad * (high - low)
        getattr(ax, f"set_{axis}lim")(low - pad, high + pad)
        getattr(ax, f"set_{axis}label")(pl.axis_label(column), fontsize=11, labelpad=8)

    ax.set_title(f"GENE7 registered into the WT morphospace ({state['SELECTED_MODEL']})",
                 fontsize=13, fontweight="bold", y=0.98)

    # Built by hand: the scatter proxies would inherit the backdrop's alpha and the colour map's
    # first colour, neither of which is legible at legend scale.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=REFERENCE_COLOR,
               markeredgecolor="none", label=f"WT reference ({len(reference):,} snips)"),
        Line2D([], [], color=CURVE_COLOR, linewidth=2.6, label=curve_name),
        Line2D([], [], marker="o", linestyle="none", markersize=7, color="#F1F5F9",
               markeredgecolor="black", markeredgewidth=0.5,
               label=f"GENE7 registered ({len(gene7)} wells)"),
    ], loc="lower left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.01, 0.02))

    ax.view_init(elev=elev, azim=0)
    figure.subplots_adjust(left=0.0, right=0.90, top=1.0, bottom=0.0)
    return figure, ax


def render(outdir: Path, *, n_frames: int = DEFAULT_FRAMES, elev: float = DEFAULT_ELEV,
           dpi: int = DEFAULT_DPI, azim_start: float = 0.0, state: dict | None = None,
           **figure_kwargs) -> list[Path]:
    """Write ``n_frames`` PNGs covering a full 360 degree turntable."""
    state = state if state is not None else load_registered_state()
    figure, ax = build_figure(state, elev=elev, **figure_kwargs)
    outdir.mkdir(parents=True, exist_ok=True)

    # Endpoint excluded so frame 0 and the last frame are not duplicates -- the loop stays seamless.
    azimuths = azim_start + np.linspace(0.0, 360.0, n_frames, endpoint=False)
    written = []
    for index, azim in enumerate(azimuths):
        ax.view_init(elev=elev, azim=float(azim))
        path = outdir / f"frame_{index:04d}.png"
        figure.savefig(path, dpi=dpi, facecolor="white")
        written.append(path)
        if (index + 1) % 30 == 0 or index == len(azimuths) - 1:
            print(f"  {index + 1:3d}/{len(azimuths)} frames  (azim {azim:6.1f})")
    import matplotlib.pyplot as plt
    plt.close(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--reference-alpha", type=float, default=0.10)
    parser.add_argument("--outdir", type=Path, default=None,
                        help="default: figures/morphospace_rotation/")
    args = parser.parse_args()

    import gene7_config as cfg
    outdir = args.outdir or (cfg.figure_dir("morphospace_rotation"))

    state = load_registered_state()
    frames = render(outdir, n_frames=args.n_frames, elev=args.elev, dpi=args.dpi,
                    reference_alpha=args.reference_alpha, state=state)
    print(f"\n{len(frames)} frames -> {outdir}")
    print("compile with, e.g.:")
    print(f"  ffmpeg -framerate 30 -i {outdir}/frame_%04d.png "
          "-c:v libx264 -pix_fmt yuv420p -crf 18 gene7_morphospace_rotation.mp4")


if __name__ == "__main__":
    main()
