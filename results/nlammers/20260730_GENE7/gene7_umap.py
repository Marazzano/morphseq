"""UMAP views of the GENE7 morphology latents.

Companion to ``gene7_morphospace.ipynb``, which works in the linear PCA basis. This module asks
what a nonlinear embedding adds, and whether it should be run on the **full** 80-dimensional
biological latent block (``z_mu_b_*``) or on a **denoised** reduction (the first 10 GENE7-native
PCs, the same basis ``data/gene7_global_scores.csv`` carries).

Why the input space matters here
--------------------------------
n = 553 wells in 80 dimensions is a regime where most of the pairwise distance that UMAP builds its
graph from is noise: the GENE7-native spectrum puts ~all structure in the leading handful of PCs, so
the remaining ~70 dimensions contribute variance without contributing signal. Running on the first
10 PCs is therefore not merely a speed-up, it changes which neighbours the graph connects. Both are
computed here so the difference is visible rather than assumed.

Nothing is scaled before embedding. The biological latents share a common prior scale, and the PC
basis is fit unwhitened (``whiten=False``, matching ``cohort_axes.fit_global_basis``), so a
per-feature standardisation would re-inflate the degenerate tail that the reduction exists to drop.

Reading a UMAP honestly
-----------------------
Distances between well-separated clusters are not meaningful, only the neighbourhood structure is,
and the layout changes with ``n_neighbors``. ``sweep_n_neighbors`` and ``embedding_quality`` exist so
that sensitivity and neighbourhood preservation are reported alongside every layout rather than
inferred from how tidy it looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

import gene7_config as cfg

# UMAP defaults. random_state is fixed so a layout is reproducible; it also forces single-threaded
# execution, which is irrelevant at n = 553.
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "euclidean"
RANDOM_STATE = 42
N_PCS = 10

LATENT_PREFIX = "z_mu_b"

# Categorical colours for the four perturbation groups. Control is deliberately neutral grey so the
# three crispant groups read as the signal.
GROUP_COLORS = {
    "Control": "#9CA3AF",
    "atf6": "#EF4444",
    "ctcf": "#06B6D4",
    "wfs1a,wfs1b": "#8B5CF6",
}
STAGE_COLORS = {24: "#FDE68A", 30: "#F59E0B", 36: "#B45309"}


def latent_columns(frame: pd.DataFrame) -> list[str]:
    """The ``z_mu_b_*`` columns in ascending dimension order."""
    cols = [c for c in frame.columns if c.startswith(LATENT_PREFIX)]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


@dataclass
class Space:
    """One input space for UMAP: a name, a matrix, and how it was built."""

    name: str
    label: str
    matrix: np.ndarray
    note: str = ""
    variance_explained: float | None = None


def build_spaces(frame: pd.DataFrame, *, n_pcs: int = N_PCS) -> dict[str, Space]:
    """The two candidate input spaces: the full latent block and its leading PCs."""
    cols = latent_columns(frame)
    if not cols:
        raise ValueError("frame carries no z_mu_b_* columns; this is not a latent frame.")
    full = frame.loc[:, cols].to_numpy(dtype=float)

    pca = PCA(n_components=n_pcs, whiten=False, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(full)
    captured = float(pca.explained_variance_ratio_.sum())

    return {
        "full": Space(
            name="full",
            label=f"full latent space ({len(cols)}D)",
            matrix=full,
            note=f"all {len(cols)} z_mu_b_* dimensions, unscaled",
            variance_explained=1.0,
        ),
        "pc10": Space(
            name="pc10",
            label=f"first {n_pcs} PCs",
            matrix=reduced,
            note=f"GENE7-native PCA, unwhitened, {captured:.1%} of latent variance",
            variance_explained=captured,
        ),
    }


def fit_umap(
    matrix: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = N_NEIGHBORS,
    min_dist: float = MIN_DIST,
    metric: str = METRIC,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Fit one UMAP embedding. Imported lazily so the module loads without umap-learn."""
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def embedding_quality(matrix: np.ndarray, embedding: np.ndarray, *,
                      n_neighbors: int = N_NEIGHBORS) -> dict[str, float]:
    """Neighbourhood and global-structure preservation for one embedding.

    ``trustworthiness`` is local: what fraction of each point's embedded neighbours were also
    neighbours in the input space. ``distance_spearman`` is global: whether the *ordering* of
    pairwise distances survives. UMAP optimises the former and makes no promise about the latter,
    so a high trustworthiness with a mediocre correlation is the expected result, not a fault.
    """
    trust = float(trustworthiness(matrix, embedding, n_neighbors=n_neighbors))
    rho = float(spearmanr(pdist(matrix), pdist(embedding)).statistic)
    return {"trustworthiness": trust, "distance_spearman": rho}


def knn_purity(matrix: np.ndarray, labels: np.ndarray, *, k: int = N_NEIGHBORS) -> float:
    """Mean fraction of each point's ``k`` nearest neighbours sharing its label.

    Preferred over a silhouette score here. Silhouette rewards compact, convex, well-separated
    blobs; this cloud is a curved continuum, so a real temperature gradient along the manifold
    scores ~0 by silhouette while being obvious by eye. Purity only asks whether near neighbours
    agree, which is the property UMAP actually optimises and the one a gradient satisfies.

    Compare against ``label_baseline`` (chance for the observed class proportions), never against 0.
    """
    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
    indices = neighbours.kneighbors(matrix, return_distance=False)[:, 1:]
    return float((labels[indices] == labels[:, None]).mean())


def label_baseline(labels: np.ndarray) -> float:
    """Expected purity if neighbours were drawn at random: sum of squared class proportions."""
    _, counts = np.unique(labels, return_counts=True)
    return float(((counts / counts.sum()) ** 2).sum())


def label_structure(
    frame: pd.DataFrame,
    spaces: "dict[str, Space]",
    results: "dict[tuple[str, int], UmapResult]",
    *,
    labels: "tuple[str, ...]" = ("stage", "temperature", "perturbation_group", "experiment_id"),
    k: int = N_NEIGHBORS,
    n_shuffles: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """kNN purity per label, for chance, a label shuffle, each input space and each embedding.

    The shuffle column is the control that matters: it confirms the metric returns chance when the
    labels carry no information, so any excess over baseline is structure rather than bookkeeping.
    """
    rng = np.random.default_rng(seed)
    reference = next(iter(results.values())).coords
    rows = []
    for label in labels:
        values = frame[label].astype(str).to_numpy()
        row = {
            "label": label,
            "chance": label_baseline(values),
            "shuffled": float(np.mean([knn_purity(reference, rng.permutation(values), k=k)
                                       for _ in range(n_shuffles)])),
        }
        for key, space in spaces.items():
            row[f"input:{key}"] = knn_purity(space.matrix, values, k=k)
        for (key, n_components), result in results.items():
            row[f"{key}:{n_components}D"] = knn_purity(result.coords, values, k=k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


@dataclass
class UmapResult:
    """One fitted layout plus its diagnostics."""

    space: str
    n_components: int
    n_neighbors: int
    coords: np.ndarray
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return [f"UMAP{i + 1}" for i in range(self.n_components)]

    def to_frame(self, metadata: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(self.coords, columns=self.columns, index=metadata.index)
        return pd.concat([out, metadata], axis=1)


def run_embeddings(
    spaces: "dict[str, Space]",
    *,
    dimensions: "tuple[int, ...]" = (2, 3),
    n_neighbors: int = N_NEIGHBORS,
    verbose: bool = True,
) -> "dict[tuple[str, int], UmapResult]":
    """Every (space x n_components) combination, with diagnostics attached."""
    results: dict[tuple[str, int], UmapResult] = {}
    for key, space in spaces.items():
        for n_components in dimensions:
            coords = fit_umap(space.matrix, n_components=n_components, n_neighbors=n_neighbors)
            quality = embedding_quality(space.matrix, coords, n_neighbors=n_neighbors)
            results[(key, n_components)] = UmapResult(
                space=key, n_components=n_components, n_neighbors=n_neighbors,
                coords=coords, quality=quality,
            )
            if verbose:
                print(f"  {space.label:<26} {n_components}D  "
                      f"trust={quality['trustworthiness']:.3f}  "
                      f"rho={quality['distance_spearman']:.3f}")
    return results


def sweep_n_neighbors(
    spaces: "dict[str, Space]",
    *,
    values: "tuple[int, ...]" = (5, 15, 30, 50),
    n_components: int = 2,
) -> "tuple[dict[tuple[str, int], UmapResult], pd.DataFrame]":
    """Refit 2D layouts across ``n_neighbors``. Returns the layouts and a tidy diagnostics table."""
    layouts: dict[tuple[str, int], UmapResult] = {}
    rows = []
    for key, space in spaces.items():
        for k in values:
            coords = fit_umap(space.matrix, n_components=n_components, n_neighbors=k)
            quality = embedding_quality(space.matrix, coords, n_neighbors=N_NEIGHBORS)
            layouts[(key, k)] = UmapResult(space=key, n_components=n_components, n_neighbors=k,
                                           coords=coords, quality=quality)
            rows.append({"space": space.label, "n_neighbors": k, **quality})
    return layouts, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _temperature_norm(values):
    from matplotlib.colors import Normalize

    vmin, vmax = cfg.temperature_limits(values)
    return Normalize(vmin=vmin, vmax=vmax)


def scatter_2d(ax, frame: pd.DataFrame, *, color_by: str, x: str = "UMAP1", y: str = "UMAP2",
               size: float = 26, legend: bool = True):
    """One 2D UMAP panel coloured by ``color_by``.

    ``temperature`` uses the project-wide diverging scale anchored at 28C (see ``gene7_config``);
    the categorical variables get explicit maps so colour means the same thing in every panel.
    """
    if color_by == "temperature":
        values = pd.to_numeric(frame["temperature"], errors="coerce")
        handle = ax.scatter(frame[x], frame[y], c=values, cmap=cfg.TEMPERATURE_SCALE,
                            norm=_temperature_norm(values), s=size, linewidths=0.4,
                            edgecolors="black")
        return handle
    if color_by == "stage":
        groups, colors = sorted(frame["stage"].dropna().unique()), STAGE_COLORS
    elif color_by == "perturbation_group":
        groups = [g for g in GROUP_COLORS if g in set(frame["perturbation_group"])]
        colors = GROUP_COLORS
    else:
        groups = sorted(frame[color_by].dropna().unique())
        colors = {g: c for g, c in zip(groups, plt_default_cycle(len(groups)))}
    for group in groups:
        subset = frame.loc[frame[color_by] == group]
        ax.scatter(subset[x], subset[y], s=size, label=str(group),
                   color=colors.get(group, "#374151"), linewidths=0.4, edgecolors="black")
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best", handletextpad=0.3)
    return None


def plt_default_cycle(n: int) -> list[str]:
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [cycle[i % len(cycle)] for i in range(n)]


def grid_2d(
    results: "dict[tuple[str, int], UmapResult]",
    metadata: pd.DataFrame,
    spaces: "dict[str, Space]",
    *,
    color_bys: "tuple[str, ...]" = ("temperature", "stage", "perturbation_group"),
    space_order: "tuple[str, ...]" = ("full", "pc10"),
    figsize: "tuple[float, float]" = (15.0, 9.5),
):
    """Rows = input space, columns = colour variable, for the 2D layouts."""
    import matplotlib.pyplot as plt

    rows = [s for s in space_order if (s, 2) in results]
    figure, axes = plt.subplots(len(rows), len(color_bys), figsize=figsize, squeeze=False)
    for r, key in enumerate(rows):
        result = results[(key, 2)]
        frame = result.to_frame(metadata)
        for c, color_by in enumerate(color_bys):
            ax = axes[r][c]
            handle = scatter_2d(ax, frame, color_by=color_by, legend=(r == 0))
            if handle is not None:
                figure.colorbar(handle, ax=ax, label="temperature (C)")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_title(f"{spaces[key].label}  ·  {color_by}", fontsize=10)
            if c == 0:
                ax.text(0.02, 0.98,
                        f"trust={result.quality['trustworthiness']:.3f}\n"
                        f"rho={result.quality['distance_spearman']:.3f}",
                        transform=ax.transAxes, va="top", fontsize=8, color="#475569")
            ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def sweep_figure(
    layouts: "dict[tuple[str, int], UmapResult]",
    metadata: pd.DataFrame,
    spaces: "dict[str, Space]",
    *,
    color_by: str = "temperature",
    space_order: "tuple[str, ...]" = ("full", "pc10"),
):
    """One row per input space, one column per ``n_neighbors`` value."""
    import matplotlib.pyplot as plt

    values = sorted({k for _, k in layouts})
    rows = [s for s in space_order if any(key == s for key, _ in layouts)]
    figure, axes = plt.subplots(len(rows), len(values),
                                figsize=(3.6 * len(values), 3.6 * len(rows)), squeeze=False)
    for r, key in enumerate(rows):
        for c, k in enumerate(values):
            ax = axes[r][c]
            result = layouts[(key, k)]
            scatter_2d(ax, result.to_frame(metadata), color_by=color_by, size=14, legend=False)
            ax.set_title(f"{spaces[key].label}  ·  n_neighbors={k}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    figure.tight_layout()
    return figure


def umap_3d_figure(
    frame: pd.DataFrame,
    *,
    color_by: str = "temperature",
    title: str = "GENE7 morphology UMAP (3D)",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """Interactive 3D UMAP. Symbol encodes perturbation group, colour encodes ``color_by``.

    3D is exported as HTML rather than a static image on purpose: a fixed camera angle on a 3D
    point cloud is one arbitrary projection, and the rotation is the part that carries information.
    """
    figure = go.Figure()
    x, y, z = "UMAP1", "UMAP2", "UMAP3"
    groups = [g for g in GROUP_COLORS if g in set(frame["perturbation_group"])]
    symbols = ["circle", "diamond", "square", "x"]

    if color_by == "temperature":
        temps = pd.to_numeric(frame["temperature"], errors="coerce")
        cmin, cmax = cfg.temperature_limits(temps)
    for index, group in enumerate(groups):
        subset = frame.loc[frame["perturbation_group"] == group]
        if color_by == "temperature":
            marker = dict(
                size=4.5,
                symbol=symbols[index % len(symbols)],
                color=pd.to_numeric(subset["temperature"], errors="coerce"),
                colorscale=cfg.TEMPERATURE_SCALE,
                cmin=cmin, cmax=cmax,
                line=dict(color="black", width=0.5),
                colorbar=dict(title=f"temp (C)<br>white={cfg.REFERENCE_TEMPERATURE:.0f}C",
                              len=0.5, y=0.5, x=1.02),
                showscale=(index == 0),
            )
        else:
            marker = dict(size=4.5, symbol=symbols[index % len(symbols)],
                          color=GROUP_COLORS.get(group, "#374151"),
                          line=dict(color="black", width=0.5))
        figure.add_trace(
            go.Scatter3d(
                x=subset[x], y=subset[y], z=subset[z], mode="markers",
                name=group, marker=marker,
                customdata=np.stack([subset["temperature"], subset["target"],
                                     subset["stage"], subset["well_id"]], axis=-1),
                hovertemplate=("temp: %{customdata[0]}C<br>target: %{customdata[1]}"
                               "<br>stage: %{customdata[2]} hpf<br>%{customdata[3]}"
                               "<extra></extra>"),
            )
        )
    figure.update_layout(
        title=title, width=width, height=height,
        scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z),
        legend=dict(title="perturbation", orientation="h", y=-0.05),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return figure
