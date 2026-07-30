"""Cohort-specific axes of INTRA-cohort morphological variability.

The question this answers is *not* "where does each cohort sit in the GENE7 morphospace" — it is
"along which directions does each cohort vary internally, and are those directions cohort-specific?"

Design, as settled with the user
-------------------------------
1. **A GENE7-native 10D basis, used only as a denoised coordinate system.** One PCA fit on all 567
   GENE7 wells' ``z_mu_b_*`` latents. GENE7-native matters: a reference-fit basis is
   wildtype-dominated and would pre-filter out exactly the perturbation directions we are looking
   for. Fitting on GENE7 also folds the known batch offset into the mean, where it stops competing
   for variance. The global PCs themselves are of no interest here.

2. **Whitening.** Each of the 10 axes is scaled to unit variance before per-cohort PCA. Without it,
   every cohort's leading axis drifts toward whichever global axis has the most room to vary, and all
   cohorts look alike for an uninteresting reason. Cost: axes are no longer in morphology-variance
   units, so eigenvalues are "fraction of this cohort's whitened variance", not absolute morphology.

3. **Per-cohort centering.** Each cohort's PCA is centered on its OWN mean. Centering globally would
   let the cohort's offset from the GENE7 centroid load onto its PC1, recovering cohort *position*
   rather than intra-cohort *variability* — the precise confusion this analysis exists to avoid.

4. **Cohort = target x temperature x timepoint.** 48 cohorts of 9-12 wells. Pooling timepoints would
   reintroduce developmental stage as the dominant intra-cohort axis.

5. **Top 2-3 axes only.** At n~12 in 10D there are ~11 non-trivial components and the tail is
   guaranteed noise. Eigenvalues are bootstrapped so the drop-off to noise is visible rather than
   assumed, and a permutation null gives the scale at which "structure" is meaningless.

6. **Principal angles for cross-cohort comparison**, never PC1-to-PC1 pairing: at this n the
   eigenvalue gaps are small, so component *ordering* is unstable even when the *subspace* is stable.

Sample-size honesty
-------------------
n~12 in 10 dimensions is under-determined. Nothing here pretends otherwise: every cohort spectrum
carries bootstrap error bars, ``permutation_null`` supplies the no-structure reference, and
``subsample_stability`` reports whether an axis survives dropping embryos. Treat an axis as real only
when it clears the null AND is stable under subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

N_GLOBAL_COMPONENTS = 10
N_COHORT_AXES = 3

COHORT_KEYS: tuple[str, ...] = ("perturbation_group", "temperature", "timepoint_seq")
GLOBAL_COLUMNS: tuple[str, ...] = tuple(f"G{p:02d}" for p in range(N_GLOBAL_COMPONENTS))

LATENT_PATTERN = "z_mu_b"
RANDOM_SEED = 42


def biological_latent_columns(columns: "list[str]") -> list[str]:
    """The ``z_mu_b_*`` columns in ascending dimension order."""
    import re

    hits = []
    for column in columns:
        if str(column).startswith(LATENT_PATTERN):
            match = re.search(r"(\d+)$", str(column))
            hits.append((int(match.group(1)) if match else -1, str(column)))
    return [name for _, name in sorted(hits)]


def base_target(target: object) -> str:
    """Crispant target with the thermal-arm suffix removed (``atf6,hot`` -> ``atf6``)."""
    parts = [part for part in str(target).split(",") if part not in ("hot", "cold")]
    return ",".join(parts) if parts else str(target)


# ---------------------------------------------------------------------------
# The shared 10D coordinate system
# ---------------------------------------------------------------------------


@dataclass
class GlobalBasis:
    """A GENE7-native 10D basis, optionally whitened.

    ``scores`` holds every GENE7 well's coordinates plus the metadata needed to form cohorts.
    ``whitened`` records whether the axes carry unit variance.
    """

    pca: PCA
    latent_columns: tuple[str, ...]
    scores: pd.DataFrame
    whitened: bool
    scale: np.ndarray

    @property
    def explained_variance(self) -> pd.DataFrame:
        ratio = self.pca.explained_variance_ratio_
        return pd.DataFrame(
            {
                "component": np.arange(1, len(ratio) + 1),
                "explained_variance_ratio": ratio,
                "cumulative": np.cumsum(ratio),
            }
        )


def fit_global_basis(
    gene7: pd.DataFrame,
    *,
    n_components: int = N_GLOBAL_COMPONENTS,
    whiten: bool = True,
) -> GlobalBasis:
    """Fit the GENE7-native basis and project every well into it.

    Args:
        gene7: One row per GENE7 well, carrying ``z_mu_b_*`` and the cohort metadata.
        whiten: Scale each component to unit variance (recommended — see module docstring).
    """
    latent_columns = tuple(biological_latent_columns(list(gene7.columns)))
    if not latent_columns:
        raise ValueError("no z_mu_b_* columns found; this is not a legacy latent frame.")

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(gene7.loc[:, list(latent_columns)])

    # Whitening uses the FITTED component variances, so it is a property of the basis rather than of
    # whichever subset happens to be transformed later.
    scale = np.sqrt(pca.explained_variance_) if whiten else np.ones(n_components)
    scale = np.where(scale > 0, scale, 1.0)
    if whiten:
        coordinates = coordinates / scale

    scores = pd.DataFrame(coordinates, columns=list(GLOBAL_COLUMNS[:n_components]))
    carried = [
        column
        for column in (
            "snip_id",
            "well_id",
            "physical_embryo_id",
            "experiment_id",
            "target",
            "temperature",
            "timepoint_seq",
            "seq_sample_id",
        )
        if column in gene7.columns
    ]
    scores[carried] = gene7.loc[:, carried].to_numpy()
    scores["perturbation_group"] = [base_target(value) for value in scores["target"]]

    return GlobalBasis(
        pca=pca,
        latent_columns=latent_columns,
        scores=scores,
        whitened=whiten,
        scale=scale,
    )


def cohort_label(keys: tuple) -> str:
    """``('atf6', 35, 24)`` -> ``atf6 | 35C | 24hpf``."""
    target, temperature, timepoint = keys
    return f"{target} | {int(float(temperature))}C | {int(float(timepoint))}hpf"


def cohort_table(basis: GlobalBasis) -> pd.DataFrame:
    """Per-cohort sizes — the sanity check on how thin each fit will be."""
    grouped = basis.scores.groupby(list(COHORT_KEYS), sort=True)
    out = grouped.size().reset_index(name="n_wells")
    out["cohort"] = [cohort_label(tuple(row)) for row in out[list(COHORT_KEYS)].to_numpy()]
    return out


# ---------------------------------------------------------------------------
# Per-cohort intra-cohort PCA
# ---------------------------------------------------------------------------


@dataclass
class CohortAxes:
    """One cohort's intra-cohort principal axes in the shared 10D basis.

    ``components`` is ``(n_axes, 10)`` — each row a unit loading vector over the global axes.
    ``eigenvalues`` are variances along them; ``variance_ratio`` normalizes by the cohort's total.
    ``projections`` gives each member's coordinate along each axis (for ordering image strips).
    """

    label: str
    keys: tuple
    n_wells: int
    components: np.ndarray
    eigenvalues: np.ndarray
    variance_ratio: np.ndarray
    mean: np.ndarray
    projections: pd.DataFrame
    eigenvalue_ci: np.ndarray | None = None
    stability: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_axes(self) -> int:
        return self.components.shape[0]


def _score_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GLOBAL_COLUMNS if column in frame.columns]


def fit_cohort_axes(
    members: pd.DataFrame,
    *,
    label: str,
    keys: tuple,
    n_axes: int = N_COHORT_AXES,
) -> CohortAxes:
    """Intra-cohort PCA for one cohort, centered on the cohort's own mean.

    Raises:
        ValueError: if the cohort has fewer than ``n_axes + 1`` members, since a k-th component
            needs at least k+1 points to be anything but an artifact of the sample.
    """
    columns = _score_columns(members)
    matrix = members.loc[:, columns].to_numpy(float)
    if len(matrix) <= n_axes:
        raise ValueError(
            f"cohort {label!r} has {len(matrix)} wells; need > {n_axes} for {n_axes} axes."
        )

    # sklearn's PCA centers on the data it is given, which IS the cohort — this is the per-cohort
    # centering the design calls for, not global centering.
    n_fit = min(n_axes, min(matrix.shape) - 1) or 1
    pca = PCA(n_components=n_fit, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(matrix)

    total_variance = matrix.var(axis=0, ddof=1).sum()
    projections = pd.DataFrame(
        coordinates, columns=[f"axis_{i + 1}" for i in range(n_fit)], index=members.index
    )
    for column in ("snip_id", "well_id", "physical_embryo_id", "experiment_id"):
        if column in members.columns:
            projections[column] = members[column].to_numpy()

    return CohortAxes(
        label=label,
        keys=keys,
        n_wells=len(matrix),
        components=pca.components_,
        eigenvalues=pca.explained_variance_,
        variance_ratio=pca.explained_variance_ / total_variance if total_variance > 0 else pca.explained_variance_,
        mean=pca.mean_,
        projections=projections,
    )


def bootstrap_eigenvalues(
    members: pd.DataFrame,
    *,
    n_axes: int = N_COHORT_AXES,
    n_boots: int = 200,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Percentile CIs for a cohort's leading eigenvalues.

    Returns ``(n_axes, 2)`` of 5th/95th percentiles over resamples. Wide intervals that overlap
    each other mean the component ordering is not resolved at this sample size.
    """
    columns = _score_columns(members)
    matrix = members.loc[:, columns].to_numpy(float)
    rng = np.random.RandomState(seed)
    collected = []
    for _ in range(n_boots):
        sample = matrix[rng.choice(len(matrix), size=len(matrix), replace=True)]
        n_fit = min(n_axes, min(sample.shape) - 1) or 1
        try:
            pca = PCA(n_components=n_fit, random_state=seed).fit(sample)
        except Exception:
            continue
        values = np.full(n_axes, np.nan)
        values[:n_fit] = pca.explained_variance_[:n_fit]
        collected.append(values)
    if not collected:
        return np.full((n_axes, 2), np.nan)
    stacked = np.vstack(collected)
    return np.stack(
        [np.nanpercentile(stacked, 5, axis=0), np.nanpercentile(stacked, 95, axis=0)], axis=-1
    )


def subsample_stability(
    members: pd.DataFrame,
    *,
    n_axes: int = N_COHORT_AXES,
    n_draws: int = 100,
    drop_fraction: float = 0.25,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """How reproducible each axis is when a fraction of the cohort is dropped.

    For each draw, refit on the retained members and take ``|cos angle|`` between the subsample's
    k-th axis and the full-cohort k-th axis. Absolute value because eigenvector sign is arbitrary.

    Returns the mean over draws per axis: ~1.0 means the direction is robust; ~0.5 or below on 10D
    data means the axis is essentially resampling noise.
    """
    full = fit_cohort_axes(members, label="_", keys=(), n_axes=n_axes)
    columns = _score_columns(members)
    matrix = members.loc[:, columns].to_numpy(float)
    keep = max(n_axes + 1, int(round(len(matrix) * (1 - drop_fraction))))
    if keep >= len(matrix):
        keep = len(matrix) - 1
    if keep <= n_axes:
        return np.full(n_axes, np.nan)

    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_draws):
        sample = matrix[rng.choice(len(matrix), size=keep, replace=False)]
        n_fit = min(n_axes, min(sample.shape) - 1) or 1
        try:
            pca = PCA(n_components=n_fit, random_state=seed).fit(sample)
        except Exception:
            continue
        row = np.full(n_axes, np.nan)
        for axis in range(min(n_fit, full.n_axes)):
            row[axis] = abs(float(np.dot(pca.components_[axis], full.components[axis])))
        scores.append(row)
    return np.nanmean(np.vstack(scores), axis=0) if scores else np.full(n_axes, np.nan)


def fit_all_cohorts(
    basis: GlobalBasis,
    *,
    n_axes: int = N_COHORT_AXES,
    min_wells: int = 6,
    bootstrap: bool = True,
    stability: bool = True,
) -> list[CohortAxes]:
    """Fit intra-cohort axes for every cohort, attaching bootstrap CIs and stability scores."""
    results = []
    for keys, members in basis.scores.groupby(list(COHORT_KEYS), sort=True):
        if len(members) < min_wells:
            continue
        label = cohort_label(keys)
        axes = fit_cohort_axes(members, label=label, keys=keys, n_axes=n_axes)
        if bootstrap:
            axes.eigenvalue_ci = bootstrap_eigenvalues(members, n_axes=n_axes)
        if stability:
            axes.stability = subsample_stability(members, n_axes=n_axes)
        results.append(axes)
    return results


# ---------------------------------------------------------------------------
# Nulls and cross-cohort comparison
# ---------------------------------------------------------------------------


def permutation_null(
    basis: GlobalBasis,
    *,
    cohort_size: int = 12,
    n_draws: int = 200,
    n_axes: int = N_COHORT_AXES,
    within: "tuple[str, ...]" = ("temperature", "timepoint_seq"),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """The no-structure reference: random same-arm pseudo-cohorts of the same size.

    Draws are taken WITHIN a temperature x timepoint stratum by default, so the null shares the
    cohort's stage and thermal composition and differs only in that its membership is arbitrary.
    Any real cohort axis must beat this; at n~12 in 10D an arbitrary group still yields a leading
    axis with substantial variance ratio, which is exactly the point of measuring it.

    Returns one row per draw with the leading ``variance_ratio`` values.
    """
    rng = np.random.RandomState(seed)
    strata = [group for _, group in basis.scores.groupby(list(within), sort=True)
              if len(group) > cohort_size]
    if not strata:
        strata = [basis.scores]

    rows = []
    for draw in range(n_draws):
        stratum = strata[rng.randint(len(strata))]
        size = min(cohort_size, len(stratum))
        if size <= n_axes:
            continue
        members = stratum.iloc[rng.choice(len(stratum), size=size, replace=False)]
        axes = fit_cohort_axes(members, label=f"null_{draw}", keys=(), n_axes=n_axes)
        row = {"draw": draw, "n_wells": axes.n_wells}
        for index in range(n_axes):
            row[f"axis_{index + 1}"] = (
                float(axes.variance_ratio[index]) if index < len(axes.variance_ratio) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def principal_angles(first: np.ndarray, second: np.ndarray, *, k: int = 2) -> np.ndarray:
    """Principal angles (degrees, ascending) between two subspaces' top-``k`` axes.

    The stable way to compare cohorts: it asks whether the SPANS agree, not whether PC1 matches PC1.
    Component order flips readily at small n, so pairwise component comparison is fragile where this
    is not. 0 degrees = identical subspace; 90 = orthogonal.
    """
    a = np.atleast_2d(first)[:k]
    b = np.atleast_2d(second)[:k]
    qa = np.linalg.qr(a.T)[0]
    qb = np.linalg.qr(b.T)[0]
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def cohort_similarity_matrix(
    cohorts: "list[CohortAxes]", *, k: int = 2
) -> pd.DataFrame:
    """Pairwise subspace similarity: mean cosine of the top-``k`` principal angles.

    1.0 = the two cohorts vary along the same plane; 0 = orthogonal planes.
    """
    labels = [cohort.label for cohort in cohorts]
    size = len(cohorts)
    matrix = np.eye(size)
    for i in range(size):
        for j in range(i + 1, size):
            angles = principal_angles(cohorts[i].components, cohorts[j].components, k=k)
            value = float(np.cos(np.radians(angles)).mean())
            matrix[i, j] = matrix[j, i] = value
    return pd.DataFrame(matrix, index=labels, columns=labels)


def cohort_summary(
    cohorts: "list[CohortAxes]", null: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per cohort: variance ratios, bootstrap CIs, stability, and null percentiles."""
    rows = []
    for cohort in cohorts:
        row = {
            "cohort": cohort.label,
            "target": cohort.keys[0] if cohort.keys else "",
            "temperature": cohort.keys[1] if cohort.keys else np.nan,
            "timepoint": cohort.keys[2] if cohort.keys else np.nan,
            "n_wells": cohort.n_wells,
        }
        for index in range(cohort.n_axes):
            name = f"axis_{index + 1}"
            row[f"{name}_var_ratio"] = float(cohort.variance_ratio[index])
            if cohort.stability is not None and index < len(cohort.stability):
                row[f"{name}_stability"] = float(cohort.stability[index])
            if cohort.eigenvalue_ci is not None and index < len(cohort.eigenvalue_ci):
                low, high = cohort.eigenvalue_ci[index]
                row[f"{name}_eig_lo"] = float(low)
                row[f"{name}_eig_hi"] = float(high)
            if null is not None and name in null.columns:
                row[f"{name}_null_pctile"] = float(
                    (null[name] < cohort.variance_ratio[index]).mean() * 100
                )
        rows.append(row)
    return pd.DataFrame(rows)


def axis_loadings(cohorts: "list[CohortAxes]") -> pd.DataFrame:
    """Long-form loading vectors: one row per (cohort, axis, global component)."""
    rows = []
    for cohort in cohorts:
        for axis_index in range(cohort.n_axes):
            for component_index, loading in enumerate(cohort.components[axis_index]):
                rows.append(
                    {
                        "cohort": cohort.label,
                        "target": cohort.keys[0] if cohort.keys else "",
                        "temperature": cohort.keys[1] if cohort.keys else np.nan,
                        "timepoint": cohort.keys[2] if cohort.keys else np.nan,
                        "axis": f"axis_{axis_index + 1}",
                        "global_component": GLOBAL_COLUMNS[component_index],
                        "loading": float(loading),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Image strips
# ---------------------------------------------------------------------------

SNIP_ROOT = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output/object_extraction"
)


PIPELINE_CHANNEL = "BF"


def snip_image_path(
    well_id: str,
    physical_embryo_id: str,
    time_index: int = 0,
    *,
    experiment_id: str | None = None,
    root: Path | None = None,
    masked: bool = False,
) -> Path:
    """Locate a snip PNG on disk.

    Layout::

        {root}/{experiment_id}/snips/per_well/{well_id}/snips/{physical_embryo_id}/{snip_id}.png

    Built from ``well_id`` / ``physical_embryo_id`` rather than by slicing a ``snip_id`` string,
    because the two id generations differ in token count: the legacy id the latents carry is
    ``..._A01_e00_t0000`` (zero-based embryo, NO channel token) while the on-disk pipeline snip is
    ``..._A01_e01_BF_t0000``. Re-deriving the filename from the canonical parts avoids translating
    between the two grammars by hand.

    ``masked=True`` returns the ``*_embryo.png`` variant (background removed).

    NOTE: these images are the CURRENT pipeline's snips, a degraded raster relative to the build that
    produced the legacy latents (smaller, more saturated). Right for interpreting shape; not the
    pixels the latents were computed from.
    """
    base = root or SNIP_ROOT
    experiment = experiment_id or str(well_id).rsplit("_", 1)[0]
    snip_id = f"{physical_embryo_id}_{PIPELINE_CHANNEL}_t{int(time_index):04d}"
    name = f"{snip_id}_embryo.png" if masked else f"{snip_id}.png"
    return (
        base
        / experiment
        / "snips"
        / "per_well"
        / str(well_id)
        / "snips"
        / str(physical_embryo_id)
        / name
    )


def axis_image_strip(
    cohort: CohortAxes, *, axis: int = 1, n_images: int = 8, masked: bool = False
) -> pd.DataFrame:
    """Members sampled evenly along one cohort axis, ordered low to high.

    Returns ``snip_id``, ``well_id``, the axis coordinate, and the resolved image path with an
    existence flag, so a missing snip is visible rather than a silently blank tile.
    """
    column = f"axis_{axis}"
    if column not in cohort.projections.columns:
        raise ValueError(f"{cohort.label}: no {column} (cohort has {cohort.n_axes} axes).")

    ordered = cohort.projections.sort_values(column).reset_index(drop=True)
    picks = np.unique(np.linspace(0, len(ordered) - 1, num=min(n_images, len(ordered))).astype(int))
    strip = ordered.iloc[picks].copy()
    strip["image_path"] = [
        str(
            snip_image_path(
                row.well_id,
                row.physical_embryo_id,
                experiment_id=row.experiment_id,
                masked=masked,
            )
        )
        for row in strip.itertuples()
    ]
    strip["image_exists"] = [Path(path).is_file() for path in strip["image_path"]]
    strip["cohort"] = cohort.label
    strip["axis"] = column
    return strip.loc[
        :, ["cohort", "axis", column, "snip_id", "well_id", "image_path", "image_exists"]
    ]
