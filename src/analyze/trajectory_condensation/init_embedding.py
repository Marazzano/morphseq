"""
init_embedding.py
-----------------
Produces initial 2D positions x0 from canonical feature tensors.

Condensation never imports from here — it only sees x0 and mask.

Functions take explicit (features, mask) arguments rather than the full
CondensationData, keeping the dependency surface minimal and the boundary sharp.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import nan_euclidean_distances


def aligned_umap_init(
    features: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    alignment_regularisation: float = 1e-2,
    alignment_window_size: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Return the standard NaN-aware UMAP initialization.

    Parameters
    ----------
    features : (N_e, T, K)
    mask : (N_e, T) bool — True where embryo is observed

    Returns
    -------
    x0 : (N_e, T, 2) float array — NaN where mask is False

    Notes
    -----
    This is the default public entry point for trajectory-condensation
    initialization in this repo. It routes through the NaN-aware path even when
    the current dataset happens not to contain NaNs in observed rows. That keeps
    initialization behavior consistent across dense and sparse representations.

    ``alignment_regularisation`` and ``alignment_window_size`` are retained in
    the signature for backward compatibility, but are not used by the current
    implementation.
    """
    return nan_aware_aligned_umap_init(
        features,
        mask,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )


def nan_aware_aligned_umap_init(
    features: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    min_shared_features: int = 1,
) -> np.ndarray:
    """Fit a NaN-aware per-bin UMAP init and align consecutive bins with Procrustes.

    This path is intended for sparse pairwise coordinates where missing values mean
    "off-support", not "neutral".
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for nan_aware_aligned_umap_init")

    N_e, T, _ = features.shape
    slice_indices = _slice_embryo_indices(mask)
    x0 = np.full((N_e, T, 2), np.nan)
    prev_embedding: np.ndarray | None = None
    prev_index: np.ndarray | None = None

    for t, idx in enumerate(slice_indices):
        if len(idx) == 0:
            continue

        emb = _nan_aware_umap_slice(
            features[idx, t, :],
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state + t,
            min_shared_features=min_shared_features,
        )
        if prev_embedding is not None and prev_index is not None:
            emb = _align_with_procrustes(
                current=emb,
                current_index=idx,
                reference=prev_embedding,
                reference_index=prev_index,
            )

        x0[idx, t, :] = emb
        prev_embedding = emb
        prev_index = idx

    return x0


def global_umap_init(
    features: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    min_shared_features: int = 1,
    time_weight: float = 0.0,
    time_values: np.ndarray | None = None,
) -> np.ndarray:
    """Embed ALL observed (embryo, time) points in ONE shared UMAP frame.

    Unlike :func:`nan_aware_aligned_umap_init`, this fits a single UMAP over every
    observed point at once, so there are no per-bin coordinate frames to stitch and
    no consecutive-bin Procrustes step. That removes the batch-entry "seam" by
    construction: a batch that first appears mid-timeline is placed by its own
    feature geometry in the shared frame, rather than by a Procrustes rotation
    solved purely from anchor embryos it does not share with the previous bin.

    Trade-off: temporal ordering is NOT imposed here — points are placed purely by
    feature (optionally lightly biased by time; see ``time_weight``). The
    downstream condensation solver (elasticity + coherence) is responsible for
    resolving the time axis. Compare against the aligned path with ``ridge_score``.

    Parameters
    ----------
    features : (N_e, T, K)
    mask : (N_e, T) bool
    time_weight : float
        If > 0, append a time channel scaled by ``time_weight`` (in the same units
        as the feature distances) so the global embedding is gently biased to keep
        adjacent times together. 0.0 (default) = pure feature geometry, the cleanest
        test of whether stitching was the problem. Requires ``time_values``.
    time_values : (T,) float, optional
        Time bin centers; required when ``time_weight`` > 0.

    Returns
    -------
    x0 : (N_e, T, 2) — NaN where mask is False, in one shared coordinate frame.
    """
    try:
        import umap  # noqa: F401
    except ImportError:
        raise ImportError("umap-learn is required for global_umap_init")

    N_e, T, K = features.shape
    # Flatten to observed points, remembering their (embryo, time) origin.
    ei, ti = np.nonzero(mask)
    X = features[ei, ti, :]  # (n_points, K)
    n_points = X.shape[0]

    if time_weight > 0:
        if time_values is None:
            raise ValueError("time_values is required when time_weight > 0")
        tcol = (time_values[ti] * float(time_weight)).reshape(-1, 1)
        X = np.hstack([X, tcol])

    dist = _nan_aware_distance_matrix(X, min_shared_features=min_shared_features)
    neighbors = max(2, min(n_neighbors, n_points - 1))
    init = "spectral" if n_points >= 4 * 2 else "random"

    import umap as _umap
    emb = _umap.UMAP(
        metric="precomputed",
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=2,
        init=init,
        random_state=random_state,
    ).fit_transform(dist)

    x0 = np.full((N_e, T, 2), np.nan)
    x0[ei, ti, :] = emb
    return x0


def pca_init(
    features: np.ndarray,
    mask: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Simple PCA fallback initialization — no temporal alignment.

    Useful for smoke tests when umap-learn is unavailable or slow.
    For sparse pairwise inputs, NaNs are mean-imputed on observed rows.
    """
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    n_e, t_count, _ = features.shape
    x0 = np.full((n_e, t_count, 2), np.nan)

    all_rows = features[mask]
    imputer = SimpleImputer(strategy="mean")
    all_rows_imputed = imputer.fit_transform(all_rows)
    pca = PCA(n_components=2, random_state=random_state).fit(all_rows_imputed)

    for t_idx in range(t_count):
        observed = mask[:, t_idx]
        if observed.sum() == 0:
            continue
        rows = imputer.transform(features[observed, t_idx, :])
        x0[observed, t_idx, :] = pca.transform(rows)

    return x0


def _has_missing_features(features: np.ndarray, mask: np.ndarray) -> bool:
    observed = features[mask]
    return bool(observed.size and np.isnan(observed).any())


def _nan_aware_umap_slice(
    X: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    min_shared_features: int,
) -> np.ndarray:
    n_obs = X.shape[0]
    if n_obs == 1:
        return np.zeros((1, 2), dtype=float)
    if n_obs == 2:
        dist = _nan_aware_distance_matrix(X, min_shared_features=min_shared_features)
        span = float(dist[0, 1]) if np.isfinite(dist[0, 1]) and dist[0, 1] > 0 else 1.0
        return np.array([[-0.5 * span, 0.0], [0.5 * span, 0.0]], dtype=float)

    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for _nan_aware_umap_slice")

    dist = _nan_aware_distance_matrix(X, min_shared_features=min_shared_features)
    neighbors = max(2, min(n_neighbors, n_obs - 1))
    # Spectral init requires the graph to have > n_components connected nodes;
    # fall back to random init when the slice is too small to guarantee this.
    init = "spectral" if n_obs >= 4 * 2 else "random"
    return umap.UMAP(
        metric="precomputed",
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=2,
        init=init,
        random_state=random_state,
    ).fit_transform(dist)


def _nan_aware_distance_matrix(X: np.ndarray, *, min_shared_features: int) -> np.ndarray:
    dist = nan_euclidean_distances(X)
    overlap = _shared_feature_counts(X)
    valid = overlap >= int(min_shared_features)
    valid[np.diag_indices_from(valid)] = True

    finite = np.isfinite(dist) & valid
    if finite.any():
        fill_value = float(np.nanmax(dist[finite]))
        if not np.isfinite(fill_value) or fill_value <= 0:
            fill_value = 1.0
    else:
        fill_value = 1.0

    out = np.array(dist, copy=True)
    out[~finite] = fill_value * 1.25
    np.fill_diagonal(out, 0.0)
    return out


def _shared_feature_counts(X: np.ndarray) -> np.ndarray:
    valid = np.isfinite(X)
    return valid @ valid.T


def _build_aligned_umap_inputs(
    features: np.ndarray,
    mask: np.ndarray,
    slice_indices: list[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Build per-time slices and consecutive-time embryo relations for AlignedUMAP."""
    T = features.shape[1]
    slices = [features[idx, t, :] for t, idx in enumerate(slice_indices)]
    relations = []

    for t in range(T - 1):
        idx_t = slice_indices[t]
        idx_t1 = slice_indices[t + 1]
        shared = set(idx_t) & set(idx_t1)
        pos_in_t = {e: i for i, e in enumerate(idx_t)}
        pos_in_t1 = {e: i for i, e in enumerate(idx_t1)}
        relations.append({pos_in_t[e]: pos_in_t1[e] for e in shared})

    return slices, relations


def _align_with_procrustes(
    *,
    current: np.ndarray,
    current_index: np.ndarray,
    reference: np.ndarray,
    reference_index: np.ndarray,
) -> np.ndarray:
    ref_pos = {int(e): i for i, e in enumerate(reference_index)}
    shared_pairs = [(i_cur, ref_pos[int(e)]) for i_cur, e in enumerate(current_index) if int(e) in ref_pos]
    if len(shared_pairs) < 2:
        return current

    cur_idx = np.array([pair[0] for pair in shared_pairs], dtype=int)
    ref_idx = np.array([pair[1] for pair in shared_pairs], dtype=int)
    cur_anchor = current[cur_idx]
    ref_anchor = reference[ref_idx]

    cur_mean = cur_anchor.mean(axis=0)
    ref_mean = ref_anchor.mean(axis=0)
    cur_centered = cur_anchor - cur_mean
    ref_centered = ref_anchor - ref_mean
    if np.allclose(cur_centered, 0.0) or np.allclose(ref_centered, 0.0):
        return current - cur_mean + ref_mean

    rotation, _ = orthogonal_procrustes(cur_centered, ref_centered)
    return (current - cur_mean) @ rotation + ref_mean


def _procrustes_fit(
    *,
    current: np.ndarray,
    current_index: np.ndarray,
    reference: np.ndarray,
    reference_index: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Rigidly align ``current`` onto ``reference`` via same-embryo anchors.

    Same math as :func:`_align_with_procrustes` but also returns the number of
    shared-embryo anchors used, so callers can detect an *under-anchored* slice
    (e.g. a batch entering at this bin with no shared embryos in the reference).
    With < 2 anchors it returns ``current`` unchanged and ``0``.
    """
    ref_pos = {int(e): i for i, e in enumerate(reference_index)}
    shared_pairs = [
        (i_cur, ref_pos[int(e)])
        for i_cur, e in enumerate(current_index)
        if int(e) in ref_pos
    ]
    n_anchors = len(shared_pairs)
    if n_anchors < 2:
        return current, n_anchors

    cur_idx = np.array([p[0] for p in shared_pairs], dtype=int)
    ref_idx = np.array([p[1] for p in shared_pairs], dtype=int)
    cur_anchor = current[cur_idx]
    ref_anchor = reference[ref_idx]

    cur_mean = cur_anchor.mean(axis=0)
    ref_mean = ref_anchor.mean(axis=0)
    cur_centered = cur_anchor - cur_mean
    ref_centered = ref_anchor - ref_mean
    if np.allclose(cur_centered, 0.0) or np.allclose(ref_centered, 0.0):
        return current - cur_mean + ref_mean, n_anchors

    rotation, _ = orthogonal_procrustes(cur_centered, ref_centered)
    return (current - cur_mean) @ rotation + ref_mean, n_anchors


def bidirectional_aligned_umap_init(
    features: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    min_shared_features: int = 1,
    min_anchors: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """Per-bin UMAP init with a FORWARD pass + per-embryo FUTURE-anchor repair.

    Motivation
    ----------
    :func:`nan_aware_aligned_umap_init` aligns each bin ``t`` onto ``t-1`` using
    only same-embryo anchors shared between the two bins. A batch that *enters* at
    bin ``t`` (no rows at ``t-1``) has no such anchors, so its bundle is placed by
    a rotation solved entirely from the batch already present — an unconstrained,
    off-manifold "seam". See RELATIVE_GEOMETRY_AUDIT.md and the ridge diagnostics.

    But an entering batch is NOT information-poor: its own embryos reappear at
    ``t+1, t+2, ...``. This routine keeps the per-bin UMAP structure (correct for
    every continuing bin) and repairs only the entering embryos, using their own
    future as anchors.

    Why the repair is PER-EMBRYO, not per-bin
    -----------------------------------------
    At an entry bin, a *continuing* batch (already correctly placed by the forward
    pass) coexists with the *entering* batch. The bin as a whole therefore looks
    well-anchored (the continuing batch supplies many ``t-1`` anchors), so a
    per-bin gate would never repair the newcomer. Worse, re-rotating the whole bin
    would disturb the correctly-placed continuing batch. So the repair:

    - identifies UNANCHORED embryos: observed at ``t`` but not at ``t-1``
      (label-free — read straight from the mask, so the initializer contract that
      condensation never sees labels is preserved);
    - fits a rigid transform from those embryos' positions at ``t`` to their OWN
      positions at ``t+1`` (already placed), and applies it ONLY to those embryos.

    The continuing batch's points are never touched. Forward and future anchors
    are complementary: the embryos lacking a ``t-1`` anchor are exactly the ones
    richest in ``t+1`` self-anchors.

    ``min_anchors`` is the minimum number of future self-anchors required to trust
    the repair; unanchored groups with fewer are left in their forward placement
    and flagged when ``verbose``.

    Returns
    -------
    x0 : (N_e, T, 2) — NaN where mask is False.
    """
    try:
        import umap  # noqa: F401
    except ImportError:
        raise ImportError("umap-learn is required for bidirectional_aligned_umap_init")

    N_e, T, _ = features.shape
    slice_indices = _slice_embryo_indices(mask)

    # ── Pass 1: forward placement (identical to the current aligned init) ──────
    embeddings: list[np.ndarray | None] = [None] * T
    prev_embedding: np.ndarray | None = None
    prev_index: np.ndarray | None = None

    for t, idx in enumerate(slice_indices):
        if len(idx) == 0:
            continue
        emb = _nan_aware_umap_slice(
            features[idx, t, :],
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state + t,
            min_shared_features=min_shared_features,
        )
        if prev_embedding is not None and prev_index is not None:
            emb, _ = _procrustes_fit(
                current=emb, current_index=idx,
                reference=prev_embedding, reference_index=prev_index,
            )
        embeddings[t] = emb
        prev_embedding = emb
        prev_index = idx

    # ── Pass 2: per-embryo future-anchor repair of entering embryos ────────────
    # Walk high->low so bin t is refined against a t+1 that is already final.
    for t in range(T - 1, 0, -1):  # skip t=0 (no t-1; nothing "enters" at the origin)
        idx = slice_indices[t]
        if len(idx) == 0 or embeddings[t] is None:
            continue
        prev_set = set(int(e) for e in slice_indices[t - 1])
        # entering = observed at t but NOT at t-1 (no backward anchor)
        entering_mask = np.array([int(e) not in prev_set for e in idx], dtype=bool)
        n_entering = int(entering_mask.sum())
        if n_entering == 0:
            continue  # every embryo has a t-1 anchor; forward placement stands

        nxt = t + 1
        if nxt >= T or embeddings[nxt] is None or len(slice_indices[nxt]) == 0:
            if verbose:
                print(f"[bidir] t={t}: {n_entering} entering embryos but no usable "
                      f"t+1 — left in forward placement")
            continue

        entering_ids = idx[entering_mask]
        entering_local = np.flatnonzero(entering_mask)  # rows within embeddings[t]
        cur_entering = embeddings[t][entering_local]

        # Fit transform from entering embryos @ t to the SAME embryos @ t+1.
        _, n_anch = _procrustes_fit(
            current=cur_entering, current_index=entering_ids,
            reference=embeddings[nxt], reference_index=slice_indices[nxt],
        )
        if n_anch < min_anchors:
            if verbose:
                print(f"[bidir] t={t}: {n_entering} entering embryos, only {n_anch} "
                      f"future self-anchors (< {min_anchors}) — left as-is")
            continue

        refined_entering, _ = _procrustes_fit(
            current=cur_entering, current_index=entering_ids,
            reference=embeddings[nxt], reference_index=slice_indices[nxt],
        )
        # Apply ONLY to entering embryos; continuing batch untouched.
        embeddings[t][entering_local] = refined_entering
        if verbose:
            print(f"[bidir] t={t}: repaired {n_entering} entering embryos using "
                  f"{n_anch} future self-anchors (continuing batch untouched)")

    x0 = np.full((N_e, T, 2), np.nan)
    for t, idx in enumerate(slice_indices):
        if len(idx) and embeddings[t] is not None:
            x0[idx, t, :] = embeddings[t]
    return x0


def _slice_embryo_indices(mask: np.ndarray) -> list[np.ndarray]:
    T = mask.shape[1]
    return [np.where(mask[:, t])[0] for t in range(T)]
