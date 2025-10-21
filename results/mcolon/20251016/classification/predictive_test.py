"""
Predictive classification and permutation testing for phenotype emergence.

This module implements the core predictive signal test that evaluates whether
genotype labels can be predicted from morphological features better than chance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def predictive_signal_test(
    df_binned: pd.DataFrame,
    group_col: str = "genotype",
    time_col: str = "time_bin",
    z_cols: Optional[list] = None,
    n_splits: int = 5,
    n_perm: int = 100,
    random_state: Optional[int] = None,
    return_embryo_probs: bool = True,
    use_class_weights: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Predictive classifier + label-shuffling test across time bins.

    This test evaluates whether genotype labels can be predicted from
    morphological features (VAE embeddings) better than chance, using
    a logistic regression classifier with cross-validation.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data (output of bin_by_embryo_time).
    group_col : str, default="genotype"
        Column specifying experimental group (e.g., genotype).
    time_col : str, default="time_bin"
        Column specifying time bins.
    z_cols : list or None
        Latent columns to use as features. Auto-detected if None.
    n_splits : int, default=5
        Number of cross-validation splits.
    n_perm : int, default=100
        Number of permutations for null distribution.
    random_state : int or None
        Random seed for reproducibility.
    return_embryo_probs : bool, default=True
        If True, return per-embryo prediction probabilities in addition to aggregate stats.
    use_class_weights : bool, default=True
        If True, use balanced class weights to handle class imbalance.
        This helps prevent bias when one class (e.g., wildtype) is much larger.
        **FIXED: This parameter is now actually used in the LogisticRegression model.**

    Returns
    -------
    df_results : pd.DataFrame
        One row per time_bin with AUROC statistics and p-values.
        Columns: time_bin, AUROC_obs, AUROC_null_mean, AUROC_null_std, pval, n_samples
    df_embryo_probs : pd.DataFrame or None
        Per-embryo prediction probabilities if return_embryo_probs=True.
        Columns: embryo_id, time_bin, true_label, pred_proba, confidence,
                predicted_label, support_true, signed_margin

    Notes
    -----
    The signed margin is a key metric defined as:
        signed_margin = sign(true_label == positive_class) * (pred_prob - 0.5)

    This makes it:
    - Positive when classifier correctly predicts the positive class
    - Negative when classifier incorrectly predicts the negative class
    - Zero at the decision boundary
    """
    rng = np.random.default_rng(random_state)

    # Auto-detect latent columns if not specified
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError("No latent columns found. Specify z_cols explicitly.")

    results = []
    embryo_predictions = [] if return_embryo_probs else None

    # Process each time bin independently
    for t, sub in df_binned.groupby(time_col):
        X = sub[z_cols].values
        y = sub[group_col].values
        embryo_ids = sub['embryo_id'].values

        # Only handle two-class problems for now
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            continue

        # Check for minimum sample size
        min_samples_per_class = min([np.sum(y == c) for c in unique_classes])
        if min_samples_per_class < n_splits:
            print(f"Skipping time bin {t}: insufficient samples "
                  f"({min_samples_per_class} < {n_splits})")
            continue

        # --- Configure class weights ---
        # FIXED: Actually use the use_class_weights parameter!
        if use_class_weights:
            class_weight = 'balanced'
        else:
            class_weight = None

        # --- True AUROC via cross-validation ---
        skf = StratifiedKFold(
            n_splits=min(n_splits, min_samples_per_class),
            shuffle=True,
            random_state=random_state
        )

        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            # FIXED: Pass class_weight parameter to LogisticRegression
            model = LogisticRegression(
                max_iter=200,
                random_state=random_state,
                class_weight=class_weight  # <-- THIS WAS MISSING!
            )
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])

            # Ensure we consistently reference the positive-class column
            class_order = model.classes_
            if len(class_order) != 2:
                raise ValueError("Expected binary classification with two classes.")

            # IMPORTANT: Determine which class is "mutant" (non-WT)
            # Assumes WT labels contain 'wildtype', 'wik', or 'ab'
            wt_classes = [c for c in class_order if 'wildtype' in str(c).lower() or str(c).lower() in ['wik', 'ab', 'wik-ab']]
            mutant_classes = [c for c in class_order if c not in wt_classes]

            if len(wt_classes) == 1 and len(mutant_classes) == 1:
                # Explicitly identify mutant class
                mutant_class = mutant_classes[0]
                wt_class = wt_classes[0]
                mutant_idx = np.where(class_order == mutant_class)[0][0]

                # pred_prob_mutant = probability of MUTANT class (regardless of alphabetical order)
                positive_prob = proba[:, mutant_idx]
                positive_class = mutant_class
            else:
                # Fallback to alphabetical second class if can't determine WT/mutant
                positive_class = class_order[1]
                positive_prob = proba[:, 1]

            aucs.append(roc_auc_score(y[test_idx], positive_prob))

            # Collect per-embryo predictions
            if return_embryo_probs:
                for i, idx in enumerate(test_idx):
                    true_label = y[idx]
                    p_pos = positive_prob[i]

                    # Support for true class
                    support_true = p_pos if true_label == positive_class else 1.0 - p_pos

                    # Signed margin: positive = correct direction, negative = wrong direction
                    signed_margin = (1 if true_label == positive_class else -1) * (p_pos - 0.5)

                    embryo_predictions.append({
                        'embryo_id': embryo_ids[idx],
                        'time_bin': t,
                        'true_label': true_label,
                        'pred_proba': p_pos,  # Now explicitly P(mutant)
                        'confidence': np.abs(p_pos - 0.5),
                        'predicted_label': positive_class if p_pos > 0.5 else class_order[0] if positive_class == class_order[1] else class_order[1],
                        'support_true': support_true,
                        'signed_margin': signed_margin,
                        'mutant_class': mutant_class if len(wt_classes) == 1 else None,
                        'wt_class': wt_class if len(wt_classes) == 1 else None
                    })

        true_auc = np.mean(aucs)

        # --- Null distribution via shuffled labels ---
        null_aucs = []
        for _ in range(n_perm):
            y_shuff = rng.permutation(y)
            perm_aucs = []

            for train_idx, test_idx in skf.split(X, y_shuff):
                # FIXED: Also apply class_weight in permutation tests
                model = LogisticRegression(
                    max_iter=200,
                    random_state=random_state,
                    class_weight=class_weight  # <-- THIS WAS ALSO MISSING!
                )
                model.fit(X[train_idx], y_shuff[train_idx])
                proba_perm = model.predict_proba(X[test_idx])

                # Use same mutant class detection logic for consistency
                class_order_perm = model.classes_
                wt_classes_perm = [c for c in class_order_perm if 'wildtype' in str(c).lower() or str(c).lower() in ['wik', 'ab', 'wik-ab']]
                mutant_classes_perm = [c for c in class_order_perm if c not in wt_classes_perm]

                if len(wt_classes_perm) == 1 and len(mutant_classes_perm) == 1:
                    mutant_idx_perm = np.where(class_order_perm == mutant_classes_perm[0])[0][0]
                    prob = proba_perm[:, mutant_idx_perm]
                else:
                    prob = proba_perm[:, 1]

                perm_aucs.append(roc_auc_score(y_shuff[test_idx], prob))

            null_aucs.append(np.mean(perm_aucs))

        null_aucs = np.array(null_aucs)
        pval = (np.sum(null_aucs >= true_auc) + 1) / (len(null_aucs) + 1)

        results.append({
            "time_bin": t,
            "AUROC_obs": true_auc,
            "AUROC_null_mean": null_aucs.mean(),
            "AUROC_null_std": null_aucs.std(),
            "pval": pval,
            "n_samples": len(y)
        })

    df_results = pd.DataFrame(results)

    if return_embryo_probs:
        df_embryo_probs = pd.DataFrame(embryo_predictions)
        return df_results, df_embryo_probs

    return df_results, None
