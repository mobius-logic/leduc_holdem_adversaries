"""LDA classification pipeline with PCA preprocessing.

Fits and evaluates Linear Discriminant Analysis classifiers across a
predefined set of hyperparameter combinations. Reports cross-validation
scores, confusion matrices, and mislabeling percentages to stdout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


# All LDA parameter combinations to evaluate.
_LDA_PARAM_GRID: List[Dict[str, Any]] = [
    {"solver": "svd",   "shrinkage": None},
    {"solver": "lsqr",  "shrinkage": "auto"},
    {"solver": "lsqr",  "shrinkage": 0.0},
    {"solver": "lsqr",  "shrinkage": 0.5},
    {"solver": "lsqr",  "shrinkage": 1.0},
    {"solver": "eigen", "shrinkage": "auto"},
    {"solver": "eigen", "shrinkage": 0.5},
]


def fit_pca(
    X_train: npt.NDArray[np.float32],
    X_test: npt.NDArray[np.float32],
) -> Tuple[PCA, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fit PCA on training data and transform both splits.

    n_components is set to min(n_samples - 1, n_features) to avoid
    rank deficiency in LDA.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        Tuple of (fitted_pca, X_train_pca, X_test_pca).
    """
    n_components = min(X_train.shape[0] - 1, X_train.shape[1])
    print(
        f"[LDA Pipeline] Fitting PCA: n_samples={X_train.shape[0]}, "
        f"n_features={X_train.shape[1]}, n_components={n_components}"
    )
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(
        f"[LDA Pipeline] PCA complete. "
        f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}"
    )
    return pca, X_train_pca, X_test_pca


def evaluate_lda_params(
    params: Dict[str, Any],
    X_train: npt.NDArray[np.float32],
    X_test: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int32],
    y_test: npt.NDArray[np.int32],
    kfold_splits: int,
    personalities: List[str],
) -> Dict[str, Any]:
    """Fit and evaluate one LDA parameter configuration.

    Args:
        params: Dict with keys 'solver' and 'shrinkage'.
        X_train: PCA-transformed training features.
        X_test: PCA-transformed test features.
        y_train: Integer training labels.
        y_test: Integer test labels.
        kfold_splits: Number of stratified K-fold splits.
        personalities: Ordered list of personality names for display.

    Returns:
        Result dict with keys:
          params, cv_mean, cv_std, mislabel_pct, confusion, lda.
    """
    label = f"solver={params['solver']}, shrinkage={params['shrinkage']}"
    print(f"\n[LDA Pipeline] Evaluating: {label}")

    lda = LinearDiscriminantAnalysis(
        solver=params["solver"],
        shrinkage=params["shrinkage"],
    )
    lda.fit(X_train, y_train)

    # Cross-validation.
    skf = StratifiedKFold(n_splits=kfold_splits)
    try:
        cv_scores = cross_val_score(lda, X_train, y_train, cv=skf)
    except (ValueError, Exception) as exc:
        print(f"  CV failed ({type(exc).__name__}): {exc}")
        print("  Skipping this configuration (insufficient data or singular matrix).")
        return {
            "params": params,
            "cv_mean": -1.0,
            "cv_std": 0.0,
            "mislabel_pct": 100.0,
            "confusion": None,
            "lda": None,
        }
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(
        f"  CV scores ({kfold_splits}-fold): "
        f"{cv_scores} | mean={cv_mean:.4f}, std={cv_std:.4f}"
    )

    # Test set evaluation.
    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    n_incorrect = int((y_pred != y_test).sum())
    mislabel_pct = 100.0 * n_incorrect / len(y_test)

    print(f"  Mislabeled: {n_incorrect}/{len(y_test)} ({mislabel_pct:.1f}%)")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    _print_confusion_matrix(cm, personalities)

    return {
        "params": params,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "mislabel_pct": mislabel_pct,
        "confusion": cm,
        "lda": lda,
    }


def run_lda_pipeline(
    X_train: npt.NDArray[np.float32],
    X_test: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int32],
    y_test: npt.NDArray[np.int32],
    cfg: dict,
    kfold_splits_override: int | None = None,
) -> Tuple[PCA, List[Dict[str, Any]]]:
    """Run the full LDA evaluation pipeline over all parameter sets.

    Applies PCA first, then evaluates all defined LDA configurations.

    Args:
        X_train: Raw training feature matrix.
        X_test: Raw test feature matrix.
        y_train: Integer training labels.
        y_test: Integer test labels.
        cfg: Full parsed config dict.
        kfold_splits_override: When provided, use this fold count instead
            of cfg["lda"]["kfold_splits"]. Pass 3 for first-pass mode,
            leave None to use the config value (typically 5).

    Returns:
        Tuple of (fitted_pca, list_of_result_dicts). Each result dict
        contains: params, cv_mean, cv_std, mislabel_pct, confusion, lda.
    """
    personalities: List[str] = cfg["training"]["personalities"]

    # Caller may override the fold count (e.g. 3 for first-pass mode).
    kfold_splits: int = (
        kfold_splits_override
        if kfold_splits_override is not None
        else cfg["lda"]["kfold_splits"]
    )

    # Defensive cap: StratifiedKFold requires n_splits ≤ smallest class size.
    min_class_count = int(np.bincount(y_train).min())
    if kfold_splits > min_class_count:
        print(
            f"[LDA Pipeline] WARNING: kfold_splits={kfold_splits} exceeds "
            f"smallest class size ({min_class_count}). "
            f"Reducing to {min_class_count}."
        )
        kfold_splits = min_class_count

    print(f"[LDA Pipeline] Using {kfold_splits}-fold stratified cross-validation.")

    pca, X_train_pca, X_test_pca = fit_pca(X_train, X_test)

    results = []
    for params in _LDA_PARAM_GRID:
        result = evaluate_lda_params(
            params=params,
            X_train=X_train_pca,
            X_test=X_test_pca,
            y_train=y_train,
            y_test=y_test,
            kfold_splits=kfold_splits,
            personalities=personalities,
        )
        results.append(result)

    return pca, results


def _print_confusion_matrix(cm: npt.NDArray, personalities: List[str]) -> None:
    """Pretty-print a confusion matrix to stdout.

    Args:
        cm: Square confusion matrix array.
        personalities: Ordered labels for rows and columns.
    """
    header = "  " + "  ".join(f"{p[:4]:>6}" for p in personalities)
    print(header)
    for i, row_label in enumerate(personalities):
        row = "  ".join(f"{cm[i, j]:6d}" for j in range(len(personalities)))
        print(f"  {row_label[:4]:>6}  {row}")
