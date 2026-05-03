"""Model selection and persistence for the LDA classifier.

Selects the best LDA configuration from the pipeline results, refits
it on training data combined with its PCA transformer, and saves both
to a pickle file for future inference.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the best LDA result from the evaluation list.

    Selection criterion: highest mean cross-validation accuracy, with
    lowest mislabeling percentage as the tiebreaker.

    Args:
        results: List of result dicts from ``run_lda_pipeline``. Each
            dict has keys: params, cv_mean, cv_std, mislabel_pct,
            confusion, lda.

    Returns:
        The result dict corresponding to the best parameter set.
    """
    print("\n[ModelSelector] Selecting best model...")

    # Filter out configs that failed (cv_mean == -1.0, lda == None).
    viable = [r for r in results if r.get("lda") is not None]
    pool = viable if viable else results  # fall back to all if none viable

    # Sort by cv_mean descending, then mislabel_pct ascending.
    ranked = sorted(
        pool,
        key=lambda r: (-r["cv_mean"], r["mislabel_pct"]),
    )

    best = ranked[0]
    params = best["params"]
    print(
        f"[ModelSelector] Best: solver={params['solver']}, "
        f"shrinkage={params['shrinkage']} | "
        f"CV mean={best['cv_mean']:.4f}, "
        f"mislabel={best['mislabel_pct']:.1f}%"
    )
    return best


def save_best_model(
    best_result: Dict[str, Any],
    pca: PCA,
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int32],
    cfg: dict,
) -> str:
    """Refit the best LDA on training data and save with PCA transformer.

    The saved pickle contains a dict:
      {
        "pca":    fitted PCA transformer,
        "lda":    fitted LDA classifier,
        "params": hyperparameter dict used,
      }

    Args:
        best_result: Best result dict from ``select_best_model``.
        pca: Fitted PCA transformer (applied to raw features).
        X_train: Raw (pre-PCA) training features.
        y_train: Training labels.
        cfg: Full parsed config dict.

    Returns:
        Absolute path to the saved pickle file.
    """
    model_dir: str = cfg["paths"]["model_dir"]
    lda_filename: str = cfg["paths"]["lda_model_filename"]
    os.makedirs(model_dir, exist_ok=True)

    params = best_result["params"]
    lda = LinearDiscriminantAnalysis(
        solver=params["solver"],
        shrinkage=params["shrinkage"],
    )

    # Transform training data with PCA before fitting.
    X_train_pca = pca.transform(X_train)
    lda.fit(X_train_pca, y_train)

    payload = {"pca": pca, "lda": lda, "params": params}
    filepath = os.path.join(model_dir, lda_filename)
    with open(filepath, "wb") as fh:
        pickle.dump(payload, fh)

    print(f"[ModelSelector] Saved best model to {filepath}")
    return filepath
