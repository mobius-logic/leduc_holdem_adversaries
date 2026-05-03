"""Build LDA_1 through LDA_60 sequential personality-classification models.

For LDA_k the feature vector is:
  - The first k observation slots (k × 28 features, from the flat obs matrix)
  - Concatenated with the 15 tournament-level aggregate features
Total features: k × 28 + 15

All models are saved to  <workspace_root>/LDA_models/LDA_{k}.pkl

Usage::

    cd leduc_holdem
    python non_instruction/build_sequential_lda.py
    python non_instruction/build_sequential_lda.py --config path/to/config.yaml
    python non_instruction/build_sequential_lda.py --start 1 --end 59   # subset
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict

import numpy as np
import numpy.typing as npt
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Ensure the leduc_holdem package root is importable.
_NI_DIR = os.path.dirname(os.path.abspath(__file__))
_LEDUC_ROOT = os.path.dirname(_NI_DIR)
if _LEDUC_ROOT not in sys.path:
    sys.path.insert(0, _LEDUC_ROOT)

from eval_runs.lda_pipeline import run_lda_pipeline          # noqa: E402
from eval_runs.model_selector import select_best_model       # noqa: E402
from non_instruction.runner_ni import (                      # noqa: E402
    build_train_test_arrays,
    load_personality_ndarrays,
)

# ── Constants (must match observer.py / config) ───────────────────────────────
_OBS_FEATURES_PER_SLOT = 28   # vector_length
_TOTAL_OBS_SLOTS = 60         # 15 hands × 4 slots
_OBS_FLAT_LENGTH = _TOTAL_OBS_SLOTS * _OBS_FEATURES_PER_SLOT  # 1680
_AGG_LENGTH = 15


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slice_to_k(
    full_arrays: Dict[str, npt.NDArray[np.float32]],
    k: int,
) -> Dict[str, npt.NDArray[np.float32]]:
    """Return arrays truncated to the first k obs slots + all agg features.

    Args:
        full_arrays: Dict personality → ndarray of shape (N, 1695).
        k: Number of observation slots to retain (1–60).

    Returns:
        Dict personality → ndarray of shape (N, k*28 + 15).
    """
    obs_end = k * _OBS_FEATURES_PER_SLOT
    result: Dict[str, npt.NDArray[np.float32]] = {}
    for p, arr in full_arrays.items():
        obs_part = arr[:, :obs_end]              # (N, k*28)
        agg_part = arr[:, _OBS_FLAT_LENGTH:]     # (N, 15)
        result[p] = np.concatenate([obs_part, agg_part], axis=1).astype(np.float32)
    return result


def _save_model(
    best_result: dict,
    pca,
    X_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.int32],
    output_path: str,
) -> None:
    """Refit the best LDA on PCA-transformed training data and pickle it.

    Pickle payload keys: ``pca``, ``lda``, ``params``.

    Args:
        best_result: Best result dict from ``select_best_model``.
        pca: Fitted PCA transformer.
        X_train: Raw (pre-PCA) training features.
        y_train: Training labels.
        output_path: Destination .pkl file path.
    """
    params = best_result["params"]
    lda = LinearDiscriminantAnalysis(
        solver=params["solver"],
        shrinkage=params["shrinkage"],
    )
    X_train_pca = pca.transform(X_train)
    lda.fit(X_train_pca, y_train)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump({"pca": pca, "lda": lda, "params": params}, fh)
    print(f"  -> Saved {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LDA_1–LDA_60 sequential models"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_LEDUC_ROOT, "config.yaml"),
        help="Path to config.yaml (default: <leduc_root>/config.yaml)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First k value to build (default: 1)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=_TOTAL_OBS_SLOTS,
        help=f"Last k value to build, inclusive (default: {_TOTAL_OBS_SLOTS})",
    )
    args = parser.parse_args()

    # ── Load config ──
    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    config_dir = os.path.dirname(config_path)
    for key, val in cfg["paths"].items():
        if not os.path.isabs(str(val)):
            cfg["paths"][key] = os.path.normpath(os.path.join(config_dir, val))

    num_tournaments: int = cfg["training"]["num_tournaments"]
    output_dir = os.path.join(os.path.dirname(_LEDUC_ROOT), "LDA_models")
    os.makedirs(output_dir, exist_ok=True)

    k_range = range(args.start, args.end + 1)
    print(
        f"\n{'#' * 60}\n"
        f"Sequential LDA build: k={args.start}..{args.end}\n"
        f"Features per model:   k×{_OBS_FEATURES_PER_SLOT} obs + {_AGG_LENGTH} agg\n"
        f"Tournaments/personality: {num_tournaments}\n"
        f"Output dir: {output_dir}\n"
        f"{'#' * 60}"
    )

    # ── Load full tournament data once ──
    print("\n[Step 1] Loading tournament CSV data...")
    full_arrays = load_personality_ndarrays(cfg, num_tournaments)

    # ── Build one LDA model per k ──
    summary_rows = []
    for k in k_range:
        n_features = k * _OBS_FEATURES_PER_SLOT + _AGG_LENGTH
        print(
            f"\n{'=' * 60}\n"
            f"LDA_{k}  ({k} obs slot{'s' if k > 1 else ''}, {n_features} features total)\n"
            f"{'=' * 60}"
        )

        sliced = _slice_to_k(full_arrays, k)
        X_train, X_test, y_train, y_test = build_train_test_arrays(sliced, cfg)

        pca, results = run_lda_pipeline(
            X_train, X_test, y_train, y_test, cfg,
            kfold_splits_override=None,
        )

        best = select_best_model(results)
        output_path = os.path.join(output_dir, f"LDA_{k}.pkl")
        _save_model(best, pca, X_train, y_train, output_path)

        summary_rows.append(
            (k, n_features, best["cv_mean"], best["mislabel_pct"])
        )
        print(
            f"[LDA_{k}] Done — "
            f"CV mean={best['cv_mean']:.4f}, "
            f"mislabel={best['mislabel_pct']:.1f}%"
        )

    # ── Final summary ──
    print(f"\n{'#' * 60}\nSUMMARY\n{'#' * 60}")
    print(f"{'k':>4}  {'features':>10}  {'CV mean':>10}  {'mislabel%':>10}")
    for k, n_feat, cv, mis in summary_rows:
        print(f"{k:>4}  {n_feat:>10}  {cv:>10.4f}  {mis:>10.1f}")
    print(f"\nAll models saved to: {output_dir}")


if __name__ == "__main__":
    main()
