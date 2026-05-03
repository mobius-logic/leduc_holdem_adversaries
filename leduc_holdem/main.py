"""Entry point and CLI for the Leduc Hold'em Personality Classification System.

Usage:
    python main.py                    # Full run: 100 tournaments per personality
    python main.py --first-pass       # Smoke test: 3 tournaments per personality
    python main.py --eval-only        # Skip data collection; run LDA on existing CSVs
    python main.py --config path.yaml # Use a custom config file
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_paths_relative_to_config(cfg: dict, config_dir: str) -> dict:
    """Make all path values in cfg absolute, resolved from the config directory.

    Args:
        cfg: Parsed config dict (paths section contains relative paths).
        config_dir: Directory containing config.yaml.

    Returns:
        Config dict with absolute path values.
    """
    paths = cfg["paths"]
    for key, val in paths.items():
        if not os.path.isabs(val):
            paths[key] = os.path.normpath(os.path.join(config_dir, val))
    return cfg


def run_data_collection(cfg: dict, num_tournaments: int) -> None:
    """Run tournament data collection for all personalities in parallel.

    Args:
        cfg: Full parsed config dict.
        num_tournaments: Number of tournaments to run per personality.
    """
    from training.runner import run_all_personalities

    print(
        f"\n{'#' * 60}\n"
        f"DATA COLLECTION: {num_tournaments} tournaments per personality\n"
        f"{'#' * 60}"
    )
    run_all_personalities(cfg=cfg, num_tournaments=num_tournaments)


def run_lda_evaluation(cfg: dict, num_tournaments: int) -> None:
    """Load collected CSVs, build arrays, fit LDA, and save best model.

    Args:
        cfg: Full parsed config dict.
        num_tournaments: Number of tournaments that were collected.
    """
    from eval_runs.lda_pipeline import run_lda_pipeline
    from eval_runs.model_selector import save_best_model, select_best_model
    from training.runner import build_train_test_arrays, load_personality_ndarrays

    print(f"\n{'#' * 60}\nLDA EVALUATION PIPELINE\n{'#' * 60}")

    personality_arrays = load_personality_ndarrays(cfg, num_tournaments)
    X_train, X_test, y_train, y_test = build_train_test_arrays(
        personality_arrays, cfg
    )

    pca, results = run_lda_pipeline(X_train, X_test, y_train, y_test, cfg)

    print(f"\n{'=' * 60}\nSUMMARY OF ALL PARAMETER SETS\n{'=' * 60}")
    for r in results:
        p = r["params"]
        print(
            f"  solver={p['solver']:6s}, shrinkage={str(p['shrinkage']):6s} | "
            f"CV mean={r['cv_mean']:.4f} ± {r['cv_std']:.4f} | "
            f"Mislabel={r['mislabel_pct']:.1f}%"
        )

    best = select_best_model(results)
    save_best_model(
        best_result=best,
        pca=pca,
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
    )


def main() -> None:
    """Parse CLI arguments, load config, and orchestrate the pipeline."""
    parser = argparse.ArgumentParser(
        description="Leduc Hold'em Personality Classification System"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml (default: config.yaml next to main.py)",
    )
    parser.add_argument(
        "--first-pass",
        action="store_true",
        help=(
            "Run only first_pass_tournaments per personality "
            "(smoke test mode)."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip data collection; run LDA pipeline on existing CSVs.",
    )
    args = parser.parse_args()

    # Load and resolve config.
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    cfg = _resolve_paths_relative_to_config(cfg, os.path.dirname(config_path))

    # Determine tournament count.
    if args.first_pass:
        num_tournaments = cfg["training"]["first_pass_tournaments"]
        print(
            f"[Main] --first-pass mode: {num_tournaments} tournaments "
            "per personality."
        )
    else:
        num_tournaments = cfg["training"]["num_tournaments"]

    # Ensure data directories exist.
    os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["tournament_dir"], exist_ok=True)

    print(
        f"\n[Main] Config loaded from {config_path}\n"
        f"[Main] Personalities: {cfg['training']['personalities']}\n"
        f"[Main] Tournaments per personality: {num_tournaments}\n"
        f"[Main] Data dir: {cfg['paths']['data_dir']}\n"
        f"[Main] Model dir: {cfg['paths']['model_dir']}"
    )

    if not args.eval_only:
        run_data_collection(cfg, num_tournaments)

    run_lda_evaluation(cfg, num_tournaments)

    print("\n[Main] Pipeline complete.")


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows.
    import multiprocessing
    multiprocessing.freeze_support()

    # Add the leduc_holdem package root to sys.path so imports work
    # when invoked as `python main.py` from the leduc_holdem/ directory.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    main()
