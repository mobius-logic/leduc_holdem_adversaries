"""CLI entry point for the rule-based (non-instruction) automation pipeline.

Mirrors ``main.py`` but uses ``non_instruction/runner_ni.py`` which drives
fully deterministic rule-based agents instead of LLM-backed PersonalityAgents.
No API key or Azure OpenAI access is required.

Usage::

    cd leduc_holdem
    python non_instruction/main_ni.py                    # all tournaments from config
    python non_instruction/main_ni.py --first-pass       # first_pass_tournaments count
    python non_instruction/main_ni.py --eval-only        # skip collection, run LDA only
    python non_instruction/main_ni.py --config path.yaml # custom config

Key config.yaml settings:
    training.num_tournaments         – number of tournaments per personality (full run)
    training.first_pass_tournaments  – number for smoke-test / first-pass run
    training.personalities           – list of personality names
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# Ensure the leduc_holdem package root is importable.
_NI_DIR     = os.path.dirname(os.path.abspath(__file__))
_LEDUC_ROOT = os.path.dirname(_NI_DIR)
if _LEDUC_ROOT not in sys.path:
    sys.path.insert(0, _LEDUC_ROOT)


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_paths(cfg: dict, config_dir: str) -> dict:
    for key, val in cfg["paths"].items():
        if not os.path.isabs(str(val)):
            cfg["paths"][key] = os.path.normpath(os.path.join(config_dir, val))
    return cfg


def run_data_collection(cfg: dict, num_tournaments: int) -> None:
    """Run rule-based tournament data collection for all personalities."""
    # Import here so the module is only loaded when needed.
    from non_instruction.runner_ni import run_all_personalities

    print(
        f"\n{'#' * 60}\n"
        f"DATA COLLECTION (rule-based): {num_tournaments} tournaments per personality\n"
        f"{'#' * 60}"
    )
    run_all_personalities(cfg=cfg, num_tournaments=num_tournaments)


def run_lda_evaluation(cfg: dict, num_tournaments: int) -> None:
    """Load collected CSVs, build arrays, fit LDA, and save best model."""
    from eval_runs.lda_pipeline import run_lda_pipeline
    from eval_runs.model_selector import save_best_model, select_best_model
    from non_instruction.runner_ni import build_train_test_arrays, load_personality_ndarrays

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
    parser = argparse.ArgumentParser(
        description="Rule-based Leduc Hold'em automation pipeline (no LLM required)"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_LEDUC_ROOT, "config.yaml"),
        help="Path to config.yaml (default: config.yaml in leduc_holdem/)",
    )
    parser.add_argument(
        "--first-pass",
        action="store_true",
        help=(
            "Run only first_pass_tournaments per personality "
            "(smoke-test mode, default count is 5 in config)."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip data collection; run LDA pipeline on existing CSVs.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = _load_config(config_path)
    cfg = _resolve_paths(cfg, os.path.dirname(config_path))

    if args.first_pass:
        num_tournaments = cfg["training"]["first_pass_tournaments"]
        print(
            f"[Main-NI] --first-pass mode: {num_tournaments} tournaments "
            "per personality."
        )
    else:
        num_tournaments = cfg["training"]["num_tournaments"]

    os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["tournament_dir"], exist_ok=True)

    print(
        f"\n[Main-NI] Config: {config_path}\n"
        f"[Main-NI] Personalities: {cfg['training']['personalities']}\n"
        f"[Main-NI] Tournaments per personality: {num_tournaments}\n"
        f"[Main-NI] Agent type: rule-based (no LLM)\n"
        f"[Main-NI] Data dir: {cfg['paths']['data_dir']}\n"
        f"[Main-NI] Model dir: {cfg['paths']['model_dir']}"
    )

    if not args.eval_only:
        run_data_collection(cfg, num_tournaments)

    run_lda_evaluation(cfg, num_tournaments)

    print("\n[Main-NI] Pipeline complete.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    main()
