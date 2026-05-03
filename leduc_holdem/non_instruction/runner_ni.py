"""Rule-based tournament runner for the Leduc Hold'em automation pipeline.

Mirrors ``training/runner.py`` but replaces the LLM-driven PersonalityAgent
with fully deterministic rule-based agents.  No API key or network call is
required.

The rule agents live in ``non_instruction/agents/`` and implement the same
``BaseAgent`` interface (``act(state, legal_actions) -> str``), so the game
engine, observer, tournament logger, and CSV output are 100% unchanged.

Usage (from the leduc_holdem/ directory)::

    python non_instruction/main_ni.py               # num_tournaments from config
    python non_instruction/main_ni.py --first-pass  # first_pass_tournaments
    python non_instruction/main_ni.py --eval-only   # skip data collection

Configuration keys used (config.yaml):
    training.personalities           – list of personality names to run
    training.num_tournaments         – full-run tournament count per personality
    training.first_pass_tournaments  – smoke-test tournament count
    training.n_workers               – parallel workers (one per personality)
    training.random_seed_base        – seed = base + tournament_index
    paths.data_dir                   – where observation CSVs are written
    paths.tournament_dir             – where tournament JSON logs are written
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Path setup — ensure both the leduc_holdem package root and the agents
# sub-directory are importable regardless of working directory.
# ---------------------------------------------------------------------------
_NI_DIR    = os.path.dirname(os.path.abspath(__file__))           # non_instruction/
_LEDUC_ROOT = os.path.dirname(_NI_DIR)                             # leduc_holdem/
_AGENTS_DIR = os.path.join(_NI_DIR, "agents")                     # non_instruction/agents/

for _p in (_LEDUC_ROOT, _AGENTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aggressive_rule_agent   import AggressiveRuleAgent    # noqa: E402
from analytical_rule_agent   import AnalyticalRuleAgent    # noqa: E402
from conservative_rule_agent import ConservativeRuleAgent  # noqa: E402
from reckless_rule_agent     import RecklessRuleAgent      # noqa: E402

from agents.random_agent import RandomAgent                # noqa: E402
from game.leduc_holdem   import LeducHoldemGame            # noqa: E402
from game.state          import PERSONALITY                # noqa: E402
from training.observer   import TournamentObserver, save_tournament_csv, save_tournament_agg_csv  # noqa: E402
from training.tournament_logger import TournamentLogger    # noqa: E402
from training.win_probability   import compute_win_probability  # noqa: E402


def _make_agent(personality: str, win_prob_fn):
    """Return the rule-based agent instance for the given personality name."""
    if personality == "aggressive":
        return AggressiveRuleAgent(win_prob_fn=win_prob_fn)
    if personality == "analytical":
        return AnalyticalRuleAgent(win_prob_fn=win_prob_fn)
    if personality == "conservative":
        return ConservativeRuleAgent(win_prob_fn=win_prob_fn)
    if personality == "reckless":
        return RecklessRuleAgent(win_prob_fn=win_prob_fn)
    raise ValueError(f"Unknown personality: {personality!r}")


# ---------------------------------------------------------------------------
# Per-personality worker
# ---------------------------------------------------------------------------

def _run_personality_tournaments(
    personality: str,
    num_tournaments: int,
    cfg: dict,
) -> str:
    """Run all tournaments for a single personality (one worker body).

    Output is buffered and returned as a single string so the parent
    process can print each personality block without interleaving.

    Args:
        personality: Lowercase personality name.
        num_tournaments: Number of tournaments to run.
        cfg: Full parsed config dictionary.

    Returns:
        Full buffered log string for this personality.
    """
    # Re-import inside the worker process (required for multiprocessing).
    import io
    import os
    import sys

    # Redirect stdout to a buffer.
    _buffer = io.StringIO()
    _real_stdout = sys.stdout
    sys.stdout = _buffer

    # Re-add paths inside the worker (new process on Windows).
    _ni_dir     = os.path.dirname(os.path.abspath(__file__))
    _leduc_root = os.path.dirname(_ni_dir)
    _agents_dir = os.path.join(_ni_dir, "agents")
    for _p in (_leduc_root, _agents_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    try:
        seed_base: int = cfg["training"]["random_seed_base"]
        data_dir: str = cfg["paths"]["data_dir"]
        tournament_dir: str = cfg["paths"]["tournament_dir"]
        hands_per_tournament: int = cfg["game"]["hands_per_tournament"]

        seeds_log_path = os.path.join(data_dir, "seeds.log")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(tournament_dir, exist_ok=True)

        game = LeducHoldemGame(cfg)

        def win_prob_fn(state):
            return compute_win_probability(state)

        personality_agent = _make_agent(personality, win_prob_fn)

        print(
            f"\n{'=' * 60}\n"
            f"[{personality.upper()}] Starting {num_tournaments} tournaments "
            f"(rule-based, no LLM)\n"
            f"{'=' * 60}"
        )

        for t_idx in range(num_tournaments):
            seed = seed_base + t_idx

            log_line = f"[{personality.upper()}] tournament={t_idx} seed={seed}"
            print(log_line)
            with open(seeds_log_path, "a", encoding="utf-8") as log_fh:
                log_fh.write(log_line + "\n")

            print(
                f"\n[{personality.upper()}] Tournament {t_idx + 1}/{num_tournaments} "
                f"(seed={seed})"
            )

            rng = random.Random(seed)
            random_agent = RandomAgent(rng=rng)

            observer = TournamentObserver(hands_per_tournament=hands_per_tournament)
            observer.set_win_prob_fn(win_prob_fn)
            observer.set_starting_chips(cfg["game"]["starting_chips"])

            def obs_callback(state, slot_index, _obs=observer):
                _obs.record(state, slot_index)

            tournament_logger = TournamentLogger(
                personality=personality,
                seed=seed,
                tournament_index=t_idx,
            )

            hand_states = game.play_tournament(
                seed_base=seed_base,
                tournament_index=t_idx,
                personality_agent=personality_agent,
                random_agent=random_agent,
                obs_callback=obs_callback,
                tournament_logger=tournament_logger,
                personality_action_callback=observer.record_personality_action,
            )

            # Record hand outcomes and final stack for aggregate features.
            for hs in hand_states:
                if hs.winner == PERSONALITY:
                    won: Optional[bool] = True
                elif hs.winner is None:
                    won = None
                else:
                    won = False
                observer.record_hand_result(won)
            if hand_states:
                observer.set_final_stack(hand_states[-1].stacks[PERSONALITY])

            tournament_logger.save(tournament_dir)

            matrix = observer.to_matrix()
            save_tournament_csv(
                matrix=matrix,
                data_dir=data_dir,
                seed=seed,
                personality=personality,
            )
            agg = observer.to_aggregates()
            save_tournament_agg_csv(
                agg=agg,
                data_dir=data_dir,
                seed=seed,
                personality=personality,
            )

        print(
            f"\n[{personality.upper()}] Completed {num_tournaments} tournaments. "
            "All CSVs saved."
        )

    finally:
        sys.stdout = _real_stdout

    return _buffer.getvalue()


# ---------------------------------------------------------------------------
# Public API — called by main_ni.py
# ---------------------------------------------------------------------------

def run_all_personalities(cfg: dict, num_tournaments: int) -> None:
    """Run tournaments for all personalities in parallel.

    No API key is required — every agent is a pure Python rule engine.

    Args:
        cfg: Full parsed config dict.
        num_tournaments: Number of tournaments per personality.
    """
    personalities: List[str] = cfg["training"]["personalities"]
    n_workers: int = cfg["training"]["n_workers"]

    print(
        f"\n[Runner-NI] Launching {len(personalities)} personalities × "
        f"{num_tournaments} tournaments with {n_workers} workers "
        f"(rule-based agents, no LLM)."
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_personality_tournaments, p, num_tournaments, cfg
            ): p
            for p in personalities
        }
        for future in concurrent.futures.as_completed(futures):
            personality = futures[future]
            try:
                log_output = future.result()
                print(
                    f"\n{'#' * 60}\n"
                    f"GAME LOG: {personality.upper()}\n"
                    f"{'#' * 60}"
                )
                print(log_output, end="")
                print(
                    f"[Runner-NI] Personality '{personality}' finished successfully."
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"[Runner-NI] Personality '{personality}' raised an exception: {exc}",
                    file=sys.stderr,
                )

    print("\n[Runner-NI] All personalities complete.")


def load_personality_ndarrays(
    cfg: dict, num_tournaments: int
) -> Dict[str, npt.NDArray[np.float32]]:
    """Load all CSV files for each personality and stack into NDArrays.

    Each tournament contributes a flat vector of length ``flat_length``
    (hands*4*22 observation features + 5 aggregate features). Stacking N
    tournaments yields shape (N, flat_length).

    Args:
        cfg: Full parsed config dict.
        num_tournaments: Expected number of tournament CSVs per personality.

    Returns:
        Dict mapping personality name → NDArray of shape (N, flat_length).
    """
    personalities: List[str] = cfg["training"]["personalities"]
    data_dir: str = cfg["paths"]["data_dir"]
    seed_base: int = cfg["training"]["random_seed_base"]
    flat_length: int = cfg["observation"]["flat_length"]
    result: Dict[str, npt.NDArray[np.float32]] = {}

    for personality in personalities:
        rows = []
        for t_idx in range(num_tournaments):
            seed = seed_base + t_idx
            filename = f"run_{seed}_{personality}.csv"
            filepath = os.path.join(data_dir, filename)
            mat = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
            agg_path = os.path.join(data_dir, f"run_{seed}_{personality}_agg.csv")
            agg = np.loadtxt(agg_path, delimiter=",", dtype=np.float32).flatten()
            rows.append(np.concatenate([mat.flatten(), agg]))

        stacked = np.stack(rows, axis=0)  # (N, 380)
        assert stacked.shape == (num_tournaments, flat_length), (
            f"Unexpected NDArray shape {stacked.shape} for {personality}"
        )
        result[personality] = stacked
        print(f"[Runner-NI] Loaded {personality}: shape {stacked.shape}")

    return result


def build_train_test_arrays(
    personality_arrays: Dict[str, npt.NDArray[np.float32]],
    cfg: dict,
):
    """Split, concatenate, and label personality arrays for LDA.

    Args:
        personality_arrays: Dict from load_personality_ndarrays().
        cfg: Full parsed config dict.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) as float32 / int32 arrays.
        Label encoding: Analytical=0, Conservative=1, Aggressive=2, Reckless=3.
    """
    personalities: List[str] = cfg["training"]["personalities"]
    train_split: float = cfg["training"]["train_split"]

    label_map = {p: i for i, p in enumerate(personalities)}

    x_trains, x_tests, y_trains, y_tests = [], [], [], []

    for personality in personalities:
        arr = personality_arrays[personality]
        n = arr.shape[0]
        split_idx = int(n * train_split)

        x_trains.append(arr[:split_idx])
        x_tests.append(arr[split_idx:])

        label = label_map[personality]
        y_trains.append(np.full(split_idx, label, dtype=np.int32))
        y_tests.append(np.full(n - split_idx, label, dtype=np.int32))

    X_train = np.concatenate(x_trains, axis=0)
    X_test = np.concatenate(x_tests, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    y_test = np.concatenate(y_tests, axis=0)

    print(
        f"[Runner-NI] Dataset shapes: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )
    return X_train, X_test, y_train, y_test
