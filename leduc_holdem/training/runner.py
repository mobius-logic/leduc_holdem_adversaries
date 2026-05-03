"""Tournament runner with per-personality parallelisation.

Runs ``num_tournaments`` tournaments for each personality using a
``ProcessPoolExecutor`` — one worker per personality. All tournaments
within a worker are executed strictly sequentially to respect OpenAI
rate limits.

Seeds are deterministic: seed = random_seed_base + tournament_index.
Every seed is logged to both stdout and a ``seeds.log`` file.
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

from agents.personality_agent import PersonalityAgent
from agents.random_agent import RandomAgent
from game.leduc_holdem import LeducHoldemGame
from training.observer import TournamentObserver, save_tournament_csv
from training.tournament_logger import TournamentLogger
from training.win_probability import compute_win_probability


def _run_personality_tournaments(
    personality: str,
    num_tournaments: int,
    cfg: dict,
) -> str:
    """Run all tournaments for a single personality (one worker body).

    All print output is buffered internally and returned as a single
    string, so the parent process can print each personality's full log
    in one uninterlepted block once the worker finishes.

    Args:
        personality: Lowercase personality name.
        num_tournaments: Number of tournaments to run for this personality.
        cfg: Full parsed config dictionary.

    Returns:
        Full buffered log string for this personality.
    """
    # Re-import inside the worker process (needed for multiprocessing).
    import io
    import os
    import sys

    # Redirect stdout to a buffer for the lifetime of this worker.
    _buffer = io.StringIO()
    _real_stdout = sys.stdout
    sys.stdout = _buffer

    try:
        seed_base: int = cfg["training"]["random_seed_base"]
        data_dir: str = cfg["paths"]["data_dir"]
        tournament_dir: str = cfg["paths"]["tournament_dir"]
        hands_per_tournament: int = cfg["game"]["hands_per_tournament"]

        seeds_log_path = os.path.join(data_dir, "seeds.log")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(tournament_dir, exist_ok=True)

        game = LeducHoldemGame(cfg)

        # Win probability function bound to compute_win_probability.
        def win_prob_fn(state):
            return compute_win_probability(state)

        personality_agent = PersonalityAgent(
            personality=personality,
            cfg=cfg,
            win_prob_fn=win_prob_fn,
        )

        print(
            f"\n{'=' * 60}\n"
            f"[{personality.upper()}] Starting {num_tournaments} tournaments\n"
            f"{'=' * 60}"
        )

        for t_idx in range(num_tournaments):
            seed = seed_base + t_idx

            # Log the seed.
            log_line = f"[{personality.upper()}] tournament={t_idx} seed={seed}"
            print(log_line)
            with open(seeds_log_path, "a", encoding="utf-8") as log_fh:
                log_fh.write(log_line + "\n")

            print(
                f"\n[{personality.upper()}] Tournament {t_idx + 1}/{num_tournaments} "
                f"(seed={seed})"
            )

            # Create the random agent with a seeded RNG derived from the tournament seed.
            rng = random.Random(seed)
            random_agent = RandomAgent(rng=rng)

            observer = TournamentObserver(hands_per_tournament=hands_per_tournament)
            observer.set_win_prob_fn(win_prob_fn)

            def obs_callback(state, slot_index, _obs=observer):
                _obs.record(state, slot_index)

            tournament_logger = TournamentLogger(
                personality=personality,
                seed=seed,
                tournament_index=t_idx,
            )

            game.play_tournament(
                seed_base=seed_base,
                tournament_index=t_idx,
                personality_agent=personality_agent,
                random_agent=random_agent,
                obs_callback=obs_callback,
                tournament_logger=tournament_logger,
            )

            tournament_logger.save(tournament_dir)

            matrix = observer.to_matrix()
            save_tournament_csv(
                matrix=matrix,
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


def run_all_personalities(cfg: dict, num_tournaments: int) -> None:
    """Run tournaments for all personalities in parallel (4 workers).

    Reads the API key from the environment once in the parent process and
    embeds it into the config dict before passing it to worker processes.
    This ensures child processes (which may not inherit env vars on Windows)
    always have the key available.

    Args:
        cfg: Full parsed config dict.
        num_tournaments: Number of tournaments per personality.
    """
    personalities: List[str] = cfg["training"]["personalities"]
    n_workers: int = cfg["training"]["n_workers"]

    # Resolve the API key here in the parent and inject it so workers
    # never need to read environment variables themselves.
    key_env_var = cfg["api"]["key_env_var"]
    api_key = os.environ.get(key_env_var)
    if not api_key:
        raise EnvironmentError(
            f"Azure OpenAI API key not found. "
            f"Set the {key_env_var} environment variable before running."
        )
    cfg["api"]["_resolved_key"] = api_key

    print(
        f"\n[Runner] Launching {len(personalities)} personalities × "
        f"{num_tournaments} tournaments with {n_workers} workers."
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
                # Print the full buffered game log in one uninterrupted block.
                print(
                    f"\n{'#' * 60}\n"
                    f"GAME LOG: {personality.upper()}\n"
                    f"{'#' * 60}"
                )
                print(log_output, end="")
                print(
                    f"[Runner] Personality '{personality}' finished successfully."
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"[Runner] Personality '{personality}' raised an exception: {exc}",
                    file=sys.stderr,
                )

    print("\n[Runner] All personalities complete.")


def load_personality_ndarrays(
    cfg: dict, num_tournaments: int
) -> Dict[str, npt.NDArray[np.float32]]:
    """Load all CSV files for each personality and stack into NDArrays.

    Each CSV is a 20×19 matrix (flattened to length 380). Stacking N
    tournaments yields shape (N, 380).

    Args:
        cfg: Full parsed config dict.
        num_tournaments: Expected number of tournament CSVs per personality.

    Returns:
        Dict mapping personality name → NDArray of shape (N, 380).
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
            rows.append(mat.flatten())

        stacked = np.stack(rows, axis=0)  # (N, 380)
        assert stacked.shape == (num_tournaments, flat_length), (
            f"Unexpected NDArray shape {stacked.shape} for {personality}"
        )
        result[personality] = stacked
        print(
            f"[Runner] Loaded {personality}: shape {stacked.shape}"
        )

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
        X_train shape: (300, 380), X_test shape: (100, 380) [for 100 tournaments].
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
        f"[Runner] Dataset shapes: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )
    return X_train, X_test, y_train, y_test
