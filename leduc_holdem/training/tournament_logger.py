"""Tournament game-log recorder.

Captures hand metadata, full action sequences, and outcomes for every
hand in a tournament and writes a structured JSON file for offline
analysis.

Directory layout::

    data/tournament/
        aggressive/
            t0000_seed42.json
            t0001_seed43.json
        conservative/
            ...

Each JSON file represents one complete tournament::

    {
      "personality": "aggressive",
      "seed": 42,
      "tournament_index": 0,
      "hands": [
        {
          "hand_index": 0,
          "hand_seed": 142,
          "stacks_start": {"personality": 50, "opponent": 50},
          "personality_card": "Q\u2665",
          "opponent_card": "K\u2660",
          "community_card": "J\u2665",        // null if preflop fold
          "personality_acts_first": false,
          "preflop_actions":  [{"player": "Opponent",     "action": "Check"},
                               {"player": "Personality",  "action": "Raise"}],
          "postflop_actions": [{"player": "Personality",  "action": "Check"},
                               {"player": "Opponent",     "action": "Check"}],
          "outcome": "showdown",             // preflop_fold | postflop_fold | showdown
          "winner": "Opponent",             // Personality | Opponent | Tie
          "stacks_end": {"personality": 48, "opponent": 52}
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from game.state import PERSONALITY, GameState


def _outcome_label(state: GameState) -> str:
    """Return a string label describing how the hand ended.

    Args:
        state: Final GameState returned by play_hand.

    Returns:
        One of 'preflop_fold', 'postflop_fold', or 'showdown'.
    """
    if not state.preflop_done:
        return "preflop_fold"
    if state.hand_over:
        return "postflop_fold"
    return "showdown"


def _winner_label(state: GameState) -> str:
    """Return a human-readable winner label.

    Args:
        state: Final GameState with winner already set.

    Returns:
        One of 'Personality', 'Opponent', or 'Tie'.
    """
    if state.winner is None:
        return "Tie"
    return "Personality" if state.winner == PERSONALITY else "Opponent"


class TournamentLogger:
    """Records one tournament's gameplay and writes a JSON log file.

    Intended usage (orchestrated by the game runner)::

        logger = TournamentLogger(personality, seed, t_idx)

        for each hand:
            logger.start_hand(hand_index, hand_seed, stacks_before_ante)
            # game loop fires logger.record_action(...) for every action
            logger.end_hand(final_state)

        logger.save(tournament_dir)

    Attributes:
        personality: Lowercase personality name for this tournament.
    """

    def __init__(
        self,
        personality: str,
        seed: int,
        tournament_index: int,
    ) -> None:
        """Initialise a logger for a single tournament.

        Args:
            personality: Lowercase personality name.
            seed: Tournament-level RNG seed.
            tournament_index: Zero-based tournament index within the run.
        """
        self.personality = personality
        self._data: Dict[str, Any] = {
            "personality": personality,
            "seed": seed,
            "tournament_index": tournament_index,
            "hands": [],
        }
        self._current_hand: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Event methods (called by the game engine / runner)
    # ------------------------------------------------------------------

    def start_hand(
        self,
        hand_index: int,
        hand_seed: int,
        stacks_start: List[int],
    ) -> None:
        """Begin recording a new hand.

        Called by the runner immediately before play_hand is invoked,
        while stacks are still the pre-ante values.

        Args:
            hand_index: Zero-based hand index within the tournament.
            hand_seed: RNG seed used for this specific hand's shuffle.
            stacks_start: Chip counts [personality, opponent] before
                this hand's ante is collected.
        """
        self._current_hand = {
            "hand_index": hand_index,
            "hand_seed": hand_seed,
            "stacks_start": {
                "personality": stacks_start[0],
                "opponent": stacks_start[1],
            },
            "personality_card": None,
            "opponent_card": None,
            "community_card": None,
            "personality_acts_first": None,
            "preflop_actions": [],
            "postflop_actions": [],
            "outcome": None,
            "winner": None,
            "stacks_end": None,
        }

    def record_action(
        self,
        player_label: str,
        action: str,
        round_name: str,
    ) -> None:
        """Append one action to the current hand's action sequence.

        Intended to be passed directly as the ``action_callback``
        parameter of ``LeducHoldemGame.play_hand``.

        Args:
            player_label: 'Personality' or 'Opponent'.
            action: One of 'Check', 'Call', 'Raise', 'Fold'.
            round_name: 'preflop' or 'postflop'.
        """
        if self._current_hand is None:
            return
        entry: Dict[str, str] = {"player": player_label, "action": action}
        if round_name == "preflop":
            self._current_hand["preflop_actions"].append(entry)
        else:
            self._current_hand["postflop_actions"].append(entry)

    def end_hand(self, state: GameState) -> None:
        """Finalise the current hand record using the returned GameState.

        Called by the runner immediately after play_hand returns.

        Args:
            state: The final GameState for the completed hand.
        """
        if self._current_hand is None:
            return
        self._current_hand["personality_card"] = str(state.personality_card)
        self._current_hand["opponent_card"] = str(state.opponent_card)
        self._current_hand["community_card"] = (
            str(state.community_card) if state.community_card is not None else None
        )
        self._current_hand["personality_acts_first"] = state.personality_acts_first
        self._current_hand["outcome"] = _outcome_label(state)
        self._current_hand["winner"] = _winner_label(state)
        self._current_hand["stacks_end"] = {
            "personality": state.stacks[0],
            "opponent": state.stacks[1],
        }
        self._data["hands"].append(self._current_hand)
        self._current_hand = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, tournament_dir: str) -> None:
        """Write the tournament log as a JSON file.

        Creates the per-personality subdirectory if it does not exist.

        Args:
            tournament_dir: Root directory for tournament logs
                (e.g. ``data/tournament/``).
        """
        personality = self._data["personality"]
        seed = self._data["seed"]
        t_idx = self._data["tournament_index"]

        subdir = os.path.join(tournament_dir, personality)
        os.makedirs(subdir, exist_ok=True)

        filename = f"t{t_idx:04d}_seed{seed}.json"
        path = os.path.join(subdir, filename)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)

        print(f"  [TournamentLogger] Saved: {path}")
