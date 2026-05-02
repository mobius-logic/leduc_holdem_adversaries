"""Observation collection, padding, and CSV persistence.

Collects 19-element observation vectors before each personality agent
action, manages the 4-slot-per-hand padding structure, and serialises
the resulting 20×19 matrix to CSV at the end of each tournament.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from game.deck import ALL_CARDS
from game.state import GameState, Round

# Constant for padding value.
_PAD = -1.0
_VECTOR_LENGTH = 19
_SLOTS_PER_HAND = 4

# One-hot indices for last opponent action encoding.
# [0] = Check or Call, [1] = Raise, [2] = Fold
_OPP_ACTION_CHECK_CALL = 0
_OPP_ACTION_RAISE = 1
_OPP_ACTION_FOLD = 2


def _pad_vector() -> npt.NDArray[np.float32]:
    """Return a 19-element padding vector filled with -1.0.

    Returns:
        float32 array of shape (19,) with all values -1.0.
    """
    return np.full(_VECTOR_LENGTH, _PAD, dtype=np.float32)


def _build_observation_vector(
    state: GameState, win_prob: float
) -> npt.NDArray[np.float32]:
    """Construct the 19-element observation vector from game state.

    Index mapping:
      [0]    Win probability (float 0.0–1.0)
      [1]    Personality agent chip stack (raw)
      [2]    Opponent chip stack (raw)
      [3]    Pot size (raw)
      [4–6]  Last opponent action one-hot [Check/Call, Raise, Fold].
             All zeros if no prior opponent action this hand.
      [7–12] Private card one-hot over [J♥,Q♥,K♥,J♠,Q♠,K♠]
      [13–18]Community card one-hot over same order.
             ALL ZEROS pre-flop.

    Args:
        state: Current fully-updated game state.
        win_prob: Exact win probability for the personality agent.

    Returns:
        float32 ndarray of shape (19,).

    Raises:
        ValueError: If the constructed vector does not have shape (19,).
    """
    vec = np.zeros(_VECTOR_LENGTH, dtype=np.float32)

    # [0] Win probability.
    vec[0] = float(win_prob)

    # [1] Personality stack.
    vec[1] = float(state.personality_stack)

    # [2] Opponent stack.
    vec[2] = float(state.opponent_stack)

    # [3] Pot.
    vec[3] = float(state.pot)

    # [4–6] Last opponent action one-hot.
    opp_action = state.last_opponent_action
    if opp_action is not None:
        if opp_action in ("Check", "Call"):
            vec[4 + _OPP_ACTION_CHECK_CALL] = 1.0
        elif opp_action == "Raise":
            vec[4 + _OPP_ACTION_RAISE] = 1.0
        elif opp_action == "Fold":
            vec[4 + _OPP_ACTION_FOLD] = 1.0

    # [7–12] Private card one-hot.
    card_list = list(ALL_CARDS)
    priv_idx = card_list.index(state.personality_card)
    vec[7 + priv_idx] = 1.0

    # [13–18] Community card one-hot (all zeros pre-flop).
    if state.community_card is not None:
        comm_idx = card_list.index(state.community_card)
        vec[13 + comm_idx] = 1.0

    if vec.shape != (_VECTOR_LENGTH,):
        raise ValueError(
            f"Observation vector shape {vec.shape} != ({_VECTOR_LENGTH},)"
        )

    return vec


class TournamentObserver:
    """Manages observation collection for one tournament.

    Maintains a buffer of (hands × slots) observation vectors and
    provides a callback suitable for injection into the game loop.

    Each hand has exactly 4 slots:
      Slot 0: Personality's 1st pre-flop action (or pad)
      Slot 1: Personality's 2nd pre-flop action (or pad)
      Slot 2: Personality's 1st post-flop action (or pad)
      Slot 3: Personality's 2nd post-flop action (or pad)

    Attributes:
        hands_per_tournament: Number of hands in this tournament.
        data: 3D buffer of shape (hands, 4, 19) for collected vectors.
    """

    def __init__(self, hands_per_tournament: int) -> None:
        """Initialise with all slots padded.

        Args:
            hands_per_tournament: Number of hands in this tournament.
        """
        self.hands_per_tournament = hands_per_tournament
        # Initialise all slots as padding.
        self.data: npt.NDArray[np.float32] = np.full(
            (hands_per_tournament, _SLOTS_PER_HAND, _VECTOR_LENGTH),
            _PAD,
            dtype=np.float32,
        )
        self._win_prob_fn = None

    def set_win_prob_fn(self, fn) -> None:
        """Attach a win probability function.

        Args:
            fn: Callable(state) -> float.
        """
        self._win_prob_fn = fn

    def record(self, state: GameState, slot_index: int) -> None:
        """Record an observation vector into the correct slot.

        Called immediately before the personality agent's action is
        applied. Slot index determines placement within the hand's 4
        slots (0 = 1st pre-flop, 1 = 2nd pre-flop, 2 = 1st post-flop,
        3 = 2nd post-flop).

        Args:
            state: Current game state at observation time.
            slot_index: Target slot (0–3) within the current hand.
        """
        assert self._win_prob_fn is not None, (
            "Call set_win_prob_fn() before using the observer."
        )
        hand = state.hand_index
        if slot_index >= _SLOTS_PER_HAND:
            print(
                f"    [Observer] slot_index={slot_index} exceeds max "
                f"({_SLOTS_PER_HAND - 1}); skipping."
            )
            return

        win_prob = self._win_prob_fn(state)
        vec = _build_observation_vector(state, win_prob)
        self.data[hand, slot_index] = vec
        print(
            f"    [Observer] Recorded slot {slot_index} for hand "
            f"{hand + 1} | win_prob={win_prob:.1%}"
        )

    def to_matrix(self) -> npt.NDArray[np.float32]:
        """Flatten observation buffer to a 20×19 matrix.

        Returns:
            float32 ndarray of shape (20, 19).

        Raises:
            ValueError: If the resulting matrix lacks shape (20, 19).
        """
        total_slots = self.hands_per_tournament * _SLOTS_PER_HAND
        matrix = self.data.reshape(total_slots, _VECTOR_LENGTH)
        if matrix.shape != (total_slots, _VECTOR_LENGTH):
            raise ValueError(
                f"Observation matrix shape {matrix.shape} != "
                f"({total_slots}, {_VECTOR_LENGTH})"
            )
        return matrix


def save_tournament_csv(
    matrix: npt.NDArray[np.float32],
    data_dir: str,
    seed: int,
    personality: str,
) -> str:
    """Save the 20×19 observation matrix to a CSV file.

    Args:
        matrix: float32 ndarray of shape (20, 19).
        data_dir: Directory path where the CSV should be saved.
        seed: Tournament seed used in filename.
        personality: Personality name used in filename.

    Returns:
        Absolute path to the saved file.

    Raises:
        ValueError: If ``matrix`` does not have shape (20, 19).
    """
    expected_rows = 20
    expected_cols = _VECTOR_LENGTH
    if matrix.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"CSV matrix shape {matrix.shape} != "
            f"({expected_rows}, {expected_cols})"
        )

    os.makedirs(data_dir, exist_ok=True)
    filename = f"run_{seed}_{personality}.csv"
    filepath = os.path.join(data_dir, filename)
    np.savetxt(filepath, matrix, delimiter=",", fmt="%.6g")
    print(f"    [Observer] Saved {filepath}")
    return filepath
