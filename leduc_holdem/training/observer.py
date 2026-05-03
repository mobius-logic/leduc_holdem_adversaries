"""Observation collection, padding, and CSV persistence.

Collects 22-element observation vectors before each personality agent
action, manages the 4-slot-per-hand padding structure, and serialises
the resulting 20×22 matrix to CSV at the end of each tournament.
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
_VECTOR_LENGTH = 22
_SLOTS_PER_HAND = 4

# One-hot indices for last opponent action encoding.
# [0] = Check or Call, [1] = Raise, [2] = Fold
_OPP_ACTION_CHECK_CALL = 0
_OPP_ACTION_RAISE = 1
_OPP_ACTION_FOLD = 2

# One-hot indices for last personality action encoding (same layout).
_PERS_ACTION_CHECK_CALL = 0
_PERS_ACTION_RAISE = 1
_PERS_ACTION_FOLD = 2

# Number of tournament-level aggregate features appended after the flat obs matrix.
# [raise_rate, call_check_rate, fold_rate, net_winnings_norm, hand_win_rate]
_AGG_LENGTH = 5


def _pad_vector() -> npt.NDArray[np.float32]:
    """Return a 22-element padding vector filled with -1.0.

    Returns:
        float32 array of shape (22,) with all values -1.0.
    """
    return np.full(_VECTOR_LENGTH, _PAD, dtype=np.float32)


def _build_observation_vector(
    state: GameState, win_prob: float
) -> npt.NDArray[np.float32]:
    """Construct the 22-element observation vector from game state.

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
      [19–21]Last personality action one-hot [Check/Call, Raise, Fold].
             All zeros if no prior personality action this hand.

    Args:
        state: Current fully-updated game state.
        win_prob: Exact win probability for the personality agent.

    Returns:
        float32 ndarray of shape (22,).

    Raises:
        ValueError: If the constructed vector does not have shape (22,).
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

    # [19–21] Last personality action one-hot.
    pers_action = state.last_personality_action
    if pers_action is not None:
        if pers_action in ("Check", "Call"):
            vec[19 + _PERS_ACTION_CHECK_CALL] = 1.0
        elif pers_action == "Raise":
            vec[19 + _PERS_ACTION_RAISE] = 1.0
        elif pers_action == "Fold":
            vec[19 + _PERS_ACTION_FOLD] = 1.0

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
        data: 3D buffer of shape (hands, 4, 22) for collected vectors.
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

        # Tournament-level aggregate tracking.
        self._action_log: List[str] = []
        self._hand_wins: float = 0.0   # 1.0=win, 0.5=tie, 0.0=loss per hand
        self._total_hands: int = 0
        self._starting_chips: int = 0
        self._final_stack: int = 0

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

    def set_starting_chips(self, chips: int) -> None:
        """Set the starting chip count used for net-winnings normalisation."""
        self._starting_chips = chips

    def record_personality_action(self, action: str) -> None:
        """Append one personality action to the tournament action log.

        Called from the game loop immediately after each personality action.

        Args:
            action: One of 'Check', 'Call', 'Raise', 'Fold'.
        """
        self._action_log.append(action)

    def record_hand_result(self, won: Optional[bool]) -> None:
        """Record the outcome of one hand.

        Args:
            won: True if personality won, False if lost, None if tied.
        """
        if won is True:
            self._hand_wins += 1.0
        elif won is None:
            self._hand_wins += 0.5
        # won=False contributes 0.0
        self._total_hands += 1

    def set_final_stack(self, stack: int) -> None:
        """Record the personality agent's chip count at tournament end."""
        self._final_stack = stack

    def to_aggregates(self) -> npt.NDArray[np.float32]:
        """Compute 5 tournament-level aggregate features.

        Features (by index):
            [0] raise_rate        — Raise count / total personality actions
            [1] call_check_rate   — (Call + Check) count / total actions
            [2] fold_rate         — Fold count / total actions
            [3] net_winnings_norm — (final_stack - starting_chips) / starting_chips
            [4] hand_win_rate     — weighted wins / total hands (tie = 0.5)

        Returns:
            float32 ndarray of shape (5,).
        """
        total = len(self._action_log)
        if total > 0:
            raise_rate = self._action_log.count("Raise") / total
            fold_rate = self._action_log.count("Fold") / total
            call_check_rate = 1.0 - raise_rate - fold_rate
        else:
            raise_rate = fold_rate = call_check_rate = 0.0

        net_norm = (
            (self._final_stack - self._starting_chips) / self._starting_chips
            if self._starting_chips > 0
            else 0.0
        )
        hand_win_rate = (
            self._hand_wins / self._total_hands
            if self._total_hands > 0
            else 0.0
        )

        return np.array(
            [raise_rate, call_check_rate, fold_rate, net_norm, hand_win_rate],
            dtype=np.float32,
        )

    def to_matrix(self) -> npt.NDArray[np.float32]:
        """Flatten observation buffer to a 20×22 matrix.

        Returns:
            float32 ndarray of shape (20, 22).

        Raises:
            ValueError: If the resulting matrix lacks shape (20, 22).
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


def save_tournament_agg_csv(
    agg: npt.NDArray[np.float32],
    data_dir: str,
    seed: int,
    personality: str,
) -> str:
    """Save the 5-element aggregate feature vector to a companion CSV.

    Filename pattern: ``run_{seed}_{personality}_agg.csv``.

    Features saved (one row, 5 columns):
        raise_rate, call_check_rate, fold_rate, net_winnings_norm, hand_win_rate

    Args:
        agg: float32 ndarray of shape (5,).
        data_dir: Directory to write to.
        seed: Tournament seed used in filename.
        personality: Personality name used in filename.

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = f"run_{seed}_{personality}_agg.csv"
    filepath = os.path.join(data_dir, filename)
    np.savetxt(filepath, agg.reshape(1, -1), delimiter=",", fmt="%.6g")
    print(f"    [Observer] Saved {filepath}")
    return filepath
