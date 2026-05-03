"""GameState dataclass for Leduc Hold'em.

Encapsulates all mutable state for a single hand in progress,
including cards, chip stacks, betting state, and round metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from game.deck import Card


class Round(Enum):
    """Betting round within a hand."""
    PRE_FLOP = auto()
    POST_FLOP = auto()


# Player index constants for clarity.
PERSONALITY = 0   # LLM-driven personality agent
OPPONENT = 1      # Random agent


@dataclass
class GameState:
    """Complete mutable state for one hand of Leduc Hold'em.

    Player 0 is always the personality agent.
    Player 1 is always the random (opponent) agent.

    Attributes:
        hand_index: Zero-based index of the current hand (0–4).
        personality_acts_first: True when the personality agent acts
            first in this hand.
        personality_card: The personality agent's private card.
        opponent_card: The random agent's private card.
        community_card: The community card, or None pre-flop.
        stacks: Chip counts indexed by player (0=personality, 1=opp).
        pot: Chips currently in the pot.
        round: Current betting round (PRE_FLOP or POST_FLOP).
        round_bets: How many chips each player has committed in the
            current round (above the ante).
        raises_this_round: Total raises taken in the current round.
        raise_cap: Maximum raises allowed in the current round.
        raise_size: Chip cost of a raise in the current round.
        actions_this_round: Ordered list of actions taken so far in
            the current round (used to detect Check-Check termination).
        current_player: Index of the next player to act (0 or 1).
        last_opponent_action: Most recent action taken by the opponent
            agent, or None if no opponent action has occurred this hand.
        last_personality_action: Most recent action taken by the personality
            agent, or None if no personality action has occurred this hand.
        hand_over: True when the hand has ended.
        winner: Player index who won the pot, or None for a tie.
        preflop_done: True once the pre-flop round is resolved.
    """

    hand_index: int
    personality_acts_first: bool
    personality_card: Card
    opponent_card: Card
    community_card: Optional[Card]
    stacks: List[int]
    pot: int
    round: Round
    round_bets: List[int]
    raises_this_round: int
    raise_cap: int
    raise_size: int
    actions_this_round: List[str]
    current_player: int
    last_opponent_action: Optional[str]
    last_personality_action: Optional[str]
    hand_over: bool
    winner: Optional[int]        # None means tie
    preflop_done: bool

    @classmethod
    def start_hand(
        cls,
        hand_index: int,
        personality_card: Card,
        opponent_card: Card,
        stacks: List[int],
        ante: int,
        preflop_raise_size: int,
        preflop_raise_cap: int,
    ) -> "GameState":
        """Create a fresh GameState for the start of a new hand.

        Antes are deducted from both stacks and added to the pot here.

        Args:
            hand_index: Zero-based hand number within the tournament.
            personality_card: Dealt private card for personality agent.
            opponent_card: Dealt private card for random agent.
            stacks: Current chip counts [personality, opponent].
            ante: Chips each player must ante before the hand.
            preflop_raise_size: Chip cost of a pre-flop raise.
            preflop_raise_cap: Maximum raises in pre-flop.

        Returns:
            A fully initialised GameState ready for pre-flop betting.
        """
        personality_acts_first = (hand_index % 2 == 1)
        first_player = PERSONALITY if personality_acts_first else OPPONENT

        new_stacks = [stacks[0] - ante, stacks[1] - ante]
        pot = ante * 2

        return cls(
            hand_index=hand_index,
            personality_acts_first=personality_acts_first,
            personality_card=personality_card,
            opponent_card=opponent_card,
            community_card=None,
            stacks=new_stacks,
            pot=pot,
            round=Round.PRE_FLOP,
            round_bets=[0, 0],
            raises_this_round=0,
            raise_cap=preflop_raise_cap,
            raise_size=preflop_raise_size,
            actions_this_round=[],
            current_player=first_player,
            last_opponent_action=None,
            last_personality_action=None,
            hand_over=False,
            winner=None,
            preflop_done=False,
        )

    def get_legal_actions(self) -> List[str]:
        """Return the set of legal actions for the current player.

        Returns:
            List of action strings from {'Check', 'Call', 'Raise', 'Fold'}.
            Order is deterministic for reproducibility.
        """
        to_call = max(self.round_bets) - self.round_bets[self.current_player]
        actions: List[str] = ["Fold"]
        if to_call == 0:
            actions.append("Check")
        else:
            actions.append("Call")
        if self.raises_this_round < self.raise_cap:
            actions.append("Raise")
        return actions

    def to_call_amount(self) -> int:
        """Return chips needed for the current player to call."""
        return max(self.round_bets) - self.round_bets[self.current_player]

    @property
    def personality_stack(self) -> int:
        """Chip count for the personality agent."""
        return self.stacks[PERSONALITY]

    @property
    def opponent_stack(self) -> int:
        """Chip count for the random agent."""
        return self.stacks[OPPONENT]
