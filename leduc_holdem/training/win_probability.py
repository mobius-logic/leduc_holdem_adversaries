"""Exact win probability computation via full card enumeration.

Computes the exact probability that the personality agent wins the
current hand by enumerating all possible unseen card combinations.
No sampling — this is a closed-form enumeration over the tiny Leduc deck.
"""

from __future__ import annotations

from typing import Optional

from game.deck import ALL_CARDS, Card, hand_strength
from game.state import GameState, Round


def compute_win_probability(state: GameState) -> float:
    """Return the exact win probability for the personality agent.

    Pre-flop: enumerate all (opponent_card, community_card) pairs from
    unseen cards.
    Post-flop: community card is fixed; enumerate all opponent cards
    from unseen cards.

    Unseen cards = ALL_CARDS minus own private card minus community card
    (if visible).

    Args:
        state: Current game state. community_card is None pre-flop.

    Returns:
        Float in [0.0, 1.0]. Ties contribute 0.5 to the numerator.
    """
    own_card: Card = state.personality_card
    community_card: Optional[Card] = state.community_card

    # Determine which cards are already known and should be excluded.
    known = {own_card}
    if community_card is not None:
        known.add(community_card)

    unseen = [c for c in ALL_CARDS if c not in known]

    wins = 0.0
    total = 0

    if state.round == Round.PRE_FLOP:
        # Enumerate (opp_card, comm_card) — both from unseen, and different.
        for i, opp_card in enumerate(unseen):
            for j, comm_card in enumerate(unseen):
                if i == j:
                    continue
                p_str = hand_strength(own_card, comm_card)
                o_str = hand_strength(opp_card, comm_card)
                if p_str > o_str:
                    wins += 1.0
                elif p_str == o_str:
                    wins += 0.5
                total += 1
    else:
        # Post-flop: community card is fixed.
        assert community_card is not None
        for opp_card in unseen:
            p_str = hand_strength(own_card, community_card)
            o_str = hand_strength(opp_card, community_card)
            if p_str > o_str:
                wins += 1.0
            elif p_str == o_str:
                wins += 0.5
            total += 1

    if total == 0:
        return 0.5  # Degenerate guard; should not occur.

    return wins / total
