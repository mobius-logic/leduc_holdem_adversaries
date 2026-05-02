"""Deck module for Leduc Hold'em.

Defines the Card dataclass, the ordered deck of 6 cards, and
the Deck class responsible for shuffling with a seeded RNG.
"""

import random
from dataclasses import dataclass
from typing import List


# Card rank ordering: J < Q < K
RANK_ORDER = {"J": 0, "Q": 1, "K": 2}

# All 6 cards in canonical index order used for one-hot encoding.
# Index 0=J♥, 1=Q♥, 2=K♥, 3=J♠, 4=Q♠, 5=K♠
CARD_ORDER = [
    ("J", "♥"),
    ("Q", "♥"),
    ("K", "♥"),
    ("J", "♠"),
    ("Q", "♠"),
    ("K", "♠"),
]


@dataclass(frozen=True)
class Card:
    """A single playing card with rank and suit.

    Attributes:
        rank: One of 'J', 'Q', 'K'.
        suit: One of '♥', '♠'.
    """

    rank: str
    suit: str

    def __str__(self) -> str:
        """Return human-readable card string, e.g. 'K♥'."""
        return f"{self.rank}{self.suit}"

    @property
    def rank_value(self) -> int:
        """Return integer rank value (J=0, Q=1, K=2)."""
        return RANK_ORDER[self.rank]

    @property
    def card_index(self) -> int:
        """Return position in CARD_ORDER for one-hot encoding."""
        return CARD_ORDER.index((self.rank, self.suit))


# Immutable tuple of all 6 cards in canonical order.
ALL_CARDS: tuple = tuple(Card(rank, suit) for rank, suit in CARD_ORDER)


def hand_strength(private: Card, community: Card) -> int:
    """Compute integer hand strength for showdown comparison.

    Pairs beat any high card. Among same hand type, higher rank wins.

    Args:
        private: The player's private card.
        community: The revealed community card.

    Returns:
        Integer strength where higher is better.
        Pairs yield values 3–5; high cards yield values 0–2.
    """
    if private.rank == community.rank:
        # Pair: offset by 3 so all pairs beat all high cards.
        return 3 + private.rank_value
    return private.rank_value


class Deck:
    """A shuffleable 6-card Leduc deck.

    Attributes:
        cards: Current ordered list of cards after shuffle.
        rng: The seeded random.Random instance used for shuffling.
    """

    def __init__(self) -> None:
        """Initialise the deck with all 6 cards in canonical order."""
        self.cards: List[Card] = list(ALL_CARDS)
        self.rng: random.Random = random.Random()

    def shuffle(self, seed: int) -> None:
        """Reset the deck to all 6 cards and shuffle with a deterministic seed.

        Restoring the full card list before shuffling ensures each hand
        starts with a complete deck regardless of how many cards were
        dealt in the previous hand.

        Args:
            seed: Integer seed for the RNG. Must be logged by caller
                for reproducibility.
        """
        self.cards = list(ALL_CARDS)
        self.rng.seed(seed)
        self.rng.shuffle(self.cards)

    def deal(self) -> Card:
        """Remove and return the top card of the deck.

        Returns:
            The next Card from the shuffled deck.

        Raises:
            IndexError: If the deck is empty.
        """
        return self.cards.pop(0)
