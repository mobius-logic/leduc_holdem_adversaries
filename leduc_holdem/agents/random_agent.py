"""Random agent for Leduc Hold'em.

Selects uniformly at random from the set of currently legal actions.
Does not use an LLM or any instruction set.
"""

from __future__ import annotations

import random
from typing import List

from agents.base_agent import BaseAgent
from game.state import GameState


class RandomAgent(BaseAgent):
    """Uniform random agent.

    On each turn, this agent picks one of the legal actions with equal
    probability. It does not inspect game state beyond obtaining the
    legal action list.

    Attributes:
        rng: A seeded random.Random instance for reproducibility.
    """

    def __init__(self, rng: random.Random) -> None:
        """Initialise the random agent with a seeded RNG.

        Args:
            rng: A ``random.Random`` instance. The caller is responsible
                for seeding it before tournament start.
        """
        self.rng = rng

    def act(self, state: GameState, legal_actions: List[str]) -> str:
        """Return a uniformly random legal action.

        Args:
            state: Current game state (used only for type consistency).
            legal_actions: List of currently legal action strings.

        Returns:
            A randomly selected action from ``legal_actions``.
        """
        return self.rng.choice(legal_actions)
