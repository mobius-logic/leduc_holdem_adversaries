"""Abstract base class for all Leduc Hold'em agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from game.state import GameState


class BaseAgent(ABC):
    """Abstract base class that all agents must subclass.

    Every agent must implement the ``act`` method, which receives the
    current game state and the set of legal actions, and returns a
    chosen action string.
    """

    @abstractmethod
    def act(self, state: GameState, legal_actions: List[str]) -> str:
        """Choose and return an action given the current game state.

        The returned action MUST be one of the strings in
        ``legal_actions``.

        Args:
            state: The fully-updated GameState for the current moment.
            legal_actions: List of currently legal action strings.

        Returns:
            One of {'Check', 'Call', 'Raise', 'Fold'}.
        """
        raise NotImplementedError
