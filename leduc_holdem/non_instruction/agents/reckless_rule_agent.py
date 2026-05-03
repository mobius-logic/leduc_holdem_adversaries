"""Rule-based reckless agent for Leduc Hold'em.

Implements the reckless personality strategy as deterministic Python logic,
derived directly from reckless_personality_instructions.txt.  No LLM or
network call is made.

Personality summary
-------------------
- Never checks — always raises or folds (calling only when cap reached).
- Bluffs indiscriminately: raises on impulse regardless of hand strength.
- Reacts to opponent raises aggressively if win_prob > 35 %; folds below.
- Gets MORE erratic as stack shrinks (desperation mode < 15 chips).
- Pre-flop: King/Queen always raise; Jack raises as bluff, folds at cap.
- Post-flop: Pair → raise to cap; King high → bluff-raise; Queen high →
  raise once then fold to re-raise; Jack high → raise bluff then fold.

Win-probability thresholds
---------------------------
REACT_RAISE_MIN  = 0.35   raise to opponent raise if above this
DESPERATION      = 15     chips; below = desperation mode (raise harder)
"""

from __future__ import annotations

from typing import Callable, List

from agents.base_agent import BaseAgent
from game.state import GameState, Round

_REACT_RAISE_MIN  = 0.35
_DESPERATION_STACK = 15


class RecklessRuleAgent(BaseAgent):
    """Deterministic reckless agent — impulsive, never checks, high bluff rate.

    Args:
        win_prob_fn: Callable ``(state) -> float`` returning exact win
            probability for the personality (slot 0) agent.
    """

    def __init__(self, win_prob_fn: Callable[[GameState], float]) -> None:
        self._win_prob_fn = win_prob_fn

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: GameState, legal_actions: List[str]) -> str:
        if state.round == Round.PRE_FLOP:
            return self._act_preflop(state, legal_actions)
        return self._act_postflop(state, legal_actions)

    # ------------------------------------------------------------------
    # Pre-flop
    # ------------------------------------------------------------------

    def _act_preflop(self, state: GameState, legal: List[str]) -> str:
        rank      = state.personality_card.rank
        can_raise = "Raise" in legal
        last_opp  = state.last_opponent_action   # None | Check | Call | Raise | Fold
        win_prob  = self._win_prob_fn(state)

        # ── KING — raise immediately; never check/call/fold ──────────
        if rank == "K":
            if can_raise:
                return "Raise"
            # Cap reached — call (never fold a King).
            return _pick(["Call"], legal)

        # ── QUEEN — raise immediately; call only at cap ───────────────
        if rank == "Q":
            if can_raise:
                return "Raise"
            # Cap reached (opponent raised twice) — call; never fold a Queen.
            return _pick(["Call"], legal)

        # ── JACK — raise as bluff; fold at cap or low prob ───────────
        if not can_raise:
            # Cap reached — fold.
            return _pick(["Fold", "Call"], legal)
        if last_opp in (None, "Check"):
            # Acting first or opponent checked → bluff raise.
            return "Raise"
        if last_opp == "Raise":
            # Opponent raised once → re-raise as bluff if win_prob > 35 %.
            if win_prob >= _REACT_RAISE_MIN:
                return "Raise"
            return _pick(["Fold", "Call"], legal)
        # Any other state (called): raise.
        return "Raise"

    # ------------------------------------------------------------------
    # Post-flop
    # ------------------------------------------------------------------

    def _act_postflop(self, state: GameState, legal: List[str]) -> str:
        private_rank   = state.personality_card.rank
        community_rank = state.community_card.rank      # guaranteed post-flop
        can_raise      = "Raise" in legal
        last_opp       = state.last_opponent_action
        my_stack       = state.stacks[0]    # personality = slot 0
        win_prob       = self._win_prob_fn(state)

        desperation = my_stack < _DESPERATION_STACK

        # ── PAIR ──────────────────────────────────────────────────────
        # Raise immediately to the cap — no slowplay, no exceptions.
        if private_rank == community_rank:
            if can_raise:
                return "Raise"
            return _pick(["Call"], legal)

        # ── Desperation mode ─────────────────────────────────────────
        # Below 15 chips: raise with almost anything; fold only Jack high
        # after two opponent raises (cap reached).
        if desperation:
            if not can_raise and private_rank == "J":
                return _pick(["Fold", "Call"], legal)
            if can_raise:
                return "Raise"
            return _pick(["Call"], legal)

        # ── KING HIGH (no pair) ───────────────────────────────────────
        # Bluff-raise; committed to the hand; call at cap.
        if private_rank == "K":
            if can_raise:
                return "Raise"
            # Cap reached — call (King high feels strong enough).
            return _pick(["Call"], legal)

        # ── QUEEN HIGH (no pair) ─────────────────────────────────────
        # Bluff-raise; fold to confirmed re-raise (opponent raised back).
        if private_rank == "Q":
            if can_raise:
                # If opponent has already raised → we are being re-raised.
                # "Opponent raised: fold" per instructions.
                if last_opp == "Raise" and not _acting_first(state):
                    return _pick(["Fold", "Call"], legal)
                return "Raise"
            # Cap reached — fold; Queen has its limits.
            return _pick(["Fold", "Call"], legal)

        # ── JACK HIGH (no pair) ──────────────────────────────────────
        # Raise once as a bluff; fold to opponent bet.
        if private_rank == "J":
            if can_raise:
                if last_opp in (None, "Check"):
                    # Acting first or opp checked → bluff raise.
                    return "Raise"
                # Opponent bet or raised → fold.
                return _pick(["Fold", "Call"], legal)
            # Cap reached → fold.
            return _pick(["Fold", "Call"], legal)

        # Fallback.
        return _pick(["Raise", "Call", "Fold"], legal)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _acting_first(state: GameState) -> bool:
    """True if no actions have been taken in the current round yet."""
    return not state.actions_this_round


def _pick(candidates: List[str], legal: List[str]) -> str:
    """Return the first candidate that is legal."""
    for c in candidates:
        if c in legal:
            return c
    return legal[0]
