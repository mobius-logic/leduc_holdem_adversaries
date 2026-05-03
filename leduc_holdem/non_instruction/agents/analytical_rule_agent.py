"""Rule-based analytical agent for Leduc Hold'em.

Implements the analytical personality strategy as deterministic Python logic,
derived directly from analytical_personality_instructions.txt.  No LLM or
network call is made.

Personality summary
-------------------
- Strictly probability-driven: win_prob > 60 % → bet/raise; 40–60 % → call
  only; < 40 % → check/fold.
- Never bluffs, never slow-plays.
- Pre-flop: King raises; Queen checks/calls; Jack checks/folds to any bet.
- Post-flop: Pair → raise to cap immediately; King high → bet when prob > 60%;
  Queen high → check/call on pot odds only; Jack high → check/fold.
- Low-stack (< 10 chips) conservative mode: only raise with a pair;
  fold below King high.

Win-probability thresholds (no bonus applied; analytical is EV-pure)
----------------------------------------------------------------------
BET_RAISE_MIN   = 0.60   # must exceed this to bet/raise (not at cap)
CALL_MIN        = 0.40   # must meet this to call; below→check or fold
LOW_STACK       = 10     # chips; triggers conservative mode
"""

from __future__ import annotations

from typing import Callable, List

from agents.base_agent import BaseAgent
from game.state import GameState, Round

_BET_RAISE_MIN = 0.60
_CALL_MIN      = 0.40
_LOW_STACK     = 10


class AnalyticalRuleAgent(BaseAgent):
    """Deterministic analytical agent — expected-value driven, no bluffs.

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

        # ── KING —never fold pre-flop ─────────────────────────────────
        if rank == "K":
            if can_raise:
                return "Raise"
            # Cap reached — call (never fold with King).
            return _pick(["Call", "Check"], legal)

        # ── QUEEN — check / call only, never raise ────────────────────
        if rank == "Q":
            # Never fold or raise pre-flop with a Queen.
            return _pick(["Call", "Check"], legal)

        # ── JACK — check only; fold to any bet ───────────────────────
        # Jack: never call or raise pre-flop.
        if last_opp in ("Raise",):
            return _pick(["Fold", "Call"], legal)
        return _pick(["Check", "Call"], legal)

    # ------------------------------------------------------------------
    # Post-flop
    # ------------------------------------------------------------------

    def _act_postflop(self, state: GameState, legal: List[str]) -> str:
        private_rank   = state.personality_card.rank
        community_rank = state.community_card.rank      # guaranteed post-flop
        can_raise      = "Raise" in legal
        last_opp       = state.last_opponent_action
        my_stack       = state.stacks[0]   # personality = slot 0

        win_prob = self._win_prob_fn(state)   # no bonus — analytical is pure EV

        # Low-stack conservative mode.
        low_stack = my_stack < _LOW_STACK

        # ── PAIR ──────────────────────────────────────────────────────
        if private_rank == community_rank:
            if can_raise:
                return "Raise"
            return _pick(["Call", "Check"], legal)

        # Low-stack: only raise with pair (handled above); fold below K high.
        if low_stack and private_rank != "K":
            return _pick(["Fold", "Check"], legal)

        # ── KING HIGH (no pair) ───────────────────────────────────────
        if private_rank == "K":
            if win_prob > _BET_RAISE_MIN:
                # Bet once — analytical never raises without a pair post-flop.
                # If cap not reached and no raise yet this round, bet/raise.
                if can_raise and last_opp not in ("Raise",):
                    return "Raise"   # opening bet
                # Facing a raise: re-raise only if still above threshold.
                if can_raise and last_opp == "Raise":
                    return "Raise"
                # Cap reached: call.
                return _pick(["Call", "Check"], legal)
            elif win_prob >= _CALL_MIN:
                # 40–60 %: call only, do not raise.
                return _pick(["Call", "Check"], legal)
            else:
                # Below 40 %: check or fold; fold to raises.
                if last_opp == "Raise":
                    return _pick(["Fold", "Call"], legal)
                return _pick(["Check", "Call"], legal)

        # ── QUEEN HIGH (no pair) ─────────────────────────────────────
        if private_rank == "Q":
            # Check; call only if pot odds justify (to_call ≤ 2 and prob ≥ 40%).
            to_call = state.to_call_amount()
            if last_opp == "Raise":
                return _pick(["Fold", "Call"], legal)
            if to_call > 0:
                # Pot-odds call: (to_call / (pot + to_call)) < win_prob
                pot_odds = to_call / (state.pot + to_call)
                if win_prob >= _CALL_MIN and pot_odds < win_prob and to_call <= 2:
                    return _pick(["Call", "Check"], legal)
                return _pick(["Fold", "Check"], legal)
            return _pick(["Check", "Call"], legal)

        # ── JACK HIGH (no pair) ──────────────────────────────────────
        # Check; fold to any bet.
        if last_opp in ("Raise",) or state.to_call_amount() > 0:
            return _pick(["Fold", "Check"], legal)
        return _pick(["Check", "Call"], legal)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick(candidates: List[str], legal: List[str]) -> str:
    """Return the first candidate that is legal."""
    for c in candidates:
        if c in legal:
            return c
    return legal[0]
