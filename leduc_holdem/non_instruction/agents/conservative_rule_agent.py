"""Rule-based conservative agent for Leduc Hold'em.

Implements the conservative personality strategy as deterministic Python
logic, derived directly from conservative_personality_instructions.txt.
No LLM or network call is made.

Personality summary
-------------------
- Applies a personal -15 % discount to win probability before deciding.
- Never open-bets without a confirmed pair; never raises pre-flop.
- Never bluffs; never chases with large pot.
- Folds much earlier than EV would suggest.
- Only aggressive holding: confirmed pair post-flop (prefer call-over-raise).

Win-probability thresholds (applied AFTER −15 % discount)
----------------------------------------------------------
The agent uses  eff_prob = win_prob − 0.15  for all decisions.

Conservative mode tiers (own stack)
-------------------------------------
STACK_STANDARD  = 35   # > 35 → standard conservative play
STACK_TIGHT     = 20   # 20–35 → tighter: King-high minimum to call
STACK_LOCKDOWN  = 20   # < 20 → near-lockdown: pair only to continue
"""

from __future__ import annotations

from typing import Callable, List

from agents.base_agent import BaseAgent
from game.state import GameState, Round

# Personal probability discount — conservative always assumes worse.
_PROB_DISCOUNT    = 0.15
_CALL_MIN_STD     = 0.55   # effective prob needed to call (standard mode)
_CALL_MIN_KING    = 0.55   # effective prob needed to call with King high post-flop
_STACK_TIGHT      = 35
_STACK_LOCKDOWN   = 20


class ConservativeRuleAgent(BaseAgent):
    """Deterministic conservative agent — passive, risk-averse, no bluffs.

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
        last_opp  = state.last_opponent_action   # None | Check | Call | Raise | Fold

        # Conservative never raises pre-flop under any circumstance.
        # ── KING ──────────────────────────────────────────────────────
        if rank == "K":
            # Never fold with King; never raise.
            if last_opp in (None, "Check"):
                return _pick(["Check"], legal)
            # Opponent bet or raised → call (reluctantly at cap).
            return _pick(["Call", "Check"], legal)

        # ── QUEEN ─────────────────────────────────────────────────────
        if rank == "Q":
            if last_opp in (None, "Check"):
                return _pick(["Check"], legal)
            if last_opp == "Call":
                return _pick(["Check", "Call"], legal)
            # Opponent raised (once) → fold.
            return _pick(["Fold", "Call"], legal)

        # ── JACK ──────────────────────────────────────────────────────
        # Check; fold to any bet or raise.
        if last_opp not in (None, "Check"):
            return _pick(["Fold", "Call"], legal)
        return _pick(["Check"], legal)

    # ------------------------------------------------------------------
    # Post-flop
    # ------------------------------------------------------------------

    def _act_postflop(self, state: GameState, legal: List[str]) -> str:
        private_rank   = state.personality_card.rank
        community_rank = state.community_card.rank      # guaranteed post-flop
        last_opp       = state.last_opponent_action
        my_stack       = state.stacks[0]    # personality = slot 0
        opp_stack      = state.stacks[1]

        win_prob = self._win_prob_fn(state)
        eff_prob = win_prob - _PROB_DISCOUNT   # personal discount

        # Opponent raised twice (at cap) → only a pair survives.
        at_cap     = "Raise" not in legal
        opp_outgun = opp_stack > my_stack

        # ── Stack-tier checks ─────────────────────────────────────────
        in_lockdown = my_stack < _STACK_LOCKDOWN
        in_tight    = _STACK_LOCKDOWN <= my_stack <= _STACK_TIGHT

        # ── PAIR ──────────────────────────────────────────────────────
        if private_rank == community_rank:
            # Prefer reactive style: check first, raise when opponent bets.
            if last_opp in (None,) and not at_cap:
                # Acting first or no opponent action yet → check.
                return _pick(["Check"], legal)
            if last_opp == "Raise":
                # Opponent raised — call (do not re-raise; one raise is enough).
                return _pick(["Call", "Check"], legal)
            if last_opp in ("Check", "Call"):
                # Opponent checked or called → raise now.
                return _pick(["Raise", "Call", "Check"], legal)
            # Cap reached → call.
            return _pick(["Call", "Check"], legal)

        # ── Lockdown mode: pair only ──────────────────────────────────
        if in_lockdown:
            # No pair → fold to any bet; check otherwise.
            if last_opp == "Raise" or state.to_call_amount() > 0:
                return _pick(["Fold", "Check"], legal)
            return _pick(["Check", "Call"], legal)

        # ── KING HIGH (no pair) ───────────────────────────────────────
        if private_rank == "K":
            # In tight mode: King high is minimum to call.
            # Call threshold: eff_prob > 55 %.
            # Opponent outguns us: fold one step earlier.
            fold_threshold = _CALL_MIN_KING + (0.05 if opp_outgun else 0.0)
            if in_tight and last_opp == "Raise":
                # Tight mode: fold to raise regardless.
                return _pick(["Fold", "Call"], legal)
            if last_opp == "Raise":
                # Standard mode: fold unless eff_prob > threshold.
                if eff_prob >= fold_threshold:
                    return _pick(["Call", "Check"], legal)
                return _pick(["Fold", "Call"], legal)
            # No aggression from opponent: check only.
            if last_opp in (None, "Check"):
                return _pick(["Check"], legal)
            # Opponent called: check/call.
            return _pick(["Check", "Call"], legal)

        # ── QUEEN HIGH (no pair) ─────────────────────────────────────
        if private_rank == "Q":
            if in_tight:
                # Tight mode: King high minimum → fold Queen.
                if last_opp == "Raise" or state.to_call_amount() > 0:
                    return _pick(["Fold", "Check"], legal)
                return _pick(["Check"], legal)
            # Standard: check; fold to any bet.
            if last_opp == "Raise" or state.to_call_amount() > 0:
                return _pick(["Fold", "Check"], legal)
            return _pick(["Check"], legal)

        # ── JACK HIGH (no pair) ──────────────────────────────────────
        # Check; fold to any bet.
        if last_opp == "Raise" or state.to_call_amount() > 0:
            return _pick(["Fold", "Check"], legal)
        return _pick(["Check"], legal)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick(candidates: List[str], legal: List[str]) -> str:
    """Return the first candidate that is legal."""
    for c in candidates:
        if c in legal:
            return c
    return legal[0]
