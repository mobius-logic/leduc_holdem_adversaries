"""Rule-based aggressive agent for Leduc Hold'em.

Implements the aggressive personality strategy as explicit Python logic,
derived directly from aggressive_personality_instructions.txt.  No LLM
call is made — every decision is computed deterministically from the
current GameState.

Card ranks:  J (weakest, 0) < Q (1) < K (strongest, 2)
Win probabilities include a +10% confidence bonus per the instruction set.

Decision rules per round and hand strength:
  Pre-flop
    KING  : Always Raise; Call at cap (never fold)
    QUEEN : Always Raise; Call at cap (never fold)
    JACK  : Raise as bluff when acting first or opp checked;
            Call if opp raised once (aggressive read);
            Fold when cap reached

  Post-flop
    PAIR          : Raise to cap unconditionally; Call if cap reached
    KING HIGH     : Bet/raise aggressively; call at cap if win_prob >= 40%
    QUEEN HIGH    : Bluff-bet; call one raise; fold to re-raise
    JACK HIGH     : Bluff-bet if opp showed weakness; fold to any bet
"""

from __future__ import annotations

from typing import Callable, List

from agents.base_agent import BaseAgent
from game.state import GameState, Round


# Aggressive personality applies a +10% confidence bonus for borderline calls.
_WIN_PROB_BONUS = 0.10

# Win probability thresholds used for borderline post-flop calls.
_KING_HIGH_CALL_THRESHOLD   = 0.40  # call at cap with King high if above this
_KING_HIGH_FOLD_THRESHOLD   = 0.35  # secondary threshold when already at cap


class AggressiveRuleAgent(BaseAgent):
    """Deterministic aggressive player.  Requires no API key or network call.

    Args:
        win_prob_fn: Callable ``(state) -> float`` that returns the
            personality agent's exact win probability (0.0–1.0).
    """

    def __init__(self, win_prob_fn: Callable[[GameState], float]) -> None:
        self._win_prob_fn = win_prob_fn

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: GameState, legal_actions: List[str]) -> str:
        """Return an action following the aggressive strategy rules."""
        if state.round == Round.PRE_FLOP:
            return self._act_preflop(state, legal_actions)
        return self._act_postflop(state, legal_actions)

    # ------------------------------------------------------------------
    # Pre-flop
    # ------------------------------------------------------------------

    def _act_preflop(self, state: GameState, legal: List[str]) -> str:
        rank     = state.personality_card.rank   # 'J', 'Q', or 'K'
        can_raise = "Raise" in legal
        last_opp  = state.last_opponent_action   # None | 'Check' | 'Raise' | 'Call'

        # ── KING ──────────────────────────────────────────────────────
        # Never check or call pre-flop with a King. Never fold pre-flop.
        if rank == "K":
            if can_raise:
                return "Raise"
            # Cap reached — still call, never fold.
            return _pick(["Call", "Check"], legal)

        # ── QUEEN ─────────────────────────────────────────────────────
        # Always raise; call only when cap is reached.  Never fold.
        if rank == "Q":
            if can_raise:
                return "Raise"
            return _pick(["Call", "Check"], legal)

        # ── JACK ──────────────────────────────────────────────────────
        # Cap reached → fold (only time to give up).
        if not can_raise:
            return _pick(["Fold", "Call"], legal)

        # Acting first or opponent checked → bluff raise.
        if last_opp is None or last_opp == "Check":
            return "Raise"

        # Opponent raised once → read as aggressive, call.
        if last_opp == "Raise":
            return _pick(["Call", "Fold"], legal)

        # Fallback (opponent called, etc.) → raise as bluff.
        return "Raise" if can_raise else _pick(["Call", "Check"], legal)

    # ------------------------------------------------------------------
    # Post-flop
    # ------------------------------------------------------------------

    def _act_postflop(self, state: GameState, legal: List[str]) -> str:
        private_rank  = state.personality_card.rank
        community_rank = state.community_card.rank    # guaranteed not None post-flop
        can_raise      = "Raise" in legal
        last_opp       = state.last_opponent_action

        # Whether we are acting first this post-flop round (no actions yet).
        acting_first = not state.actions_this_round

        # Win probability with the aggressive confidence bonus.
        win_prob = self._win_prob_fn(state) + _WIN_PROB_BONUS

        # ── PAIR ──────────────────────────────────────────────────────
        # Bet or raise immediately to the cap.  Never check, fold, or
        # slow-play with a pair.
        if private_rank == community_rank:
            if can_raise:
                return "Raise"
            # Cap reached — call; a pair always fights to the end.
            return _pick(["Call", "Check"], legal)

        # ── KING HIGH ─────────────────────────────────────────────────
        if private_rank == "K":
            if acting_first or last_opp == "Check":
                # First action or opponent checked → bet (raise).
                return "Raise" if can_raise else _pick(["Check", "Call"], legal)
            if last_opp == "Raise":
                if can_raise:
                    # Opponent raised once → semi-bluff re-raise.
                    return "Raise"
                # Cap reached: call only if win probability exceeds threshold.
                if win_prob >= _KING_HIGH_CALL_THRESHOLD:
                    return _pick(["Call", "Check"], legal)
                return _pick(["Fold", "Call"], legal)
            # Opponent called/other → keep pressure on.
            return "Raise" if can_raise else _pick(["Check", "Call"], legal)

        # ── QUEEN HIGH ────────────────────────────────────────────────
        if private_rank == "Q":
            if acting_first or last_opp == "Check":
                # Bluff bet.
                return "Raise" if can_raise else _pick(["Check", "Call"], legal)
            if last_opp == "Raise":
                if can_raise:
                    # Opponent raised us — call, do not re-raise.
                    return _pick(["Call", "Fold"], legal)
                # Cap reached (they re-raised our bluff) → fold.
                return _pick(["Fold", "Call"], legal)
            return _pick(["Call", "Check"], legal)

        # ── JACK HIGH ─────────────────────────────────────────────────
        # (private_rank == "J", no pair)
        if acting_first:
            # Bet as a bluff only if the opponent showed prior weakness
            # (last_opp from pre-flop is None, "Check", or "Call").
            if last_opp in (None, "Check", "Call"):
                return "Raise" if can_raise else _pick(["Check", "Call"], legal)
            # Opponent showed strength pre-flop ("Raise") → check it down.
            return _pick(["Check", "Call"], legal)

        if last_opp == "Check":
            # Opponent checked this round → single bluff bet.
            return "Raise" if can_raise else _pick(["Check", "Call"], legal)

        # Opponent bet or raised → fold; Jack is not worth chasing.
        return _pick(["Fold", "Call"], legal)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick(candidates: List[str], legal: List[str]) -> str:
    """Return the first candidate that appears in legal actions."""
    for c in candidates:
        if c in legal:
            return c
    return legal[0]
