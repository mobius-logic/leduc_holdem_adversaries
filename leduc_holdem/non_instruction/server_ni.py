"""Non-instruction interactive Leduc Hold'em web server.

Serves the same browser frontend as server.py but uses a fully
deterministic rule-based agent (AggressiveRuleAgent) instead of an LLM.
No API key or network call is required.

Usage::

    cd leduc_holdem
    python non_instruction/server_ni.py           # port 5001
    python non_instruction/server_ni.py --port 8080

The agent responds instantly (no network latency).

API (identical contract to server.py)
--------------------------------------
POST /api/new_game  {}
    Start a new tournament against the aggressive rule agent.

POST /api/action    {"action": "Raise"}
    Apply a human action and advance agent turns.

POST /api/next_hand
    Advance past the hand-summary screen to the next hand.

GET  /api/state
    Return current game state.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

import yaml

# Ensure leduc_holdem package root is importable regardless of cwd.
_LEDUC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _LEDUC_ROOT)

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from game.deck import Deck          # noqa: E402
from game.state import GameState, Round  # noqa: E402
from game.leduc_holdem import LeducHoldemGame  # noqa: E402
from training.win_probability import compute_win_probability  # noqa: E402

# Rule-based agents live in the agents/ subfolder.
_NI_AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
sys.path.insert(0, _NI_AGENTS_DIR)
from aggressive_rule_agent   import AggressiveRuleAgent    # noqa: E402
from analytical_rule_agent   import AnalyticalRuleAgent    # noqa: E402
from conservative_rule_agent import ConservativeRuleAgent  # noqa: E402
from reckless_rule_agent     import RecklessRuleAgent      # noqa: E402

# Player-index aliases (PERSONALITY=0 = agent, OPPONENT=1 = human).
_AGENT = 0
_HUMAN = 1

# All supported rule-based personalities.
_VALID_PERSONALITIES = {"aggressive", "analytical", "conservative", "reckless"}

# Leduc Hold'em betting rules: maximum raises allowed per round.
# After this many raises the Raise action is removed from legal moves.
_PREFLOP_RAISE_CAP  = 2   # pre-flop:  +$2 per raise, max 2 raises
_POSTFLOP_RAISE_CAP = 2   # post-flop: +$4 per raise, max 2 raises


def _make_agent(personality: str):
    """Instantiate the rule-based agent for the given personality name."""
    if personality == "aggressive":
        return AggressiveRuleAgent(win_prob_fn=compute_win_probability)
    if personality == "analytical":
        return AnalyticalRuleAgent(win_prob_fn=compute_win_probability)
    if personality == "conservative":
        return ConservativeRuleAgent(win_prob_fn=compute_win_probability)
    if personality == "reckless":
        return RecklessRuleAgent(win_prob_fn=compute_win_probability)
    raise ValueError(f"Unknown personality: {personality!r}")

_WEB_DIR = os.path.join(_LEDUC_ROOT, "web_ni")
app = Flask(__name__, static_folder=_WEB_DIR)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    config_path = os.path.join(_LEDUC_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    config_dir = os.path.dirname(config_path)
    for key, val in cfg["paths"].items():
        if not os.path.isabs(str(val)):
            cfg["paths"][key] = os.path.normpath(os.path.join(config_dir, val))
    return cfg


_CFG: dict = _load_cfg()


# ---------------------------------------------------------------------------
# GameSession
# ---------------------------------------------------------------------------

_session: Optional["GameSession"] = None


class GameSession:
    """Manages one interactive tournament against a rule-based personality agent.

    The human occupies player-index 1 (_HUMAN).
    The rule agent occupies player-index 0 (_AGENT).
    """

    def __init__(self, personality: str = "aggressive") -> None:
        gcfg = _CFG["game"]
        self.personality_name = personality
        self._game  = LeducHoldemGame(_CFG)
        self._agent = _make_agent(personality)

        self.hands_per_tournament: int = gcfg["hands_per_tournament"]
        self.stacks: list = [gcfg["starting_chips"], gcfg["starting_chips"]]
        self.hands_played: int = 0
        self.tournament_seed: int = random.randint(10_000, 99_999)
        self.state: Optional[GameState] = None
        self._hidden_community = None
        self._playing_hand_idx: int = 0
        self.phase: str = "idle"
        self.log: list = []
        # Acting-order tracker: None until the first deal, then strictly
        # alternated each hand.  Hand 1 is random; hand N+1 flips hand N.
        self._agent_acts_first: Optional[bool] = None
        self._deal_hand()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hand_seed(self) -> int:
        return self.tournament_seed + self.hands_played * 97

    def _deal_hand(self) -> None:
        self._playing_hand_idx = self.hands_played
        seed = self._hand_seed()
        self._game.deck.shuffle(seed)
        agent_card  = self._game.deck.deal()
        human_card  = self._game.deck.deal()
        self._hidden_community = self._game.deck.deal()

        gcfg = _CFG["game"]
        # Hand 1: random.  Every subsequent hand: flip previous first-actor.
        if self._agent_acts_first is None:
            self._agent_acts_first = random.choice([True, False])
        else:
            self._agent_acts_first = not self._agent_acts_first
        agent_acts_first = self._agent_acts_first
        # GameState.start_hand uses (hand_index % 2 == 1) to derive
        # personality_acts_first, so pass 1 for agent-first, 0 for human-first.
        acting_hand_idx = 1 if agent_acts_first else 0

        self.state = GameState.start_hand(
            hand_index=acting_hand_idx,
            personality_card=agent_card,
            opponent_card=human_card,
            stacks=list(self.stacks),
            ante=gcfg["ante"],
            preflop_raise_size=gcfg["preflop_raise_size"],
            preflop_raise_cap=_PREFLOP_RAISE_CAP,   # 2 raises max pre-flop
        )

        pname = self.personality_name.title()
        first_label = f"{pname} acts first" if agent_acts_first else "You act first"
        self.log.append(
            f"── Hand {self._playing_hand_idx + 1} / {self.hands_per_tournament} ──  "
            f"{first_label}  │  "
            f"Stacks → You ${self.stacks[_HUMAN]} · {pname} ${self.stacks[_AGENT]}"
        )
        self.log.append(f"Your card: {human_card}  │  Pot: ${self.state.pot}")
        self._advance_agent()

    def _advance_agent(self) -> None:
        """Run agent turns until the human needs to act or the hand ends."""
        while (
            self.state is not None
            and not self.state.hand_over
            and self.state.current_player == _AGENT
        ):
            legal  = self.state.get_legal_actions()
            action = self._agent.act(self.state, legal)
            self.log.append(f"{self.personality_name.title()}: {action}")
            round_ended = self._game._apply_action(self.state, action)
            if self.state.hand_over:
                self._end_hand()
                return
            if round_ended:
                self._transition_round()
                if self.state is None or self.state.hand_over:
                    return
        if self.state and not self.state.hand_over:
            self.phase = "your_turn"

    def _transition_round(self) -> None:
        if self.state.round == Round.PRE_FLOP:
            gcfg = _CFG["game"]
            self.state.preflop_done   = True
            self.state.community_card = self._hidden_community
            self.state.round          = Round.POST_FLOP
            self.state.round_bets     = [0, 0]
            self.state.raises_this_round = 0
            self.state.raise_cap      = _POSTFLOP_RAISE_CAP   # 2 raises max post-flop
            self.state.raise_size     = gcfg["postflop_raise_size"]
            self.state.actions_this_round = []
            self.state.current_player = (
                _AGENT if self.state.personality_acts_first else _HUMAN
            )
            self.log.append(
                f"── Post-Flop  Community: {self._hidden_community}"
            )
        else:
            self._game._resolve_showdown(self.state)
            self.state.hand_over = True
            self._end_hand()

    def _end_hand(self) -> None:
        self.state.hand_over = True
        self._game._finalise_stacks(self.state)
        self.stacks = list(self.state.stacks)

        cc_str = (
            f"  Community: {self.state.community_card}"
            if self.state.community_card
            else ""
        )
        pname = self.personality_name.title()
        if self.state.winner is None:
            self.log.append("Tie! Pot split.")
        elif self.state.winner == _AGENT:
            self.log.append(
                f"{pname} wins! "
                f"[{self.state.personality_card} vs {self.state.opponent_card}"
                f"{cc_str}]"
            )
        else:
            self.log.append(
                f"You win! "
                f"[{self.state.opponent_card} vs {self.state.personality_card}"
                f"{cc_str}]"
            )
        self.log.append(
            f"Stacks → You ${self.stacks[_HUMAN]} · {pname} ${self.stacks[_AGENT]}"
        )
        self.hands_played += 1
        if self.hands_played >= self.hands_per_tournament:
            self.phase = "tournament_over"
            self.log.append(f"── Tournament Over ── Your opponent was {pname}!")
            if self.stacks[_HUMAN] > self.stacks[_AGENT]:
                self.log.append("You win the tournament!")
            elif self.stacks[_AGENT] > self.stacks[_HUMAN]:
                self.log.append(f"{pname} wins the tournament!")
            else:
                self.log.append("Tournament tied!")
        else:
            self.phase = "hand_summary"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply_action(self, action: str) -> None:
        """Validate and apply a human action, then advance agent turns."""
        if self.phase != "your_turn":
            raise ValueError("It is not your turn.")
        if action not in self.state.get_legal_actions():
            raise ValueError(
                f"Illegal action {action!r}. "
                f"Legal: {self.state.get_legal_actions()}"
            )
        self.log.append(f"You: {action}")
        # Update last_opponent_action so the rule agent can read it.
        self.state.last_opponent_action = action
        round_ended = self._game._apply_action(self.state, action)
        if self.state.hand_over:
            self._end_hand()
            return
        if round_ended:
            self._transition_round()
            if self.state is None or self.state.hand_over:
                return
        self._advance_agent()

    def start_next_hand(self) -> None:
        if self.phase != "hand_summary":
            raise ValueError("Not in hand-summary phase.")
        self._deal_hand()

    def to_dict(self) -> dict:
        s = self.state
        reveal_cards = self.phase in ("hand_summary", "tournament_over")

        legal_actions: list = []
        if self.phase == "your_turn" and s:
            legal_actions = s.get_legal_actions()

        hand_winner = None
        if reveal_cards and s:
            if s.winner is None:
                hand_winner = "tie"
            elif s.winner == _AGENT:
                hand_winner = "llm"
            else:
                hand_winner = "you"

        tournament_winner = None
        if self.phase == "tournament_over":
            if self.stacks[_HUMAN] > self.stacks[_AGENT]:
                tournament_winner = "you"
            elif self.stacks[_AGENT] > self.stacks[_HUMAN]:
                tournament_winner = "llm"
            else:
                tournament_winner = "tie"

        your_stack  = s.stacks[_HUMAN]  if s else self.stacks[_HUMAN]
        agent_stack = s.stacks[_AGENT] if s else self.stacks[_AGENT]

        return {
            "personality": self.personality_name,   # always revealed (rule agent)
            "hand_number": self._playing_hand_idx + 1,
            "total_hands": self.hands_per_tournament,
            "round": s.round.name.lower() if s else None,
            "your_card": str(s.opponent_card) if s else None,
            "community_card": (
                str(s.community_card) if (s and s.community_card) else None
            ),
            "llm_card": (
                str(s.personality_card) if (s and reveal_cards) else None
            ),
            "your_stack": your_stack,
            "llm_stack": agent_stack,
            "pot": s.pot if s else 0,
            "legal_actions": legal_actions,
            "phase": self.phase,
            "hand_winner": hand_winner,
            "tournament_winner": tournament_winner,
            "log": list(self.log[-60:]),
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(_WEB_DIR, "index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global _session
    body = request.get_json(force=True, silent=True) or {}
    personality = body.get("personality", "").lower().strip()
    if not personality:
        personality = random.choice(sorted(_VALID_PERSONALITIES))
    if personality not in _VALID_PERSONALITIES:
        return jsonify({"error": f"Unknown personality: {personality!r}"}), 400
    _session = GameSession(personality=personality)
    return jsonify(_session.to_dict())


@app.route("/api/action", methods=["POST"])
def action():
    if _session is None:
        return jsonify({"error": "No active game. Start a new game first."}), 400
    body = request.get_json(force=True, silent=True) or {}
    act  = body.get("action", "").strip()
    try:
        _session.apply_action(act)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_session.to_dict())


@app.route("/api/next_hand", methods=["POST"])
def next_hand():
    if _session is None:
        return jsonify({"error": "No active game."}), 400
    try:
        _session.start_next_hand()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_session.to_dict())


@app.route("/api/state")
def state():
    if _session is None:
        return jsonify({"error": "No active game."}), 404
    return jsonify(_session.to_dict())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Non-instruction Leduc Hold'em web server (rule-based agent)"
    )
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to listen on (default: 5001)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"[server_ni] Serving at http://{args.host}:{args.port}")
    print(f"[server_ni] Agents: deterministic rule engines for all 4 personalities (no LLM)")
    print(f"[server_ni] Web dir: {_WEB_DIR}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
