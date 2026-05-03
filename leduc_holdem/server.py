"""Interactive Leduc Hold'em web server.

Serves a browser-based frontend at GET / and a JSON API so a human
can play one-on-one against an LLM personality agent in real time.

Usage::

    cd leduc_holdem
    python server.py             # listens on http://localhost:5000
    python server.py --port 8080

The LLM response may take 5–30 seconds per action depending on the
Azure endpoint. The server is intentionally single-threaded (one game
at a time) so actions are processed strictly in order.

API summary
-----------
POST /api/new_game  {"personality": "aggressive"}
    Start a new tournament.  Runs the LLM automatically if it goes
    first pre-flop, then returns the state waiting for the human.

POST /api/action    {"action": "Raise"}
    Apply a human action.  The server runs all LLM turns that follow
    until the next human decision point (or the hand ends), then
    returns the new state.

POST /api/next_hand
    Advance to the next hand after reviewing the hand-summary screen.

GET  /api/state
    Return the current game state (useful on page reload).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

import yaml

# Ensure package imports work when invoked as `python server.py` from
# the leduc_holdem/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from game.deck import Deck  # noqa: E402  (imported for type resolution)
from game.state import GameState, Round  # noqa: E402
from game.leduc_holdem import LeducHoldemGame  # noqa: E402
from agents.personality_agent import PersonalityAgent  # noqa: E402
from training.win_probability import compute_win_probability  # noqa: E402

# Player-index aliases matching state.py (PERSONALITY=0, OPPONENT=1).
_LLM = 0    # LLM personality agent occupies slot 0
_HUMAN = 1  # Human player occupies slot 1

_VALID_PERSONALITIES = {"analytical", "conservative", "aggressive", "reckless"}

app = Flask(__name__, static_folder="web")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    """Load config.yaml and resolve all relative paths to absolute."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    config_dir = os.path.dirname(config_path)
    for key, val in cfg["paths"].items():
        if not os.path.isabs(str(val)):
            cfg["paths"][key] = os.path.normpath(os.path.join(config_dir, val))
    # Resolve API key from environment.
    api_key = os.environ.get(cfg["api"]["key_env_var"], "")
    cfg["api"]["_resolved_key"] = api_key
    return cfg


_CFG: dict = _load_cfg()


# ---------------------------------------------------------------------------
# GameSession – one human-vs-LLM tournament
# ---------------------------------------------------------------------------

_session: Optional["GameSession"] = None


class GameSession:
    """Manages state and turn logic for one interactive tournament.

    The human always occupies player-index 1 (the opponent slot in the
    existing game engine). The LLM personality agent occupies index 0.

    Phases
    ------
    idle          Initial state before the first action.
    your_turn     Waiting for the human to submit an action.
    hand_summary  A hand just finished; human must call /api/next_hand.
    tournament_over  All hands played; human sees final result.
    """

    def __init__(self, personality_name: str, personality_chosen: bool = False) -> None:
        gcfg = _CFG["game"]
        self.personality_name = personality_name
        # True when the human explicitly picked the personality (vs random).
        # When True the name is shown on-screen throughout the game.
        self.personality_chosen = personality_chosen
        self._game = LeducHoldemGame(_CFG)
        self._agent = PersonalityAgent(
            personality=personality_name,
            cfg=_CFG,
            win_prob_fn=compute_win_probability,
        )
        self.hands_per_tournament: int = gcfg["hands_per_tournament"]
        self.stacks: list = [gcfg["starting_chips"], gcfg["starting_chips"]]
        self.hands_played: int = 0
        self.tournament_seed: int = random.randint(10_000, 99_999)
        self.state: Optional[GameState] = None
        self._hidden_community = None   # Card object, revealed post-flop
        self._playing_hand_idx: int = 0  # 0-based index of the current hand
        self.phase: str = "idle"
        self.log: list = []
        self._deal_hand()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _opponent_label(self) -> str:
        """Display name for the LLM player used in log entries."""
        return self.personality_name.title() if self.personality_chosen else "Opponent"

    def _hand_seed(self) -> int:
        return self.tournament_seed + self.hands_played * 97

    def _deal_hand(self) -> None:
        """Deal cards and initialise GameState for the current hand."""
        self._playing_hand_idx = self.hands_played
        seed = self._hand_seed()
        self._game.deck.shuffle(seed)
        llm_card = self._game.deck.deal()
        human_card = self._game.deck.deal()
        self._hidden_community = self._game.deck.deal()

        gcfg = _CFG["game"]
        # Randomly assign who acts first this hand. We pass hand_index as
        # 1 (LLM first) or 0 (human first) so that GameState.start_hand's
        # `personality_acts_first = (hand_index % 2 == 1)` resolves correctly.
        # This same flag is then reused for post-flop order in _transition_round,
        # guaranteeing pre-flop and post-flop acting order are always identical.
        llm_acts_first = random.choice([True, False])
        acting_hand_idx = 1 if llm_acts_first else 0

        self.state = GameState.start_hand(
            hand_index=acting_hand_idx,
            personality_card=llm_card,
            opponent_card=human_card,
            stacks=list(self.stacks),
            ante=gcfg["ante"],
            preflop_raise_size=gcfg["preflop_raise_size"],
            preflop_raise_cap=gcfg["preflop_raise_cap"],
        )
        first_label = f"{self._opponent_label} acts first" if llm_acts_first else "You act first"
        self.log.append(
            f"── Hand {self._playing_hand_idx + 1} / {self.hands_per_tournament} ──  "
            f"{first_label}  │  "
            f"Stacks → You ${self.stacks[_HUMAN]} · {self._opponent_label} ${self.stacks[_LLM]}"
        )
        self.log.append(f"Your card: {human_card}  │  Pot: ${self.state.pot}")
        # Run LLM immediately if it has first action.
        self._advance_llm()

    def _advance_llm(self) -> None:
        """Run LLM turns until the human needs to act or the hand ends."""
        while (
            self.state is not None
            and not self.state.hand_over
            and self.state.current_player == _LLM
        ):
            legal = self.state.get_legal_actions()
            action = self._agent.act(self.state, legal)
            self.log.append(f"{self._opponent_label}: {action}")
            round_ended = self._game._apply_action(self.state, action)
            if self.state.hand_over:
                self._end_hand()
                return
            if round_ended:
                self._transition_round()
                if self.state is None or self.state.hand_over:
                    return
                # After transition the LLM may still need to act (first
                # post-flop action); the while-loop condition re-evaluates.
        if self.state and not self.state.hand_over:
            self.phase = "your_turn"

    def _transition_round(self) -> None:
        """Handle pre-flop → post-flop transition or trigger showdown."""
        if self.state.round == Round.PRE_FLOP:
            gcfg = _CFG["game"]
            self.state.preflop_done = True
            self.state.community_card = self._hidden_community
            self.state.round = Round.POST_FLOP
            self.state.round_bets = [0, 0]
            self.state.raises_this_round = 0
            self.state.raise_cap = gcfg["postflop_raise_cap"]
            self.state.raise_size = gcfg["postflop_raise_size"]
            self.state.actions_this_round = []
            # Post-flop acting order is always the same as pre-flop:
            # state.personality_acts_first was locked in at deal time from
            # the random coin flip in _deal_hand and never mutated.
            self.state.current_player = (
                _LLM if self.state.personality_acts_first else _HUMAN
            )
            self.log.append(
                f"── Post-Flop  Community: {self._hidden_community}"
            )
        else:
            # Post-flop round ended without a fold → showdown.
            self._game._resolve_showdown(self.state)
            self.state.hand_over = True
            self._end_hand()

    def _end_hand(self) -> None:
        """Finalise stacks, log result, and advance phase."""
        self.state.hand_over = True
        self._game._finalise_stacks(self.state)
        self.stacks = list(self.state.stacks)

        cc_str = (
            f"  Community: {self.state.community_card}"
            if self.state.community_card
            else ""
        )
        if self.state.winner is None:
            self.log.append("Tie! Pot split.")
        elif self.state.winner == _LLM:
            self.log.append(
                f"{self._opponent_label} wins! "
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
            f"Stacks → You ${self.stacks[_HUMAN]} · {self._opponent_label} ${self.stacks[_LLM]}"
        )
        self.hands_played += 1
        if self.hands_played >= self.hands_per_tournament:
            self.phase = "tournament_over"
            self.log.append(
                f"── Tournament Over ── Your opponent was {self.personality_name.title()}!"
            )
            if self.stacks[_HUMAN] > self.stacks[_LLM]:
                self.log.append("You win the tournament!")
            elif self.stacks[_LLM] > self.stacks[_HUMAN]:
                self.log.append("Opponent wins the tournament!")
            else:
                self.log.append("Tournament tied!")
        else:
            self.phase = "hand_summary"

    # ------------------------------------------------------------------
    # Public methods (called by route handlers)
    # ------------------------------------------------------------------

    def apply_action(self, action: str) -> None:
        """Validate and apply a human action, then advance LLM turns."""
        if self.phase != "your_turn":
            raise ValueError("It is not your turn.")
        if action not in self.state.get_legal_actions():
            raise ValueError(
                f"Illegal action {action!r}. "
                f"Legal: {self.state.get_legal_actions()}"
            )
        self.log.append(f"You: {action}")
        round_ended = self._game._apply_action(self.state, action)
        if self.state.hand_over:
            self._end_hand()
            return
        if round_ended:
            self._transition_round()
            if self.state is None or self.state.hand_over:
                return
        self._advance_llm()

    def start_next_hand(self) -> None:
        """Begin the next hand after the human has reviewed hand summary."""
        if self.phase != "hand_summary":
            raise ValueError("Not in hand-summary phase.")
        self._deal_hand()

    def to_dict(self) -> dict:
        """Serialise game state to a JSON-safe dict for the frontend."""
        s = self.state
        reveal_cards = self.phase in ("hand_summary", "tournament_over")
        # Personality name is revealed immediately when the human chose it,
        # otherwise hidden until the tournament ends (random mode).
        revealed_personality = (
            self.personality_name
            if (self.personality_chosen or self.phase == "tournament_over")
            else None
        )

        legal_actions: list = []
        if self.phase == "your_turn" and s:
            legal_actions = s.get_legal_actions()

        hand_winner = None
        if reveal_cards and s:
            if s.winner is None:
                hand_winner = "tie"
            elif s.winner == _LLM:
                hand_winner = "llm"
            else:
                hand_winner = "you"

        tournament_winner = None
        if self.phase == "tournament_over":
            if self.stacks[_HUMAN] > self.stacks[_LLM]:
                tournament_winner = "you"
            elif self.stacks[_LLM] > self.stacks[_HUMAN]:
                tournament_winner = "llm"
            else:
                tournament_winner = "tie"

        # Use live in-hand stacks (reflects ante + any bets placed so far).
        # Fall back to tournament-level stacks between hands.
        your_stack = s.stacks[_HUMAN] if s else self.stacks[_HUMAN]
        llm_stack  = s.stacks[_LLM]  if s else self.stacks[_LLM]

        return {
            # personality is None until tournament_over (hidden from human)
            "personality": revealed_personality,
            "hand_number": self._playing_hand_idx + 1,
            "total_hands": self.hands_per_tournament,
            "round": s.round.name.lower() if s else None,
            "your_card": str(s.opponent_card) if s else None,
            "community_card": (
                str(s.community_card) if (s and s.community_card) else None
            ),
            # LLM card hidden during play; revealed at hand_summary / tournament_over.
            "llm_card": (
                str(s.personality_card) if (s and reveal_cards) else None
            ),
            "your_stack": your_stack,
            "llm_stack": llm_stack,
            "pot": s.pot if s else 0,
            "legal_actions": legal_actions,
            "phase": self.phase,
            "hand_winner": hand_winner,
            "tournament_winner": tournament_winner,
            "log": list(self.log[-60:]),  # cap at last 60 entries
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global _session
    body = request.get_json(force=True, silent=True) or {}
    personality = body.get("personality", "").lower().strip()
    personality_chosen = bool(personality)
    if not personality:
        # Random mode: server picks the personality; the human won't know
        # until the tournament ends.
        personality = random.choice(sorted(_VALID_PERSONALITIES))
    if personality not in _VALID_PERSONALITIES:
        return jsonify({"error": f"Unknown personality: {personality!r}"}), 400
    _session = GameSession(personality, personality_chosen=personality_chosen)
    return jsonify(_session.to_dict())


@app.route("/api/action", methods=["POST"])
def action():
    if _session is None:
        return jsonify({"error": "No active game. Start a new game first."}), 400
    body = request.get_json(force=True, silent=True) or {}
    act = body.get("action", "").strip()
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
        return jsonify({"phase": "no_game"})
    return jsonify(_session.to_dict())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leduc Hold'em interactive server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    args = parser.parse_args()
    print(f"[Server] Leduc Hold'em running at http://localhost:{args.port}")
    print("[Server] LLM responses may take 5–30s per action — this is expected.")
    # threaded=False keeps game state access single-threaded (one session).
    app.run(debug=False, port=args.port, threaded=False)
