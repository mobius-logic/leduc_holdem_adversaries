"""LLM-driven personality agent for Leduc Hold'em.

Calls the OpenAI API exactly once per action. Every call is stateless —
no conversation history is retained between actions, hands, or
tournaments. The full instruction set and game state are injected into
every prompt.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from jinja2 import Template
from json_repair import json_repair

from agents.base_agent import BaseAgent
from game.state import GameState, Round


_VALID_ACTIONS = {"Check", "Call", "Raise", "Fold"}

# Substitution priority when parsed action is illegal.
# Never substitute to Fold automatically.
_SUBSTITUTE_ORDER = ["Raise", "Call", "Check"]


def _load_instruction_file(instructions_dir: str, personality: str) -> str:
    """Load the instruction set text for a given personality.

    Uses explicit if-statements to select the instruction filename
    based on the personality name, as required by the spec.

    Args:
        instructions_dir: Path to the directory containing instruction
            set .txt files.
        personality: Personality name (analytical / conservative /
            aggressive / reckless).

    Returns:
        Full text of the instruction set.

    Raises:
        FileNotFoundError: If the instruction file does not exist.
        ValueError: If the personality name is unrecognised.
    """
    if personality == "analytical":
        filename = "analytical_personality_instructions.txt"
    elif personality == "conservative":
        filename = "conservative_personality_instructions.txt"
    elif personality == "aggressive":
        filename = "aggressive_personality_instructions.txt"
    elif personality == "reckless":
        filename = "reckless_personality_instructions.txt"
    else:
        raise ValueError(f"Unknown personality: {personality!r}")

    path = os.path.join(instructions_dir, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _build_system_prompt(personality: str, instruction_text: str) -> str:
    """Construct the system prompt for an LLM call.

    Args:
        personality: Display name of the personality (e.g. 'Analytical').
        instruction_text: Full instruction set text loaded from disk.

    Returns:
        Formatted system prompt string.
    """
    template = Template(
        "You are playing Leduc Hold'em as a {{ personality }} player.\n"
        "Your instruction set follows in full.\n\n"
        "{{ instruction_text }}"
    )
    return template.render(
        personality=personality.capitalize(),
        instruction_text=instruction_text,
    )


def _build_user_prompt(state: GameState, win_prob: float) -> str:
    """Construct the user prompt describing the current game state.

    Args:
        state: Current game state.
        win_prob: Pre-computed win probability for the personality agent
            (float 0.0–1.0).

    Returns:
        Formatted user prompt string.
    """
    round_name = "pre-flop" if state.round == Round.PRE_FLOP else "post-flop"
    acting_order = "first" if state.personality_acts_first else "second"

    community_str = (
        str(state.community_card)
        if state.community_card is not None
        else "NOT YET REVEALED"
    )
    last_opp = state.last_opponent_action if state.last_opponent_action else "None"
    win_pct = round(win_prob * 100, 1)

    return (
        f"Current game state:\n"
        f"  - Round: {round_name}\n"
        f"  - You are acting {acting_order} this hand\n"
        f"  - Your private card: {state.personality_card}\n"
        f"  - Community card: {community_str}\n"
        f"  - Your chip stack: ${state.personality_stack}\n"
        f"  - Opponent chip stack: ${state.opponent_stack}\n"
        f"  - Pot: ${state.pot}\n"
        f"  - Your win probability: {win_pct}%\n"
        f"  - Last opponent action: {last_opp}\n"
        f"  - Raises so far this round: {state.raises_this_round} of "
        f"{state.raise_cap} maximum\n\n"
        f"Respond with exactly one word: Check, Call, Raise, or Fold."
    )


def _parse_response(raw: str) -> str:
    """Extract and normalise the action word from an LLM response.

    Args:
        raw: Raw text returned by the API.

    Returns:
        Capitalised action string. Defaults to 'Check' if unrecognised.
    """
    cleaned = raw.strip()
    # Capitalise first letter, lowercase the rest.
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:].lower()
    if cleaned not in _VALID_ACTIONS:
        print(
            f"    [PersonalityAgent] Unrecognised response {raw!r}, "
            "defaulting to Check."
        )
        return "Check"
    return cleaned


def _validate_action(action: str, legal_actions: List[str]) -> str:
    """Ensure the action is legal; substitute if not.

    Substitution order: Raise → Call → Check. Never Fold automatically.

    Args:
        action: Parsed action from LLM response.
        legal_actions: Currently legal actions.

    Returns:
        A legal action string.
    """
    if action in legal_actions:
        return action
    print(
        f"    [PersonalityAgent] Action '{action}' is illegal "
        f"(legal: {legal_actions}). Substituting."
    )
    for substitute in _SUBSTITUTE_ORDER:
        if substitute in legal_actions:
            return substitute
    # Fallback: first available legal action (should never reach here).
    return legal_actions[0]


class PersonalityAgent(BaseAgent):
    """LLM-driven agent embodying a specific personality type.

    Makes one fresh, stateless OpenAI API call per action. The full
    instruction set text and game state are included in every prompt.

    Attributes:
        personality: Lowercase personality name string.
        cfg: Full parsed config dictionary.
        _instruction_text: Cached instruction file content.
        _client: Initialised OpenAI client.
    """

    def __init__(self, personality: str, cfg: dict, win_prob_fn) -> None:
        """Initialise the personality agent.

        Args:
            personality: Lowercase personality name (analytical /
                conservative / aggressive / reckless).
            cfg: Full parsed config dictionary.
            win_prob_fn: Callable(state) -> float that returns the
                exact win probability for the personality agent.

        Raises:
            EnvironmentError: If the OpenAI API key env var is not set.
            FileNotFoundError: If the instruction file is not found.
        """
        self.personality = personality
        self._cfg = cfg
        self._win_prob_fn = win_prob_fn

        api_cfg = cfg["api"]
        instructions_dir = cfg["paths"]["instructions_dir"]

        key = os.environ.get(api_cfg["key_env_var"])
        if not key:
            raise EnvironmentError(
                f"OpenAI API key not found. "
                f"Set the {api_cfg['key_env_var']} environment variable."
            )

        from openai import OpenAI  # Imported here to keep module-level imports clean.
        self._client = OpenAI(api_key=key)
        self._model = api_cfg["model"]

        self._instruction_text = _load_instruction_file(
            instructions_dir, personality
        )

    def act(self, state: GameState, legal_actions: List[str]) -> str:
        """Call the LLM once and return a validated legal action.

        Args:
            state: Current game state at the moment of this action.
            legal_actions: Currently legal action strings.

        Returns:
            A legal action string chosen by the LLM (or substituted on
            parse/API failure).
        """
        win_prob = self._win_prob_fn(state)
        system_prompt = _build_system_prompt(
            self.personality, self._instruction_text
        )
        user_prompt = _build_user_prompt(state, win_prob)

        raw_response = self._call_llm(system_prompt, user_prompt)
        action = _parse_response(raw_response)
        action = _validate_action(action, legal_actions)
        print(
            f"    [PersonalityAgent:{self.personality}] "
            f"win_prob={win_prob:.1%} → {action}"
        )
        return action

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single stateless API call to OpenAI.

        Args:
            system_prompt: System-role message content.
            user_prompt: User-role message content.

        Returns:
            Raw text content of the first response choice.
            Returns 'Check' on any API failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"    [PersonalityAgent:{self.personality}] "
                f"OpenAI API error: {exc}. Defaulting to Check."
            )
            return "Check"
