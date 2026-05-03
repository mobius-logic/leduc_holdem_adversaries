"""LLM-driven personality agent for Leduc Hold'em.

Calls the Azure OpenAI Chat Completions API exactly once per action via
direct HTTP (requests library). Every call is stateless — no conversation
history is retained between actions, hands, or tournaments. The full
instruction set and game state are injected into every prompt.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List

import requests
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
    """LLM-driven agent calling Azure OpenAI via direct HTTP requests.

    Makes one fresh, stateless HTTP POST per action. The full instruction
    set text and game state are included in every request. Retries
    automatically on HTTP 429 rate-limit responses.

    Attributes:
        personality: Lowercase personality name string.
        _url: Fully constructed Azure OpenAI deployment URL.
        _headers: HTTP headers including the API key.
        _max_retries: Maximum retry attempts on rate-limit errors.
        _retry_delay: Seconds to wait between retries (overridden by
            Retry-After header if present).
        _instruction_text: Cached instruction file content.
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
            EnvironmentError: If the API key env var is not set.
            FileNotFoundError: If the instruction file is not found.
        """
        self.personality = personality
        self._cfg = cfg
        self._win_prob_fn = win_prob_fn

        api_cfg = cfg["api"]
        instructions_dir = cfg["paths"]["instructions_dir"]

        key = api_cfg.get("_resolved_key") or os.environ.get(api_cfg["key_env_var"])
        if not key:
            raise EnvironmentError(
                f"Azure OpenAI API key not found. "
                f"Set the {api_cfg['key_env_var']} environment variable."
            )

        endpoint = api_cfg["endpoint"].rstrip("/")
        deployment = api_cfg["deployment"]
        api_version = api_cfg["api_version"]
        self._url = (
            f"{endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )
        self._headers = {
            "Content-Type": "application/json",
            "api-key": key,
        }
        self._max_retries: int = api_cfg.get("max_retries", 5)
        self._retry_delay: int = api_cfg.get("retry_delay", 10)

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
        """POST to Azure OpenAI Chat Completions with retry on rate limits.

        Constructs the URL from config at init time. Retries up to
        ``max_retries`` times on HTTP 429, honouring the Retry-After
        header when present.

        Args:
            system_prompt: System-role message content.
            user_prompt: User-role message content.

        Returns:
            Raw content string from the first choice.
            Returns 'Check' on any unrecoverable error.
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = requests.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", self._retry_delay)
                    )
                    print(
                        f"    [PersonalityAgent:{self.personality}] "
                        f"Rate limit (attempt {attempt}/{self._max_retries}). "
                        f"Waiting {retry_after}s ..."
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"] or ""

            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"    [PersonalityAgent:{self.personality}] "
                    f"API error (attempt {attempt}/{self._max_retries}): {exc}. "
                    "Defaulting to Check."
                )
                return "Check"

        print(
            f"    [PersonalityAgent:{self.personality}] "
            f"Failed after {self._max_retries} attempts. Defaulting to Check."
        )
        return "Check"
