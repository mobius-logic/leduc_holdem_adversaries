"""Leduc Hold'em game loop and action resolution.

Drives a complete tournament (N hands) between a personality agent and
a random agent, calling observation hooks at the correct timing points.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from game.deck import Card, Deck, hand_strength
from game.state import OPPONENT, PERSONALITY, GameState, Round


# Type alias for an observation callback.
# Called with (state, slot_index) immediately before the personality
# agent's action is applied.
ObservationCallback = Callable[[GameState, int], None]


class LeducHoldemGame:
    """Orchestrates hands and tournaments for Leduc Hold'em.

    This class owns the Deck, resolves actions, manages chip transfers,
    and fires observation callbacks at the correct moments.

    Attributes:
        cfg: Loaded config dictionary (game sub-section).
        deck: The 6-card Leduc deck instance.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialise the game with configuration.

        Args:
            cfg: The full parsed config dict (from config.yaml). The
                 game and observation sub-sections are accessed directly.
        """
        self._game_cfg = cfg["game"]
        self.deck = Deck()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def play_hand(
        self,
        hand_index: int,
        stacks: List[int],
        seed: int,
        personality_agent,
        random_agent,
        obs_callback: Optional[ObservationCallback] = None,
        action_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> Tuple[List[int], GameState]:
        """Play one complete hand and return updated stacks plus final state.

        Args:
            hand_index: Zero-based index of this hand within the
                tournament (0–4).
            stacks: Current chip counts [personality, opponent].
            seed: RNG seed used to shuffle the deck for this hand.
            personality_agent: The LLM personality agent instance.
            random_agent: The random agent instance.
            obs_callback: Optional callable fired immediately before
                each personality agent action. Receives (state,
                preflop_action_count, postflop_action_count).

        Returns:
            Tuple of (updated_stacks, final_game_state).
        """
        cfg = self._game_cfg
        self.deck.shuffle(seed)

        personality_card: Card = self.deck.deal()
        opponent_card: Card = self.deck.deal()
        community_card: Card = self.deck.deal()

        state = GameState.start_hand(
            hand_index=hand_index,
            personality_card=personality_card,
            opponent_card=opponent_card,
            stacks=list(stacks),
            ante=cfg["ante"],
            preflop_raise_size=cfg["preflop_raise_size"],
            preflop_raise_cap=cfg["preflop_raise_cap"],
        )

        agents = [personality_agent, random_agent]

        # Track how many times the personality agent has acted per round.
        preflop_personality_actions = 0
        postflop_personality_actions = 0

        # ---- Pre-flop round ----
        print(
            f"    [Hand {hand_index + 1}] Pre-flop | "
            f"Personality: {personality_card} | Pot: ${state.pot}"
        )

        while not state.hand_over:
            player_idx = state.current_player
            legal = state.get_legal_actions()

            if player_idx == PERSONALITY:
                # Determine slot index (0 or 1 for pre-flop).
                slot = preflop_personality_actions
                if obs_callback is not None:
                    obs_callback(state, slot)
                action = personality_agent.act(state, legal)
                state.last_personality_action = action
                preflop_personality_actions += 1
            else:
                action = random_agent.act(state, legal)
                state.last_opponent_action = action

            print(
                f"      {'Personality' if player_idx == PERSONALITY else 'Opponent'} "
                f"action: {action}"
            )

            if action_callback is not None:
                action_callback(
                    "Personality" if player_idx == PERSONALITY else "Opponent",
                    action,
                    "preflop",
                )
            round_ended = self._apply_action(state, action)
            if round_ended or state.hand_over:
                break

        if state.hand_over:
            self._finalise_stacks(state)
            print(
                f"    [Hand {hand_index + 1}] Pre-flop fold. "
                f"Winner: {'Personality' if state.winner == PERSONALITY else 'Opponent'}"
            )
            return state.stacks, state

        # ---- Transition to post-flop ----
        state.preflop_done = True
        state.community_card = community_card
        state.round = Round.POST_FLOP
        state.round_bets = [0, 0]
        state.raises_this_round = 0
        state.raise_cap = cfg["postflop_raise_cap"]
        state.raise_size = cfg["postflop_raise_size"]
        state.actions_this_round = []
        # Reset acting order: same player who went first pre-flop goes first post-flop.
        state.current_player = PERSONALITY if state.personality_acts_first else OPPONENT

        print(
            f"    [Hand {hand_index + 1}] Post-flop | Community: {community_card} | Pot: ${state.pot}"
        )

        # ---- Post-flop round ----
        while not state.hand_over:
            player_idx = state.current_player
            legal = state.get_legal_actions()

            if player_idx == PERSONALITY:
                slot = 2 + postflop_personality_actions
                if obs_callback is not None:
                    obs_callback(state, slot)
                action = personality_agent.act(state, legal)
                state.last_personality_action = action
                postflop_personality_actions += 1
            else:
                action = random_agent.act(state, legal)
                state.last_opponent_action = action

            print(
                f"      {'Personality' if player_idx == PERSONALITY else 'Opponent'} "
                f"action: {action}"
            )

            if action_callback is not None:
                action_callback(
                    "Personality" if player_idx == PERSONALITY else "Opponent",
                    action,
                    "postflop",
                )
            round_ended = self._apply_action(state, action)
            if round_ended or state.hand_over:
                break

        if state.hand_over:
            # Folded post-flop.
            self._finalise_stacks(state)
            print(
                f"    [Hand {hand_index + 1}] Post-flop fold. "
                f"Winner: {'Personality' if state.winner == PERSONALITY else 'Opponent'}"
            )
            return state.stacks, state

        # ---- Showdown ----
        self._resolve_showdown(state)
        self._finalise_stacks(state)
        winner_label = (
            "Tie" if state.winner is None
            else ("Personality" if state.winner == PERSONALITY else "Opponent")
        )
        print(
            f"    [Hand {hand_index + 1}] Showdown: "
            f"Personality={personality_card}, Opponent={opponent_card}, "
            f"Community={community_card} | Winner: {winner_label}"
        )
        return state.stacks, state

    def play_tournament(
        self,
        seed_base: int,
        tournament_index: int,
        personality_agent,
        random_agent,
        obs_callback: Optional[ObservationCallback] = None,
        tournament_logger=None,
    ) -> List[GameState]:
        """Play a complete tournament of N hands.

        Args:
            seed_base: Base seed value from config.
            tournament_index: Index of this tournament (0-based).
            personality_agent: LLM personality agent instance.
            random_agent: Random agent instance.
            obs_callback: Optional observation callback.

        Returns:
            List of final GameState objects, one per hand.
        """
        cfg = self._game_cfg
        starting_chips = cfg["starting_chips"]
        stacks = [starting_chips, starting_chips]
        hand_states: List[GameState] = []

        for hand_index in range(cfg["hands_per_tournament"]):
            hand_seed = seed_base + tournament_index * 100 + hand_index
            print(
                f"  [Hand {hand_index + 1}/{cfg['hands_per_tournament']}] "
                f"Stacks: Personality=${stacks[0]}, Opponent=${stacks[1]} | "
                f"Seed: {hand_seed}"
            )

            if tournament_logger is not None:
                tournament_logger.start_hand(hand_index, hand_seed, list(stacks))

            action_callback = (
                tournament_logger.record_action
                if tournament_logger is not None
                else None
            )

            updated_stacks, final_state = self.play_hand(
                hand_index=hand_index,
                stacks=stacks,
                seed=hand_seed,
                personality_agent=personality_agent,
                random_agent=random_agent,
                obs_callback=obs_callback,
                action_callback=action_callback,
            )
            stacks = updated_stacks
            hand_states.append(final_state)

            if tournament_logger is not None:
                tournament_logger.end_hand(final_state)

        print(
            f"  Tournament {tournament_index} complete. "
            f"Final stacks: Personality=${stacks[0]}, Opponent=${stacks[1]}"
        )
        return hand_states

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_action(self, state: GameState, action: str) -> bool:
        """Apply an action to the game state in-place.

        Args:
            state: Current mutable game state.
            action: One of 'Check', 'Call', 'Raise', 'Fold'.

        Returns:
            True if the current betting round has ended (not necessarily
            the hand), False if more betting is needed.
        """
        player = state.current_player
        to_call = state.to_call_amount()

        if action == "Fold":
            state.hand_over = True
            state.winner = 1 - player
            return True

        if action == "Call":
            # Pay the outstanding bet.
            state.stacks[player] -= to_call
            state.pot += to_call
            state.round_bets[player] = max(state.round_bets)
            state.actions_this_round.append(action)
            return True  # Round ends on a call.

        if action == "Check":
            # Valid only when to_call == 0.
            # Check BEFORE appending — actions_this_round is mutable and
            # a reference copy would point to the same list.
            both_checked = bool(
                state.actions_this_round
                and state.actions_this_round[-1] == "Check"
            )
            state.actions_this_round.append(action)
            if both_checked:
                # Both players have checked consecutively → round over.
                return True
            # Pass to other player.
            state.current_player = 1 - player
            return False

        if action == "Raise":
            # Pay to_call (if any) plus the raise size.
            cost = to_call + state.raise_size
            state.stacks[player] -= cost
            state.pot += cost
            state.round_bets[player] = max(state.round_bets) + state.raise_size
            state.raises_this_round += 1
            state.actions_this_round.append(action)
            state.current_player = 1 - player
            return False

        raise ValueError(f"Unknown action: {action!r}")

    def _resolve_showdown(self, state: GameState) -> None:
        """Determine hand winner at showdown and set state.winner.

        Args:
            state: GameState at end of post-flop betting.
        """
        assert state.community_card is not None
        p_strength = hand_strength(state.personality_card, state.community_card)
        o_strength = hand_strength(state.opponent_card, state.community_card)

        if p_strength > o_strength:
            state.winner = PERSONALITY
        elif o_strength > p_strength:
            state.winner = OPPONENT
        else:
            state.winner = None  # Tie

    def _finalise_stacks(self, state: GameState) -> None:
        """Transfer pot chips to the winner(s) after hand resolution.

        For ties, each player receives floor(pot / 2). Any remainder
        stays in the pot (accounted for per spec: "round down").

        Args:
            state: GameState with winner already set.
        """
        if state.winner is None:
            # Tie: split pot, remainder stays out.
            share = state.pot // 2
            state.stacks[PERSONALITY] += share
            state.stacks[OPPONENT] += share
            state.pot -= share * 2  # Remainder (0 or 1) is left over.
        else:
            state.stacks[state.winner] += state.pot
            state.pot = 0
