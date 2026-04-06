"""
game_engine.py - Core Pazaak game engine.

Pure Python, no RL or Gymnasium dependencies.
Implements the full Pazaak rules as specified in game_rules.md and implementation_plan.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class PlayerState:
    """Represents the persistent + per-round state of one player."""

    total: int = 0
    # Fixed 4-slot hand. Each slot is an int value or None if consumed.
    hand: list = field(default_factory=lambda: [None, None, None, None])
    # All hand cards played so far this game (visible to both players on the field)
    used_hand: list = field(default_factory=list)
    # Has this player stood in the current round?
    stood: bool = False


# ---------------------------------------------------------------------------
# PazaakGame
# ---------------------------------------------------------------------------


class PazaakGame:
    """
    Manages a full Pazaak game (multiple rounds, first to 3 wins).

    Turn phases
    -----------
    "field_draw"  - Waiting to auto-draw a field card (handled internally).
    "decision"    - Active player must choose: hit / stand / play hand card.
    "round_over"  - A round just finished. Call start_round() to continue.
    "game_over"   - A player has 3 round wins. Game is finished.

    Usage
    -----
    game = PazaakGame(side_deck_a, side_deck_b, seed=42)
    # game is already in "decision" phase with Player A's first field card drawn.

    while game.phase != "game_over":
        legal = game.legal_actions()
        action = choose(legal)           # "hit" | "stand" | ("play", i)
        game.step(action)

        if game.phase == "round_over":
            game.start_round()           # caller must call this to begin next round
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        side_deck_a: list[int],
        side_deck_b: list[int],
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        side_deck_a : list of 20 ints, values in {-6..−1, +1..+6}
        side_deck_b : list of 20 ints, same constraint
        seed        : RNG seed for reproducible games (optional)
        """
        self._rng = random.Random(seed)

        self.side_decks = [list(side_deck_a), list(side_deck_b)]

        # Draw 4 hand cards per player from their side decks (once per game)
        self.players: list[PlayerState] = []
        for deck in self.side_decks:
            hand_cards = self._rng.sample(deck, 4)
            ps = PlayerState(hand=hand_cards)
            self.players.append(ps)

        # Round wins for each player
        self.round_wins: list[int] = [0, 0]

        # Outcome trackers (set during round/game resolution)
        self.last_round_winner: Optional[int] = None  # 0, 1, or None (draw)
        self.game_winner: Optional[int] = None  # 0 or 1

        # Phase and active player
        self.phase: str = "field_draw"
        self.active_player: int = 0

        # Start the first round (draws Player A's first field card)
        self.start_round()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_round(self) -> None:
        """
        Reset per-round state and begin a new round.

        - Resets totals and stood flags for both players.
        - Does NOT reset hand cards, used_hand, or round_wins.
        - Player A always goes first.
        - Automatically draws the first field card for Player A.
        """
        for ps in self.players:
            ps.total = 0
            ps.stood = False

        self.active_player = 0
        self.phase = "field_draw"
        self._draw_field_card()  # → sets phase = "decision"

    def step(self, action) -> None:
        """
        Apply one action for the active player.

        Parameters
        ----------
        action : "hit" | "stand" | ("play", slot_index)

        Raises
        ------
        ValueError  if the action is illegal in the current phase/state.
        """
        if self.phase != "decision":
            raise ValueError(
                f"step() called in phase '{self.phase}'; expected 'decision'."
            )

        if isinstance(action, tuple) and action[0] == "play":
            self._apply_play_card(action[1])

        elif action == "hit":
            self._apply_hit()

        elif action == "stand":
            self._apply_stand()

        else:
            raise ValueError(f"Unknown action: {action!r}")

    def legal_actions(self) -> list:
        """
        Return all legal actions for the active player in the current phase.

        Returns an empty list when not in "decision" phase.
        """
        if self.phase != "decision":
            return []

        actions = ["hit", "stand"]
        hand = self.players[self.active_player].hand
        for i in range(len(hand)):
            if hand[i] is not None:
                actions.append(("play", i))
        return actions

    def get_observation(self, player_idx: int) -> dict:
        """
        Return what player_idx can observe (per Section 7 of game_rules.md).

        Opponent's unrevealed hand values are hidden (only the count is shown).
        """
        me = self.players[player_idx]
        opp = self.players[1 - player_idx]
        return {
            "my_total": me.total,
            "opp_total": opp.total,
            "my_stood": me.stood,
            "opp_stood": opp.stood,
            "my_hand": list(me.hand),  # 4 slots: value or None
            "my_hand_count": sum(1 for c in me.hand if c is not None),
            "opp_hand_count": sum(1 for c in opp.hand if c is not None),
            "opp_used_hand": list(opp.used_hand),  # played cards are visible
            "my_round_wins": self.round_wins[player_idx],
            "opp_round_wins": self.round_wins[1 - player_idx],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_field_card(self) -> int:
        """Draw one field card (1–10, uniform) and add it to the active player's total."""
        card = self._rng.randint(1, 10)
        self.players[self.active_player].total += card
        self.phase = "decision"
        return card

    def _apply_play_card(self, slot: int) -> None:
        """Play the hand card at the given slot index for the active player."""
        player = self.players[self.active_player]
        if slot < 0 or slot >= len(player.hand):
            raise ValueError(f"Hand slot {slot} out of range.")
        if player.hand[slot] is None:
            raise ValueError(f"Hand slot {slot} is empty (already used).")

        card_value = player.hand[slot]
        player.total += card_value
        player.used_hand.append(card_value)
        player.hand[slot] = None  # consume the slot - never shift!
        # Phase stays "decision"; the player can play more cards or hit/stand

    def _apply_hit(self) -> None:
        """End the active player's turn and continue (they may play again next turn)."""
        player = self.players[self.active_player]

        # Bust check
        if player.total > 20:
            self._resolve_bust(self.active_player)
            return

        # Determine who goes next
        other = 1 - self.active_player
        if self.players[other].stood:
            # The other player has stood; current player takes another turn
            next_player = self.active_player
        else:
            next_player = other

        self._begin_turn(next_player)

    def _apply_stand(self) -> None:
        """Lock the active player's total. They take no further actions this round."""
        player = self.players[self.active_player]

        # Bust check (a player could stand on a negative total, but > 20 is still bust)
        if player.total > 20:
            self._resolve_bust(self.active_player)
            return

        player.stood = True
        other = 1 - self.active_player

        if self.players[other].stood:
            # Both players have stood → compare totals
            self._resolve_both_stood()
        else:
            # Other player still needs to complete their rounds
            self._begin_turn(other)

    def _begin_turn(self, player_idx: int) -> None:
        """Switch to player_idx and auto-draw their field card."""
        self.active_player = player_idx
        self.phase = "field_draw"
        self._draw_field_card()

    # ------------------------------------------------------------------
    # Round resolution
    # ------------------------------------------------------------------

    def _resolve_bust(self, busted_player: int) -> None:
        """The busted player loses the round immediately."""
        winner = 1 - busted_player
        self._end_round(winner)

    def _resolve_both_stood(self) -> None:
        """Both players stood - compare totals to determine round winner."""
        t0 = self.players[0].total
        t1 = self.players[1].total

        if t0 > t1:
            self._end_round(winner=0)
        elif t1 > t0:
            self._end_round(winner=1)
        else:
            self._end_round(winner=None)  # draw

    def _end_round(self, winner: Optional[int]) -> None:
        """
        Record the round result, update round_wins, and check for game end.

        Sets phase to "round_over" (or "game_over" if the match is decided).
        The caller (or the gymnasium wrapper) is responsible for calling
        start_round() to continue, unless the game is over.
        """
        self.last_round_winner = winner

        if winner is not None:
            self.round_wins[winner] += 1

        if self.round_wins[0] >= 3:
            self.game_winner = 0
            self.phase = "game_over"
        elif self.round_wins[1] >= 3:
            self.game_winner = 1
            self.phase = "game_over"
        else:
            self.phase = "round_over"

    # ------------------------------------------------------------------
    # Convenience / debug helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        p0 = self.players[0]
        p1 = self.players[1]
        return (
            f"PazaakGame(phase={self.phase!r}, active={self.active_player}, "
            f"score={self.round_wins}, "
            f"P0(total={p0.total}, stood={p0.stood}), "
            f"P1(total={p1.total}, stood={p1.stood}))"
        )
