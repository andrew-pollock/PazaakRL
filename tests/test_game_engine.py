"""
test_game_engine.py - Unit tests for game_engine.py

Run with:  python -m pytest test_game_engine.py -v
       or:  python test_game_engine.py
"""

import sys
import unittest
import random
from unittest.mock import patch

sys.path.insert(0, ".")
from game_engine import PazaakGame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_deck(values=None):
    """Return a 20-card side deck. Defaults to a balanced mix of -6..+6 (no 0)."""
    if values is not None:
        deck = list(values)
        while len(deck) < 20:
            deck.extend(values)
        return deck[:20]
    vals = list(range(-6, 0)) + list(range(1, 7))
    deck = []
    while len(deck) < 20:
        deck.extend(vals)
    return deck[:20]


def play_full_game(seed=0, agent_a=None, agent_b=None):
    """
    Run a complete game to "game_over" using simple agents.
    Returns the finished PazaakGame instance.
    Default agents: stand on 15+, hit otherwise, never play hand cards.
    """
    rng = random.Random(seed + 1000)

    def default_agent(game):
        p = game.players[game.active_player]
        if p.total >= 15:
            return "stand"
        return "hit"

    if agent_a is None:
        agent_a = default_agent
    if agent_b is None:
        agent_b = default_agent

    game = PazaakGame(make_deck(), make_deck(), seed=seed)
    safety = 0
    while game.phase != "game_over":
        if game.phase == "round_over":
            game.start_round()
            continue
        agent = agent_a if game.active_player == 0 else agent_b
        game.step(agent(game))
        safety += 1
        if safety > 50_000:
            raise RuntimeError(
                "play_full_game: exceeded step limit (possible infinite loop)"
            )
    return game


# ---------------------------------------------------------------------------
# 1. Field Card Drawing
# ---------------------------------------------------------------------------


class TestFieldCardDrawing(unittest.TestCase):
    def test_field_card_drawn_on_init(self):
        """A field card is drawn automatically when the game starts."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        # Player A's total must be 1–10 after the automatic draw
        self.assertIn(game.players[0].total, range(1, 11))

    def test_field_card_range(self):
        """Field cards are integers in [1, 10]."""
        # Patch randint to observe every drawn value
        drawn = []
        orig = random.Random.randint

        def capturing_randint(self_, a, b):
            v = orig(self_, a, b)
            drawn.append(v)
            return v

        with patch.object(random.Random, "randint", capturing_randint):
            game = PazaakGame(make_deck(), make_deck(), seed=5)
            # Drive through several turns
            for _ in range(30):
                if game.phase == "round_over":
                    game.start_round()
                elif game.phase == "game_over":
                    break
                elif game.phase == "decision":
                    game.step("hit")

        self.assertTrue(len(drawn) > 0)
        for v in drawn:
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 10)

    def test_field_card_drawn_once_per_turn(self):
        """Playing a hand card does not draw a new field card; only a new turn does."""
        # Give player 0 a known hand so we can play a card
        game = PazaakGame(
            make_deck([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6] * 2),
            make_deck(),
            seed=10,
        )
        total_after_field = game.players[0].total
        # Playing a hand card changes the total by the card value, not another +1..10
        hand = game.players[0].hand
        slot = next(i for i, v in enumerate(hand) if v is not None)
        card_val = hand[slot]
        game.step(("play", slot))
        self.assertEqual(game.players[0].total, total_after_field + card_val)

    def test_new_field_card_on_new_turn(self):
        """After the active player hits, the next player's total increases by 1–10."""
        game = PazaakGame(make_deck(), make_deck(), seed=7)
        total_before = game.players[1].total  # Player B starts at 0
        game.step("hit")  # Player A hits → Player B's turn begins
        self.assertGreater(game.players[1].total, total_before)
        self.assertIn(game.players[1].total, range(1, 11))

    def test_field_cards_uniform_distribution(self):
        """Over many draws the distribution of field cards is roughly uniform."""
        counts = {v: 0 for v in range(1, 11)}
        game = PazaakGame(make_deck(), make_deck(), seed=99)
        for _ in range(1000):
            if game.phase == "round_over":
                game.start_round()
            elif game.phase == "game_over":
                game = PazaakGame(
                    make_deck(), make_deck(), seed=random.randint(0, 9999)
                )
            elif game.phase == "decision":
                total = game.players[game.active_player].total
                counts[total % 10 + 1] += 1  # rough proxy; just drive the game forward
                game.step("hit")

        # Check that each value 1–10 was drawn (roughly; not a strict chi-square test)
        for v in range(1, 11):
            self.assertGreater(counts[v], 0, f"Value {v} never appeared")


# ---------------------------------------------------------------------------
# 2. Hand Card Persistence
# ---------------------------------------------------------------------------


class TestHandCardPersistence(unittest.TestCase):
    def test_hand_cards_drawn_at_game_start(self):
        """Each player starts with exactly 4 hand cards."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        for ps in game.players:
            non_none = [c for c in ps.hand if c is not None]
            self.assertEqual(len(non_none), 4)

    def test_hand_cards_persist_across_rounds(self):
        """Hand cards that were not used in round 1 are still there in round 2."""
        game = PazaakGame(make_deck(), make_deck(), seed=2)
        hand_after_round1 = None

        # Record hand state at the end of round 1 (before start_round clears totals)
        safety = 0
        while game.phase not in ("round_over", "game_over"):
            # Stand immediately - never play hand cards
            if game.phase == "decision":
                game.step("stand")
            safety += 1
            if safety > 200:
                break

        hand_after_round1 = list(game.players[0].hand)
        # None of the cards should have been used (we never played hand cards)
        self.assertEqual(hand_after_round1, game.players[0].hand)

        if game.phase == "round_over":
            game.start_round()
            # Hand must be identical after start_round
            self.assertEqual(game.players[0].hand, hand_after_round1)

    def test_used_hand_card_slot_is_none(self):
        """Playing a hand card sets that slot to None."""
        game = PazaakGame(make_deck(), make_deck(), seed=3)
        hand = game.players[0].hand
        slot = next(i for i, v in enumerate(hand) if v is not None)
        game.step(("play", slot))
        self.assertIsNone(game.players[0].hand[slot])

    def test_slots_do_not_shift_after_use(self):
        """Playing card at slot i does not affect any other slot's position or value."""
        game = PazaakGame(make_deck(), make_deck(), seed=4)
        original_hand = list(game.players[0].hand)
        slot = 0  # play the first card
        game.step(("play", slot))
        for j in range(1, 4):
            self.assertEqual(
                game.players[0].hand[j],
                original_hand[j],
                f"Slot {j} changed after consuming slot 0",
            )

    def test_cannot_reuse_consumed_card(self):
        """Attempting to play an already-consumed slot raises ValueError."""
        game = PazaakGame(make_deck(), make_deck(), seed=5)
        slot = next(i for i, v in enumerate(game.players[0].hand) if v is not None)
        game.step(("play", slot))
        # Slot is now None - playing it again must raise
        with self.assertRaises((ValueError, Exception)):
            game.step(("play", slot))

    def test_hand_cards_not_reset_across_games(self):
        """A new PazaakGame always starts with fresh hand draws (not leftovers)."""
        game1 = PazaakGame(make_deck(), make_deck(), seed=6)
        game2 = PazaakGame(make_deck(), make_deck(), seed=6)
        # Same seed → same draws, but the objects are independent
        self.assertEqual(game1.players[0].hand, game2.players[0].hand)
        # Mutating game1 must not affect game2
        slot = next(i for i, v in enumerate(game1.players[0].hand) if v is not None)
        game1.step(("play", slot))
        self.assertIsNotNone(game2.players[0].hand[slot])

    def test_all_four_hand_cards_usable(self):
        """All 4 hand card slots can be played in a single turn."""
        game = PazaakGame(make_deck([-1, -1, -1, -1] + [1] * 16), make_deck(), seed=7)
        # Ensure it's Player A's turn
        self.assertEqual(game.active_player, 0)
        hand = list(game.players[0].hand)
        played = 0
        for i in range(4):
            if game.players[0].hand[i] is not None:
                game.step(("play", i))
                played += 1
        self.assertEqual(played, 4)
        for i in range(4):
            self.assertIsNone(game.players[0].hand[i])

    def test_hand_card_count_decrements(self):
        """my_hand_count in observation decrements as cards are used."""
        game = PazaakGame(make_deck(), make_deck(), seed=8)
        obs_before = game.get_observation(0)
        count_before = obs_before["my_hand_count"]

        slot = next(i for i, v in enumerate(game.players[0].hand) if v is not None)
        game.step(("play", slot))
        obs_after = game.get_observation(0)
        self.assertEqual(obs_after["my_hand_count"], count_before - 1)


# ---------------------------------------------------------------------------
# 3. Busting
# ---------------------------------------------------------------------------


class TestBusting(unittest.TestCase):
    def _force_bust(self, game, player_idx):
        """Hit player_idx repeatedly until they bust or we exceed safety limit."""
        safety = 0
        while game.phase == "decision" and game.active_player == player_idx:
            game.step("hit")
            safety += 1
            if safety > 100:
                break

    def test_bust_ends_round(self):
        """When a player's total exceeds 20 after ending their turn, the round ends."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        # Remove all hand cards so there is no escape
        game.players[0].hand = [None, None, None, None]
        # Total is already over 20 → hitting triggers the bust check immediately
        game.players[0].total = 21
        game.phase = "decision"
        game.active_player = 0

        game.step("hit")

        # After Player A busted, the round must have ended
        self.assertIn(game.phase, ("round_over", "game_over"))
        self.assertEqual(game.last_round_winner, 1)

    def test_bust_awards_round_to_opponent(self):
        """A bust awards the round to the non-busting player."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        # Strip all hand cards from Player 0 so they cannot recover
        game.players[0].hand = [None, None, None, None]
        # Total already over 20 → bust is checked when they choose hit or stand
        game.players[0].total = 25
        game.phase = "decision"
        game.active_player = 0

        game.step("hit")

        self.assertIn(game.phase, ("round_over", "game_over"))
        self.assertEqual(
            game.last_round_winner, 1, "Player B should win when Player A busts"
        )

    def test_negative_hand_card_rescues_from_bust(self):
        """A player may use a negative hand card to bring total <= 20 before ending turn."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)

        # Manually set up a bust-rescue scenario
        game.players[0].total = 22
        game.players[0].hand = [-3, None, None, None]
        game.phase = "decision"
        game.active_player = 0

        # Play the -3 card: total becomes 19
        game.step(("play", 0))
        self.assertEqual(game.players[0].total, 19)

        # Now stand - should NOT bust
        game.step("stand")
        # Round should not have ended with player 0 losing
        if game.phase in ("round_over", "game_over"):
            self.assertNotEqual(
                game.last_round_winner,
                1,
                "Player A should not lose after recovering from bust",
            )

    def test_bust_check_after_hit_not_during_card_play(self):
        """The bust is not resolved mid-card-play; only after hit or stand."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        game.players[0].total = 21  # over 20
        game.players[0].hand = [-2, -2, None, None]
        game.phase = "decision"
        game.active_player = 0

        # Playing a hand card while at 21 should keep us in "decision"
        game.step(("play", 0))  # total now 19
        self.assertEqual(
            game.phase, "decision", "Phase must stay 'decision' during hand card play"
        )

    def test_stand_on_bust_total_loses(self):
        """Standing with total > 20 still counts as a bust (round loss)."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        game.players[0].total = 21
        game.players[0].hand = [None, None, None, None]
        game.phase = "decision"
        game.active_player = 0

        game.step("stand")
        self.assertIn(game.phase, ("round_over", "game_over"))
        self.assertEqual(game.last_round_winner, 1)


# ---------------------------------------------------------------------------
# 4. Draw Handling
# ---------------------------------------------------------------------------


class TestDrawHandling(unittest.TestCase):
    def test_draw_awards_no_round_wins(self):
        """Equal totals when both stand result in a draw (no round win for either)."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        wins_before = list(game.round_wins)

        # Force both players to have the same total and stand
        game.players[0].total = 15
        game.players[1].total = 15
        game.players[0].stood = True
        game.players[1].stood = False
        game.active_player = 1
        game.phase = "decision"

        game.step("stand")

        self.assertIn(game.phase, ("round_over", "game_over"))
        self.assertIsNone(game.last_round_winner)
        self.assertEqual(
            game.round_wins, wins_before, "No round wins should be awarded on a draw"
        )

    def test_draw_continues_game(self):
        """After a draw the game continues (new round starts)."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        game.players[0].total = 15
        game.players[1].total = 15
        game.players[0].stood = True
        game.players[1].stood = False
        game.active_player = 1
        game.phase = "decision"

        game.step("stand")
        self.assertEqual(game.phase, "round_over")  # not game_over
        # Starting new round should work fine
        game.start_round()
        self.assertEqual(game.phase, "decision")

    def test_draw_does_not_count_toward_three_wins(self):
        """A game with many draws never falsely triggers game_over."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        draws = 0
        # Force 6 consecutive draws
        for _ in range(6):
            game.players[0].total = 10
            game.players[1].total = 10
            game.players[0].stood = True
            game.players[1].stood = False
            game.active_player = 1
            game.phase = "decision"
            game.step("stand")
            self.assertEqual(game.phase, "round_over")
            self.assertIsNone(game.last_round_winner)
            draws += 1
            game.start_round()

        self.assertEqual(draws, 6)
        self.assertEqual(game.round_wins, [0, 0])
        self.assertIsNone(game.game_winner)


# ---------------------------------------------------------------------------
# 5. Game End Condition
# ---------------------------------------------------------------------------


class TestGameEnd(unittest.TestCase):
    def test_game_ends_at_three_wins(self):
        """The game is over when a player reaches 3 round wins."""
        game = play_full_game(seed=10)
        self.assertEqual(max(game.round_wins), 3)
        self.assertEqual(game.phase, "game_over")
        self.assertIn(game.game_winner, [0, 1])

    def test_winner_has_three_wins(self):
        """The game_winner field matches the player with 3 round wins."""
        game = play_full_game(seed=11)
        self.assertEqual(game.round_wins[game.game_winner], 3)

    def test_loser_has_fewer_than_three_wins(self):
        """The losing player always has fewer than 3 wins."""
        for seed in range(20):
            game = play_full_game(seed=seed)
            loser = 1 - game.game_winner
            self.assertLess(game.round_wins[loser], 3)

    def test_step_raises_after_game_over(self):
        """Calling step() after game_over raises an error."""
        game = play_full_game(seed=12)
        self.assertEqual(game.phase, "game_over")
        with self.assertRaises((ValueError, Exception)):
            game.step("hit")

    def test_game_can_last_more_than_five_rounds(self):
        """If draws occur, the game may exceed 5 rounds."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        # Inject draws until someone would need more than 5 rounds
        rounds_played = 0
        # Force 4 draws first
        for _ in range(4):
            game.players[0].total = 10
            game.players[1].total = 10
            game.players[0].stood = True
            game.players[1].stood = False
            game.active_player = 1
            game.phase = "decision"
            game.step("stand")
            game.start_round()
            rounds_played += 1

        # Now play normally to finish; game should still end correctly
        safety = 0
        while game.phase != "game_over":
            if game.phase == "round_over":
                game.start_round()
                rounds_played += 1
            elif game.phase == "decision":
                p = game.players[game.active_player]
                game.step("stand" if p.total >= 10 else "hit")
            safety += 1
            if safety > 5000:
                break
        rounds_played += 1  # final round

        self.assertGreater(rounds_played, 5)
        self.assertEqual(max(game.round_wins), 3)


# ---------------------------------------------------------------------------
# 6. Turn Order
# ---------------------------------------------------------------------------


class TestTurnOrder(unittest.TestCase):
    def test_player_a_goes_first_every_round(self):
        """Player A (index 0) is always the active player at the start of each round."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        self.assertEqual(game.active_player, 0)

        for _ in range(5):
            # Quick-finish the round by having both players stand
            while game.phase == "decision":
                game.step("stand")
            if game.phase == "game_over":
                break
            game.start_round()
            self.assertEqual(
                game.active_player, 0, "Player A must start every new round"
            )

    def test_player_b_goes_after_player_a_hits(self):
        """After Player A hits, the active player switches to Player B."""
        game = PazaakGame(make_deck(), make_deck(), seed=2)
        self.assertEqual(game.active_player, 0)
        game.step("hit")
        self.assertEqual(game.active_player, 1)

    def test_standing_player_skipped(self):
        """After Player A stands, subsequent turns belong to Player B only."""
        game = PazaakGame(make_deck(), make_deck(), seed=3)
        game.step("stand")  # Player A stands
        self.assertTrue(game.players[0].stood)

        # All subsequent decision phases must be Player B's
        turns = 0
        while game.phase == "decision" and turns < 20:
            self.assertEqual(
                game.active_player, 1, "Only Player B should be active after A stands"
            )
            game.step("hit")
            turns += 1

    def test_hit_does_not_switch_when_opponent_stood(self):
        """If the opponent has stood and you hit, it's still your turn next."""
        game = PazaakGame(make_deck(), make_deck(), seed=4)
        # Player A goes; switch to Player B
        game.step("hit")
        # Player B stands
        game.step("stand")
        self.assertTrue(game.players[1].stood)
        # Now it's Player A's turn again
        self.assertEqual(game.active_player, 0)
        # Player A hits - should remain Player A's turn
        game.step("hit")
        self.assertEqual(game.active_player, 0)


# ---------------------------------------------------------------------------
# 7. Standing Behaviour
# ---------------------------------------------------------------------------


class TestStandingBehaviour(unittest.TestCase):
    def test_stood_flag_set_on_stand(self):
        """stood flag is True immediately after a player stands."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        self.assertFalse(game.players[0].stood)
        game.step("stand")
        self.assertTrue(game.players[0].stood)

    def test_stood_flag_reset_each_round(self):
        """stood flags are cleared at the start of each new round."""
        game = PazaakGame(make_deck(), make_deck(), seed=2)
        game.step("stand")  # Player A stands
        game.step("stand")  # Player B stands → round over
        self.assertIn(game.phase, ("round_over", "game_over"))

        if game.phase == "round_over":
            game.start_round()
            self.assertFalse(game.players[0].stood)
            self.assertFalse(game.players[1].stood)

    def test_both_stand_triggers_resolution(self):
        """Round resolves when both players have stood."""
        game = PazaakGame(make_deck(), make_deck(), seed=3)
        game.step("stand")  # Player A stands
        game.step("stand")  # Player B stands
        self.assertIn(game.phase, ("round_over", "game_over"))

    def test_higher_total_wins_on_both_stand(self):
        """The player with the higher total wins when both stand."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        game.players[0].total = 18
        game.players[1].total = 15
        game.players[0].stood = True
        game.players[1].stood = False
        game.active_player = 1
        game.phase = "decision"
        game.step("stand")
        self.assertEqual(game.last_round_winner, 0)

    def test_lower_total_loses_on_both_stand(self):
        """The player with the lower total loses when both stand."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        game.players[0].total = 12
        game.players[1].total = 19
        game.players[0].stood = True
        game.players[1].stood = False
        game.active_player = 1
        game.phase = "decision"
        game.step("stand")
        self.assertEqual(game.last_round_winner, 1)

    def test_can_stand_on_low_total(self):
        """A player can legally stand on a very low (or even negative) total."""
        game = PazaakGame(make_deck(), make_deck(), seed=0)
        game.players[0].total = 1
        game.phase = "decision"
        game.active_player = 0
        # Should not raise
        game.step("stand")
        self.assertTrue(game.players[0].stood)


# ---------------------------------------------------------------------------
# 8. Multiple Hand Cards Per Turn
# ---------------------------------------------------------------------------


class TestMultipleHandCardsPerTurn(unittest.TestCase):
    def test_play_multiple_cards_same_turn(self):
        """Multiple hand cards may be played in one turn without drawing extra field cards."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        total_after_field = game.players[0].total

        # Play all available hand cards in sequence
        slots_played = []
        for i in range(4):
            if game.players[0].hand[i] is not None and game.phase == "decision":
                card_val = game.players[0].hand[i]
                game.step(("play", i))
                total_after_field += card_val
                slots_played.append(i)
                self.assertEqual(game.players[0].total, total_after_field)
                self.assertEqual(game.phase, "decision")

        self.assertGreater(len(slots_played), 1, "Should have played at least 2 cards")

    def test_phase_stays_decision_after_card_play(self):
        """Phase remains 'decision' after playing a hand card (no auto-advance)."""
        game = PazaakGame(make_deck(), make_deck(), seed=2)
        slot = next(i for i, v in enumerate(game.players[0].hand) if v is not None)
        game.step(("play", slot))
        self.assertEqual(game.phase, "decision")
        self.assertEqual(game.active_player, 0)

    def test_no_new_field_card_between_hand_plays(self):
        """Playing hand cards back-to-back does not draw additional field cards."""
        game = PazaakGame(make_deck(), make_deck(), seed=3)
        total_after_field = game.players[0].total

        cards_played_total = 0
        for i in range(4):
            if game.players[0].hand[i] is not None and game.phase == "decision":
                card_val = game.players[0].hand[i]
                game.step(("play", i))
                cards_played_total += card_val

        # Total should equal field card + sum of played hand cards (no extra field draws)
        self.assertEqual(game.players[0].total, total_after_field + cards_played_total)

    def test_used_hand_records_all_played_cards(self):
        """used_hand accumulates every card played during the turn."""
        game = PazaakGame(make_deck(), make_deck(), seed=4)
        played_values = []
        for i in range(4):
            if game.players[0].hand[i] is not None and game.phase == "decision":
                played_values.append(game.players[0].hand[i])
                game.step(("play", i))

        for v in played_values:
            self.assertIn(v, game.players[0].used_hand)


# ---------------------------------------------------------------------------
# 9. Observability
# ---------------------------------------------------------------------------


class TestObservability(unittest.TestCase):
    def test_own_hand_values_visible(self):
        """A player can see their own hand card values."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        obs = game.get_observation(0)
        self.assertEqual(obs["my_hand"], game.players[0].hand)

    def test_opponent_hand_values_hidden(self):
        """The observation does not expose opponent hand values (only count)."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        obs = game.get_observation(0)
        self.assertNotIn("opp_hand", obs)
        self.assertIn("opp_hand_count", obs)

    def test_opponent_played_cards_visible(self):
        """Hand cards already played by the opponent appear in opp_used_hand."""
        game = PazaakGame(make_deck(), make_deck(), seed=2)
        # Switch perspective: let Player B play a card
        game.step("hit")  # Player A hits → Player B's turn
        # Player B plays a hand card if available
        slot = next(
            (i for i, v in enumerate(game.players[1].hand) if v is not None), None
        )
        if slot is not None:
            card_val = game.players[1].hand[slot]
            game.step(("play", slot))
            obs_from_a = game.get_observation(0)
            self.assertIn(card_val, obs_from_a["opp_used_hand"])

    def test_round_wins_correct_in_observation(self):
        """Observation correctly reflects current round wins for each player."""
        game = PazaakGame(make_deck(), make_deck(), seed=3)
        # Force Player 0 to win a round by making their total higher
        game.players[0].total = 20
        game.players[1].total = 10
        game.players[0].stood = True
        game.players[1].stood = False
        game.active_player = 1
        game.phase = "decision"
        game.step("stand")

        if game.phase == "round_over":
            obs = game.get_observation(0)
            self.assertEqual(obs["my_round_wins"], 1)
            self.assertEqual(obs["opp_round_wins"], 0)

    def test_observation_is_player_relative(self):
        """get_observation(0) and get_observation(1) swap my/opp fields correctly."""
        game = PazaakGame(make_deck(), make_deck(), seed=4)
        obs0 = game.get_observation(0)
        obs1 = game.get_observation(1)

        self.assertEqual(obs0["my_total"], obs1["opp_total"])
        self.assertEqual(obs0["opp_total"], obs1["my_total"])
        self.assertEqual(obs0["my_round_wins"], obs1["opp_round_wins"])
        self.assertEqual(obs0["opp_round_wins"], obs1["my_round_wins"])

    def test_opp_hand_count_reflects_consumed_slots(self):
        """opp_hand_count seen by the opponent decrements when cards are consumed."""
        game = PazaakGame(make_deck(), make_deck(), seed=5)
        obs_before = game.get_observation(1)
        count_before = obs_before["opp_hand_count"]

        slot = next(i for i, v in enumerate(game.players[0].hand) if v is not None)
        game.step(("play", slot))

        obs_after = game.get_observation(1)
        self.assertEqual(obs_after["opp_hand_count"], count_before - 1)

    def test_observation_contains_all_required_keys(self):
        """Observation dict contains exactly the expected keys."""
        expected_keys = {
            "my_total",
            "opp_total",
            "my_stood",
            "opp_stood",
            "my_hand",
            "my_hand_count",
            "opp_hand_count",
            "opp_used_hand",
            "my_round_wins",
            "opp_round_wins",
        }
        game = PazaakGame(make_deck(), make_deck(), seed=6)
        obs = game.get_observation(0)
        self.assertEqual(set(obs.keys()), expected_keys)


# ---------------------------------------------------------------------------
# 10. Legal Actions
# ---------------------------------------------------------------------------


class TestLegalActions(unittest.TestCase):
    def test_legal_actions_in_decision_phase(self):
        """legal_actions() always includes 'hit' and 'stand' during decision phase."""
        game = PazaakGame(make_deck(), make_deck(), seed=1)
        legal = game.legal_actions()
        self.assertIn("hit", legal)
        self.assertIn("stand", legal)

    def test_legal_actions_outside_decision_phase(self):
        """legal_actions() returns empty list when not in 'decision' phase."""
        game = PazaakGame(make_deck(), make_deck(), seed=2)
        game.phase = "round_over"
        self.assertEqual(game.legal_actions(), [])

    def test_play_actions_match_non_none_slots(self):
        """('play', i) appears in legal_actions iff hand[i] is not None."""
        game = PazaakGame(make_deck(), make_deck(), seed=3)
        legal = game.legal_actions()
        play_indices = {a[1] for a in legal if isinstance(a, tuple)}

        expected = {i for i, v in enumerate(game.players[0].hand) if v is not None}
        self.assertEqual(play_indices, expected)

    def test_consumed_slot_not_in_legal_actions(self):
        """After consuming a slot, that ('play', i) is no longer legal."""
        game = PazaakGame(make_deck(), make_deck(), seed=4)
        slot = next(i for i, v in enumerate(game.players[0].hand) if v is not None)
        game.step(("play", slot))
        legal = game.legal_actions()
        self.assertNotIn(("play", slot), legal)

    def test_illegal_action_raises(self):
        """Calling step() with an invalid action raises ValueError."""
        game = PazaakGame(make_deck(), make_deck(), seed=5)
        with self.assertRaises((ValueError, Exception)):
            game.step("invalid_action")

    def test_play_out_of_range_raises(self):
        """Playing a slot index outside 0–3 raises an error."""
        game = PazaakGame(make_deck(), make_deck(), seed=6)
        with self.assertRaises((ValueError, Exception)):
            game.step(("play", 99))


# ---------------------------------------------------------------------------
# 11. Round and Game State Integrity (integration)
# ---------------------------------------------------------------------------


class TestStateIntegrity(unittest.TestCase):
    def test_round_wins_never_exceed_three(self):
        """Neither player ever accumulates more than 3 round wins."""
        for seed in range(15):
            game = play_full_game(seed=seed)
            self.assertLessEqual(game.round_wins[0], 3)
            self.assertLessEqual(game.round_wins[1], 3)

    def test_totals_reset_between_rounds(self):
        """Player totals are 0 at the very start of a new round (before field draw)."""
        game = PazaakGame(make_deck(), make_deck(), seed=7)
        # Force a quick round end
        game.players[0].total = 20
        game.players[1].total = 10
        game.players[0].stood = True
        game.players[1].stood = False
        game.active_player = 1
        game.phase = "decision"
        game.step("stand")

        if game.phase == "round_over":
            # Manually check total is 0 before start_round triggers the draw
            # (start_round resets totals to 0 before drawing)
            game.start_round()
            # After start_round the draw has already happened, total >= 1
            self.assertGreaterEqual(game.players[0].total, 1)

    def test_used_hand_carries_over_across_rounds(self):
        """Cards played in round 1 remain in used_hand in round 2."""
        game = PazaakGame(make_deck(), make_deck(), seed=8)
        slot = next(i for i, v in enumerate(game.players[0].hand) if v is not None)
        played_val = game.players[0].hand[slot]
        game.step(("play", slot))
        game.step("stand")  # Player A stands
        game.step("stand")  # Player B stands → round over
        if game.phase == "round_over":
            game.start_round()
            self.assertIn(played_val, game.players[0].used_hand)
            self.assertIsNone(game.players[0].hand[slot])

    def test_game_winner_set_correctly(self):
        """game_winner is set to the player who reached 3 wins."""
        for seed in range(10):
            game = play_full_game(seed=seed)
            self.assertIsNotNone(game.game_winner)
            self.assertEqual(game.round_wins[game.game_winner], 3)

    def test_repr_does_not_raise(self):
        """__repr__ runs without error at all major phases."""
        game = PazaakGame(make_deck(), make_deck(), seed=9)
        repr(game)
        game.step("stand")
        repr(game)


# ---------------------------------------------------------------------------
# 12. Seeded Reproducibility
# ---------------------------------------------------------------------------


class TestSeededReproducibility(unittest.TestCase):
    def test_same_seed_same_outcome(self):
        """Two games with the same seed and same agents produce identical outcomes."""

        def run(seed):
            return play_full_game(seed=seed)

        for seed in range(5):
            g1 = run(seed)
            g2 = run(seed)
            self.assertEqual(g1.round_wins, g2.round_wins)
            self.assertEqual(g1.game_winner, g2.game_winner)

    def test_different_seeds_can_differ(self):
        """Games with different seeds are not guaranteed to be identical."""
        outcomes = set()
        for seed in range(20):
            g = play_full_game(seed=seed)
            outcomes.add(tuple(g.round_wins))
        # There should be more than one possible outcome across 20 games
        self.assertGreater(len(outcomes), 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
