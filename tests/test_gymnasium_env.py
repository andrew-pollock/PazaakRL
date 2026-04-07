"""
test_gymnasium_env.py - Basic checks for PazaakGymnasiumEnv.

Run with:  python -m pytest test_gymnasium_env.py -v
       or:  python -m pytest tests/test_gymnasium_env.py -v

gymnasium is not available in this environment, so a minimal stub is
installed into sys.modules before any imports from gymnasium_env.
"""

import sys
import types
import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Minimal gymnasium stub (installed before importing gymnasium_env)
# ---------------------------------------------------------------------------


def _install_gymnasium_stub():
    gym_stub = types.ModuleType("gymnasium")
    gym_spaces = types.ModuleType("gymnasium.spaces")

    class Discrete:
        def __init__(self, n):
            self.n = n

    class Box:
        def __init__(self, low, high, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self._low = low
            self._high = high

        def contains(self, x):
            return (
                x.shape == self.shape
                and np.all(x >= self._low)
                and np.all(x <= self._high)
            )

    class Env:
        """Minimal base-class stub."""

        def reset(self, *, seed=None, options=None):
            pass

        def step(self, action): ...
        def render(self): ...
        def close(self): ...

    gym_spaces.Discrete = Discrete
    gym_spaces.Box = Box
    gym_stub.spaces = gym_spaces
    gym_stub.Env = Env
    sys.modules["gymnasium"] = gym_stub
    sys.modules["gymnasium.spaces"] = gym_spaces


_install_gymnasium_stub()

# Now safe to import our modules
from pazaakrl.game_engine import PazaakGame  # noqa: E402
from pazaakrl.heuristic import simple_heuristic_agent, heuristic_agent  # noqa: E402
from pazaakrl.gymnasium_env import (  # noqa: E402
    PazaakGymnasiumEnv,
    observation_to_array,
    int_to_action,
    action_to_int,
    mask_fn,
    make_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_deck(values=None):
    base = values or (list(range(-6, 0)) + list(range(1, 7)))
    deck = []
    while len(deck) < 20:
        deck.extend(base)
    return deck[:20]


def fresh_env(opponent=None, seed=None):
    """Return a reset PazaakGymnasiumEnv."""
    env = PazaakGymnasiumEnv(
        opponent_agent=opponent or simple_heuristic_agent,
        seed=seed,
    )
    env.reset()
    return env


def run_episode(env, rng=None):
    """
    Play one full episode to completion using random legal actions.
    Returns (total_reward, game_winner).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    total_r = 0.0
    done = False
    steps = 0
    while not done:
        legal = np.where(env.action_masks())[0]
        a = int(rng.choice(legal))
        _, r, term, trunc, _ = env.step(a)
        total_r += r
        done = term or trunc
        steps += 1
        assert steps < 10_000, "Episode step limit exceeded"
    return total_r, env.game.game_winner


# ---------------------------------------------------------------------------
# 1. Spaces
# ---------------------------------------------------------------------------


class TestSpaces(unittest.TestCase):
    def setUp(self):
        self.env = fresh_env()

    def test_action_space_size(self):
        self.assertEqual(self.env.action_space.n, 6)

    def test_observation_space_shape(self):
        self.assertEqual(self.env.observation_space.shape, (33,))

    def test_observation_space_dtype(self):
        self.assertEqual(self.env.observation_space.dtype, np.float32)

    def test_reset_obs_shape(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.shape, (33,))

    def test_reset_obs_dtype(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.dtype, np.float32)

    def test_reset_obs_finite(self):
        obs, _ = self.env.reset()
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_reset_obs_in_space(self):
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------


class TestReset(unittest.TestCase):
    def test_reset_returns_obs_and_info(self):
        env = PazaakGymnasiumEnv(opponent_agent=simple_heuristic_agent)
        result = env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_reset_info_keys(self):
        env = fresh_env()
        _, info = env.reset()
        for key in ("phase", "round_wins", "active_player", "game_winner"):
            self.assertIn(key, info)

    def test_reset_starts_in_decision_phase(self):
        env = fresh_env()
        self.assertEqual(env.game.phase, "decision")

    def test_reset_starts_player_a(self):
        env = fresh_env()
        self.assertEqual(env.game.active_player, 0)

    def test_reset_round_wins_zeroed(self):
        env = fresh_env()
        self.assertEqual(env.game.round_wins, [0, 0])

    def test_reset_clears_mid_game_state(self):
        """Resetting mid-game produces a clean slate."""
        env = fresh_env()
        for _ in range(8):
            if env.game.phase == "decision":
                env.step(0)  # hit a few times
        env.reset()
        self.assertEqual(env.game.round_wins, [0, 0])
        self.assertEqual(env.game.phase, "decision")
        self.assertEqual(env.game.active_player, 0)

    def test_reset_before_step_does_not_raise(self):
        env = PazaakGymnasiumEnv(opponent_agent=simple_heuristic_agent)
        env.reset()  # must not raise

    def test_step_without_reset_raises(self):
        env = PazaakGymnasiumEnv(opponent_agent=simple_heuristic_agent)
        with self.assertRaises((RuntimeError, AttributeError, Exception)):
            env.step(0)


# ---------------------------------------------------------------------------
# 3. Action masking
# ---------------------------------------------------------------------------


class TestActionMasking(unittest.TestCase):
    def setUp(self):
        self.env = fresh_env()

    def test_mask_shape(self):
        self.assertEqual(self.env.action_masks().shape, (6,))

    def test_mask_dtype(self):
        self.assertEqual(self.env.action_masks().dtype, bool)

    def test_hit_always_legal(self):
        self.assertTrue(self.env.action_masks()[0])

    def test_stand_always_legal(self):
        self.assertTrue(self.env.action_masks()[1])

    def test_play_slots_match_hand(self):
        """Slots 2-5 are legal iff the corresponding hand slot is not None."""
        mask = self.env.action_masks()
        hand = self.env.game.players[0].hand
        for i in range(4):
            self.assertEqual(
                mask[2 + i],
                hand[i] is not None,
                f"Slot {i}: hand={hand[i]}, mask={mask[2 + i]}",
            )

    def test_consumed_slot_masked_out(self):
        slot = next(
            i for i, v in enumerate(self.env.game.players[0].hand) if v is not None
        )
        self.env.step(2 + slot)
        self.assertFalse(self.env.action_masks()[2 + slot])

    def test_mask_fn_matches_action_masks(self):
        np.testing.assert_array_equal(mask_fn(self.env), self.env.action_masks())

    def test_mask_has_at_least_two_legal_actions(self):
        """Hit and stand are always available."""
        self.assertGreaterEqual(self.env.action_masks().sum(), 2)

    def test_mask_empty_hand_still_has_hit_stand(self):
        """With all hand slots consumed, only hit and stand remain."""
        self.env.game.players[0].hand = [None, None, None, None]
        mask = self.env.action_masks()
        self.assertTrue(mask[0])  # hit
        self.assertTrue(mask[1])  # stand
        self.assertFalse(any(mask[2:]))


# ---------------------------------------------------------------------------
# 4. Action conversion helpers
# ---------------------------------------------------------------------------


class TestActionConversion(unittest.TestCase):
    _mapping = [
        (0, "hit"),
        (1, "stand"),
        (2, ("play", 0)),
        (3, ("play", 1)),
        (4, ("play", 2)),
        (5, ("play", 3)),
    ]

    def test_int_to_action(self):
        for i, expected in self._mapping:
            with self.subTest(i=i):
                self.assertEqual(int_to_action(i), expected)

    def test_action_to_int(self):
        for i, action in self._mapping:
            with self.subTest(action=action):
                self.assertEqual(action_to_int(action), i)

    def test_round_trip_int_to_action(self):
        for i, _ in self._mapping:
            self.assertEqual(action_to_int(int_to_action(i)), i)

    def test_invalid_int_raises(self):
        with self.assertRaises(ValueError):
            int_to_action(99)

    def test_invalid_action_raises(self):
        with self.assertRaises(ValueError):
            action_to_int("invalid")


# ---------------------------------------------------------------------------
# 5. Step - hand card play
# ---------------------------------------------------------------------------


class TestStepHandCard(unittest.TestCase):
    def setUp(self):
        self.env = fresh_env()
        self.slot = next(
            i for i, v in enumerate(self.env.game.players[0].hand) if v is not None
        )

    def test_hand_card_step_reward_zero(self):
        _, r, _, _, _ = self.env.step(2 + self.slot)
        self.assertEqual(r, 0.0)

    def test_hand_card_step_not_terminated(self):
        _, _, term, trunc, _ = self.env.step(2 + self.slot)
        self.assertFalse(term)
        self.assertFalse(trunc)

    def test_hand_card_step_stays_agent_turn(self):
        self.env.step(2 + self.slot)
        self.assertEqual(self.env.game.active_player, 0)

    def test_hand_card_step_stays_decision_phase(self):
        self.env.step(2 + self.slot)
        self.assertEqual(self.env.game.phase, "decision")

    def test_hand_card_step_returns_valid_obs(self):
        obs, _, _, _, _ = self.env.step(2 + self.slot)
        self.assertEqual(obs.shape, (33,))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_hand_card_updates_total(self):
        total_before = self.env.game.players[0].total
        card_val = self.env.game.players[0].hand[self.slot]
        self.env.step(2 + self.slot)
        self.assertEqual(self.env.game.players[0].total, total_before + card_val)

    def test_hand_card_slot_consumed(self):
        self.env.step(2 + self.slot)
        self.assertIsNone(self.env.game.players[0].hand[self.slot])


# ---------------------------------------------------------------------------
# 6. Step - hit and stand
# ---------------------------------------------------------------------------


class TestStepHitStand(unittest.TestCase):
    def test_hit_returns_to_agent_turn(self):
        env = fresh_env()
        _, _, term, _, _ = env.step(0)  # hit
        if not term:
            self.assertEqual(env.game.active_player, 0)
            self.assertEqual(env.game.phase, "decision")

    def test_stand_returns_valid_obs(self):
        env = fresh_env()
        obs, _, _, _, _ = env.step(1)  # stand
        self.assertEqual(obs.shape, (33,))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_hit_returns_valid_obs(self):
        env = fresh_env()
        obs, _, _, _, _ = env.step(0)
        self.assertEqual(obs.shape, (33,))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_agent_never_sees_opponent_turn(self):
        """After any hit/stand step, the returned state is always the agent's turn."""
        env = fresh_env()
        rng = np.random.default_rng(7)
        for _ in range(30):
            if env.game.phase != "decision":
                break
            legal = np.where(env.action_masks())[0]
            a = int(rng.choice(legal))
            _, _, term, _, _ = env.step(a)
            if term:
                break
            self.assertEqual(env.game.active_player, 0)
            self.assertEqual(env.game.phase, "decision")


# ---------------------------------------------------------------------------
# 7. Reward shaping
# ---------------------------------------------------------------------------


class TestRewardShaping(unittest.TestCase):
    def _setup_state(self, my_total, opp_total, round_wins=(0, 0), opp_stood=True):
        """Return an env manually placed in the given decision state."""
        env = PazaakGymnasiumEnv(opponent_agent=simple_heuristic_agent)
        env.reset()
        env.game.round_wins = list(round_wins)
        env.game.players[0].total = my_total
        env.game.players[1].total = opp_total
        env.game.players[0].hand = [None] * 4
        env.game.players[1].hand = [None] * 4
        env.game.players[0].stood = False
        env.game.players[1].stood = opp_stood
        env.game.active_player = 0
        env.game.phase = "decision"
        return env

    def test_mid_game_round_win_reward(self):
        """Winning a non-final round gives +0.3 plus shaping bonus for standing on 20."""
        env = self._setup_state(20, 10, round_wins=(0, 0))
        _, r, term, _, _ = env.step(1)  # agent stands on 20 → wins round
        self.assertFalse(term)
        # +0.3 round win + 0.05 stand 18-20 + 0.05 stand on 20 = 0.4
        self.assertAlmostEqual(r, 0.4)

    def test_mid_game_round_loss_reward(self):
        """Losing a non-final round gives -0.3 (no shaping bonus for standing on 10)."""
        env = self._setup_state(10, 20, round_wins=(0, 0))
        _, r, term, _, _ = env.step(1)
        self.assertFalse(term)
        self.assertAlmostEqual(r, -0.3)

    def test_round_draw_reward(self):
        """A drawn round gives shaping bonus only (stand on 15 = no bonus)."""
        env = self._setup_state(15, 15, round_wins=(0, 0))
        _, r, term, _, _ = env.step(1)
        self.assertFalse(term)
        self.assertAlmostEqual(r, 0.0)

    def test_game_win_reward(self):
        """Winning the final round terminates with game reward + shaping."""
        env = self._setup_state(20, 10, round_wins=(2, 2))
        _, r, term, _, _ = env.step(1)
        self.assertTrue(term)
        # +1.0 game win + 0.05 stand 18-20 + 0.05 stand on 20 = 1.1
        # (round_over is skipped when the round ends the game)
        self.assertAlmostEqual(r, 1.1)

    def test_game_loss_reward(self):
        """Losing the final round terminates with game loss reward."""
        env = self._setup_state(10, 20, round_wins=(2, 2))
        _, r, term, _, _ = env.step(1)
        self.assertTrue(term)
        # -1.0 game loss (no shaping for standing on 10, no round_over for final round)
        self.assertAlmostEqual(r, -1.0)

    def test_hand_card_step_reward(self):
        """Playing a hand card yields 0.0 unless it reaches exactly 20."""
        env = fresh_env()
        # Find a slot that won't reach exactly 20 (most won't)
        for i, v in enumerate(env.game.players[0].hand):
            if v is not None and env.game.players[0].total + v != 20:
                _, r, _, _, _ = env.step(2 + i)
                self.assertEqual(r, 0.0)
                return
        # If all cards reach 20 (extremely unlikely), skip this test
        self.skipTest("All hand cards reach 20 in this seed")

    def test_total_episode_reward_in_range(self):
        """Over a full episode the total reward is within a reasonable range."""
        for seed in range(20):
            env = fresh_env(seed=seed)
            total_r, _ = run_episode(env, rng=np.random.default_rng(seed))
            # With shaping rewards, the range is wider than [-2, 2]
            self.assertGreaterEqual(total_r, -3.0)
            self.assertLessEqual(total_r, 3.0)


# ---------------------------------------------------------------------------
# 8. Episode lifecycle
# ---------------------------------------------------------------------------


class TestEpisodeLifecycle(unittest.TestCase):
    def test_episode_terminates(self):
        """Every episode eventually returns terminated=True."""
        for seed in range(15):
            env = fresh_env(seed=seed)
            _, winner = run_episode(env, rng=np.random.default_rng(seed))
            self.assertEqual(env.game.phase, "game_over")

    def test_game_winner_is_set_on_termination(self):
        for seed in range(10):
            env = fresh_env(seed=seed)
            _, winner = run_episode(env, rng=np.random.default_rng(seed))
            self.assertIn(winner, (0, 1))

    def test_winner_has_three_round_wins(self):
        for seed in range(10):
            env = fresh_env(seed=seed)
            _, winner = run_episode(env, rng=np.random.default_rng(seed))
            self.assertEqual(env.game.round_wins[winner], 3)

    def test_no_step_after_termination(self):
        """Stepping into a terminated env raises an error."""
        env = fresh_env()
        done = False
        while not done:
            legal = np.where(env.action_masks())[0]
            _, _, term, trunc, _ = env.step(int(np.random.choice(legal)))
            done = term or trunc
        with self.assertRaises(Exception):
            env.step(0)

    def test_reset_after_episode_works(self):
        """Calling reset() after a completed episode produces a fresh game."""
        env = fresh_env()
        run_episode(env)
        env.reset()
        self.assertEqual(env.game.round_wins, [0, 0])
        self.assertEqual(env.game.phase, "decision")
        self.assertEqual(env.game.active_player, 0)

    def test_multiple_rounds_within_episode(self):
        """A game contains at least 3 rounds (each player needs 3 wins)."""
        env = fresh_env(seed=3)
        run_episode(env)
        total_rounds = sum(env.game.round_wins)
        self.assertGreaterEqual(total_rounds, 3)

    def test_hand_cards_persist_across_rounds(self):
        """Slots consumed in one round remain None in the next."""
        env = fresh_env()
        # Play all hand cards in the first turn
        for i in range(4):
            if env.game.players[0].hand[i] is not None and env.game.phase == "decision":
                env.step(2 + i)
        consumed = [i for i in range(4) if env.game.players[0].hand[i] is None]

        # Play through to the next round
        safety = 0
        initial_round_wins = sum(env.game.round_wins)
        while sum(env.game.round_wins) == initial_round_wins:
            if env.game.phase == "decision" and env.game.active_player == 0:
                env.step(1)  # stand
            safety += 1
            if safety > 200 or env.game.phase == "game_over":
                break

        for i in consumed:
            self.assertIsNone(
                env.game.players[0].hand[i],
                f"Slot {i} should still be None after round transition",
            )


# ---------------------------------------------------------------------------
# 9. observation_to_array
# ---------------------------------------------------------------------------


class TestObservationToArray(unittest.TestCase):
    def _raw_obs(self):
        game = PazaakGame(make_deck(), make_deck(), seed=42)
        return game.get_observation(0)

    def test_shape(self):
        self.assertEqual(observation_to_array(self._raw_obs()).shape, (33,))

    def test_dtype(self):
        self.assertEqual(observation_to_array(self._raw_obs()).dtype, np.float32)

    def test_my_total_normalised(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        self.assertAlmostEqual(arr[0], raw["my_total"] / 20.0)

    def test_opp_total_normalised(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        self.assertAlmostEqual(arr[1], raw["opp_total"] / 20.0)

    def test_stood_flags(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        self.assertEqual(arr[2], float(raw["my_stood"]))
        self.assertEqual(arr[3], float(raw["opp_stood"]))

    def test_hand_count_normalised(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        self.assertAlmostEqual(arr[4], raw["my_hand_count"] / 4.0)
        self.assertAlmostEqual(arr[5], raw["opp_hand_count"] / 4.0)

    def test_round_wins_normalised(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        self.assertAlmostEqual(arr[6], raw["my_round_wins"] / 3.0)
        self.assertAlmostEqual(arr[7], raw["opp_round_wins"] / 3.0)

    def test_hand_slots_normalised(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        for i in range(4):
            v = raw["my_hand"][i]
            expected = (v if v is not None else 0) / 6.0
            self.assertAlmostEqual(arr[8 + i], expected, places=6)

    def test_opp_used_hand_normalised(self):
        raw = self._raw_obs()
        arr = observation_to_array(raw)
        used = raw["opp_used_hand"]
        for i in range(4):
            expected = (used[i] if i < len(used) else 0) / 6.0
            self.assertAlmostEqual(arr[12 + i], expected, places=6)

    def test_none_slots_encode_as_zero(self):
        raw = self._raw_obs()
        raw["my_hand"] = [None, 3, None, -2]
        arr = observation_to_array(raw)
        self.assertEqual(arr[8], 0.0)
        self.assertAlmostEqual(arr[9], 3 / 6.0)
        self.assertEqual(arr[10], 0.0)
        self.assertAlmostEqual(arr[11], -2 / 6.0)

    def test_player_perspective_swap(self):
        """observation_to_array on P0 obs and P1 obs swap my/opp correctly."""
        game = PazaakGame(make_deck(), make_deck(), seed=5)
        arr0 = observation_to_array(game.get_observation(0))
        arr1 = observation_to_array(game.get_observation(1))
        # Index 0 (my_total for P0) should equal index 1 (opp_total for P1)
        self.assertAlmostEqual(arr0[0], arr1[1])
        self.assertAlmostEqual(arr0[1], arr1[0])
        self.assertAlmostEqual(arr0[6], arr1[7])
        self.assertAlmostEqual(arr0[7], arr1[6])


# ---------------------------------------------------------------------------
# 10. Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility(unittest.TestCase):
    def _run(self, seed):
        rng = np.random.default_rng(seed + 5000)
        env = PazaakGymnasiumEnv(opponent_agent=simple_heuristic_agent, seed=seed)
        env.reset()
        total_r = 0.0
        done = False
        while not done:
            legal = np.where(env.action_masks())[0]
            a = int(rng.choice(legal))
            _, r, term, trunc, _ = env.step(a)
            total_r += r
            done = term or trunc
        return total_r, env.game.game_winner

    def test_same_seed_same_winner(self):
        for seed in range(5):
            _, w1 = self._run(seed)
            _, w2 = self._run(seed)
            self.assertEqual(w1, w2, f"seed={seed}: {w1} vs {w2}")

    def test_same_seed_same_reward(self):
        for seed in range(5):
            r1, _ = self._run(seed)
            r2, _ = self._run(seed)
            self.assertAlmostEqual(r1, r2, places=6, msg=f"seed={seed}")

    def test_different_seeds_vary(self):
        winners = {self._run(s)[1] for s in range(20)}
        self.assertGreater(
            len(winners),
            1,
            "All seeds produced the same winner - suspiciously deterministic",
        )


# ---------------------------------------------------------------------------
# 11. make_env factory
# ---------------------------------------------------------------------------


class TestMakeEnv(unittest.TestCase):
    def test_raw_env_resets(self):
        env = make_env(wrap_for_maskable_ppo=False)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (33,))

    def test_raw_env_has_action_masks(self):
        env = make_env(wrap_for_maskable_ppo=False)
        env.reset()
        self.assertTrue(hasattr(env, "action_masks"))

    def test_custom_opponent_used(self):
        """make_env passes the opponent_agent through correctly."""
        env = make_env(
            opponent_agent=heuristic_agent,
            wrap_for_maskable_ppo=False,
        )
        env.reset()
        self.assertIs(env.opponent_agent, heuristic_agent)

    def test_default_opponent_is_simple_heuristic(self):
        env = make_env(wrap_for_maskable_ppo=False)
        env.reset()
        from pazaakrl.heuristic import simple_heuristic_agent as sha

        self.assertIs(env.opponent_agent, sha)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
