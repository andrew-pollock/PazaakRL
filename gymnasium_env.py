"""
gymnasium_env.py - Gymnasium wrapper for the Pazaak game engine.

Wraps PazaakGame as a standard gymnasium.Env for RL training with
sb3_contrib.MaskablePPO.

Episode boundary
----------------
One episode = one full game (first player to 3 round wins).
``terminated=True`` is only returned when the game is over.
Between rounds the environment continues seamlessly.

Action space (Discrete 6)
--------------------------
  0  →  hit
  1  →  stand
  2  →  play hand card at slot 0
  3  →  play hand card at slot 1
  4  →  play hand card at slot 2
  5  →  play hand card at slot 3

Invalid actions are masked via ``action_masks()``.

Observation space (Box, 16 floats)
-----------------------------------
  [0]   my_total          / 20
  [1]   opp_total         / 20
  [2]   my_stood          (0 or 1)
  [3]   opp_stood         (0 or 1)
  [4]   my_hand_count     / 4
  [5]   opp_hand_count    / 4
  [6]   my_round_wins     / 3
  [7]   opp_round_wins    / 3
  [8]   hand slot 0       / 6  (0 if empty)
  [9]   hand slot 1       / 6
  [10]  hand slot 2       / 6
  [11]  hand slot 3       / 6
  [12]  opp used card 0   / 6  (0 if fewer cards played)
  [13]  opp used card 1   / 6
  [14]  opp used card 2   / 6
  [15]  opp used card 3   / 6

Reward shaping
--------------
  Round win   : +0.3
  Round loss  : -0.3
  Round draw  :  0.0
  Game win    : +1.0
  Game loss   : -1.0

Rewards accumulate within a single step() call when multiple events
occur (e.g. the agent wins a round and the game ends together).

Usage
-----
Basic (for testing / manual play)::

    env = PazaakGymnasiumEnv(opponent_agent=simple_heuristic_agent)
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample(mask=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

With MaskablePPO (sb3-contrib)::

    from sb3_contrib.common.wrappers import ActionMasker
    env = ActionMasker(PazaakGymnasiumEnv(opponent_agent=heuristic_agent), mask_fn)
    model = MaskablePPO("MlpPolicy", env, ...)

A convenience factory ``make_env()`` is provided at the bottom of this
module for use with ``stable_baselines3.common.env_util.make_vec_env``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import gymnasium

from game_engine import PazaakGame


# ---------------------------------------------------------------------------
# Default side decks
# ---------------------------------------------------------------------------


def _default_side_deck() -> list[int]:
    """Balanced 20-card side deck: two copies of each value in {-6..−1, +1..+6}."""
    vals = list(range(-6, 0)) + list(range(1, 7))  # 12 distinct values
    # Repeat to fill 20 slots (12 + 8 more from the start of the list)
    deck = (vals * 2)[:20]
    return deck


# ---------------------------------------------------------------------------
# Observation helper (shared with training scripts / snapshot opponents)
# ---------------------------------------------------------------------------


def observation_to_array(obs: dict) -> np.ndarray:
    """
    Convert a ``get_observation()`` dict into the 16-float normalised array
    used as the RL input.

    This function is intentionally module-level so it can be imported and
    reused by the self-play training script when building observations for
    snapshot opponents (see implementation_plan.md §4.3.2).
    """
    hand_slots: list = obs["my_hand"]  # 4 slots: int or None
    opp_used: list = obs["opp_used_hand"]  # 0–4 values already played

    arr = np.array(
        [
            obs["my_total"] / 20.0,  # [0]
            obs["opp_total"] / 20.0,  # [1]
            float(obs["my_stood"]),  # [2]
            float(obs["opp_stood"]),  # [3]
            obs["my_hand_count"] / 4.0,  # [4]
            obs["opp_hand_count"] / 4.0,  # [5]
            obs["my_round_wins"] / 3.0,  # [6]
            obs["opp_round_wins"] / 3.0,  # [7]
            (hand_slots[0] if hand_slots[0] is not None else 0) / 6.0,  # [8]
            (hand_slots[1] if hand_slots[1] is not None else 0) / 6.0,  # [9]
            (hand_slots[2] if hand_slots[2] is not None else 0) / 6.0,  # [10]
            (hand_slots[3] if hand_slots[3] is not None else 0) / 6.0,  # [11]
            (opp_used[0] if len(opp_used) > 0 else 0) / 6.0,  # [12]
            (opp_used[1] if len(opp_used) > 1 else 0) / 6.0,  # [13]
            (opp_used[2] if len(opp_used) > 2 else 0) / 6.0,  # [14]
            (opp_used[3] if len(opp_used) > 3 else 0) / 6.0,  # [15]
        ],
        dtype=np.float32,
    )

    return arr


# ---------------------------------------------------------------------------
# Action conversion helpers
# ---------------------------------------------------------------------------


def int_to_action(action_int: int):
    """Convert a Discrete(6) integer to a game engine action."""
    if action_int == 0:
        return "hit"
    if action_int == 1:
        return "stand"
    if 2 <= action_int <= 5:
        return ("play", action_int - 2)
    raise ValueError(f"Invalid action integer: {action_int}")


def action_to_int(action) -> int:
    """Convert a game engine action to a Discrete(6) integer."""
    if action == "hit":
        return 0
    if action == "stand":
        return 1
    if isinstance(action, tuple) and action[0] == "play":
        return 2 + action[1]
    raise ValueError(f"Cannot convert action to int: {action!r}")


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class PazaakGymnasiumEnv(gymnasium.Env):
    """
    Gymnasium environment for Pazaak.

    Parameters
    ----------
    opponent_agent : callable
        Any agent with signature ``(observation: dict, game: PazaakGame) → action``.
        Defaults to ``simple_heuristic_agent`` if not supplied.
    side_deck_a : list[int], optional
        20-card side deck for the RL agent (Player 0).
        Defaults to a balanced deck of {-6..−1, +1..+6}.
    side_deck_b : list[int], optional
        20-card side deck for the opponent (Player 1).
        Defaults to the same balanced deck.
    seed : int, optional
        Fixed RNG seed.  When None (default) each reset draws a fresh seed
        from the environment's own RNG so training remains stochastic.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        opponent_agent: Optional[Callable] = None,
        side_deck_a: Optional[list[int]] = None,
        side_deck_b: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Lazy import so the module works even when heuristic.py is absent
        # (e.g. during isolated unit testing of the env).
        if opponent_agent is None:
            from heuristic import simple_heuristic_agent

            opponent_agent = simple_heuristic_agent

        self.opponent_agent: Callable = opponent_agent
        self.side_deck_a: list[int] = side_deck_a or _default_side_deck()
        self.side_deck_b: list[int] = side_deck_b or _default_side_deck()
        self._fixed_seed: Optional[int] = seed

        # Action space: 6 discrete actions (hit, stand, play slots 0-3)
        self.action_space = gymnasium.spaces.Discrete(6)

        # Observation space: 16 normalised floats
        # Most values map to [-1, 1]; totals can go slightly negative (rare).
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=2.0,  # totals > 20 are possible before bust check fires
            shape=(16,),
            dtype=np.float32,
        )

        # Game instance - populated in reset()
        self.game: Optional[PazaakGame] = None
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Determine game seed
        if self._fixed_seed is not None:
            game_seed = self._fixed_seed
        elif seed is not None:
            game_seed = seed
        else:
            game_seed = int(self._rng.integers(0, 2**31))

        self.game = PazaakGame(
            side_deck_a=list(self.side_deck_a),
            side_deck_b=list(self.side_deck_b),
            seed=game_seed,
        )

        # The game starts with Player A's first field card already drawn and
        # in "decision" phase.  If it somehow starts on the opponent's turn
        # (should never happen by the rules), run the opponent now.
        self._run_opponent_if_needed()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.game is None:
            raise RuntimeError("Call reset() before step().")

        engine_action = int_to_action(int(action))

        # Apply the agent's action to the engine
        self.game.step(engine_action)

        # --- Hand card played: return immediately, no reward, not done ---
        if isinstance(engine_action, tuple):
            return self._get_obs(), 0.0, False, False, self._get_info()

        # --- Hit or stand: drive the game until it's the agent's turn again ---
        reward = 0.0

        while True:
            # Game over?
            if self.game.phase == "game_over":
                reward += 1.0 if self.game.game_winner == 0 else -1.0
                return self._get_obs(), reward, True, False, self._get_info()

            # Round just ended (game continues)?
            if self.game.phase == "round_over":
                if self.game.last_round_winner == 0:
                    reward += 0.3
                elif self.game.last_round_winner == 1:
                    reward -= 0.3
                # else: draw - no round reward
                self.game.start_round()
                # Fall through: check whose turn it is after the new round starts

            # Agent's turn: hand back control
            if self.game.phase == "decision" and self.game.active_player == 0:
                return self._get_obs(), reward, False, False, self._get_info()

            # Opponent's turn: run all of their sequential decisions
            while self.game.phase == "decision" and self.game.active_player == 1:
                opp_obs = self.game.get_observation(player_idx=1)
                opp_action = self.opponent_agent(opp_obs, self.game)
                self.game.step(opp_action)

            # Loop back to check for round_over / game_over after opponent's turn

    def render(self) -> None:
        """Rendering is not implemented (use playback.py for human-readable output)."""
        pass

    def close(self) -> None:
        self.game = None

    # ------------------------------------------------------------------
    # Action masking (required by sb3_contrib.MaskablePPO)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask of shape (6,) indicating which actions are
        currently legal for the RL agent (Player 0).

        Must be called when the env is in a state where the agent acts
        (i.e. game.phase == "decision" and game.active_player == 0).
        """
        mask = np.zeros(6, dtype=bool)
        if self.game is None:
            return mask
        for action in self.game.legal_actions():
            if action == "hit":
                mask[0] = True
            elif action == "stand":
                mask[1] = True
            elif isinstance(action, tuple) and action[0] == "play":
                mask[2 + action[1]] = True
        return mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the 16-float observation array from the agent's (Player 0) perspective."""
        raw = self.game.get_observation(player_idx=0)
        return observation_to_array(raw)

    def _get_info(self) -> dict:
        """Return auxiliary info dict (non-essential; useful for logging)."""
        if self.game is None:
            return {}
        return {
            "round_wins": list(self.game.round_wins),
            "phase": self.game.phase,
            "active_player": self.game.active_player,
            "last_round_winner": self.game.last_round_winner,
            "game_winner": self.game.game_winner,
        }

    def _run_opponent_if_needed(self) -> None:
        """
        If the game is in decision phase with the opponent as active player,
        run the opponent's full turn.  This should never be needed under normal
        Pazaak rules (Player A always goes first), but is included as a safety
        guard in case of future rule changes or unusual initialisation.
        """
        while self.game.phase == "decision" and self.game.active_player == 1:
            opp_obs = self.game.get_observation(player_idx=1)
            opp_action = self.opponent_agent(opp_obs, self.game)
            self.game.step(opp_action)


# ---------------------------------------------------------------------------
# ActionMasker factory (convenience wrapper for sb3-contrib)
# ---------------------------------------------------------------------------


def mask_fn(env: PazaakGymnasiumEnv) -> np.ndarray:
    """Mask function compatible with ``sb3_contrib.common.wrappers.ActionMasker``."""
    return env.action_masks()


def make_env(
    opponent_agent: Optional[Callable] = None,
    side_deck_a: Optional[list[int]] = None,
    side_deck_b: Optional[list[int]] = None,
    seed: Optional[int] = None,
    wrap_for_maskable_ppo: bool = True,
) -> gymnasium.Env:
    """
    Factory that creates a ``PazaakGymnasiumEnv`` and optionally wraps it
    with ``ActionMasker`` for use with ``sb3_contrib.MaskablePPO``.

    Parameters
    ----------
    opponent_agent : callable, optional
        Agent callable.  Defaults to ``simple_heuristic_agent``.
    side_deck_a, side_deck_b : list[int], optional
        Side decks for Player A and B.  Defaults to balanced decks.
    seed : int, optional
        RNG seed.
    wrap_for_maskable_ppo : bool
        If True (default), wraps with ``ActionMasker``.
        Set to False when you want the raw env (e.g. for ``check_env``).

    Returns
    -------
    gymnasium.Env
        Either a raw ``PazaakGymnasiumEnv`` or an ``ActionMasker``-wrapped one.

    Example
    -------
    ::

        from stable_baselines3.common.env_util import make_vec_env
        from gymnasium_env import make_env

        vec_env = make_vec_env(
            lambda: make_env(opponent_agent=heuristic_agent),
            n_envs=8,
        )
    """
    env = PazaakGymnasiumEnv(
        opponent_agent=opponent_agent,
        side_deck_a=side_deck_a,
        side_deck_b=side_deck_b,
        seed=seed,
    )

    if wrap_for_maskable_ppo:
        try:
            from sb3_contrib.common.wrappers import ActionMasker

            return ActionMasker(env, mask_fn)
        except ImportError:
            # sb3_contrib not installed - return the raw env so the module
            # remains importable in environments without RL dependencies.
            return env

    return env
