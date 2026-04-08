"""
train_self_play.py - Phase 3 self-play training for Pazaak RL.

Loads the Phase 2 checkpoint and progressively improves the agent by playing
against a mixed opponent pool:

     5%  simple heuristic
    15%  aggressive heuristic
    25%  full heuristic
    55%  random snapshot from the pool (up to 10 most recent)

Each self-play iteration:
  1. Trains for 1,000,000 timesteps against the current pool
  2. Evaluates vs the full heuristic and the latest snapshot
  3. Saves the model as a new snapshot and adds it to the pool

Usage
-----
# Default: 10 iterations, loads checkpoints/phase2_final.zip
python train_self_play.py

# Custom:
python train_self_play.py --iterations 8 --load checkpoints/phase2_final.zip
python train_self_play.py --timesteps 300000 --n-envs 4
python train_self_play.py --no-eval
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Callable, Optional

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO

from pazaakrl.gymnasium_env import (
    PazaakGymnasiumEnv,
    observation_to_array,
    int_to_action,
)
from pazaakrl.heuristic import (
    simple_heuristic_agent,
    aggressive_heuristic_agent,
    heuristic_agent,
)
from pazaakrl.game_engine import PazaakGame
from train_vs_heuristic import evaluate_vs_heuristic, TrainingProgressCallback


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints"
DEFAULT_LOAD_FROM = "checkpoints/phase2_final.zip"
DEFAULT_ITERATIONS = 10
DEFAULT_TIMESTEPS = 1_000_000
MAX_POOL_SNAPSHOTS = 10  # keep at most this many snapshots in the pool

# Probability weights: [simple_heuristic, aggressive_heuristic, full_heuristic, snapshot_pool]
OPPONENT_WEIGHTS = [0.05, 0.15, 0.25, 0.55]

PPO_KWARGS = dict(
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=512,
    gamma=0.995,
    ent_coef=0.02,
    n_epochs=10,
    clip_range=0.2,
    verbose=1,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
)


# ---------------------------------------------------------------------------
# Snapshot opponent
# ---------------------------------------------------------------------------


def make_snapshot_agent(model_path: str) -> Callable:
    """
    Return an agent callable that uses a saved MaskablePPO snapshot.

    The agent operates from Player 1's perspective - it receives
    get_observation(player_idx=1) which already flips "my" / "opp" fields,
    so the same observation_to_array() function works without extra flipping.

    The model is loaded lazily on first call and cached inside the closure.
    """
    _cache: dict = {}

    def _build_action_mask(game: PazaakGame) -> np.ndarray:
        """Build a 6-element boolean mask for Player 1's legal actions."""
        mask = np.zeros(6, dtype=bool)
        for action in game.legal_actions():
            if action == "hit":
                mask[0] = True
            elif action == "stand":
                mask[1] = True
            elif isinstance(action, tuple) and action[0] == "play":
                mask[2 + action[1]] = True
        return mask

    def agent(obs_dict: dict, game: PazaakGame):
        if "model" not in _cache:
            _cache["model"] = MaskablePPO.load(model_path)
        snapshot_model: MaskablePPO = _cache["model"]

        obs_array = observation_to_array(obs_dict)
        mask = _build_action_mask(game)
        action_int, _ = snapshot_model.predict(
            obs_array, action_masks=mask, deterministic=False
        )
        return int_to_action(int(action_int))

    agent.__name__ = f"snapshot({os.path.basename(model_path)})"
    return agent


# ---------------------------------------------------------------------------
# Dynamic opponent environment
# ---------------------------------------------------------------------------


class DynamicOpponentEnv(PazaakGymnasiumEnv):
    """
    A PazaakGymnasiumEnv whose opponent is re-sampled from a pool at the
    start of every episode (every reset() call).

    The pool manager is a callable with no arguments that returns an agent
    callable.  This indirection lets the training loop update the pool between
    iterations without rebuilding the vectorised environment.

    Parameters
    ----------
    pool_manager : callable() → agent_callable
        Called on every reset() to pick the opponent for the coming episode.
    kwargs
        Forwarded to PazaakGymnasiumEnv.__init__ (except opponent_agent).
    """

    def __init__(self, pool_manager: Callable, **kwargs):
        # Pass a placeholder opponent; it will be replaced on every reset().
        super().__init__(opponent_agent=simple_heuristic_agent, **kwargs)
        self._pool_manager = pool_manager

    def reset(self, *, seed=None, options=None):
        # Sample a fresh opponent for this episode
        self.opponent_agent = self._pool_manager()
        return super().reset(seed=seed, options=options)


# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------


class OpponentPool:
    """
    Manages the mixed opponent pool for self-play.

    At any time the pool contains:
      - simple_heuristic_agent      (always present)
      - aggressive_heuristic_agent  (always present)
      - heuristic_agent             (always present)
      - up to MAX_POOL_SNAPSHOTS snapshot agents (loaded on demand)

    Sampling:
      - 15% → simple heuristic
      - 15% → aggressive heuristic
      - 20% → full heuristic
      - 50% → uniform random from snapshot list
              (falls back to a random heuristic if no snapshots yet)
    """

    def __init__(self):
        self._snapshot_paths: list[str] = []
        # Cache of loaded agents keyed by path (lazy-loaded on first use)
        self._snapshot_agents: dict[str, Callable] = {}

    def add_snapshot(self, path: str) -> None:
        """Register a new snapshot. Evicts the oldest if the pool is full."""
        self._snapshot_paths.append(path)
        if len(self._snapshot_paths) > MAX_POOL_SNAPSHOTS:
            evicted = self._snapshot_paths.pop(0)
            self._snapshot_agents.pop(evicted, None)
            print(f"  Pool: evicted oldest snapshot ({os.path.basename(evicted)})")
        print(
            f"  Pool: added snapshot '{os.path.basename(path)}' "
            f"({len(self._snapshot_paths)}/{MAX_POOL_SNAPSHOTS} snapshots)"
        )

    def sample(self) -> Callable:
        """Return one opponent agent sampled according to OPPONENT_WEIGHTS."""
        if not self._snapshot_paths:
            # No snapshots yet - distribute evenly across all three heuristics
            return random.choice(
                [simple_heuristic_agent, aggressive_heuristic_agent, heuristic_agent]
            )

        choice = random.choices(
            ["simple", "aggressive", "full", "snapshot"],
            weights=OPPONENT_WEIGHTS,
            k=1,
        )[0]

        if choice == "simple":
            return simple_heuristic_agent
        if choice == "aggressive":
            return aggressive_heuristic_agent
        if choice == "full":
            return heuristic_agent

        # Snapshot - load lazily and cache
        path = random.choice(self._snapshot_paths)
        if path not in self._snapshot_agents:
            self._snapshot_agents[path] = make_snapshot_agent(path)
        return self._snapshot_agents[path]

    def __len__(self) -> int:
        return len(self._snapshot_paths)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_vs_snapshot(
    model: MaskablePPO,
    snapshot_path: str,
    n_games: int = 2500,
    verbose: bool = True,
) -> dict:
    """Evaluate the current model against the most recent snapshot."""
    snapshot_agent = make_snapshot_agent(snapshot_path)
    if verbose:
        print(f"  Evaluating vs snapshot '{os.path.basename(snapshot_path)}' …")
    return evaluate_vs_heuristic(
        model=model,
        opponent_agent=snapshot_agent,
        n_games=n_games,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Core self-play loop
# ---------------------------------------------------------------------------


def _discover_snapshots(checkpoint_dir: str) -> list[str]:
    """
    Return all snapshot_N.zip files in *checkpoint_dir*, sorted by N ascending.

    Only files whose names match the pattern ``snapshot_<int>.zip`` are
    included; ``self_play_final.zip`` and other checkpoints are ignored so
    the pool only contains true per-iteration snapshots.
    """
    import re

    pattern = re.compile(r"^snapshot_(\d+)\.zip$")
    matches = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            matches.append((int(m.group(1)), os.path.join(checkpoint_dir, fname)))
    matches.sort(key=lambda x: x[0])
    return [path for _, path in matches]


def run_self_play(
    load_from: str = DEFAULT_LOAD_FROM,
    iterations: int = DEFAULT_ITERATIONS,
    timesteps_per_iter: int = DEFAULT_TIMESTEPS,
    n_envs: int = 8,
    run_eval: bool = True,
    eval_games: int = 2500,
    warm_start: bool = False,
) -> MaskablePPO:
    """
    Execute the full self-play training loop.

    Parameters
    ----------
    load_from           : Path to the checkpoint to start from.
    iterations          : Number of self-play iterations.
    timesteps_per_iter  : Training steps per iteration.
    n_envs              : Parallel environments.
    run_eval            : Whether to evaluate after each iteration.
    eval_games          : Games per evaluation.
    warm_start          : If True, scan CHECKPOINT_DIR for existing
                          snapshot_N.zip files and pre-populate the pool
                          before training begins. Useful when resuming from
                          a previous run's self_play_final.zip.

    Returns
    -------
    The final trained model.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Phase 3: Self-Play Training")
    print(f"  Iterations          : {iterations}")
    print(f"  Timesteps/iteration : {timesteps_per_iter:,}")
    print(f"  Parallel envs       : {n_envs}")
    print(f"  Loading from        : {load_from}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------ #
    # 1. Load checkpoint and seed the snapshot pool
    # ------------------------------------------------------------------ #
    if not os.path.exists(load_from):
        raise FileNotFoundError(
            f"Checkpoint not found: '{load_from}'. "
            f"Run train_vs_heuristic.py --phase 2 first, or point --load at "
            f"an existing checkpoint."
        )

    pool = OpponentPool()

    if warm_start:
        # Pre-populate the pool with any snapshot_N.zip files already on disk,
        # up to MAX_POOL_SNAPSHOTS (oldest are naturally evicted by add_snapshot).
        existing = _discover_snapshots(CHECKPOINT_DIR)
        if existing:
            print(f"  Warm-start: found {len(existing)} existing snapshot(s) on disk.")
            for path in existing:
                pool.add_snapshot(path)
            print()
        else:
            print(
                "  Warm-start: no existing snapshots found; pool will be seeded from checkpoint.\n"
            )

    # Always register the load_from checkpoint as snapshot_0 so the loaded
    # model itself is always in the pool, even on a warm start.
    snapshot_0_path = os.path.join(CHECKPOINT_DIR, "snapshot_0.zip")
    _seed_model = MaskablePPO.load(load_from)
    _seed_model.save(snapshot_0_path)
    pool.add_snapshot(snapshot_0_path)
    print("  Saved loaded checkpoint as snapshot_0\n")
    del _seed_model

    # ------------------------------------------------------------------ #
    # 2. Build vectorised environment with a dynamic opponent
    # ------------------------------------------------------------------ #
    def pool_sampler() -> Callable:
        return pool.sample()

    vec_env = make_vec_env(
        lambda: _make_dynamic_env(pool_sampler),
        n_envs=n_envs,
    )

    # ------------------------------------------------------------------ #
    # 3. Load the model into the vec_env
    # ------------------------------------------------------------------ #
    model = MaskablePPO.load(
        load_from,
        env=vec_env,
        **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
        verbose=PPO_KWARGS["verbose"],
    )

    # ------------------------------------------------------------------ #
    # 4. Self-play iterations
    # ------------------------------------------------------------------ #
    total_start = time.time()

    for iteration in range(1, iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"  Self-play iteration {iteration}/{iterations}")
        print(f"  Snapshot pool size: {len(pool)} snapshot(s)")
        print(f"{'─' * 60}")

        # -- Train --
        progress_cb = TrainingProgressCallback(log_freq=100_000)
        iter_start = time.time()

        model.learn(
            total_timesteps=timesteps_per_iter,
            callback=[progress_cb],
            reset_num_timesteps=False,  # keep a continuous step counter
        )

        elapsed = time.time() - iter_start
        print(f"\n  Iteration {iteration} training complete in {elapsed:.1f}s")

        # -- Save snapshot --
        snapshot_name = f"snapshot_{iteration}"
        snapshot_path = os.path.join(CHECKPOINT_DIR, f"{snapshot_name}.zip")
        model.save(snapshot_path)
        pool.add_snapshot(snapshot_path)
        print(f"  Saved {snapshot_name}.zip")

        # -- Evaluate --
        if run_eval:
            print(f"\n  === Evaluation (iteration {iteration}) ===")

            print(f"  vs simple heuristic ({eval_games} games):")
            evaluate_vs_heuristic(
                model=model,
                opponent_agent=simple_heuristic_agent,
                n_games=eval_games,
                verbose=True,
            )

            print(f"  vs aggressive heuristic ({eval_games} games):")
            evaluate_vs_heuristic(
                model=model,
                opponent_agent=aggressive_heuristic_agent,
                n_games=eval_games,
                verbose=True,
            )

            print(f"  vs full heuristic ({eval_games} games):")
            evaluate_vs_heuristic(
                model=model,
                opponent_agent=heuristic_agent,
                n_games=eval_games,
                verbose=True,
            )

            # Evaluate vs the snapshot we just saved (self-play mirror)
            print(f"  vs {snapshot_name} ({eval_games} games):")
            evaluate_vs_snapshot(
                model=model,
                snapshot_path=snapshot_path,
                n_games=eval_games,
                verbose=True,
            )

    # ------------------------------------------------------------------ #
    # 5. Save final model
    # ------------------------------------------------------------------ #
    final_path = os.path.join(CHECKPOINT_DIR, "self_play_final")
    model.save(final_path)
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(
        f"  Self-play complete in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)"
    )
    print(f"  Final model saved to {final_path}.zip")
    print(f"{'=' * 60}\n")

    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# Internal factory (needs to be a top-level def for multiprocessing pickle)
# ---------------------------------------------------------------------------

_GLOBAL_POOL_SAMPLER: Optional[Callable] = None


def _make_dynamic_env(pool_sampler: Callable) -> "DynamicOpponentEnv":
    """
    Factory used by make_vec_env.

    make_vec_env spawns worker processes, so the factory must be picklable.
    We achieve this by stashing the sampler in a module-level global that each
    worker process inherits via fork (on Linux) or re-imports (on Windows).
    """
    global _GLOBAL_POOL_SAMPLER
    _GLOBAL_POOL_SAMPLER = pool_sampler

    from sb3_contrib.common.wrappers import ActionMasker
    from pazaakrl.gymnasium_env import mask_fn

    env = DynamicOpponentEnv(pool_manager=_get_global_sampler)
    return ActionMasker(env, mask_fn)


def _get_global_sampler() -> Callable:
    """Retrieve the pool sampler stored in the module global."""
    if _GLOBAL_POOL_SAMPLER is None:
        # Fallback in case of unusual initialisation order
        return heuristic_agent
    return _GLOBAL_POOL_SAMPLER()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 self-play training for the Pazaak RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load",
        type=str,
        default=DEFAULT_LOAD_FROM,
        dest="load_from",
        help="Phase 2 checkpoint .zip to start from.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of self-play iterations.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Training timesteps per iteration.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=2500,
        help="Games per evaluation (after each iteration).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after each iteration.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "Pre-populate the opponent pool with any snapshot_N.zip files "
            "already in the checkpoint directory. Use this when resuming from "
            "a previous run (e.g. --load checkpoints/self_play_final.zip) so "
            "the pool is not reset to a single snapshot."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_self_play(
        load_from=args.load_from,
        iterations=args.iterations,
        timesteps_per_iter=args.timesteps,
        n_envs=args.n_envs,
        run_eval=not args.no_eval,
        eval_games=args.eval_games,
        warm_start=args.warm_start,
    )


if __name__ == "__main__":
    main()
