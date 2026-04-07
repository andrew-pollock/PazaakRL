"""
train_vs_heuristic.py - Phase 1 and Phase 2 heuristic training for Pazaak RL.

Phase 1: Train from scratch against the simple heuristic (stand on 17+, no hand cards).
         Target: >70% win rate.  Saves checkpoint as "checkpoints/phase1_final.zip".

Phase 2: Continue from the Phase 1 checkpoint against the full 6-rule heuristic.
         Target: >50% win rate.  Saves checkpoint as "checkpoints/phase2_final.zip".

Usage
-----
# Phase 1 (default):
python train_vs_heuristic.py

# Phase 2 (loads Phase 1 checkpoint automatically):
python train_vs_heuristic.py --phase 2

# Override timesteps or checkpoint path:
python train_vs_heuristic.py --phase 1 --timesteps 500000
python train_vs_heuristic.py --phase 2 --load checkpoints/phase1_final.zip --timesteps 800000

# Skip evaluation after training:
python train_vs_heuristic.py --no-eval

Dependencies
------------
    gymnasium>=0.29
    stable-baselines3>=2.1
    sb3-contrib>=2.1
    numpy
    torch
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import MaskablePPO

from pazaakrl.gymnasium_env import PazaakGymnasiumEnv, make_env
from pazaakrl.heuristic import simple_heuristic_agent, heuristic_agent


# ---------------------------------------------------------------------------
# Mixed opponent (Change 3: prevents catastrophic forgetting in Phase 2)
# ---------------------------------------------------------------------------


class MixedOpponent:
    """
    Randomly selects an opponent strategy per call-sequence.

    The gymnasium env calls the opponent in a loop within a single turn.
    We pick a strategy once per game (on the first call after reset) and
    stick with it for the entire game.  The env resets the game in reset(),
    and we detect a new game by checking if the opponent's total is 0 and
    no cards have been used (i.e. fresh game state).

    Parameters
    ----------
    agents : list of (weight, callable) pairs.
        Weights are relative (they don't need to sum to 1).
    """

    def __init__(self, agents: list[tuple[float, callable]]):
        self._agents = agents
        weights = [w for w, _ in agents]
        total = sum(weights)
        self._probs = [w / total for w in weights]
        self._current = agents[0][1]
        self._rng = np.random.default_rng()
        self._last_game_id = None

    def __call__(self, observation: dict, game) -> str | tuple:
        # Detect new game: pick a new opponent when game state looks fresh
        game_id = id(game)
        if game_id != self._last_game_id:
            self._last_game_id = game_id
            idx = self._rng.choice(len(self._agents), p=self._probs)
            self._current = self._agents[idx][1]
        return self._current(observation, game)


def _random_agent(observation: dict, game) -> str | tuple:
    """Pick a random legal action. Used as a diversity opponent."""
    import random as _rand
    legal = game.legal_actions()
    return _rand.choice(legal)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints"

# Hyperparameters (tuned per docs/training_analysis.md)
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

PHASE_CONFIG = {
    1: dict(
        opponent=simple_heuristic_agent,
        opponent_name="simple heuristic",
        timesteps=1_500_000,
        target_win_rate=0.70,
        save_name="phase1_final",
        load_from=None,
    ),
    2: dict(
        opponent=MixedOpponent([
            (0.6, heuristic_agent),
            (0.3, simple_heuristic_agent),
            (0.1, _random_agent),
        ]),
        opponent_name="mixed (60% heuristic / 30% simple / 10% random)",
        timesteps=3_000_000,
        target_win_rate=0.30,
        save_name="phase2_final",
        load_from="checkpoints/phase1_final.zip",
    ),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_vs_heuristic(
    model: MaskablePPO,
    opponent_agent,
    n_games: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Play n_games full Pazaak games and return win/loss/draw statistics.

    Uses the raw PazaakGymnasiumEnv (no vectorisation) so that game_winner
    is always accessible via env.game.
    """
    env = PazaakGymnasiumEnv(opponent_agent=opponent_agent)
    wins = losses = draws = 0
    total_rounds: list[int] = []

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            # Build the action mask directly from the unwrapped env
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        winner = env.game.game_winner
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1
        total_rounds.append(sum(env.game.round_wins))

    env.close()

    n = wins + losses + draws
    win_rate = wins / n
    stats = {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_rounds": float(np.mean(total_rounds)),
    }

    if verbose:
        print(f"\n{'─' * 50}")
        print(f"  Evaluation results ({n} games)")
        print(f"{'─' * 50}")
        print(f"  Wins   : {wins:>5}  ({win_rate * 100:.1f}%)")
        print(f"  Losses : {losses:>5}  ({losses / n * 100:.1f}%)")
        print(f"  Draws  : {draws:>5}  ({draws / n * 100:.1f}%)")
        print(f"  Avg rounds per game: {stats['avg_rounds']:.2f}")
        print(f"{'─' * 50}\n")

    return stats


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TrainingProgressCallback(BaseCallback):
    """
    Logs episode win rate to stdout every `log_freq` timesteps.

    Tracks per-episode outcomes by reading `game_winner` from the info dict
    that PazaakGymnasiumEnv includes in every terminated step.
    """

    def __init__(self, log_freq: int = 20_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._wins: list[int] = []
        self._losses: list[int] = []
        self._last_log_step = 0

    def _on_step(self) -> bool:
        # `self.locals["dones"]` is a bool array over all envs
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                winner = info.get("game_winner")
                if winner == 0:
                    self._wins.append(1)
                    self._losses.append(0)
                elif winner == 1:
                    self._wins.append(0)
                    self._losses.append(1)

        if self.num_timesteps - self._last_log_step >= self.log_freq and (
            self._wins or self._losses
        ):
            n = len(self._wins)
            w = sum(self._wins)
            wr = w / n if n > 0 else 0.0
            print(
                f"  Step {self.num_timesteps:>8,} | "
                f"Recent episodes: {n:>4} | "
                f"Win rate: {wr * 100:5.1f}%"
            )
            self._wins.clear()
            self._losses.clear()
            self._last_log_step = self.num_timesteps

        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_vec_env_for_phase(opponent_agent, n_envs: int = 8):
    """Create a vectorised, ActionMasker-wrapped environment for training."""
    return make_vec_env(
        lambda: make_env(opponent_agent=opponent_agent, wrap_for_maskable_ppo=True),
        n_envs=n_envs,
    )


def train_phase(
    phase: int,
    timesteps: Optional[int] = None,
    load_from: Optional[str] = None,
    n_envs: int = 8,
    run_eval: bool = True,
    eval_games: int = 1000,
) -> MaskablePPO:
    """
    Run one training phase.

    Parameters
    ----------
    phase       : 1 or 2
    timesteps   : Override default timestep count for this phase.
    load_from   : Override default checkpoint path to load from.
    n_envs      : Number of parallel environments.
    run_eval    : If True, evaluate after training.
    eval_games  : Number of games for post-training evaluation.

    Returns
    -------
    The trained MaskablePPO model.
    """
    cfg = PHASE_CONFIG[phase]
    opponent = cfg["opponent"]
    opponent_name = cfg["opponent_name"]
    total_timesteps = timesteps if timesteps is not None else cfg["timesteps"]
    checkpoint_path = load_from if load_from is not None else cfg["load_from"]
    save_name = cfg["save_name"]
    target_wr = cfg["target_win_rate"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Phase {phase}: Training vs {opponent_name}")
    print(f"  Timesteps : {total_timesteps:,}")
    if checkpoint_path:
        print(f"  Loading   : {checkpoint_path}")
    print(f"  Save to   : {CHECKPOINT_DIR}/{save_name}.zip")
    print(f"  Target WR : >{target_wr * 100:.0f}%")
    print(f"{'=' * 60}\n")

    # ---- Build vectorised training environment ----
    vec_env = make_vec_env_for_phase(opponent, n_envs=n_envs)

    # ---- Build or load model ----
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading existing model from {checkpoint_path} …")
        model = MaskablePPO.load(
            checkpoint_path,
            env=vec_env,
            **{k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
            verbose=PPO_KWARGS["verbose"],
        )
    else:
        if checkpoint_path:
            print(
                f"  WARNING: checkpoint '{checkpoint_path}' not found. "
                f"Starting from scratch."
            )
        print("  Initialising new MaskablePPO model …")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            **PPO_KWARGS,
        )

    # ---- Callbacks ----
    progress_cb = TrainingProgressCallback(log_freq=20_000)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix=f"phase{phase}",
        verbose=0,
    )

    # ---- Train ----
    start = time.time()
    print("  Training started …\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_cb, checkpoint_cb],
        reset_num_timesteps=(checkpoint_path is None),  # continue step count if loading
    )
    elapsed = time.time() - start
    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ---- Save final checkpoint ----
    final_path = os.path.join(CHECKPOINT_DIR, save_name)
    model.save(final_path)
    print(f"  Model saved to {final_path}.zip")

    # ---- Evaluate ----
    if run_eval:
        # Always evaluate against both opponents for a complete picture
        for eval_name, eval_agent in [
            ("simple heuristic", simple_heuristic_agent),
            ("full heuristic", heuristic_agent),
        ]:
            print(f"\n  Evaluating vs {eval_name} over {eval_games:,} games …")
            stats = evaluate_vs_heuristic(
                model=model,
                opponent_agent=eval_agent,
                n_games=eval_games,
                verbose=True,
            )
        wr = stats["win_rate"]  # last eval (full heuristic) for target check
        if phase == 1:
            # Phase 1 target is measured against simple heuristic
            pass  # both results already printed
        if wr >= target_wr:
            print(f"  ✓ Win rate {wr * 100:.1f}% vs full heuristic meets target >{target_wr * 100:.0f}%")
        else:
            print(
                f"  ✗ Win rate {wr * 100:.1f}% vs full heuristic is BELOW target >{target_wr * 100:.0f}%. "
                f"Consider more training or tuning hyperparameters."
            )

    vec_env.close()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Pazaak RL agent against heuristic opponents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training phase: 1 = vs simple heuristic, 2 = vs full heuristic.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override the default timestep count for this phase.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        dest="load_from",
        help="Checkpoint .zip file to load and continue from.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments for training.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=1000,
        help="Number of games to play in post-training evaluation.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-training evaluation.",
    )
    parser.add_argument(
        "--both-phases",
        action="store_true",
        help="Run Phase 1 then Phase 2 sequentially (ignores --phase).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.both_phases:
        print("Running both phases sequentially.")
        # Phase 1
        train_phase(
            phase=1,
            timesteps=args.timesteps,
            load_from=args.load_from,
            n_envs=args.n_envs,
            run_eval=not args.no_eval,
            eval_games=args.eval_games,
        )
        # Phase 2 always loads the Phase 1 output
        train_phase(
            phase=2,
            timesteps=args.timesteps,
            load_from=None,  # let it auto-load phase1_final.zip
            n_envs=args.n_envs,
            run_eval=not args.no_eval,
            eval_games=args.eval_games,
        )
    else:
        train_phase(
            phase=args.phase,
            timesteps=args.timesteps,
            load_from=args.load_from,
            n_envs=args.n_envs,
            run_eval=not args.no_eval,
            eval_games=args.eval_games,
        )


if __name__ == "__main__":
    main()
