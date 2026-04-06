"""
evaluate.py - Standalone evaluation script for trained Pazaak RL agents.

Loads a model checkpoint and plays N games against a specified opponent,
reporting win rate, draw rate, loss rate, and average rounds per game.
Optionally prints a condensed game-by-game log for a sample of games.

Supported opponent types
------------------------
  simple      simple_heuristic_agent (stand on 17+, no hand cards)
  heuristic   full 6-rule heuristic_agent
  random      picks a uniformly random legal action each step
  model:<path>  a second trained MaskablePPO checkpoint

Usage
-----
  # Evaluate a model vs the full heuristic (default, 1000 games):
  python evaluate.py checkpoints/phase2_final.zip

  # Choose opponent and game count:
  python evaluate.py checkpoints/self_play_final.zip --opponent simple --games 2000

  # Model vs model:
  python evaluate.py checkpoints/self_play_final.zip \\
      --opponent model:checkpoints/phase2_final.zip

  # Show a sample of 5 games as a condensed log:
  python evaluate.py checkpoints/phase1_final.zip --show 5

  # Show all game outcomes line by line:
  python evaluate.py checkpoints/phase2_final.zip --verbose
"""

from __future__ import annotations

import argparse
import random as stdlib_random
from typing import Callable

import numpy as np
from sb3_contrib import MaskablePPO

from game_engine import PazaakGame
from gymnasium_env import (
    PazaakGymnasiumEnv,
    observation_to_array,
    int_to_action,
    _default_side_deck,
)
from heuristic import simple_heuristic_agent, heuristic_agent


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def random_agent(obs: dict, game: PazaakGame):
    """Picks a uniformly random legal action - useful as a baseline."""
    return stdlib_random.choice(game.legal_actions())


def make_model_agent(model_path: str) -> Callable:
    """
    Load a MaskablePPO checkpoint and return an agent callable.

    The agent operates from whatever player perspective it is called with -
    it uses observation_to_array() and builds its own action mask from the
    game's legal_actions(), so it works correctly as either Player 0 or
    Player 1 (the gymnasium wrapper always calls it as Player 1).
    """
    model = MaskablePPO.load(model_path)

    def _build_mask(game: PazaakGame) -> np.ndarray:
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
        obs_array = observation_to_array(obs_dict)
        mask = _build_mask(game)
        action_int, _ = model.predict(obs_array, action_masks=mask, deterministic=False)
        return int_to_action(int(action_int))

    agent.__name__ = f"model({model_path})"
    return agent


def resolve_opponent(opponent_str: str) -> Callable:
    """Parse the --opponent argument and return the corresponding agent."""
    if opponent_str == "simple":
        return simple_heuristic_agent
    if opponent_str == "heuristic":
        return heuristic_agent
    if opponent_str == "random":
        return random_agent
    if opponent_str.startswith("model:"):
        path = opponent_str[len("model:") :]
        print(f"  Loading opponent model from '{path}' …")
        return make_model_agent(path)
    raise ValueError(
        f"Unknown opponent '{opponent_str}'. "
        f"Expected: simple | heuristic | random | model:<path>"
    )


def opponent_display_name(opponent_str: str) -> str:
    if opponent_str == "simple":
        return "simple heuristic"
    if opponent_str == "heuristic":
        return "full heuristic"
    if opponent_str == "random":
        return "random agent"
    if opponent_str.startswith("model:"):
        return f"model ({opponent_str[6:]})"
    return opponent_str


# ---------------------------------------------------------------------------
# Single-game condensed log
# ---------------------------------------------------------------------------


def _hand_display(hand: list, hide: bool) -> str:
    """Format a hand for display, optionally hiding values."""
    if hide:
        count = sum(1 for c in hand if c is not None)
        return f"[hidden – {count} card{'s' if count != 1 else ''}]"
    cards = [f"{'+' if c > 0 else ''}{c}" if c is not None else "___" for c in hand]
    return "[" + ", ".join(cards) + "]"


def play_and_log_game(
    model_agent: Callable,
    opponent_agent: Callable,
    game_number: int,
    god_mode: bool = False,
) -> dict:
    """
    Play one game, print a condensed per-round summary, and return stats.

    Returns a dict with keys: winner (0, 1, or None), rounds.
    """
    deck = _default_side_deck()
    game = PazaakGame(side_deck_a=list(deck), side_deck_b=list(deck))

    print(f"\n  ── Game {game_number} ──")
    print(f"  Agent hand  : {_hand_display(game.players[0].hand, hide=False)}")
    print(f"  Opponent    : {_hand_display(game.players[1].hand, hide=not god_mode)}")

    round_num = 0
    while game.phase != "game_over":
        if game.phase == "round_over":
            game.start_round()
            continue

        round_num += 1
        round_start_score = list(game.round_wins)

        # Play one round by driving the engine directly
        while game.phase not in ("round_over", "game_over"):
            if game.phase != "decision":
                break
            p = game.active_player
            agent = model_agent if p == 0 else opponent_agent
            obs = game.get_observation(p)
            action = agent(obs, game)
            game.step(action)

        rw = game.last_round_winner
        result_str = "Agent wins" if rw == 0 else "Opponent wins" if rw == 1 else "Draw"
        score = game.round_wins
        print(
            f"    Round {round_num}: {result_str:14s}  "
            f"totals {game.players[0].total:>3} vs {game.players[1].total:<3}  "
            f"score {score[0]}-{score[1]}"
        )

    gw = game.game_winner
    outcome = "AGENT WINS" if gw == 0 else "OPPONENT WINS" if gw == 1 else "DRAW"
    print(f"  → {outcome}  (final score {game.round_wins[0]}-{game.round_wins[1]})")

    return {"winner": game.game_winner, "rounds": sum(game.round_wins)}


# ---------------------------------------------------------------------------
# Bulk evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model_path: str,
    opponent_str: str = "heuristic",
    n_games: int = 1000,
    show_games: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Load a model and evaluate it over n_games against the specified opponent.

    Parameters
    ----------
    model_path   : Path to a MaskablePPO .zip checkpoint.
    opponent_str : Opponent identifier string (see resolve_opponent).
    n_games      : Total games to play.
    show_games   : Print a condensed log for the first N games.
    verbose      : Print a one-line result for every game.

    Returns
    -------
    dict with keys: games, wins, losses, draws, win_rate, loss_rate,
                    draw_rate, avg_rounds.
    """
    print(f"\n{'=' * 60}")
    print("  Pazaak RL - Evaluation")
    print(f"{'=' * 60}")
    print(f"  Model    : {model_path}")
    print(f"  Opponent : {opponent_display_name(opponent_str)}")
    print(f"  Games    : {n_games:,}")
    print(f"{'=' * 60}\n")

    # Load model and opponent
    print(f"  Loading model from '{model_path}' …")
    model = MaskablePPO.load(model_path)
    model_agent = make_model_agent(model_path)  # for direct-engine calls (show_games)
    opponent_agent = resolve_opponent(opponent_str)

    # ---- Bulk evaluation via gymnasium env ----
    env = PazaakGymnasiumEnv(opponent_agent=opponent_agent)

    wins = losses = draws = 0
    total_rounds: list[int] = []
    shown = 0

    for game_idx in range(1, n_games + 1):
        # Condensed game log for the first `show_games` games
        if shown < show_games:
            stats = play_and_log_game(
                model_agent=model_agent,
                opponent_agent=opponent_agent,
                game_number=shown + 1,
                god_mode=False,
            )
            winner = stats["winner"]
            rounds = stats["rounds"]
            shown += 1
        else:
            # Fast path: use the gymnasium wrapper
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            winner = env.game.game_winner
            rounds = sum(env.game.round_wins)

        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1
        total_rounds.append(rounds)

        if verbose and shown >= show_games:
            outcome = "W" if winner == 0 else "L" if winner == 1 else "D"
            wr_so_far = wins / game_idx
            print(
                f"  Game {game_idx:>5} : {outcome}  "
                f"rounds={rounds}  "
                f"running WR={wr_so_far * 100:5.1f}%"
            )

    env.close()

    # ---- Summary ----
    n = wins + losses + draws
    win_rate = wins / n
    loss_rate = losses / n
    draw_rate = draws / n
    avg_rounds = float(np.mean(total_rounds))

    print(f"\n{'─' * 60}")
    print(f"  Results - {n:,} games vs {opponent_display_name(opponent_str)}")
    print(f"{'─' * 60}")
    print(f"  Wins   : {wins:>6,}  ({win_rate * 100:5.1f}%)")
    print(f"  Losses : {losses:>6,}  ({loss_rate * 100:5.1f}%)")
    print(f"  Draws  : {draws:>6,}  ({draw_rate * 100:5.1f}%)")
    print(f"  Avg rounds/game : {avg_rounds:.2f}")
    print(f"{'─' * 60}\n")

    return dict(
        games=n,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        loss_rate=loss_rate,
        draw_rate=draw_rate,
        avg_rounds=avg_rounds,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pazaak RL model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="Path to the MaskablePPO checkpoint .zip to evaluate.",
    )
    parser.add_argument(
        "--opponent",
        default="heuristic",
        help=("Opponent type: simple | heuristic | random | model:<path>. "),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to play.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="Print a condensed round-by-round log for the first N games.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a one-line result for every game (after any --show games).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        opponent_str=args.opponent,
        n_games=args.games,
        show_games=args.show,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
