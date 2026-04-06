# Pazaak RL

A reinforcement learning agent trained to play **Pazaak** - a two-player card game from the *Star Wars: Knights of the Old Republic* series, and variant of Blackjack.

---

## The Game

Pazaak is a zero-sum, stochastic, imperfect-information card game. Players take turns drawing field cards (1–10) and optionally playing hand cards (±1–6) to get their total as close to 20 as possible without going over. The first player to win **3 rounds** wins the game.

What makes it different from Blackjack:
- Each player has a **private hand** of 4 cards (positive and negative) that persist across rounds
- Hand cards are a limited resource - once used, they're gone for the game
- You can see your opponent's total and how many hand cards they have left, but not their values

Full rules are detailed in [`game_rules.md`](docs/game_rules.md)

---

## Architecture

```
game_engine.py          Pure Python game logic - no ML dependencies
heuristic.py            Two rule-based opponents (simple and full)
gymnasium_env.py        Gymnasium wrapper for RL training
train_vs_heuristic.py   Training against basic heuristic opponents
train_self_play.py      Training against older versions of itself
```

The RL agent is trained using **MaskablePPO** (from `sb3-contrib`), which handles illegal action masking natively. One episode = one full game. The action space is 6 discrete actions: hit, stand, or play one of 4 hand card slots.

---

## Training Pipeline

Training proceeds in three graduated phases:

| Phase | Opponent | Steps | Target |
|-------|----------|-------|--------|
| 1 | Simple heuristic (stand on 17+) | 300k | >70% win rate |
| 2 | Full heuristic (uses hand cards) | 500k | >50% win rate |
| 3 | Self-play against snapshot pool | 200k × 5 | - |

Phases 1 and 2 use `train_vs_heuristic.py`. Phase 3 (self-play) is `train_self_play.py`.

---

## Installation

```bash
uv add gymnasium stable-baselines3 sb3-contrib numpy torch
```

---

## Usage

**Phase 1** - train from scratch against the simple heuristic:
```bash
python train_vs_heuristic.py
```

**Phase 2** - continue from Phase 1 against the full heuristic:
```bash
python train_vs_heuristic.py --phase 2
```

**Both phases** sequentially:
```bash
python train_vs_heuristic.py --both-phases
```

Checkpoints are saved to `checkpoints/` as `phase1_final.zip` and `phase2_final.zip`.

**Options:**
```
--timesteps N       Override default step count
--load PATH         Load a specific checkpoint to continue from
--n-envs N          Number of parallel environments (default: 8)
--eval-games N      Games to play in post-training evaluation (default: 1000)
--no-eval           Skip evaluation after training
```

---

## File Reference

| File | Purpose |
|------|---------|
| `game_engine.py` | Core game logic |
| `heuristic.py` | `simple_heuristic_agent`, `heuristic_agent` |
| `gymnasium_env.py` | `PazaakGymnasiumEnv`, `make_env()`, `observation_to_array()` |
| `train_vs_heuristic.py` | Phase 1 & 2 training |
| `game_rules.md` | Full game rules |
| `implementation_plan.md` | Full technical design |
