# Training Analysis and Path Forward

## Current Results Summary

| Metric | Value |
|--------|-------|
| First-mover disadvantage (baseline) | ~5% (55-45 split) |
| Heuristic vs simple (heuristic plays first) | 89.7% win rate |
| Phase 1 RL vs simple (3M steps) | 61.9% win rate |
| Phase 2 RL vs heuristic (best, 16M steps) | ~16% win rate |
| Phase 2 RL vs simple (regression) | 50.2% win rate |
| Training speed | ~300k steps/min |

---

## Diagnosis: Why Training Is Plateauing

### 1. The observation space forces the network to rediscover arithmetic

This is the single biggest problem. The current 16-float observation gives the agent raw normalized values:

```
my_total / 20     →  0.85  (total is 17)
hand_slot_0 / 6   →  0.50  (card is +3)
```

To learn "play this card to reach 20", the network must discover that `0.85 * 20 + 0.50 * 6 = 20`. That's a multiplicative relationship between two inputs that a small 2x64 MLP has to learn purely from delayed reward signal. The heuristic agent has this as an explicit `if total + card == 20` check.

Similarly, "am I beating my opponent?" requires comparing `my_total/20` with `opp_total/20` — a subtraction the network has to learn implicitly. Every rule in the heuristic involves a derived relationship between observation features. The RL agent gets none of these for free.

### 2. Reward signal is too sparse and delayed

The agent only receives reward when:
- A round ends (+/-0.3)
- The game ends (+/-1.0)

A typical game has 3-5 rounds, each with 3-8 turns. The agent might take 20-40 actions per game but only receives 4-6 reward signals. Most individual decisions (play a hand card, hit, stand) get zero reward. The agent cannot learn which specific decisions were good because the credit assignment problem is severe.

For comparison: the heuristic agent doesn't need reward at all — its logic directly encodes "standing on 20 is good" and "busting is bad." The RL agent must discover these from game-level outcomes across thousands of episodes.

### 3. The Phase 1 to Phase 2 jump is too steep

Phase 1 trains against an opponent that never plays hand cards and has a fixed stand threshold of 17. The agent overfits to this specific, predictable strategy. When Phase 2 introduces the full heuristic (which plays hand cards to reach 20, rescues from bust, and reacts to the agent's state), the agent's learned policy is nearly useless against this new behavior.

The regression from 61.9% vs simple to 50.2% confirms catastrophic forgetting — the Phase 2 training destabilizes what was learned in Phase 1 without building effective new strategies.

### 4. The network capacity is adequate but not the bottleneck

The default MlpPolicy (2x64) is small but the game state isn't that complex. The issue isn't network capacity — it's that the inputs make critical relationships invisible and the reward doesn't guide discovery.

### 5. MaskablePPO is appropriate for this task

The algorithm choice is fine. MaskablePPO handles the variable legality of actions correctly. The discrete action space is small (6 actions). PPO is well-suited to this type of sequential decision-making. Switching algorithms (DQN, SAC, etc.) would not address the fundamental representation and reward problems.

---

## Recommended Path Forward

### Change 1: Enrich the observation space with derived features (HIGH IMPACT)

Add features that make the heuristic's decision logic learnable without requiring the network to discover arithmetic relationships. Expand from 16 to 33 features:

**Keep all 16 existing features** (indices 0-15, unchanged), then append:

| Index | Feature | Calculation | Why it helps |
|-------|---------|-------------|-------------|
| 16 | Distance to 20 | `(20 - my_total) / 20` | Directly encodes proximity to goal |
| 17 | Opponent distance to 20 | `(20 - opp_total) / 20` | Encodes opponent's position |
| 18 | Am I bust? | `1.0 if my_total > 20 else 0.0` | Binary bust signal |
| 19 | Am I ahead? | `1.0 if my_total > opp_total else 0.0` | Relative position (when both active) |
| 20 | Can reach exactly 20 | `1.0 if any hand card makes total == 20 else 0.0` | Key decision trigger |
| 21 | Best card delta to 20 | `min(abs(20 - (total + card)) for card in hand) / 20` | How close can a hand card get me? |
| 22 | Have negative cards | `1.0 if any hand card < 0 else 0.0` | Can I rescue from bust? |
| 23 | Can rescue from bust | `1.0 if bust and any negative card fixes it else 0.0` | Direct bust-rescue signal |
| 24 | Round win deficit | `(opp_round_wins - my_round_wins) / 3` | Game-level pressure |
| 25 | Opponent has stood and is ahead | `1.0 if opp_stood and opp_total > my_total else 0.0` | Must-beat-or-lose signal |
| 26 | Cards remaining advantage | `(my_hand_count - opp_hand_count) / 4` | Resource advantage |
| 27 | Safe to hit (total <= 10) | `1.0 if my_total <= 10 else 0.0` | Can't bust on next draw |
| 28 | Bust probability on next hit | `max(0, my_total - 10) / 10` | Fraction of draws that bust you |
| 29 | Negative cards in hand count | `count(card < 0 for card in hand) / 4` | How many rescue chances remain |
| 30 | Positive cards in hand count | `count(card > 0 for card in hand) / 4` | How much boost potential remains |
| 31 | Would standing now win? | `1.0 if opp_stood and my_total > opp_total else 0.0` | Clearest "stop now" signal |
| 32 | Opponent bust risk | `max(0, opp_total - 10) / 10 if not opp_stood else 0.0` | Probability opponent busts on next hit |

These features encode the same relationships the heuristic uses as explicit rules. The network doesn't have to learn that 17 + 3 = 20 — it just sees `can_reach_20 = 1.0`.

**Implementation**: Modify `observation_to_array()` in `gymnasium_env.py` to compute and append these features. Update `observation_space` shape from `(16,)` to `(33,)`.

### Change 2: Add intermediate reward shaping (HIGH IMPACT)

Supplement the existing round/game rewards with immediate decision-quality signals:

```python
# On stand:
if 18 <= my_total <= 20:
    reward += 0.05   # good stand
if my_total == 20:
    reward += 0.05   # perfect stand (total: +0.10 for standing on 20)

# On bust (total > 20 after hit/stand):
reward -= 0.1        # busting is always bad

# On playing a hand card that reaches exactly 20:
reward += 0.05       # optimal hand card usage
```

These are small (0.05-0.10) so they don't overpower the game outcome (+/-1.0), but they provide per-decision gradient signal. The agent learns "standing on 20 is good" within a few hundred episodes instead of needing thousands.

**Important**: Don't reward specific actions like "always stand on 18" too heavily or the agent will overfit to a deterministic strategy. Keep shaping rewards an order of magnitude below game rewards.

### Change 3: Mix opponents during Phase 2 (MEDIUM IMPACT)

Instead of a clean Phase 1 then Phase 2 transition, mix opponents during Phase 2 to prevent catastrophic forgetting:

```python
# Phase 2 opponent selection per episode:
#   60% full heuristic
#   30% simple heuristic  
#   10% random agent
```

This keeps the agent sharp against simple strategies while learning to handle the heuristic. The random agent adds diversity and prevents overfitting to any specific strategy pattern.

**Implementation**: Create a `MixedOpponent` wrapper that randomly selects an opponent at the start of each game (in `reset()`). This is a simple change to the environment.

### Change 4: Increase network capacity slightly (LOW-MEDIUM IMPACT)

With 28 input features and richer reward signal, a slightly larger network can learn more nuanced policies:

```python
policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128])
)
```

This doubles the hidden layer size. Not strictly necessary but gives the network more room to learn combinatorial hand-card strategies.

### Change 5: Tune hyperparameters for this problem (LOW-MEDIUM IMPACT)

A few targeted adjustments:

```python
PPO_KWARGS = dict(
    learning_rate=1e-4,       # slower, more stable (was 3e-4)
    n_steps=4096,             # longer rollouts capture full games better (was 2048)
    batch_size=512,           # larger batches for more stable gradients (was 256)
    gamma=0.995,              # higher discount — hand card saves matter rounds later (was 0.99)
    ent_coef=0.02,            # more exploration early on (was 0.01)
    n_epochs=10,              # unchanged
    clip_range=0.2,           # unchanged
)
```

The higher gamma is particularly important: saving a hand card for a later round has a delayed payoff that `gamma=0.99` discounts too aggressively over 20-40 steps.

---

## Recommended Implementation Order

1. **Feature engineering** (Change 1) — Modify `observation_to_array()` and `observation_space`. This is the highest-impact change and should be done first. Re-train Phase 1 from scratch with the new observation space.

2. **Reward shaping** (Change 2) — Add to `step()` in `gymnasium_env.py`. Do this alongside Change 1 for maximum impact.

3. **Mixed opponents** (Change 3) — Add after validating that Changes 1+2 improve Phase 1 performance. Use for Phase 2 training.

4. **Network and hyperparameters** (Changes 4+5) — Apply when running the full pipeline with the above changes.

After implementing Changes 1-5, re-run the full pipeline: Phase 1 (1.5M steps vs simple), then Phase 2 (3M steps vs mixed opponents). Expected outcome: Phase 1 should reach >75% against simple, and Phase 2 should reach >30% against the heuristic — a major improvement over the current 16%.

---

## What If This Still Plateaus?

If the above changes get you to ~30-40% against the heuristic but you want to push higher:

### Self-play with league training
Implement Phase 3 (self-play) from the implementation plan, but use a league-style approach: maintain a pool of snapshots and weight opponent selection toward opponents the current agent struggles against. This prevents the agent from farming easy wins against old snapshots.

### Monte Carlo Tree Search (MCTS) hybrid
Use the trained policy network as a prior for MCTS at evaluation time (similar to AlphaZero). During each decision, run short MCTS simulations using the game engine to look ahead. This is particularly effective in Pazaak because the game tree is narrow (6 actions max) and the stochastic element (field card draws) can be sampled.

### Opponent modeling
Add features that track the opponent's behavioral patterns within a game (e.g., "opponent tends to stand early", "opponent has been playing hand cards aggressively"). This helps against the heuristic because it has predictable, exploitable rules.

---

## What About Switching Algorithms?

**MaskablePPO is still the right choice.** Here's why alternatives are less suitable:

- **DQN/Rainbow**: Works for discrete actions but PPO is generally more sample-efficient for sequential games and handles the multi-step nature of Pazaak turns better.
- **SAC**: Designed for continuous action spaces. Not ideal here.
- **AlphaZero/MuZero**: Would work extremely well but requires implementing MCTS and a value network from scratch — a much larger project. Worth considering as a v2 if PPO-based training plateaus after the above improvements.
- **IMPALA/R2D2**: Designed for partially observable environments with long episodes. Could help but the added complexity isn't justified yet.

The bottleneck is representation and reward, not algorithm choice.

---

## Expected Timeline

| Step | Effort | Expected Impact |
|------|--------|-----------------|
| Feature engineering (Change 1) | ~1 hour code, ~10 min retrain | Major — removes the arithmetic discovery burden |
| Reward shaping (Change 2) | ~30 min code | Major — provides per-decision learning signal |
| Mixed opponents (Change 3) | ~30 min code | Medium — prevents catastrophic forgetting |
| Hyperparameter tuning (Changes 4+5) | ~15 min code, multiple training runs | Low-Medium — marginal gains |
| Full pipeline re-run | ~30 min wall-clock | Validates all changes together |
