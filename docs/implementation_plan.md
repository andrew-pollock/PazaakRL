# Pazaak RL - Implementation Plan

This document is a step-by-step guide for building a Pazaak game engine with a Gymnasium environment, heuristic opponent, and reinforcement learning pipeline with self-play. It is designed so that any competent LLM can follow it to produce working code.

Refer to `game_rules.md` for the full game rules. This plan does not repeat those rules but references them frequently.

---

## Architecture Overview

The system has four layers, built bottom-up:

```
┌──────────────────────────────────────┐
│  4. Training Scripts                 │  self_play.py, evaluate.py
├──────────────────────────────────────┤
│  3. Gymnasium Wrapper                │  gymnasium_env.py
├──────────────────────────────────────┤
│  2. Heuristic Opponent               │  heuristic.py
├──────────────────────────────────────┤
│  1. Core Game Engine                 │  game_engine.py
└──────────────────────────────────────┘
```

Each layer only depends on the layers below it.

---

## Step 1: Core Game Engine (`game_engine.py`)

This module implements the full Pazaak rules with no RL or Gymnasium dependencies. It is a pure Python game engine.

### 1.1 Data Structures

Define a `PlayerState` dataclass:

```python
@dataclass
class PlayerState:
    total: int = 0                           # Current field total
    hand: list[int | None]                   # Fixed 4 slots. Value or None if consumed.
    used_hand: list[int] = field(default_factory=list)  # Hand cards played this game
    stood: bool = False                      # Has this player stood this round?
```

Define a `PazaakGame` class that manages the full game (multiple rounds, first to 3 wins).

### 1.1.1 Hand Card Fixed Slots (IMPORTANT)

Hand cards are stored as a **fixed-size list of 4 slots**, not a variable-length list. When a card is consumed, its slot is set to `None` rather than being removed from the list.

```
Game start:  hand = [+3, -2, +1, -4]     (4 cards in 4 slots)
Play slot 2: hand = [+3, -2, None, -4]   (slot 2 is now empty)
Play slot 0: hand = [None, -2, None, -4] (slot 0 is now empty)
```

**Why this matters**: The gymnasium action space maps actions 2-5 to hand card slots 0-3. If the hand were a shrinking list (e.g. using `list.pop()`), then after playing slot 1 the list shifts and action 3 would now refer to what was previously slot 3. This makes the action-to-card mapping unstable and breaks learning.

With fixed slots:
- Action 2 always means "play whatever is in slot 0" (or invalid if slot 0 is `None`)
- Action 3 always means "play whatever is in slot 1" (or invalid if slot 1 is `None`)
- etc.

The `legal_actions()` method returns `("play", i)` only for slots where `hand[i] is not None`.

**Do not use `list.remove()`, `list.pop()`, or any operation that shifts elements.** Always set the slot to `None`.

### 1.2 Game Lifecycle

```
PazaakGame.__init__(side_deck_a, side_deck_b, seed=None)
    │
    ├── Draw 4 hand cards per player from their side decks (once per game)
    ├── Initialize round_wins = [0, 0]
    │
    └── start_round()    ← same method used for the first round AND all subsequent rounds
            │
            ├── Reset both players' totals to 0
            ├── Reset both players' stood flags to False
            ├── Does NOT touch hand, used_hand, or round_wins
            ├── Set active_player = 0  (Player A always first)
            ├── Set phase = "field_draw"
            │
            └── draw_field_card()  ← automatically draw for active player
                    │
                    └── Set phase = "decision"
                            │
                            └── (Agent now chooses actions)
```

### 1.3 Turn Phases

A critical design decision: each turn has an internal phase to track where we are.

- **`"field_draw"`**: The active player needs a field card drawn. This happens automatically (the environment draws it, not the agent). After drawing, transition to `"decision"`.
- **`"decision"`**: The active player chooses one of: play a hand card (remain in `"decision"`), hit (end turn), or stand (end turn).
- **`"round_over"`**: The round has ended (bust or both stood). Evaluate winner, update `round_wins`, and either start a new round or end the game.
- **`"game_over"`**: One player has 3 round wins. The game is finished.

### 1.3.1 Field Card Draw Timing (IMPORTANT)

The field card is drawn **exactly once per turn**, at the start of the turn, before the player makes any decisions. It is NOT drawn each time the `step()` method is called.

When a player plays a hand card via `step(("play", i))`, the engine stays in `"decision"` phase. The player can then play more hand cards or choose hit/stand. **No new field card is drawn during these subsequent steps.** A new field card is only drawn when a brand new turn starts (i.e. after the previous turn ended with hit/stand and the game transitions to the next player's turn, or the same player's next turn if the opponent has stood).

The sequence within a single turn:
```
1. [AUTOMATIC] Field card drawn, added to total → phase = "decision"
2. [AGENT]     step(("play", 2))  → card played, total updated, phase stays "decision"
3. [AGENT]     step(("play", 0))  → card played, total updated, phase stays "decision"
4. [AGENT]     step("stand")      → turn ends, bust check, turn transition
                                   → next turn begins: NEW field card drawn for next player
```

Steps 2 and 3 do NOT trigger a field card draw. Only step 4 ending the turn causes the next turn to begin (which draws a new field card).

**Implementation**: The field card draw should happen inside the turn transition logic (when advancing to a new turn), not inside `step()`. The `step()` method should only process the player's chosen action. The `start_round()` method draws the first field card for Player A's first turn.

### 1.4 The `step` Method

The game engine exposes a single `step(action)` method where `action` is one of:

- `"hit"` - end turn, continue playing
- `"stand"` - end turn, lock total
- `("play", index)` - play the hand card at the given index in the player's hand list

This is the **sequential decision tree** approach. It was chosen because:
1. The action space is small and fixed: at most 6 discrete actions (hit, stand, play card at index 0/1/2/3)
2. It naturally handles the variable number of hand cards (just mask invalid indices)
3. It is trivial to implement action masking
4. Standard RL algorithms (PPO, DQN) work well with small discrete action spaces

**Why not the flattened approach (Option A)?** The flattened approach encodes "which subset of hand cards to play" and "hit or stand" as a single action. This creates up to 32 actions (2^4 subsets x 2 end actions). While workable, it creates a large action space where most actions are invalid most of the time, and it's harder to reason about. The sequential approach is simpler and equally expressive.

### 1.5 Turn Transition Logic

After the active player chooses hit or stand:

1. **Bust check**: If `player.total > 20`, the round ends immediately. The other player wins the round.
2. **Stand check**: If the active player just stood:
   - If the other player has also stood, resolve the round (compare totals).
   - If the other player has NOT stood, switch active player and begin their turn (auto-draw field card).
3. **Hit**: Switch active player and begin their turn (auto-draw field card). But if the other player has stood, the active player gets another turn instead (auto-draw again for the same player).

**Important edge case**: When one player has stood and the other hits, the non-standing player keeps taking consecutive turns. The standing player never gets another turn.

### 1.6 Round Resolution

When a round ends:
- Set `last_round_winner` to `0`, `1`, or `None` (draw). This is an explicit field on `PazaakGame`.
- Update `round_wins` for the winner (if not a draw)
- Check if either player has 3 wins. If so, set `game_winner` (explicit field) and `phase = "game_over"`.
- Otherwise, call `start_round()` which resets totals and stood flags but **preserves hand cards**.

### 1.7 Observability Helper

Provide a method `get_observation(player_index)` that returns only what that player can see (per Section 7 of game_rules.md):

```python
def get_observation(self, player_idx: int) -> dict:
    me = self.players[player_idx]
    opp = self.players[1 - player_idx]
    return {
        "my_total": me.total,
        "opp_total": opp.total,
        "my_stood": me.stood,
        "opp_stood": opp.stood,
        "my_hand": list(me.hand),            # Fixed 4 slots: values or None
        "my_hand_count": sum(1 for c in me.hand if c is not None),
        "opp_hand_count": sum(1 for c in opp.hand if c is not None),  # Non-None count, NOT len()
        "opp_used_hand": list(opp.used_hand),# Visible (played to field)
        "my_round_wins": self.round_wins[player_idx],
        "opp_round_wins": self.round_wins[1 - player_idx],
    }
```

### 1.8 Legal Actions Helper

```python
def legal_actions(self) -> list:
    if self.phase != "decision":
        return []
    actions = ["hit", "stand"]
    for i in range(len(self.players[self.active_player].hand)):
        actions.append(("play", i))
    return actions
```

### 1.9 Testing the Engine

Write unit tests that verify:
- Field cards are drawn uniformly from 1-10
- Hand cards persist across rounds but not across games
- Busting ends the round immediately
- Hand cards can rescue from bust before hit/stand
- Draws don't award round wins
- Game ends at 3 round wins
- Player A always goes first in every round
- Standing player takes no further turns
- Both players standing triggers resolution
- Multiple hand cards can be played in one turn (sequential play actions)

---

## Step 2: Heuristic Opponent (`heuristic.py`)

This module contains two rule-based opponents of increasing strength. Both receive a game observation (the dict from `get_observation`) and return an action. They share the agent interface defined in Section 2.5.

### 2.1 Simple Heuristic (`simple_heuristic_agent`)

A minimal opponent that plays like a blackjack dealer: no hand card usage, fixed stand threshold.

Rules:
1. **If total >= 17**: return `"stand"`
2. **Otherwise**: return `"hit"`

That's it. It never plays hand cards. This agent exists to give the RL model an easy first opponent that still plays a recognizable, coherent game. A model that can't consistently beat this has fundamental problems.

### 2.2 Full Heuristic (`heuristic_agent`)

A stronger opponent that uses hand cards and reacts to opponent state.

Rules, applied in priority order:

1. **If total == 20**: return `"stand"`
2. **If a hand card can make total == 20**: return `("play", index)` for that card, then on the next call return `"stand"` (the gymnasium wrapper will call this repeatedly until hit/stand is chosen)
3. **If opponent has stood and opponent's total > my total**: return `"hit"`
4. **If total > 20 and a negative hand card can bring it to <= 20**: return `("play", index)` for the best negative card (the one that brings total closest to 20 without exceeding it). After playing, if new total is 18-20 return `"stand"`, otherwise return `"hit"`. Since this is sequential, the play and hit/stand happen in separate calls.
5. **If total is 18 or 19**: return `"stand"`
6. **Otherwise**: return `"hit"`

### 2.3 Implementation Detail: Stateless Design

The heuristic must be **stateless** - it makes a decision based purely on the current observation. Since the sequential action space means the heuristic might be called multiple times per turn (once to play a card, once to hit/stand), it must re-evaluate the rules each time.

This works naturally: after playing a card that makes total == 20, the next call will see total == 20 and Rule 1 triggers "stand". After playing a negative card to recover from bust, Rule 4/5/6 will fire based on the new total. No state tracking is needed.

### 2.4 Rule 4 Detail: Bust Recovery

When total > 20, find the negative card that produces the highest total <= 20:

```python
best_card_idx = None
best_new_total = -inf
for i, card in enumerate(hand):
    if card < 0:
        new_total = total + card
        if new_total <= 20 and new_total > best_new_total:
            best_new_total = new_total
            best_card_idx = i
```

If no single negative card can fix the bust, try combinations of two or more negative cards. However, for simplicity, single-card recovery is a reasonable approximation since having to play 2+ negative cards to recover from >20 is rare and already a very bad position.

**Design decision**: Only attempt single-card bust recovery. Multi-card recovery is extremely rare (would require total > 26 with appropriate negative cards) and not worth the complexity for a heuristic opponent.

---

## Step 2.5: Agent Interface Convention

All agents (heuristic, random, trained model) must conform to the same callable interface so they can be used interchangeably in both the gymnasium wrapper (as an opponent) and the playback tool (as either player).

### 2.5.1 Interface Definition

An agent is any callable with this signature:

```python
def agent(observation: dict, game: PazaakGame) -> str | tuple:
```

- `observation`: the dict returned by `game.get_observation(player_idx)` for the agent's player
- `game`: the game engine instance (read-only access - the agent should not call `game.step()` itself, only inspect state if needed for action masking)
- Returns: `"hit"`, `"stand"`, or `("play", slot_index)`

The caller is responsible for calling the agent repeatedly within a turn until the agent returns `"hit"` or `"stand"`. The agent must return a **single action per call**, not a sequence.

### 2.5.2 Concrete Implementations

**Simple heuristic agent** (`heuristic.py`):
```python
def simple_heuristic_agent(observation: dict, game: PazaakGame) -> str | tuple:
    # Stand on 17+, otherwise hit. Never plays hand cards.
```

**Full heuristic agent** (`heuristic.py`):
```python
def heuristic_agent(observation: dict, game: PazaakGame) -> str | tuple:
    # Applies the 6 rules from Section 2.2 using observation fields
    # Does not need `game` - only uses `observation`
```

**Random agent** (can live in `playback.py` or a shared utils file - used for engine stress testing only, not training):
```python
def random_agent(observation: dict, game: PazaakGame) -> str | tuple:
    legal = game.legal_actions()
    return random.choice(legal)
```

**Trained model agent** (wrapper around MaskablePPO):
```python
class ModelAgent:
    def __init__(self, model_path: str):
        self.model = MaskablePPO.load(model_path)

    def __call__(self, observation: dict, game: PazaakGame) -> str | tuple:
        obs_array = observation_to_array(observation)
        mask = action_masks_from_game(game)
        action_int, _ = self.model.predict(obs_array, action_masks=mask, deterministic=False)
        return int_to_action(int(action_int))
        # 0 → "hit", 1 → "stand", 2 → ("play", 0), etc.
```

### 2.5.3 Usage in Gymnasium Wrapper

The gymnasium wrapper's opponent is an agent callable. During the opponent's turn, the wrapper calls it in a loop:

```python
while self.game.phase == "decision" and self.game.active_player == 1:
    obs = self.game.get_observation(player_idx=1)
    action = self.opponent_agent(obs, self.game)
    self.game.step(action)
```

### 2.5.4 Usage in Playback

The playback tool uses the same interface for both players:

```python
agent = agent_a if game.active_player == 0 else agent_b
obs = game.get_observation(game.active_player)
action = agent(obs, game)
game.step(action)
```

Because the interface is identical, any agent type can be plugged into any player slot in any context.

---

## Step 3: Gymnasium Environment (`gymnasium_env.py`)

This wraps the game engine as a `gymnasium.Env` for RL training.

### 3.1 Action Space

```python
self.action_space = gymnasium.spaces.Discrete(6)
```

Actions are:
- `0`: Hit
- `1`: Stand
- `2`: Play hand card at index 0
- `3`: Play hand card at index 1
- `4`: Play hand card at index 2
- `5`: Play hand card at index 3

Invalid actions are handled via action masking.

### 3.2 Observation Space

A flat `Box` space with 16 floats, all normalized to roughly [-1, 1]:

| Index | Field | Normalization |
|-------|-------|---------------|
| 0 | My total | / 20 |
| 1 | Opponent total | / 20 |
| 2 | My stood | 0 or 1 |
| 3 | Opponent stood | 0 or 1 |
| 4 | My hand card count | / 4 |
| 5 | Opponent hand card count | / 4 |
| 6 | My round wins | / 3 |
| 7 | Opponent round wins | / 3 |
| 8-11 | My hand cards (4 slots) | / 6 (0 if slot empty) |
| 12-15 | Opponent used hand cards (4 slots) | / 6 (0 if slot empty) |

### 3.3 Action Masking

The environment must implement `action_masks() -> np.ndarray` (boolean array of length 6) for use with `sb3_contrib.MaskablePPO`.

```python
def action_masks(self) -> np.ndarray:
    mask = np.zeros(6, dtype=bool)
    legal = self.game.legal_actions()
    for action in legal:
        if action == "hit":
            mask[0] = True
        elif action == "stand":
            mask[1] = True
        elif isinstance(action, tuple) and action[0] == "play":
            mask[2 + action[1]] = True
    return mask
```

### 3.4 Step Logic

The gymnasium `step(action)` method has branching logic depending on whether the agent played a hand card or ended their turn. The pseudocode below covers every case:

```python
def step(self, action: int):
    # 1. Convert integer action to engine action
    engine_action = self._int_to_action(action)
    #    0 → "hit", 1 → "stand", 2 → ("play", 0), 3 → ("play", 1), etc.

    # 2. Apply to engine
    self.game.step(engine_action)

    # 3. If the agent played a hand card (not hit/stand), return immediately.
    #    The agent gets another decision step. No opponent turn. No reward.
    if isinstance(engine_action, tuple):  # was a ("play", idx) action
        return self._get_obs(), 0.0, False, False, self._get_info()

    # 4. The agent chose hit or stand. Their turn is now over.
    #    The engine has already done the bust check and turn transition.
    #    Now we need to handle everything until it's the agent's turn again
    #    (or the game ends).

    reward = 0.0

    while True:
        # 4a. Check if the game is over
        if self.game.phase == "game_over":
            reward += 1.0 if self.game.game_winner == 0 else -1.0
            return self._get_obs(), reward, True, False, self._get_info()

        # 4b. Check if a round just ended (but game continues)
        if self.game.phase == "round_over":
            if self.game.last_round_winner == 0:
                reward += 0.3
            elif self.game.last_round_winner == 1:
                reward -= 0.3
            # Start the next round (this draws Player A's first field card)
            self.game.start_round()
            # Fall through to check whose turn it is now

        # 4c. If it's the agent's turn (Player 0), return control
        if self.game.active_player == 0 and self.game.phase == "decision":
            return self._get_obs(), reward, False, False, self._get_info()

        # 4d. It's the opponent's turn. Run their full turn.
        #     The opponent makes sequential decisions just like the agent.
        while (self.game.phase == "decision"
               and self.game.active_player == 1):
            opp_obs = self.game.get_observation(player_idx=1)
            opp_action = self.opponent_policy(opp_obs, self.game)
            self.game.step(opp_action)

        # Loop back to check for round_over / game_over after opponent's turn
```

**Key points about this loop**:
- After the agent's turn, the opponent might take multiple turns (if the agent has stood, the opponent keeps playing). The `while True` loop handles this by repeatedly running opponent turns until it's the agent's turn again or the game ends.
- Round transitions happen inside this loop. If the opponent busts (ending the round) and the game isn't over, a new round starts and the agent gets their first turn observation with the field card already drawn.
- Reward accumulates: if a round ends AND the game ends in the same transition, the agent gets both the round reward and the game reward.
- The agent only ever sees the game in `"decision"` phase with `active_player == 0`, or as a terminal state.

**Design decision**: The agent sees intermediate states when playing hand cards. This means a single "turn" (draw + play cards + hit/stand) may take 1-5 gymnasium steps from the agent's perspective. This is intentional - it lets the agent learn sequentially which cards to play.

### 3.5 Reward Shaping

```python
def _compute_reward(self) -> float:
    if self.game.phase == "game_over":
        return 1.0 if self.game.game_winner == 0 else -1.0
    if self.game.phase == "round_over" or self._round_just_ended:
        winner = self.game.last_round_winner
        if winner == 0:
            return 0.3
        elif winner == 1:
            return -0.3
        else:
            return 0.0  # Draw
    return 0.0
```

Game win/loss gives +/-1.0. Round win/loss gives +/-0.3. This small round reward helps the agent learn faster early on by getting more frequent signal, but the game outcome dominates.

**Design decision**: Round rewards are 0.3. The exact value is a hyperparameter. 0.3 was chosen because with 3 rounds needed to win, 3 * 0.3 = 0.9 < 1.0, so the game reward still dominates. This can be tuned later.

### 3.6 Episode Boundaries

An episode = one full game (first to 3 round wins). The environment returns `terminated=True` only when the game is over. Between rounds within a game, the environment continues.

`reset()` starts a brand new game: new hand cards drawn, round wins reset to 0.

### 3.7 Wrapping for sb3-contrib

The environment must be compatible with `sb3_contrib.MaskablePPO`. This requires:
- The `action_masks()` method defined above
- Wrapping with `sb3_contrib.common.wrappers.ActionMasker` at environment creation time

```python
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.action_masks()

env = ActionMasker(PazaakGymnasiumEnv(...), mask_fn)
```

---

## Step 4: Training Pipeline

### 4.1 Phase 1: Train vs Simple Heuristic (`train_vs_heuristic.py`)

The model first learns basic Pazaak against the simple heuristic (stand on 17+, no hand cards).

```
1. Create vectorized environments (n_envs=8) with simple_heuristic_agent as opponent
2. Initialize MaskablePPO with MlpPolicy
3. Train for 300,000 timesteps
4. Save model checkpoint as "phase1_final.zip"
5. Evaluate: play 1000 games vs simple heuristic, report win rate
```

The model should reach a high win rate (>70%) against this opponent. If it can't, something is wrong with the environment or training setup. This phase is as much a validation step as a training step.

### 4.2 Phase 2: Train vs Full Heuristic (`train_vs_heuristic.py`)

The Phase 1 model is loaded and continues training against the full 6-rule heuristic.

```
1. Load Phase 1 model
2. Create vectorized environments (n_envs=8) with heuristic_agent as opponent
3. Train for 500,000 timesteps
4. Save model checkpoint as "phase2_final.zip"
5. Evaluate: play 1000 games vs full heuristic, report win rate
```

The model should reach >50% win rate against the full heuristic. This opponent uses hand cards and reacts to the model's state, so it's a meaningful challenge.

### 4.1/4.2 Shared Hyperparameters

Both heuristic phases use the same hyperparameters:

- `learning_rate=3e-4`
- `n_steps=2048` (steps per env before update)
- `batch_size=256`
- `gamma=0.99`
- `ent_coef=0.01` (encourage exploration)
- `n_epochs=10`
- `clip_range=0.2`
- Policy network: MlpPolicy default (2 hidden layers of 64 units)

These are standard PPO defaults and should work as a starting point. Both phases use the same `train_vs_heuristic.py` script, which accepts the opponent type and starting checkpoint as arguments.

### 4.3 Phase 3: Self-Play (`train_self_play.py`)

Self-play progressively improves the agent by playing against previous versions of itself.

```
1. Load the Phase 2 model
2. Save it as snapshot_0
3. For each self-play iteration (e.g., 5 iterations):
   a. Create opponent pool: all snapshots + heuristic
   b. At each episode, randomly select an opponent from the pool
   c. Train for 200,000 timesteps
   d. Evaluate vs heuristic and vs latest snapshot
   e. Save model as new snapshot
4. Save final model
```

#### 4.3.1 Opponent Selection Strategy

At the start of each episode, choose the opponent:
- 10% chance: simple heuristic
- 10% chance: full heuristic
- 80% chance: random snapshot from the pool (uniform random)

**Design decision**: Always keep both heuristics in the opponent pool. Without them, the agent can "drift" and become specialized at beating only its own playstyle, potentially losing to simple strategies. The 20/80 split ensures the agent remains robust.

#### 4.3.2 Using a Snapshot as Opponent

When a snapshot model is the opponent:

```python
def snapshot_opponent(game, snapshot_model):
    obs = game.get_observation(player_idx=1)  # Opponent's perspective
    obs_array = observation_to_array(obs)     # Convert to numpy
    mask = game.get_action_mask_for(player_idx=1)  # Opponent's legal actions
    action, _ = snapshot_model.predict(obs_array, action_masks=mask, deterministic=False)
    return int_to_game_action(action)
```

**Critical**: The observation must be from the **opponent's perspective** (player index 1), not player 0. The snapshot model was trained as player 0, so we must flip the observation: "my total" becomes the opponent's total, "opponent total" becomes player 0's total, etc.

This is already handled by calling `game.get_observation(player_idx=1)` on the engine (which returns the game from Player 1's perspective). However, you also need a helper `observation_to_array(obs_dict) -> np.ndarray` that converts the dict into the 16-float normalized array. This helper is shared between the gymnasium wrapper (which builds obs for Player 0) and the snapshot opponent (which builds obs for Player 1). The mapping is:

```python
def observation_to_array(obs: dict) -> np.ndarray:
    hand_slots = obs["my_hand"]  # list of 4 values or None
    opp_used = obs["opp_used_hand"]  # list of 0-4 values

    arr = np.array([
        obs["my_total"] / 20,           # index 0
        obs["opp_total"] / 20,          # index 1
        float(obs["my_stood"]),         # index 2
        float(obs["opp_stood"]),        # index 3
        obs["my_hand_count"] / 4,       # index 4  (count of non-None slots)
        obs["opp_hand_count"] / 4,      # index 5
        obs["my_round_wins"] / 3,       # index 6
        obs["opp_round_wins"] / 3,      # index 7
        (hand_slots[0] or 0) / 6,       # index 8   (None → 0)
        (hand_slots[1] or 0) / 6,       # index 9
        (hand_slots[2] or 0) / 6,       # index 10
        (hand_slots[3] or 0) / 6,       # index 11
        (opp_used[0] if len(opp_used) > 0 else 0) / 6,  # index 12
        (opp_used[1] if len(opp_used) > 1 else 0) / 6,  # index 13
        (opp_used[2] if len(opp_used) > 2 else 0) / 6,  # index 14
        (opp_used[3] if len(opp_used) > 3 else 0) / 6,  # index 15
    ], dtype=np.float32)
    return arr
```

Because `get_observation(player_idx)` already swaps "my" and "opp" fields based on the player index, the same `observation_to_array` function works for both players without any additional flipping logic.

#### 4.3.3 Snapshot Storage

Save snapshots to disk as `.zip` files (MaskablePPO's built-in save format). Load them on demand rather than keeping them all in memory. Keep at most the 10 most recent snapshots in the opponent pool to bound memory usage.

### 4.4 Evaluation (`evaluate.py`)

A standalone script that loads a model and evaluates it:

```
1. Load model from checkpoint file
2. Play N games (default 1000) against a specified opponent (heuristic or another model)
3. Report: win rate, draw rate, loss rate, average rounds per game
4. Optionally render a few games to stdout for inspection
```

### 4.5 Game Playback (`playback.py`)

A standalone tool that runs a game between any two agents and prints a detailed, human-readable turn-by-turn log. This is essential for verifying that the game engine and agents behave correctly.

#### 4.5.1 Purpose

Playback is a debugging and validation tool, not part of the training loop. Use it to:
- Verify the game engine follows the rules (e.g. hand cards persist, bust is checked correctly)
- Check the heuristic opponent behaves as expected against known scenarios
- Watch trained models play and assess whether their decisions make sense
- Catch bugs like hand cards being redrawn, duplicate card plays, or incorrect turn order

#### 4.5.2 Output Format

Each game should be printed as a structured log. Example output:

```
=== GAME START ===
Player A side deck: [+1, +2, +3, -1, -2, ...] (20 cards)
Player B side deck: [+1, +2, +3, -1, -2, ...] (20 cards)
Player A hand: [+3, -2, +1, -4]
Player B hand: [hidden - 4 cards]

--- Round 1 (Score: A=0, B=0) ---

  Turn 1 (Player A):
    Field card drawn: 7        Total: 7
    Hand: [+3, -2, +1, -4]
    Action: HIT

  Turn 2 (Player B):
    Field card drawn: 4        Total: 4
    Hand: [hidden - 4 cards]
    Action: HIT

  Turn 3 (Player A):
    Field card drawn: 9        Total: 16
    Hand: [+3, -2, +1, -4]
    Action: plays +3            Total: 19
    Hand: [-2, +1, -4]
    Action: STAND

  Turn 4 (Player B):
    Field card drawn: 8        Total: 12
    Hand: [hidden - 4 cards]
    Action: HIT

  Turn 5 (Player B):
    Field card drawn: 10       Total: 22
    Hand: [hidden - 4 cards]
    Action: plays -2            Total: 20
    Opponent sees played card: -2
    Hand: [hidden - 3 cards]
    Action: STAND

  Round 1 result: Player B wins (20 vs 19)

--- Round 2 (Score: A=0, B=1) ---
  ...

=== GAME OVER ===
Player B wins the game 3-1
Total rounds played: 4
Player A hand cards remaining: [-2, -4]
Player B hand cards remaining: [+1]
```

#### 4.5.3 Key Log Details

Every turn must show:
1. **Which player** is acting
2. **Field card drawn** and **new total** after the draw
3. **Hand card(s) played** (if any), with the updated total after each card
4. **Final action**: hit or stand
5. **Hand card count** for the opponent (values hidden unless it's a "god mode" view)

At round boundaries, show:
- Round result and updated score
- Confirmation that hand cards carried over (implicitly, by showing remaining cards)

At game end, show:
- Final score and winner
- Remaining hand cards for both players (useful for verifying persistence)

#### 4.5.4 View Modes

Two viewing modes:

- **Player perspective** (default): Shows Player A's hand values but hides Player B's unplayed hand (shows count only). Played cards from both sides are visible. This matches the information a real player would see.
- **God mode** (`--god`): Shows both players' hands at all times. Useful for debugging the engine and heuristic behaviour.

#### 4.5.5 Agent Types

The playback script should accept agent types for both players via command line:

```
python playback.py --player-a simple --player-b simple
python playback.py --player-a heuristic --player-b heuristic
python playback.py --player-a model:checkpoints/phase2_final.zip --player-b heuristic
python playback.py --player-a model:checkpoints/self_play_3.zip --player-b model:checkpoints/self_play_1.zip --god
python playback.py --player-a random --player-b heuristic
```

Supported agent types:
- `simple` - the simple heuristic (stand on 17+, no hand cards) from `heuristic.py`
- `heuristic` - the full 6-rule heuristic from `heuristic.py`
- `random` - picks a random legal action each step (useful for engine stress testing only)
- `model:<path>` - a trained MaskablePPO model loaded from a checkpoint file

#### 4.5.6 Implementation

The playback tool does **not** use the gymnasium wrapper. It interacts directly with the game engine:

```python
def play_game(game: PazaakGame, agent_a, agent_b, god_mode=False):
    while game.phase != "game_over":
        player_idx = game.active_player
        agent = agent_a if player_idx == 0 else agent_b
        obs = game.get_observation(player_idx)

        # Log the field card draw (already happened in engine)
        log_field_draw(game, player_idx, god_mode)

        # Agent makes sequential decisions until hit/stand
        while game.phase == "decision" and game.active_player == player_idx:
            action = agent(obs)
            log_action(game, player_idx, action, god_mode)
            game.step(action)
            obs = game.get_observation(player_idx)

        # Log round/game results if applicable
        if game.phase in ("round_over", "game_over"):
            log_round_result(game)
```

This direct engine interaction (bypassing the gymnasium wrapper) means playback is a true end-to-end test of the engine logic itself.

#### 4.5.7 Seeded Replay

Accept an optional `--seed <int>` argument. When provided, the game engine RNG is seeded, making the field card sequence deterministic. This allows replaying the exact same game to compare different agents' decisions given identical card draws.

---

## Step 5: File Structure

```
pazaak/
├── game_engine.py          # Core game logic (Step 1)
├── heuristic.py            # Heuristic opponent (Step 2)
├── gymnasium_env.py        # Gymnasium wrapper (Step 3)
├── train_vs_heuristic.py   # Phase 1 training (Step 4.1)
├── train_self_play.py      # Phase 2 training (Step 4.2)
├── evaluate.py             # Evaluation script (Step 4.3)
├── playback.py             # Turn-by-turn game viewer (Step 4.4)
├── test_game_engine.py     # Unit tests for game engine (Step 1.9)
├── game_rules.md           # Game rules reference
├── implementation_plan.md  # This file
└── checkpoints/            # Saved model files (created at runtime)
```

---

## Step 6: Implementation Order

Follow this exact order. Each step should be fully working and tested before moving to the next.

1. **`game_engine.py`** - Implement and manually verify the game engine. This is the foundation. Get this right.
2. **`test_game_engine.py`** - Write and run unit tests for the engine.
3. **`heuristic.py`** - Implement both heuristic opponents (simple and full). Test them by running games against each other and verifying the full heuristic wins consistently.
4. **`gymnasium_env.py`** - Build the gymnasium wrapper. Test it with `gymnasium.utils.env_checker.check_env()` to verify API compliance.
5. **`playback.py`** - Build the game viewer. Use it to watch simple-vs-simple, heuristic-vs-heuristic, and simple-vs-heuristic games. Verify the engine rules are correct by reading the logs.
6. **`train_vs_heuristic.py`** - Train Phase 1 (vs simple heuristic). Verify the model reaches >70% win rate. Then train Phase 2 (vs full heuristic). Verify >50% win rate.
7. **`evaluate.py`** - Build evaluation tooling.
8. **`train_self_play.py`** - Train Phase 3.

---

## Key Design Decisions Summary

These are decisions that were made deliberately and should not be changed without good reason:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Action space | Sequential (6 discrete) | Small, fixed size, natural masking, expressive |
| Actions per turn | Multiple gym steps per game turn | Agent learns card-play sequencing |
| Field card draw | Automatic (not an action) | It's mandatory, so making it an action wastes steps |
| Round handling | Single episode = full game | Agent learns resource management across rounds |
| Observation | 16 floats, normalized | Compact, all relevant info included |
| RL algorithm | MaskablePPO (sb3-contrib) | Handles illegal action masking natively |
| Training progression | Simple heuristic → full heuristic → self-play | Graduated difficulty; Phase 1 doubles as validation |
| Self-play opponent pool | 10% simple + 10% full heuristic + 80% snapshot | Prevents catastrophic forgetting |
| Snapshot perspective | Flip observation for opponent | Snapshots trained as P0; opponent is P1 |
| Heuristic bust recovery | Single-card only | Multi-card recovery too rare to justify complexity |
| Round reward | +/- 0.3 | Frequent signal, but game outcome still dominates |

---

## Dependencies

```
gymnasium>=0.29
stable-baselines3>=2.1
sb3-contrib>=2.1
numpy
torch
```

This project uses **uv** for dependency management. Install with: `uv add gymnasium stable-baselines3 sb3-contrib numpy torch`

---

## Known Limitations and Future Work

- The observation does not encode which specific field cards have been drawn (only the total). This means the agent cannot reason about what field cards are "likely" next. Since the deck is infinite with replacement and uniform, this information has no predictive value anyway.
- The observation does not encode turn order or how many turns have elapsed in the current round. The totals and hand card counts serve as rough proxies.
- Side deck composition is fixed at training time. To generalize across different side decks, the observation would need to encode the side deck or use a meta-learning approach.
- The heuristic opponent does not reason about multi-card combinations. A stronger heuristic could consider playing multiple cards strategically.
