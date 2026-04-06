# Pazaak - Machine Learning & Game Engine Specification

## 1. Game Overview

Pazaak is a two-player, zero-sum, stochastic, imperfect-information card game inspired by Blackjack.

Each game consists of independent rounds. **The first player to win 3 rounds wins the game** (effectively best-of-5).

**Important**: Drawn rounds award no points to either player, so the game continues until one player reaches 3 round wins. This means a game could theoretically have more than 5 rounds if draws occur.

Key complexity drivers:
- Hidden opponent hand composition
- Persistent hand cards across rounds
- Sequential decision-making with stochastic draws
- Resource management across rounds

This makes Pazaak well-suited to Reinforcement Learning (RL) and Monte Carlo Tree Search (MCTS).

---

## 2. Card Systems

### 2.1 Main Deck (Field Cards)

- Infinite deck
- Card values: integers 1–10
- Uniform probability: 10% per value
- Cards are drawn with replacement
- No reshuffling required

### 2.2 Side Deck (Hand Cards)

- Each player builds a customizable 20-card side deck (specified as a parameter)
- **Each player may have a different side deck composition**
- Card values: −6 to −1 and +1 to +6
- Duplicates allowed
- At game start, each player draws 4 cards into their hand
- No additional hand cards are drawn during the game
- Unused hand cards persist across rounds
- A hand card can be used **once per game**
- **Note**: Side deck composition should be passed as a parameter when initializing the game

---

## 3. Turn Structure

### 3.1 Turn Order

- Player A always goes first in every round
- Player B always goes second
- This asymmetry is intentional and modeled explicitly

### 3.2 Turn Sequence

On a player’s turn:

1. **Draw exactly one field card** (mandatory)
2. Optionally play **one or more hand cards**
3. **Must** choose exactly one:
   - Hit (pass turn to opponent, continue playing this round)
   - Stand (lock total and take no further actions this round)

Hand cards:
- May only be played after the field card draw
- May only affect the player’s own total
- Multiple hand cards may be played in one turn in any order
- Hand cards add/subtract to the total in the order played
- However, since bust is checked only after Hit/Stand, play order does not affect outcome
- May be used to reduce total below 20 before bust resolution
- Once all 4 hand cards have been used, the player can only Hit or Stand for the remainder of the game

**Important**: A turn MUST end with either Hit or Stand. Even if no hand cards are played, the player must still choose Hit or Stand after drawing the field card.

---

## 4. Standing and Busting Rules

- Bust is evaluated **after the player ends their turn** (i.e., after choosing Hit or Stand)
- A player may use negative hand cards to reduce their total **before** choosing Hit or Stand
- Even if the field card draw causes the total to exceed 20, the player can still play negative hand cards to avoid busting before ending their turn
- Once a player chooses Hit or Stand, their current total is locked for bust evaluation
- Bust (total > 20) ends the round immediately with a loss for the busted player
- The bust check only occurs when the turn ends (Hit or Stand), not during hand card play

### Total Limits

- **Maximum**: Total > 20 results in bust (round loss)
- **Minimum**: There is no minimum total limit
- Players can theoretically stand on negative or zero totals, though this is strategically disadvantageous
- A player standing on a negative total would lose to any opponent standing on a non-negative total

---

## 5. Post-Stand Behavior

- A player who stands takes no further actions in the round
- The opposing player continues taking turns until they either:
  - Stand
  - Bust

---

## 6. Round Resolution

### 6.1 Win Conditions

- Bust → immediate loss of the round
- One stands, other busts → standing player wins the round
- Both stand:
  - Higher total wins
  - Equal totals → draw

### 6.2 Draw Handling

- Neither player scores a round win
- A new round begins, with Player A going first as always
- Hand cards that were used in the drawn round remain consumed
- The game continues until one player wins 3 rounds

---

## 7. Game State Observability

Each player observes:

- Their own field total
- Their opponent's field total
- Whether opponent has stood
- Their remaining hand cards (exact values)
- Opponent remaining hand card count
- Which hand cards opponent has already used
- Current round wins (0–2)
- Current game score (0–2)

Opponent hand values not yet played are hidden.

In other words, you can look at your own hand, and you can see how many cards are remaining in your opponent's hand. Plus both players can see everything that is on the field (the field cards that each player has drawn, and what cards they're played from their hand).
