"""
heuristic.py - Rule-based Pazaak agents.

Two agents are provided:

  simple_heuristic_agent  - Stand on 17+, never play hand cards.
  heuristic_agent         - Full 6-rule strategy using hand cards and opponent awareness.

Both conform to the agent interface defined in implementation_plan.md §2.5:

    def agent(observation: dict, game: PazaakGame) -> str | tuple

They are stateless: each call receives the current observation and returns a
single action.  The caller loops until hit or stand is returned.

Gymnasium compatibility
-----------------------
Because the gymnasium wrapper drives the opponent by calling the agent in a
loop (see implementation_plan.md §2.5.3), no wrapper class is needed here.
Both functions can be passed directly as the `opponent_agent` callable::

    env = PazaakGymnasiumEnv(opponent_agent=heuristic_agent, ...)

If a bound method or partial is more convenient, a thin ``AgentWrapper``
helper is provided at the bottom of this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game_engine import PazaakGame


# ---------------------------------------------------------------------------
# Simple heuristic
# ---------------------------------------------------------------------------


def simple_heuristic_agent(observation: dict, game: "PazaakGame") -> str | tuple:
    """
    Minimal Blackjack-dealer-style agent.

    Rules (in order):
      1. If total >= 17 → stand
      2. Otherwise      → hit

    Never plays hand cards.
    """
    if observation["my_total"] >= 17:
        return "stand"
    return "hit"


# ---------------------------------------------------------------------------
# Full heuristic
# ---------------------------------------------------------------------------


def heuristic_agent(observation: dict, game: "PazaakGame") -> str | tuple:
    """
    Six-rule heuristic that uses hand cards and reacts to the opponent.

    Rules applied in strict priority order (implementation_plan.md §2.2):

      1. Total == 20                        → stand
      2. A hand card makes total == 20      → play that card  (stand fires next call)
      3. Opponent has stood AND opp_total > my_total → hit
      4. Total > 20 AND a single negative card rescues → play best rescue card
                                              then stand (18–20) or hit (< 18)
      5. Total is 18 or 19                  → stand
      6. Otherwise                          → hit

    This function is stateless; it re-evaluates all rules on every call.
    After playing a card (rules 2 or 4) the caller will invoke the agent
    again and the updated observation will naturally trigger the next rule.
    """
    total: int = observation["my_total"]
    opp_total: int = observation["opp_total"]
    opp_stood: bool = observation["opp_stood"]
    hand: list = observation["my_hand"]  # 4 slots: int or None

    # Rule 1 - already at 20
    if total == 20:
        return "stand"

    # Rule 2 - a hand card gets us to exactly 20
    for i, card in enumerate(hand):
        if card is not None and total + card == 20:
            return ("play", i)

    # Rule 3 - opponent has stood and is beating us; we must keep drawing
    if opp_stood and opp_total > total:
        return "hit"

    # Rule 4 - bust recovery with a single negative card
    if total > 20:
        best_idx = None
        best_new_total = float("-inf")
        for i, card in enumerate(hand):
            if card is not None and card < 0:
                new_total = total + card
                if new_total <= 20 and new_total > best_new_total:
                    best_new_total = new_total
                    best_idx = i
        if best_idx is not None:
            return ("play", best_idx)
        # No rescue card available - we will bust on hit or stand; return hit
        # so the round ends (stand also ends it; either is equivalent here).
        return "hit"

    # Rule 5 - comfortable standing range
    if total >= 18:
        return "stand"

    # Rule 6 - default: keep drawing
    return "hit"


# ---------------------------------------------------------------------------
# Optional thin wrapper (useful for testing or partial application)
# ---------------------------------------------------------------------------


class AgentWrapper:
    """
    Wraps a heuristic function so it can be stored as an object and called
    with the standard ``(observation, game)`` signature.

    Example::

        opponent = AgentWrapper(heuristic_agent)
        env = PazaakGymnasiumEnv(opponent_agent=opponent, ...)
    """

    def __init__(self, agent_fn):
        self._fn = agent_fn

    def __call__(self, observation: dict, game: "PazaakGame") -> str | tuple:
        return self._fn(observation, game)

    def __repr__(self) -> str:
        return f"AgentWrapper({self._fn.__name__})"
