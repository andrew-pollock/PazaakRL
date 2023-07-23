import random
import numpy as np

def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)


def agent_random(obs, config):
    if obs.other_bust:
        return "stand"
    for card in obs.cards:
        if obs.score + card == 20:
            return "card"
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)
