import os
from typing import Optional

import numpy as np
import random

## Tic tac toe example:
## https://github.com/haje01/gym-tictactoe/blob/master/gym_tictactoe/env.py

## Texas Hold-em
## https://github.com/dickreuter/neuron_poker/blob/master/gym_env/env.py

import gym
from gym import spaces
from gym.error import DependencyNotInstalled


def cmp(a, b):
    return float(a > b) - float(a < b)


# Only main deck cards are 1-10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

side_deck = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]


# Add a card to your total
def draw_card():
    return int(random.choice(deck))


# Draws 4 cards from your side deck
def draw_hand(hand_size=4):
    return random.sample(side_deck, hand_size)


def play_card(hand, card_number, prev_total):
    new_total = prev_total + hand[card_number]
    hand[card_number] = 0
    return (hand, new_total)


def sum_table(table): 
    return sum(table)


def is_bust(table):  # Is this hand a bust?
    return sum_table(table) > 20


def score(table):  # What is the score of this table (0 if bust)
    return 0 if is_bust(table) else sum_table(table)



class PazaakEnv(gym.Env):
    """
    Pazaak is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 20 (without going over 20) than the dealers cards.

    ### Description
    Card Values:
    - Numerical cards (1-10) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with both players drawing 4 cards into their hand from their
    side deck. Then the first player draws a card onto their table.

    Each turn, the player has to take one additional card at random from the deck.
    They can then play any number of cards from their hand (including zero), 
    after which they must stand (stick with their current total and draw no more
    cards) or end their turn (allowing their opponent to play). If a players total
    is greater than 20 at the end of their turn, their opponent wins the game.

    The round continues until one player busts, or both players stand. When both 
    players stand, the player with the highest total (without exceeding 20) wins.
    If both players stand on the same total then the round is a tie.

    Play continues until one player has one 3 rounds, at which point they have 
    won the game. Cards from the players hand can only be played once per game.

    ### Action Space
    There are six possible actions: 
    - play card A from your hand, then stand
    - play card B from your hand, then stand
    - play card C from your hand, then stand
    - play card D from your hand, then stand
    - end turn
    - stand

    ### Observation Space
    The observation consists of a 6-tuple containing: 
    - how many rounds the player has won
    - how many rounds the opponent has won
    - the player's current sum
    - their opponents current sum
    - the players available hand
    - the number of remaining cards in their opponents hand

    ### Rewards
    - win round: ???
    - lose round: ???
    - draw round: ???
    - win game: ???
    - play a card: ???
    - end turn: ??? (stand-in for getting another card)

    ### Arguments

    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, sab=False):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))) ## Not sure what this should be
        self.render_mode = render_mode

    def step(self, action):
        assert self.action_space.contains(action) # First, check that this is a valid action
        if action == 0:
            print("play first card")


        if action == 4:
            print("Just end turn")

        if action == 5:
            print("Stand")


        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            terminated = True
            while sum_table(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return (sum_table(self.player), self.dealer[0], usable_ace(self.player))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        _, dealer_card_value, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


