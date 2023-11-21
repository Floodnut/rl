from typing import List, Optional
from itertools import product
import random
# import numpy as np
# import matplotlib.pyplot as plt


# 환경 정의
class Environment:
    def __init__(self):
        self.cards_count = [0] + [4] * 14
        self.theta = 0.000001  # 수렴 조건
        self.gamma = 0.8  # discount factor
        self.card_mapping = {
            1: 'A',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '10',
            11: 'J',
            12: 'Q',
            13: 'K'
        }
        self.card_values = [
            (0, ),
            (1, 11),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (10,),
            (10,),
            (10,),
        ]

class Agent:
    def __init__(self):
        self.given_cards = list()
        self.card_sum = 0
    
    def add_card(self, card: list):
        self.given_cards.append(card)
        self._sum_card(card)

    def _sum_card(self, new_cards: List[int]):
        self.card_sum += sum(new_cards)

class Dealer(Agent):
    def __init__(self):
        super().__init__()

    def need_update(self) -> bool:
        return False

class Player(Agent):
    def __init__(self):
        super().__init__()

    def need_update(self) -> bool:
        return False

class Blackjack(Environment):
    def __init__(self, dealer: Dealer, player: Player):
        super().__init__()
        self.dealer = dealer
        self.player = player
    
    def draw_random_cards(self, count: int = 1) -> List[int]:
        random.shuffle(self.card_values)
        return self.card_values.pop()

    def start(self):
        new_2_dealer_cards = self.draw_random_cards(count=2)
        self.dealer.add_card(new_2_dealer_cards)
        new_2_player_cards = self.draw_random_cards(count=2)
        self.player.add_card(new_2_player_cards)

        return self._strict_policy(self.player.card_sum, self.dealer.card_sum)
    
    def monte_carlo(self, num_episodes):
        wins = 0
        for _ in range(num_episodes):
            result = self.start()
            if result == 1:
                wins += 1
        return wins / num_episodes

    def _action_stay(self) -> None:
        return []
    
    def _action_hit(self, sum) -> Optional[List[int]]:
        if sum > 21:
            return self._action_stay()

        return self.draw_random_cards(count=1)

    def _strict_policy(self, player_sum, dealer_sum) -> int:
        if player_sum > 21:
            return -1
        elif player_sum == dealer_sum:
            return 0
        elif dealer_sum > 21 or player_sum > dealer_sum:
            return 1
        else:
            return -1

    def update_value(self):
        pass

    def update_policy(self):
        pass

    def update_reward(self):
        pass

if __name__ == "__main__" :
    blackjack = Blackjack(dealer=Dealer(), player=Player())
    result_mc = blackjack.monte_carlo(1000)