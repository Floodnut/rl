"""
사진과 같은 environment에서 policy evaluation과 value iteration을 사용해서 수렴할때의 value map을 return하는 함수를 작성하세요
"""

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import product


# 환경 정의
class Environment:
    def __init__(self, wall: tuple):
        self.width = 8
        self.height = 8
        self.start = (2, 6)
        self.goal = (6, 7)
        self.wall = wall
        self.theta = 0.000001  # 수렴 조건
        self.gamma = 0.8  # discount factor


class Evaluation(Environment):
    def __init__(self, wall):
        super().__init__(wall)
        self.actions = ["North", "East", "South", "West"]

    def policy(self, policy: dict) -> list:

        self._initiate_value()

        while True:
            # 현재 가치 함수와 이전 가치 함수의 차이
            delta = 0

            for x, y in product(size, repeat=2):
                state = (x, y)
                if state == self.goal or state in self.wall:
                    continue

                v = 0
                for action in self.actions:
                    next_state = self._get_next_state(state, action)
                    r = self._get_reward(state)
                    v += policy[state][action] * (
                        r + self.gamma * self.values[next_state[0]][next_state[1]]
                    )
                delta = max(delta, abs(self.values[x][y] - v))
                self.values[x][y] = v

            if delta < self.theta:
                break

        return self.values

    def value(self) -> list:

        self._initiate_value()

        while True:
            delta = 0
            for x, y in product(size, repeat=2):
                state = (x, y)
                if state == self.goal or state in self.wall:
                    continue

                # 갈 수 없는 곳의 v를 최소값으로 설정해서 못가게 함
                v = -np.inf

                for action in self.actions:
                    next_state = self._get_next_state(state, action)
                    r = self._get_reward(state)
                    v = max(
                        v, r + self.gamma * self.values[next_state[0]][next_state[1]]
                    )
                delta = max(delta, abs(self.values[x][y] - v))
                self.values[x][y] = v

            if delta < self.theta:
                break

        return self.values

    # 갈 수 없는 곳의 값을 0으로 설정
    def _initiate_value(self):
        self.values = np.zeros((self.width, self.height))

        for (x, y) in self.wall:
            self.values[x][y] = 0

    # 보상 함수
    def _get_reward(self, state: tuple) -> int:
        if state == self.goal:
            return 0
        else:
            return -1

    def _get_next_state(self, state: tuple, action: str) -> tuple:
        x, y = state
        if action == "North":
            x = x - 1 if x - 1 > 0 else 0
        elif action == "West":
            y = y - 1 if y - 1 > 0 else 0
        elif action == "East":
            y = y + 1 if y + 1 < self.height - 1 else self.height - 1
        elif action == "South":
            x = x + 1 if x + 1 < self.width - 1 else self.width - 1

        return x, y


if __name__ == "__main__":
    size = [0, 1, 2, 3, 4, 5, 6, 7]
    walls = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (2, 2),
        (2, 3),
        (2, 5),
        (3, 3),
        (3, 4),
        (4, 1),
        (4, 4),
        (4, 6),
        (5, 2),
        (5, 4),
        (6, 5),
    ]

    evalutation = Evaluation(walls)

    initial_policy = dict()
    for x, y in product(size, repeat=2):
        initial_policy[(x, y)] = {
            "North": 1 / 4,
            "East": 1 / 4,
            "South": 1 / 4,
            "West": 1 / 4,
        }

    policy_result = evalutation.policy(initial_policy)
    value_result = evalutation.value()

    figure = px.imshow(policy_result, text_auto=True, color_continuous_scale="Viridis")
    figure.show()
