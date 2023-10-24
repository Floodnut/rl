import numpy as np
import random


class Greedy:
    # arm의 가치
    def action(self, q_values: list) -> int:
        return np.argmax(q_values)


class E_Greedy:
    # 0~1 사이의 epsilon
    def __init__(self, e):
        self.e = e

    # arm의 가치
    def action(self, q_values: list) -> int:
        if random.random() < self.e:
            return random.randint(0, len(q_values) - 1)
        else:
            return np.argmax(q_values)


class UCB:
    # exploration과 exploitation 조절
    def __init__(self, c):
        self.c = c

    def action(self, q_values: list, n_plays: list, total_plays: int) -> int:
        ucb_values = q_values + self.c * np.sqrt(
            np.log(total_plays) / (np.array(n_plays) + 0.000001)
        )
        return np.argmax(ucb_values)


class ThompsonSampling:
    def __init__(self):
        pass

    # 성공, 실패 횟수 (베타 분포)
    def action(self, alpha: list, beta: list) -> int:
        samples = [np.random.beta(a, b) for a, b in zip(alpha, beta)]
        return np.argmax(samples)
