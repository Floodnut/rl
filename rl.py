import numpy as np
import random


class Greedy:
    def __init__(self):
        pass

    def action(self, a: int, b: int) -> int:
        return (0, a) if a > b else (1, b)


class E_Greedy:
    def __init__(self):
        pass

    def action(self, q_values: list, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return np.argmax(q_values)


class UCB:
    def __init__(self):
        pass

    def action(self, q_values: list, n_plays: list, c: float) -> int:
        total_plays = sum(n_plays)
        ucb_values = q_values + c * np.sqrt(
            np.log(total_plays) / (np.array(n_plays) + 1)
        )

        return np.argmax(ucb_values)


class ThompsonSampling:
    def __init__(self):
        pass

    def action(self, alpha: list, beta: list) -> int:
        samples = [np.random.beta(a, b) for a, b in zip(alpha, beta)]
        return np.argmax(samples)
