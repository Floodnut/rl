import numpy as np
import random


class E_Greedy:
    def __init__(self):
        pass

    def action(q_values: list, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return np.argmax(q_values)
