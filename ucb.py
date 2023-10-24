import numpy as np


class UCB:
    def __init__(self):
        pass

    def ucb(q_values: list, n_plays: list, c: float) -> int:
        total_plays = sum(n_plays)
        ucb_values = q_values + c * np.sqrt(
            np.log(total_plays) / (np.array(n_plays) + 1)
        )

        return np.argmax(ucb_values)
