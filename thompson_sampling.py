import numpy as np


class ThompsonSampling:
    def __init__(self):
        pass

    def thompson_sampling(alpha: list, beta: list) -> int:
        samples = [np.random.beta(a, b) for a, b in zip(alpha, beta)]
        return np.argmax(samples)
