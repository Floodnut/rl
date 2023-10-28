import random


class Arm:
    def __init__(self, probability):
        self.probability = probability

    def pull(self) -> int:
        if random.random() < self.probability:
            return 1

        return 0
