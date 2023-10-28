import numpy as np
import matplotlib.pyplot as plt


# 환경 정의
class Environment:
    def __init__(self, wall: tuple):
        self.width = 8
        self.height = 8
        self.start = (2, 6)
        self.goal = (6, 7)
        self.wall = wall


class Evaluation(Environment):
    def __init__(self, wall):
        super().__init__(wall)
        self.actions = ["North", "East", "South", "West"]
