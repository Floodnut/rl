import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product


class Arm:
    def __init__(self, win_rate):
        self.win_rate = win_rate

    def pull(self):
        return np.random.binomial(1, self.win_rate)


# 확률 조합 설정
win_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results = np.zeros((len(win_rates), len(win_rates), 5, 4))


class Greedy:
    def select_action(self, q_values):
        return np.argmax(q_values)


class EGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return np.argmax(q_values)


class UCB:
    def __init__(self, c):
        self.c = c

    def select_action(self, q_values, n_plays, total_plays):
        ucb_values = q_values + self.c * np.sqrt(np.log(total_plays) / (n_plays + 1e-6))
        return np.argmax(ucb_values)


class ThompsonSampling:
    def select_action(self, alpha, beta):
        samples = [np.random.beta(a, b) for a, b in zip(alpha, beta)]
        return np.argmax(samples)


algorithms = [Greedy(), EGreedy(0.1), UCB(2), ThompsonSampling()]

for i, (a, b) in enumerate(product(win_rates, repeat=2)):
    for _ in range(5):
        arm_a = Arm(a)
        arm_b = Arm(b)
        q_values = [0, 0]
        n_plays = [0, 0]
        alpha = [1, 1]
        beta = [1, 1]
        total_plays = 0
        for t in range(1000):
            actions = [
                algorithm.select_action(q_values, n_plays, total_plays)
                for algorithm in algorithms
            ]
            for k, action in enumerate(actions):
                reward = arm_a.pull() if action == 0 else arm_b.pull()
                n_plays[action] += 1
                q_values[action] += (reward - q_values[action]) / n_plays[action]
                if reward == 1:
                    alpha[action] += 1
                else:
                    beta[action] += 1
                results[i, i, _, k] = reward

plt.figure()
for k in range(4):
    avg_rewards = np.mean(results[i, i, :, k])
    plt.scatter(a, b, c=f"C{k}", s=avg_rewards * 50, label=str(algorithms[k]))

plt.xlabel("E(P(A))")
plt.ylabel("E(P(B))")
plt.legend()
plt.show()
