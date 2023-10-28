import numpy as np
import random
import matplotlib.pyplot as plt

# 1. 두 개의 arm A, B를 생성
arms = [{"win_rate": 0.1}, {"win_rate": 0.1}]

# 3. 시뮬레이션
n_experiments = 5
n_tests = 4
n_iterations = 9

win_rates_A = [0.1 + i * 0.1 for i in range(n_iterations)]
win_rates_B = [0.1 + i * 0.1 for i in range(n_iterations)]

results = np.zeros((len(win_rates_A), len(win_rates_B), n_experiments, n_tests, 4))

print(results)

for exp in range(n_experiments):
    for i, win_rate_A in enumerate(win_rates_A):
        for j, win_rate_B in enumerate(win_rates_B):
            arms[0]["win_rate"] = win_rate_A
            arms[1]["win_rate"] = win_rate_B
            for t in range(n_tests):
                q_values = [0, 0]
                n_plays = [0, 0]
                alpha = [1, 1]
                beta = [1, 1]
                for _ in range(1000):
                    action_greedy = greedy(q_values)
                    action_e_greedy = e_greedy(q_values, epsilon=0.1)
                    action_ucb = ucb(q_values, n_plays, c=2)
                    action_thompson = thompson_sampling(alpha, beta)
                    actions = [
                        action_greedy,
                        action_e_greedy,
                        action_ucb,
                        action_thompson,
                    ]
                    for k, action in enumerate(actions):
                        reward = np.random.binomial(1, arms[action]["win_rate"])
                        n_plays[action] += 1
                        q_values[action] += (reward - q_values[action]) / n_plays[
                            action
                        ]
                        if reward == 1:
                            alpha[action] += 1
                        else:
                            beta[action] += 1
                        results[i, j, exp, t, k] = reward

# 4. 결과 플롯
algorithms = ["Greedy", "Epsilon-Greedy", "UCB", "Thompson Sampling"]
colors = ["r", "g", "b", "y"]

plt.figure()
for k in range(4):
    for i, win_rate_A in enumerate(win_rates_A):
        for j, win_rate_B in enumerate(win_rates_B):
            avg_rewards = np.mean(results[i, j, :, :, k])
            plt.scatter(
                win_rate_A,
                win_rate_B,
                c=colors[k],
                s=avg_rewards * 500,
                label=algorithms[k],
            )

plt.xlabel("E(P(A))")
plt.ylabel("E(P(B))")
# plt.legend()
plt.title("MAB")
plt.show()
