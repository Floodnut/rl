# import random
# import matplotlib.pyplot as plt
import numpy as np


class Environment:
    """n-step TD error
    """

    def __init__(self, num_states: int):
        self.alpha = 0.1
        self.gamma = 0.95  # discount factor
        self.values = np.arange(num_states)
        self.num_states = num_states

        # policy
        self.behavior_policy = np.ones(num_states) / num_states  # 무작위 정책
        self.target_policy = np.zeros(num_states)
        self.target_policy[1:6] = 1

        self.target_state_values = np.arange(1, 6)


class Agent:
    """agent
    """
    def __init__(self):
        pass


class OffPolicy(Environment):
    """off-policy agent
    """

    def __init__(self, num_states: int):
        super().__init__(num_states)

    def __call__(self):
        pass

    def _action(self):
        return np.random.choice(
            np.arange(self.num_states), p=self.behavior_policy)

    def _td_error(self, current_state, next_state, reward, n):
        """TD error in a n-step TD
        """
        if n == 0:
            return 0

        target = reward + self.values[next_state] if next_state else 0
        return target + self.values[current_state] - self.values[current_state]

    def n_steps_td_error(self, current_state, n):
        """n-steps TD error

            현재 시간에서부터 n-steps 후까지의 보상과 가치 추정치 사이의 차이
                δ: 시간 t에서의 n-step TD 오차
                R(t+i): t+i의 보상.
                y: discount factor
                V: 상태 별 추정 가치

            δ = R(t+1) + yR(t+2) + ... + y^(n-1)R(t+n) + y^nV(t+n-1) - V(t)
        """
        n_step_err = 0
        for _ in range(n):
            action = self._action()
            next_state = action
            reward = 0
            n_step_err += self._td_error(current_state, next_state, reward, n)
            current_state = next_state

        return n_step_err

    def n_step_values(self, n):
        """n-step TD error
        """
        current_state = 0
        for _ in range(1000):
            n_step_err = 0
            n_step_err += self.n_steps_td_error(current_state, n)

            self.values[current_state] += self.alpha * n_step_err

        return self.values


if __name__ == "__main__":
    off_policy_agent_1 = OffPolicy(10)
    v1 = off_policy_agent_1.n_step_values(1)

    off_policy_agent_2 = OffPolicy(10)
    v2 = off_policy_agent_2.n_step_values(3)

    print(f"A: {v1}")
    print(f"B: {v2}")
