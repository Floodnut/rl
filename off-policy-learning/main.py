import random
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, num_states: int):
        self.alpha = 0.1
        
        self.values = np.arange(num_states)
        
        # policy
        self.behavior_policy = np.ones(num_states) / num_states  # 무작위 정책
        self.target_policy = np.zeros(num_states)
        self.target_policy[1:6] = 1
        
        self.target_state_values = np.arange(1, 6)

class Agent:
    def __init__(self):
        pass

class OffPolicy(Environment):
    def __init__(self, num_states: int):
        super().__init__(num_states)
        
    def __call__(self):
        pass
    
    def n_step_td_error(self, current_state, next_state, reward, values, n):
        if n == 0:
            return 0
        
        target = reward + values[next_state] if next_state is not None else 0
        return target + values[current_state] - values[current_state]

    def n_step_values(self, n):
        current_state = 0
        
        for _ in range(1000):
            n_step_error = 0
            for _ in range(n):
                action = np.random.choice(np.arange(self.num_states), p=self.behavior_policy)
                next_state = action
                reward = 0
                n_step_error += self.n_step_td_error(current_state, next_state, reward, self.values, n)
                current_state = next_state
            
            self.values[current_state] += self.alpha * n_step_error
            
        return self.values

if __name__ == "__main__":
    off_policy_agent_1 = OffPolicy(10)
    v1 = off_policy_agent_1.n_step_values(1)
    
    off_policy_agent_2 = OffPolicy(10)
    v2 = off_policy_agent_2.n_step_values(3)

    print(f"A: {v1}")
    print(f"B: {v2}")
