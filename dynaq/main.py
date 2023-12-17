import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), ) # 상하좌우

# 미로 클래스
class MazeEnvironment:
    def __init__(self, rows: int, cols: int):
        # 미로 크기
        self.rows = rows
        self.cols = cols
        
        self.start = (5, 3) # 시작
        self.goal = (0, 8) # 도착
        
        # 에이전트의 위치
        self.cur_pos = self.start
        
        # 벽
        self.walls = [
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)
        ]
        
        # environment params
        self.env_discount_factor = 1.0
        self.start_state = [2, 0]
        self.end_state = [0, 8]
        self.current_state = [None, None]
        self.timesteps = 0
        self.change_at_n = 0

    def is_goal_state(self) -> bool:
        return self.cur_pos == self.goal

    def _is_valid_pos(self, row, col) -> bool:
        
        # 벽인지 체크
        if (row, col) in self.walls:
            return False
        
        # 미로 범위 안인지 체크
        return row < 0 or row > 5 or col < 0 or col > 8
    
    def get_observation(self, state) -> int:
        return state[0] * 9 + state[1]
    
    def _env_step(self, action):
        self.timesteps += 1
        if self.timesteps == self.change_at_n:
            self.obstacles = self.obstacles[:-1]

        reward = 0.0
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]
        d = ACTIONS[action]
        
        if not self._is_valid_pos(row + d[0], col + d[1]):
            self.current_state = [row + d[0], col + d[1]]


        if self.current_state == self.end_state: # 종료 조건
            reward = 1.0
            is_terminal = True

        self.reward_obs_term = [reward, self.get_observation(self.current_state), is_terminal]

        return self.reward_obs_term

# 에이전트
class MazeRunner(MazeEnvironment):
    def __init__(self, rows: int=6, cols: int=9):
        super().__init__(rows=rows, cols=cols)
        self._reset()
    
    def _reset(self):
        # 초기 값
        self.steps = 0
        self.episodes = 0
        self.reward = 0
        
        # agent params
        self.num_actions = 4 
        self.num_states = 3
        self.epsilon = 0.01
        self.step_size = 0.1 
        self.dis_factor = 1.0 
        self.planning_steps = (0, 5, 50) # 0는 naive q learning
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.model = dict()
        self.g = 0.9
        self.aspect = 0.001
        
        # env params
        super().__init__(rows=6, cols=9)
        
        self.last_action = None

    def dyna_q(self, planning_steps):
        num_runs = 30
        num_episodes = 40
        planning_steps_all = self.planning_steps

        all_averages = np.zeros((len(planning_steps_all), num_runs, num_episodes))
        log_data = {'planning_steps_all' : planning_steps_all}

        for idx, planning_steps in enumerate(planning_steps_all):
            print(f"Planning steps: {planning_steps}")
            self.planning_steps = planning_steps  

            for i in range(num_runs):
                self.random_seed, self.planning_random_seed = i, i
                self._reset()

                for j in range(num_episodes):
                    self.run()
                    is_terminal = False
                    num_steps = 0
                    while not is_terminal:
                        reward, _, action, is_terminal = self._incr_step()
                        num_steps += 1
                    all_averages[idx][i][j] = num_steps

            log_data['all_averages'] = all_averages
    
    def _incr_step(self):
        (reward, last_state, term) = self._env_step(self.last_action)

        self.reward += reward

        if term:
            self.episodes += 1
            self.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.steps += 1
            self.last_action = self.agent_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat
    
    def agent_step(self, reward, state):
        self.q_values[self.past_state][self.past_action] += self.step_size * (
            reward + self.g * np.max(self.q_values[state]) - self.q_values[self.past_state][self.past_action]
        )
        
        self.update(self.past_state, self.past_action, state, reward)
        self.planning_step()
        
        action = self.action(state)
        self.past_state = state
        self.past_action = action
        
        return self.past_action
    
    def planning_step(self):
        for _ in range(self.planning_steps):
            past_state = random.choice(list(self.model.keys()))
            past_action = random.choice(list(self.model.get(past_state).keys()))
            state, reward = self.model[past_state][past_action]
            reward += self.aspect * np.sqrt(self.tau[past_state][past_action])
            if state != -1:
                self.q_values[past_state][past_action] += self.step_size * (
                    reward + self.g * np.max(self.q_values[state]) - self.q_values[past_state][past_action]
                )
            else:
                self.q_values[past_state][past_action] += self.step_size * (
                    reward - self.q_values[past_state][past_action]
                )

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties)
    
    def action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(1, 4)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action
    
    def update(self, past_state, past_action, state, reward):
        if past_state not in self.model:
            self.model[past_state] = {}
        self.model[past_state][past_action] = (state, reward)
    
    def run(self, state):
        self.current_state = self.start_state
        
        last_state = self.get_observation(self.current_state)
        action = self.action(state)
        self.past_state = state
        self.past_action = action
        
        return (last_state, self.last_action)

if __name__ == "__main__":
    agent = MazeRunner()
    agent.run()

    planning_step_results = {}
    for planning_step in [0, 5, 50]:
        steps = agent.dyna_q(planning_step)
        planning_step_results[planning_step] = steps

