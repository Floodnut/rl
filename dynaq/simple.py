from typing import List, Tuple
from itertools import product

import random

import numpy as np
import matplotlib.pyplot as plt

# class Model:
#     def __init__(self, state: Tuple[int], action: int):
#         self.state = state
#         self.action = action
    
#     def update(self):
#         pass
    
#     def sample(self):
#         pass
    
# class Q:
#     def __init__(self, state: Tuple[int], action: int):
#         self.state = state
#         self.action = action
    
#     def update(self, state, action, reward, next_state):
#         pass

class Maze:
    """Maze environment
    """
    def __init__(self):
        # 0 is road, 1 is wall
        self.maze = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        
        # actions
        self.actions = (
            (-1, 0), (1, 0), (0, -1), (0, 1), # 상하좌우
        )
        
        # state는 좌표
        self.start = (5, 3) # 시작
        self.goal = (0, 8) # 도착
        
        self.width = 9 # 미로 가로 길이
        self.height = 6 # 미로 세로 길이
        
        self.alpha = 0.1
        self.gamma = 0.95
        

class MazeRunner(Maze):
    """agent
    """
    def __init__(self, limit: int, model: dict, q: dict):
        super().__init__()
        self.cur_state = [self.start[0], self.start[1]]
        self.limit = limit
        
        self.model = model
        self.q_values = q
        
        self.reward = 0
        
        # for e-greedy
        self.epsilon = 0.05
        
    def reset(self):
        pass
    
    def __repr__(self):
        pass

    def _is_valid_pos(self, row, col) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width and self.maze[row][col] == 0

    def get_max(self, s: tuple):
        """get max value from q_values
        """        
        return max(q_values[s][a] for a in range(4))

    
    def e_greedy(self, state) -> int:
        """(b) a <- epsilon-greedy(s, Q)
        
        Choose action via epsilon-greedy
        """
        if random.random() < self.epsilon:
            action = random.randint(1, 4)
        else:
            values = self.q_values[state]
            action = self.get_max(values)

        return action
         
    def do_action(self, action: int) -> None:
        """(c) Execute action a;
        
        Do action to update current state (move state)
        Return next state (s') in tuple
        """
        d = self.actions[action]
        
        if self._is_valid_pos(self.cur_state[0] + d[0], self.cur_state[1] + d[1]):
            self.cur_state[0] += d[0]
            self.cur_state[1] += d[1]
            

    def observe(self) -> tuple:
        """(c) observe resultant reward, r, and state, s'
        
        Return next state (s') and reward (r) in tuple
        """
        
        s_prime = (self.cur_state[0], self.cur_state[1])
        r = -1 if s_prime != self.goal else 0
        
        return s_prime, r
    
    def q_learning(self, state: tuple, action: int):
        """(d) Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
        
        Value iteration to get optimal policy
        """
        self.q_values[state][action] += self.alpha * (
            self.reward + self.gamma * self.get_max() - self.q_values[state][action]
        )
    
    def model_update(self, s: tuple, a: int, s_prime: tuple, r: int):
        """(e) Model(s, a) <- s', r
        
        Assume deterministic environment
        """
        self.model[s][a] = (s_prime, r)

    def dyna_q(self):
        while True:
            # (a) s <- current (nonterminal) state
            s = self.start
            
            # (b) a <- epsilon-greedy(s, Q)
            a = self.e_greedy(s)
            
            # (c) Execute action a; observe resultant state, s', and reward, r 
            self.do_action(a)
            s_p, r = self.observe()
            
            # (e) Model(s, a) <- s', r
            self.model_update(s, a, s_p, r)
                        
            # (f) Repeat N times:
            repeat = 0
            while (repeat < self.limit):
                # s <- random previously observed state
                s = random.choice(self.model.keys())
                
                # a <- random action previously taken in s
                a = random.choice(self.model[s].keys())
                
                # s', r <- Model(s, a)
                s_p, r = self.model[s][a]
                
                # Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
                q_values[s][a] += self.alpha * (
                    r + self.gamma * self.get_max() - q_values[s][a]
                )
                
                repeat += 1
                
    def result(self):
        pass
    
if __name__ == "__main__":
    for planning_step in [0, 5, 50]:
        # Initialize Q(s, a) and Model(s, a) for all s ∈ S, a ∈ A(s)
        models: dict = {}
        q_values: dict = {}
        
        for w, h in product(range(9), range(6)):
            s = (w, h)
            models[s] = {}
            q_values[s] = {}
            
            for a in range(4):
                q_values[s][a] = random.random() / 100

        maze_runner = MazeRunner(limit=planning_step, model=models, q=q_values)
        maze_runner.dyna_q()