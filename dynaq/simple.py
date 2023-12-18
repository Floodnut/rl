from typing import List, Tuple
from itertools import product

import random

import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, state: Tuple[int], action: int):
        self.state = state
        self.action = action
    
    def update(self):
        pass
    
    def sample(self):
        pass
    
class Q:
    def __init__(self, state: Tuple[int], action: int):
        self.state = state
        self.action = action
    
    def update(self, state, action, reward, next_state):
        pass

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
    def __init__(self, limit: int, model, q):
        super().__init__()
        self.cur_state: tuple = self.start
        self.limit = limit
        
        self.model = model
        self.q_values: dict = q
        
        self.reward = 0
        
        # for e-greedy
        self.epsilon = 0.05
        
    def reset(self):
        pass
    
    def __repr__(self):
        pass
    
    def dyna_q(self):
        while True:
            # (a) s <- current (nonterminal) state
            s = self.start
            
            # (b) a <- epsilon-greedy(s, Q)
            a = self.e_greedy(s)
            
            # (c) Execute action a; observe resultant state, s', and reward, r 
            self.do_action(a)
            result_state, s_prime, reward, r = self.observe()
            
            # (e) Model(s, a) <- s', r
            self.model_update(self.cur_state, self.reward)
                        
            # (f) Repeat N times:
            repeat = 0
            while (repeat < self.limit):
                # s <- random previously observed state
                
                # a <- random action previously taken in s
                
                # s', r <- Model(s, a)
                
                # Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
                
                repeat += 1

    def get_max(self):
        pass
    
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
         
    def do_action(self, action: int):
        """(c) Execute action a;
        
        Do action to update current state (move state)
        """
        self.cur_state += self.actions[action]

    def observe(self):
        """(c) observe resultant reward, r, and state, s'
        """
        return None, None, None, None
    
    def q_learning(self, state, action):
        """(d) Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        self.q_values[state][action] += self.alpha * (
            self.reward + self.gamma * self.get_max() - self.q_values[state][action]
        )
    
    def model_update(self, new_state, action, reward):
        """(e) Model(s, a) <- s', r
        
        Assume deterministic environment
        """
        self.model.update(self.cur_state, action, reward, new_state)
    
if __name__ == "__main__":
    for planning_step in [0, 5, 50]:
        # Initialize Q(s, a) and Model(s, a) for all s ∈ S, a ∈ A(s)
        models: dict = []
        q_values: dict = []
        
        for w, h, a in product(range(9), range(6), range(4)):
            models[((w, h), a)] = Model((w, h), a)
            q_values[((w, h), a)] = Q((w, h), a)

        maze = MazeRunner(limit=planning_step, model=models, q=q_values)           

        maze_runner = MazeRunner(maze)
        maze_runner.dyna_q()