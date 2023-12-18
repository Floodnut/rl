from typing import List, Tuple
from itertools import product

import random

import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, state: Tuple[int], action: int):
        self.state = state
        self.action = action
    
    def update(self, state:Tuple[int], action: int, reward: int, next_state: Tuple[int]):
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
    def __init__(self, limit: int, model: List[Model], q: List[Q]):
        super().__init__()
        self.cur_state: tuple = self.start
        self.limit = limit
        
        self.model = model
        self.q = q
        
        self.reward = 0
        
        # for e-greedy
        self.epsilon = 0.05
        
    def reset(self):
        pass
    
    def __repr__(self):
        pass
    
    def observe(self):
        pass
    
    def get_max(self):
        pass
    
    def dyna_q(self):
        
        while True:
            # (a) s <- current (nonterminal) state
            s = self.start
            
            # (b) a <- epsilon-greedy(s, Q)
            a = self.e_greedy(s)
            
            # (c) Execute action a; observe resultant state, s', and reward, r 
            self.do_action(a)
            
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
    
    def e_greedy(self, state):
        """(b) a <- epsilon-greedy(s, Q)
        """
        pass    
         
    def do_action(self, state):
        """(c) Execute action a; observe resultant reward, r, and state, s'
        """
        if random.random() < self.epsilon:
            action = random.randint(1, 4)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action
    
    def q_learning(self):
        """(d) Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        pass
    
    def model_update(self, new_state, reward):
        """(e) Model(s, a) <- s', r
        
        Assume deterministic environment
        """
        self.model.update(self.cur_state, self.action, reward, new_state)
    
if __name__ == "__main__":
    
    for planning_step in [0, 5, 50]:
        models: List[Model] = []
        qs: List[Q] = []
        
        for w, h, a in product(range(9), range(6), range(4)):
            models.append(Model((w, h), a))
            qs.append(Q((w, h), a))

        maze = MazeRunner(limit=planning_step, model=models, q=qs)           
        # Initialize Q(s, a) and Model(s, a) for all s ∈ S, a ∈ A(s)

        maze_runner = MazeRunner(maze)
        maze_runner.dyna_q()