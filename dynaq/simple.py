from typing import List, Tuple
from itertools import product

import random

import numpy as np
import matplotlib.pyplot as plt


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

    def get_max(self, s: tuple) -> float:
        """get max value from q_values
        """

        return max([self.q_values[s][a] for a in range(4)])      


    def e_greedy(self, state) -> int:
        """(b) a <- epsilon-greedy(s, Q)
        
        Choose action via epsilon-greedy
        """
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return max([0, 1, 2, 3], key=self.q_values[s].get)

         
    def do_action(self, action: int) -> Tuple[int]:
        """(c) Execute action a;
        
        Do action to update current state (move state)
        Return next state (s') in tuple
        """
        d = self.actions[action]
        
        if self._is_valid_pos(self.cur_state[0] + d[0], self.cur_state[1] + d[1]):
            return (d[0], d[1])
        
        return (0, 0)

    def observe(self, movement: Tuple[int]) -> tuple:
        """(c) observe resultant reward, r, and state, s'
        
        Return next state (s') and reward (r) in tuple
        """
        
        s_prime = (self.cur_state[0] + movement[0], self.cur_state[1] + movement[1])
        r = -1 if s_prime != self.goal else 0
        
        return s_prime, r
    
    def q_learning(self, state: tuple, action: int, r: int):
        """(d) Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
        
        Value iteration to get optimal policy
        """
        
        self.q_values[state][action] += self.alpha * (
            r + self.gamma * self.get_max(state) - self.q_values[state][action]
        )

    def model_update(self, s: tuple, a: int, s_prime: tuple, r: int):
        """(e) Model(s, a) <- s', r
        
        Assume deterministic environment
        """
        self.model[s][a] = (s_prime, r)

    def dyna_q(self):
        # (a) s <- current (nonterminal) state
        s = self.start
        while True:
            # (b) a <- epsilon-greedy(s, Q)
            a = self.e_greedy(s)
            
            # (c) Execute action a; observe resultant state, s', and reward, r 
            movement = self.do_action(a)
            s_p, r = self.observe(movement)
            self.q_learning(s, a, r)
            
            # (e) Model(s, a) <- s', r
            self.model_update(s, a, s_p, r)
                        
            # (f) Repeat N times:
            repeat = 0
            while (repeat < self.limit):
                # s <- random previously observed state
                s = random.choice(self.model.keys())
                
                # a <- random action previously taken in s
                a = random.choice([0, 1, 2, 3])
                
                # s', r <- Model(s, a)
                s_p, r = self.model[s][a]
                
                # Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
                q_values[s][a] += self.alpha * (
                    r + self.gamma * self.get_max(s) - q_values[s][a]
                )
                
                repeat += 1
                
            print(repeat, s_p)
            if s_p == self.goal:
                print("Goal!")
                break
            
            s = s_p
                
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
        
        maze_runner.result()