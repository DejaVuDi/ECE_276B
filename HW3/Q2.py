# reference: https://github.com/dennybritz/reinforcement-learning

import numpy as np
import matplotlib.pyplot as plt
import pickle
# import cv2

from scipy import misc
from skimage import transform
from math import pi
from PIL import Image
from collections import defaultdict

import io
import sys
from gym.envs.toy_text import discrete

UP = 0    # north
RIGHT = 1 # east
DOWN = 2  # south
LEFT = 3  # west

class GridworldEnv(discrete.DiscreteEnv):
    def __init__(self, shape=[5,5]):
        
        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        print(grid)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            is_not_edge = lambda s: s in [6,7,8,11,12,13,16,17,18]
            
            if s == 1:
                ns_up = ns_right = ns_down = ns_left = 21
            elif s == 3:
                ns_up = ns_right = ns_down = ns_left = 13
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                
            if s in [0,2,4]:
                up_reward = -1
                right_reward = down_reward = left_reward = 0
            elif s in [0,5,10,15,20]:
                left_reward = -1
                right_reward = down_reward = up_reward = 0
            elif s in [20,21,22,23,24]:
                down_reward = -1
                right_reward = left_reward = up_reward = 0
            elif s in [4,9,14,19,24]:
                right_reward = -1
                left_reward = down_reward = up_reward = 0
            elif s == 1:
                up_reward = down_reward = left_reward = right_reward = -10
            elif s == 3:
                up_reward = down_reward = left_reward = right_reward = -5
            else:
                up_reward = down_reward = left_reward = right_reward = 0
            P[s][UP] = [(1.0, ns_up, up_reward)]
            P[s][RIGHT] = [(1.0, ns_right, right_reward)]
            P[s][DOWN] = [(1.0, ns_down, down_reward)]
            P[s][LEFT] = [(1.0, ns_left, left_reward)]

            it.iternext()
        isd = np.ones(nS) / nS
        self.P = P
        super(GridworldEnv, self).__init__(nS, nA, P, isd)

print('Welcome to GridWorld')
env = GridworldEnv()

def value_iteration(env, discount_factor=0.9): 
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            prob, next_state, reward = env.P[state][a][0]
            A[a] = reward + discount_factor * V[next_state]
        return A
    V = np.zeros(env.nS)
    u = np.arange(env.nA)    
    policy = np.zeros(env.nS)
    for _ in range(500):
        for s in range(env.nS):
            A = one_step_lookahead(s, V)
            best_action_value = np.min(A)
            V[s] = best_action_value
            policy[s] = u[np.argmin(A)]
    return policy, V

def policy_eval(policy, env, discount_factor,V):
    for _ in range(4):
        for s in range(env.nS):
            a = policy[s]
            prob, next_state, reward = env.P[s][a][0]
            V[s] = reward + discount_factor * V[next_state]
    return np.array(V)

policy, v = value_iteration(env,discount_factor=0.99)
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(policy,env.shape))

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=0.9):
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward in env.P[state][a]:
                A[a] = reward + discount_factor * V[next_state]
        return A
    
    policy = np.ones([env.nS])
    u = np.arange(env.nA)
    V = np.zeros(env.nS)
    
    for i in range(500):
        V = policy_eval_fn(policy, env, discount_factor,V)
        for s in range(env.nS):
            policy[s] = u[np.argmin(one_step_lookahead(s,V))]
    return policy, V

policy, v = policy_improvement(env,discount_factor=0.99)
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(policy.reshape(env.shape))