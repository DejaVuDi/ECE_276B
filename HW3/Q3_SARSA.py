import gym
import numpy as np
import itertools
import pandas as pd
import pickle

from collections import defaultdict,namedtuple
from matplotlib import pyplot as plt

# reference: https://learning.oreilly.com/library/view/reinforcement-learning-with/9781788835725/ffd21cf7-d907-45e6-a897-8762c9a20f2d.xhtml
# reference: https://github.com/dennybritz/reinforcement-learning

class Planner:
    def __init__(self):
        self.lr = 1
        self.discount_factor = 0.99
        self.episode = 200000
        self.eps = 0.05
        self.n_states = 40
        self.min_lr = 0.005
        self.alpha = 0.5

    def policy(self,env,Q,state):
        A = np.ones(env.action_space.n, dtype=float) * self.eps / env.action_space.n
        action = np.argmax(Q[tuple(state)])
        A[action] += (1-self.eps)
        return A

    def discretized_state(self,env,obs):
        state = np.array([0,0])
        env_low = env.observation_space.low
        env_high = env.observation_space.high
        env_dx = (env_high-env_low)/self.n_states
        state[0] = int((obs[0] - env_low[0])/env_dx[0])
        state[1] = int((obs[1] - env_low[1])/env_dx[1])
        return state
    
    def rollout(self, env, policy=None, render=False):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        stats = []    
        best_reward = np.float('-inf')
        best_episode = 0
        best_stats = []
        for i_episode in range(self.episode):
            traj = []
            t = 0
            done = False
            total_reward = 0
            alpha = max(self.min_lr,self.lr*(self.discount_factor**(i_episode//100)))
            c_state = env.reset()
            state = self.discretized_state(env,c_state)
            for j in itertools.count():
                action_prob = self.policy(env,Q,state)
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
#                 if render:
#                     env.render()                
                n_state, reward, done, _ = env.step(action)
                n_state = self.discretized_state(env,n_state)
                traj.append((state, action, reward))
            
                total_reward += reward                
                state = n_state
                
                if done:
                    stats.append((i_episode,total_reward))
                    if best_reward < total_reward:
                        best_reward = total_reward
                        best_episode = i_episode
                        best_stats.append((best_episode,best_reward))
                    break

            for i in range(len(traj)-1):
                state = traj[i][0]
                action = traj[i][1]
                n_state = traj[i+1][0]
                n_action = traj[i+1][1]
                reward = traj[i][2]
                td_delta = reward + self.discount_factor * Q[tuple(n_state)][n_action] - Q[tuple(state)][action]
                Q[tuple(state)][action] += alpha * td_delta
            if i_episode%5000 == 0:
                print(i_episode)
            if total_reward != -200:
                print("Episode {} completed with total reward {} with alpha {}".format(i_episode,total_reward,alpha)) 
        env.close()
#         self.plot_stats(stats)
        return traj,stats,best_stats


env = gym.make('MountainCar-v0')
env.seed(0)
np.random.seed(0)
planner = Planner()
traj,stats,best_stats = planner.rollout(env, policy=np.random.choice(env.action_space.n))