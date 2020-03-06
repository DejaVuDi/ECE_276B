import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import os 
import time
# import matplotlib
# matplotlib.use("TkAgg")
# %matplotlib inline

from scipy import misc,stats
from skimage import transform
from math import pi
from PIL import Image
from collections import defaultdict
from PendulumProblem import EnvAnimate

# from matplotlib import animation, rc
# from IPython.display import HTML

class IP:
    def __init__(self):
        self.iter = 80
        self.tau = 0.05
        self.gravity = 9.8
        self.a = 1
        self.b = 1
        self.sigma = 2*np.array([[1,0],[0,1]])
        self.k = 1
        self.r = 0.001
        self.gamma = 0.9
        self.n1_states = 360
        self.n2_states = 30
        self.n_actions = 40
        self.vmax = 5
        self.umax = 20
        self.thresh = 0.001
        
        self.x1 = np.round(np.linspace(-pi,pi,self.n1_states),8)
        self.x2 = np.round(np.linspace(-self.vmax,self.vmax,self.n2_states),8)
        self.u = np.linspace(-self.umax,self.umax,self.n_actions)
        
        self.x1_resolution = np.round(2*pi/(self.n1_states-1),8)
        self.x2_resolution = np.round(2*self.vmax/(self.n2_states-1),8)
        
        if os.path.exists('vi_v.npy'):
            self.vlist = np.load('vi_v.npy')
        else:
            self.vlist = np.zeros((self.n1_states,self.n2_states))
        
        self.V = {} #defaultdict(lambda: 0)
        for i,a in enumerate(list(self.x1)):
            for j,b in enumerate(list(self.x2)):
                self.V[(a,b)] = self.vlist[i][j]
        self.policy = np.zeros((self.n1_states,self.n2_states))

    def cost(self,state_tuple,u):
        x1, x2 = state_tuple
        return self.tau*(1-np.exp(self.k*np.cos(x1)-self.k)+self.r/2*u**2)
    
    def find_x1(self,x):
        return self.x1[np.digitize((x+pi)%(2*pi)-pi,self.x1,right=True)]
    
    def find_x2(self,x):
        if x > self.vmax:
            x_out = self.x2[np.digitize(x,self.x2)-1]
        else:
            x_out = self.x2[np.digitize(x,self.x2,right=True)]     
        return x_out
#         return self.x2[np.digitize((x+self.vmax)%(2*self.vmax)-self.vmax,self.x2,right=True)]
    
    def transform(self,n_state_tuple):
        y1, y2 = n_state_tuple
        y1_ = self.find_x1(y1)
        y2_ = self.find_x2(y2)    
        return (y1_,y2_)
        
    def next_state(self,action,state_tuple):
        x1, x2 = state_tuple
        y1 = x1+self.tau*x2
        y2 = x2+self.tau*(self.a*np.sin(x1)-self.b*x2+action)
        return (y1,y2)
      
    def next_state_with_noise(self,action,state_tuple):
        x1, x2 = state_tuple
        noise_a , noise_b = np.random.multivariate_normal([0,0],self.tau*self.sigma)
        y1 = x1+self.tau*x2+noise_a
        y2 = x2+self.tau*(self.a*np.sin(x1)-self.b*x2+action)+noise_b
        return(y1,y2)

    def position_x1(self,x):
        pos = (x+pi)/(2*pi)*self.n1_states
        return int(pos)
      
    def position_x2(self,x):
        pos = (x+self.vmax)/(2*self.vmax)*self.n2_states
        return int(pos)
        
    def Gaussian(self,state_tuple,action):
        x_h = np.array(state_tuple)
        x_prime = np.array(self.next_state(action,state_tuple))
        mean = np.array(self.transform(x_prime))
#         mean = x_prime
        cov = self.sigma@self.sigma.T*self.tau        
#         print(x_h)
#         print(x_prime)
        
        x1_min = x_prime[0]-3*cov[0][0]
        x1_max = x_prime[0]+3*cov[0][0]
        x2_min = x_prime[1]-3*cov[1][1]
        x2_max = x_prime[1]+3*cov[1][1] 
        
#         print(x2_min,x2_max)
#         print(self.find_x1(x1_min),self.find_x1(x1_max))

        x1_possible_state_min_index = list(self.x1).index(self.find_x1(x1_min))
        x1_possible_state_max_index = list(self.x1).index(self.find_x1(x1_max))
        x1_possible_index = list(np.arange(x1_possible_state_min_index,x1_possible_state_max_index))
        x1_possible = self.x1[x1_possible_index]
        
        if x2_max < -self.vmax:
            x2_possible_index = [0]
        elif np.logical_and((x2_min < -self.vmax),(x2_max > -self.vmax)):
            x2_possible_state_min_index = 0
            x2_possible_state_max_index = list(self.x2).index(self.find_x2(x2_max))
            if x2_possible_state_min_index == x2_possible_state_max_index:
                x2_possible_index = [0]
            else:
                x2_possible_index = list(np.arange(0,x2_possible_state_max_index))
        elif x2_min > self.vmax:
            x2_possible_index = [self.n2_states-1]
        elif np.logical_and((x2_min < self.vmax),(x2_max > self.vmax)):
            x2_possible_state_min_index = list(self.x2).index(self.find_x2(x2_min))
            x2_possible_state_max_index = self.n2_states-1
            if x2_possible_state_min_index == x2_possible_state_max_index:
                x2_possible_index = [self.n2_states-1]
            else:
                x2_possible_index = list(np.arange(x2_possible_state_min_index,self.n2_states-1))
        else:
            x2_possible_state_min_index = list(self.x2).index(self.find_x2(x2_min))
            x2_possible_state_max_index = list(self.x2).index(self.find_x2(x2_max))
            if x2_possible_state_min_index == x2_possible_state_max_index:
                x2_possible_index = [x2_possible_state_min_index]
            else:
                x2_possible_index = list(np.arange(x2_possible_state_min_index,x2_possible_state_max_index))
                
        x2_possible = self.x2[x2_possible_index]
        
        p = np.zeros((len(x1_possible),len(x2_possible)))
        
#         print(x1_possible,x2_possible)

        expection = 0
        for i in range(len(x1_possible)):
            for j in range(len(x2_possible)):
#                 print([x1_possible[i],x2_possible[j]])
#                 print('mean',mean)
                p[i,j] = stats.multivariate_normal.pdf([x1_possible[i],x2_possible[j]],mean,cov)
        
#         print(p)
        p = p/p.sum()
        
        v = self.vlist[x1_possible_index,:]
        v = v[:,x2_possible_index]
#         print(v)
        exp = (p*v).sum()
#         print(exp)
        return exp
    
  
    def new_gaussian(self,state_tuple,action):
        x_h = np.array(state_tuple)
        x_prime = np.array(self.next_state(action,state_tuple))
        mean = x_prime
        cov = self.sigma@self.sigma.T*self.tau      
    
#         print(x_h)
#         print(x_prime)
        
        x1_min_index = self.position_x1(x_prime[0]-3*cov[0][0])
        x1_max_index = self.position_x1(x_prime[0]+3*cov[0][0])
        x2_min_index = self.position_x2(x_prime[1]-3*cov[1][1]) 
        x2_max_index = self.position_x2(x_prime[1]+3*cov[1][1])
        
#         print(x_prime[1]+3*cov[1][1],x2_min_index,x2_max_index)
        
        x1_min = x1_min_index*(2*pi)/self.n1_states-pi
        if x2_max_index < 0:
            x2_min_index = 0
            x2_max_index = 0
            x2_min = -self.vmax
        elif x2_min_index < 0 & x2_max_index >= 0:
            x2_min_index = 0
            x2_min = -self.vmax
        elif x2_min_index > self.n2_states:
            x2_min_index = self.n2_states
            x2_max_index = self.n2_states
            x2_min = self.vmax
        elif x2_min_index <= self.n2_states & x2_max_index > self.n2_states:
            x2_max_index = self.n2_states
            x2_min = x2_min_index*(2*self.vmax)/self.n2_states-self.vmax
        else:
            x2_min = x2_min_index*(2*self.vmax)/self.n2_states-self.vmax
        
#         print('x2',x2_min)
        
        x1_range = x1_max_index-x1_min_index+1
        x2_range = x2_max_index-x2_min_index+1
        
        p = np.zeros((x1_range,x2_range))
        v = p
        
        for i in range(x1_range):
            for j in range(x2_range):
#                 print(x2_min+self.x2_resolution*j)
                p[i,j] = stats.multivariate_normal.pdf([x1_min+self.x1_resolution*i,x2_min+self.x2_resolution*j],mean,cov)
                v[i,j] = self.V[(self.find_x1(x1_min+self.x1_resolution*i),self.find_x2(x2_min+self.x2_resolution*j))]
        p = p/p.sum()
        exp = (p*v).sum()
        
        return exp
        
        
    def VI(self):
        def one_step_lookahead(state_tuple):
            A = np.zeros(self.n_actions)
            for a in range(self.n_actions):
#                 n_state = self.next_state(self.u[a],state_tuple)
                reward = self.cost(state_tuple,self.u[a])
#                 print(state_tuple,self.u[a])
                A[a] =  (reward + self.gamma * self.new_gaussian(state_tuple,self.u[a]))
            return A
        
        for n in range(self.iter):
            print(self.vlist)
            print(self.V)
#             vi_policy = self.get_policy(self.vlist)
#             np.save('vi_p.npy',vi_policy)
            delta = 0
            lastv = self.vlist.copy()
            start = time.time()
            for i in range(self.n1_states):
                if i%50 == 0:
                    print(i)
                for j in range(self.n2_states):
#                     print(j)
                    state_tuple = (self.x1[i],self.x2[j])
                    A = one_step_lookahead(state_tuple)
                    best_action_value = np.min(A)
                    self.V[state_tuple] = best_action_value
                    self.vlist[i][j] = best_action_value
                    self.policy[i][j] = self.u[np.argmin(A)]
#             print(self.vlist)
            delta = np.linalg.norm(self.vlist-lastv,1)
            print('number of iteration:',n+1,', time:',time.time()-start,', norm: ',delta)
            np.save('vi_v.npy',self.vlist)
            np.save('vi_p.npy',self.policy)
#             if delta < self.thresh:
#                 break        
        return self.V,self.vlist,self.policy
#     def PI(self):

    def roll_out(self,start,policy):
        state = start
        traj = []
        traj.append(state[0])
        
        for _ in range(1000):
            n_state_tuple = self.next_state_with_noise(policy[list(self.x1).index(self.find_x1(state[0])),list(self.x2).index(self.find_x2(state[1]))],state)
            # print(n_state_tuple)
            state = self.transform(n_state_tuple)
            traj.append(state[0])
        return traj

MDP = IP()
# dic_v,v,p = MDP.VI()
p = np.load('pi_p.npy')
# p = np.load('pi_p.npy')
print(p)
# p = np.load('pi_p.npy')
traj = MDP.roll_out((-pi,0),p)
print(traj)
plt.plot(traj)
plt.ylabel('angle (rad)')
plt.xlabel('time (s)')
plt.show()
# animation = EnvAnimate()
# animation.new_data(traj)
# animation.start()