import heapq
import random
import numpy as np
import time

class RRT:
    def __init__(self,boundary,blocks,start,goal,eps):
        self.eps = eps
        self.boundary = boundary
        self.blocks = blocks
        self.start = start
        self.goal = goal
        self.done = False
        self.V = [start]
        self.E = {}
        self.route = []
        
    def heuristic(self,a, b):
        return np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2) 
    
    def check(self,neighbour):
        valid = True
        if( neighbour[0] < self.boundary[0,0]+0.1 or neighbour[0] > self.boundary[0,3]-0.1 or \
            neighbour[1] < self.boundary[0,1]+0.1 or neighbour[1] > self.boundary[0,4]-0.1 or \
            neighbour[2] < self.boundary[0,2]+0.1 or neighbour[2] > self.boundary[0,5]-0.1 ):
            valid = False
        for k in range(self.blocks.shape[0]):
            if( neighbour[0] > self.blocks[k,0]-0.1 and neighbour[0] < self.blocks[k,3]+0.1 and\
                neighbour[1] > self.blocks[k,1]-0.1 and neighbour[1] < self.blocks[k,4]+0.1 and\
                neighbour[2] > self.blocks[k,2]-0.1 and neighbour[2] < self.blocks[k,5]+0.1 ):
                valid = False
        return valid 
    
    def samplefree(self):
        [x_min,y_min,z_min] = self.boundary[0,0:3]
        [x_max,y_max,z_max] = self.boundary[0,3:6]
        return np.array([random.uniform(x_min,x_max),random.uniform(y_min,y_max),random.uniform(z_min,z_max)])
    
    def near(self,v,rand):
        x_nearest = [self.heuristic(rand,v[i]) for i in range(len(v))]
        return v[np.argmin(x_nearest)]
  
    def steer(self,nearest,rand):
        if self.heuristic(self.goal,nearest) < 0.7:
            x_new = self.goal
        else:
#             new_dir = (rand-nearest)/np.linalg.norm(rand-nearest)
#             x_new = nearest + self.eps * new_dir
            [dX,dY,dZ] = np.meshgrid([-self.eps,0,self.eps],[-self.eps,0,self.eps],[-self.eps,0,self.eps])
            move_set = np.delete(np.vstack((dX.flatten(),dY.flatten(),dZ.flatten())),13,1)
            neighbour = [nearest + move_set[:,k] for k in range(26)]
            dist = [self.heuristic(rand,neighbour[i]) for i in range(len(neighbour))]
            x_new = neighbour[np.argmin(dist)]
        return x_new
    
    def collisionfree(self,nearest,new):
        return np.all([self.check(i) for i in np.array([np.linspace(new[i],nearest[i],10) for i in range(3)]).transpose()])
    
    def RRT_route(self):
        n = 0
        t0 = time.time()
        while not self.done and n < 100000:
            if n%500 == 0:
                print(n)
            x_rand = self.samplefree()
            x_nearest = self.near(self.V,x_rand)
            x_new = self.steer(x_nearest,x_rand)
            while not self.collisionfree(x_nearest,x_new) or tuple(x_new) in self.E:
                x_rand = self.samplefree()
                x_nearest = self.near(self.V,x_rand)
                x_new = self.steer(x_nearest,x_rand)
            self.V.append(x_new)
            self.E[tuple(x_new)] = x_nearest
            n+=1
            if (x_new == self.goal).all():
                self.V.append(x_new)
                self.E[tuple(x_new)] = x_nearest
                self.route.append(x_new)
                print('planing time:',time.time()-t0)
                self.done = True
                break
        while(np.linalg.norm(self.start-x_new)>1):
            x_new = self.E[tuple(x_new)]
            self.route.append(x_new)
        return self.route