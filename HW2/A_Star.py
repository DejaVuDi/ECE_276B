import heapq
import numpy as np
import time

class A_Star:
    def __init__(self,boundary,blocks,start,goal,movement):
        self.boundary = boundary
        self.blocks = blocks
        self.move = movement
        self.start = tuple(start)
        self.pos = start
        self.goal = tuple(goal)
        
    def heuristic(self,a, b):
        return np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2) 
    
    def label(self,x,gscore):
        if tuple(x) in gscore:
            return gscore.get(tuple(x),0)
        else:
            return np.inf
        
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
    
    def neighbours(self,current,move):
        numdir = 26
        neighbour = []
        [dX,dY,dZ] = np.meshgrid([-move,0,move],[-move,0,move],[-move,0,move])
        move_set = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
        move_set = np.delete(move_set,13,axis=1)
        for k in range(numdir):
            new_loc = current + move_set[:,k]
            if np.all([self.check(i) for i in np.array([np.linspace(current[i],new_loc[i],10) for i in range(3)]).transpose()]):
                neighbour.append(new_loc)
        if self.heuristic(neighbour[0],self.goal)<0.5:
            neighbour = [self.goal]
        return neighbour
                
    def A_star(self):
        gscore = {self.start:0}
        open = []
        close_set = set()
        fscore = {self.start:self.heuristic(self.start, self.goal)}
        heapq.heappush(open,(fscore[self.start],self.start))
        route = {self.start:[self.start]}
        while not self.goal in close_set:
            i = heapq.heappop(open)[1]
            close_set.add(i)
            for j in self.neighbours(i,self.move):
                tmp_g = gscore[i] + self.heuristic(i,j)
                if tuple(j) in close_set:
                    continue
                elif self.label(tuple(j),gscore) > tmp_g:
                    gscore[tuple(j)] = tmp_g
                    route[tuple(j)] = route[i].copy()
                    route[tuple(j)].append(tuple(j))
                    fscore[tuple(j)] = tmp_g+self.heuristic(j,self.goal)
                    heapq.heappush(open,(fscore[tuple(j)],tuple(j)))
        print(route[self.goal])
        data = route[self.goal]
        return data