# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:34:48 2019

@author: isgud
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


A = np.array([[0.75,-1],[1,0.75]])
B = np.array([[1,0.5],[0.5,0.5]])
T = 3
eps = 2e-3

def cost(x,t):
    q = 1/2*(x@x)
    if t == T:
        return q
    else:
        return q + np.min([cost(A@x,t+1),cost(B@x,t+1)])

for t in range(T):
    control = np.zeros([int(2/eps),int(2/eps)])
    value = []
    for i in np.arange(-1,1,eps):
        for j in np.arange(-1,1,eps):
            if cost(A@np.array([i,j]),t+1)<cost(B@np.array([i,j]),t+1):
                control[int((i+1)/eps),int((j+1)/eps)] = 1
                value.append([i,j,cost(np.array([i,j]),0)])
            else:
                control[int((i+1)/eps),int((j+1)/eps)] = 2
                value.append([i,j,cost(np.array([i,j]),0)])
                
    value = np.array(value)
    [x,y,z] = [value[:,0],value[:,1],value[:,2]]
    
    fig = plt.figure()
    plt.imshow(control, cmap='binary')
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.scatter(x, y, z, c=z/z.max())
plt.show()