import os
os.environ["PATH"] += os.pathsep + 'C:/Users/isgud/Anaconda3/Library/bin/graphviz'

import numpy as np
import sys
import os.path

def load_data(input_file):
  '''
  Read deterministic shortest path specification
  '''
  with np.load(input_file) as data:
    n = data["number_of_nodes"]
    s = data["start_node"]
    t = data["goal_node"]
    C = data["cost_matrix"]
  return n, s, t, C

def plot_graph(C,path_nodes,output_file):
  '''
  Plot a graph with edge weights sepcified in matrix C.
  Saves the output to output_file.
  '''
  from graphviz import Digraph
  
  G = Digraph(filename=output_file, format='pdf', engine='neato')
  G.attr('node', colorscheme='accent3', color='1', shape='oval', style="filled", label="")

  # Normalize the edge weights to [1,11] to fit the colorscheme  
  maxC = np.max(C[np.isfinite(C)])
  minC = np.min(C)
  norC = 10*np.nan_to_num((C-minC)/(maxC-minC))+1
  
  # Add edges with non-infinite cost to the graph 
  for i in range(C.shape[0]):
    for j in range(C.shape[1]):
      if C[i,j] < np.inf:
        G.edge(str(i), str(j), colorscheme="rdylbu11", color="{:d}".format(int(norC[i,j])))
  
  # Display path
  for n in path_nodes:
	  G.node(str(n), colorscheme='accent3', color='3', shape='oval', style="filled")
	
  G.view()

def save_results(path, cost, output_file):
  '''
  write the path and cost arrays to a text file
  '''
  with open(output_file, 'w') as fp:
    for i in range(len(path)):
      fp.write('%d ' % path[i])
    fp.write('\n')
    for i in range(len(cost)):
      fp.write('%.2f ' % cost[i])  


# reference from https://www.redblobgames.com/pathfinding/a-star/implementation.html#python
import heapq
class Queue:
    def __init__(self):
        self.elements = []    
    def empty(self):
        return len(self.elements) == 0    
    def put(self, item):
        heapq.heappush(self.elements, (item))    
    def get(self):
        return heapq.heappop(self.elements)[1]

def LC(n,s,t,C):
  # process data get the node without inf
  children = [[] for i in range(n)]
  g = []
  for i in range(n):
      g.append((np.inf,i))
      for j in range(n): 
          if C[i,j] < np.inf:
              children[i].append(j)
              
  # start node and OPEN
  g[s] = (0,s)    
  OPEN = Queue()
  OPEN.put((0,s))
  parent = [0] * n
  g_cost = [0] * n
  node = [0] * n
  
  # LC
  while not OPEN.empty():
    i = OPEN.get()
    for j in children[i]:
        if g[i][0]+C[i,j]<g[j][0] and g[i][0]+C[i,j]<g[t][0]:
          g[j] = (g[i][0]+C[i,j], j)
          parent[j] = i
          g_cost[j] = g[i][0]+C[i,j]
          node[j] = j
          if j!=t:
            OPEN.put(g[j])
  path = []
  cost = []
  current = t
  while current != s:
      path.append(node[current])
      cost.append(g_cost[current])
      current = parent[current]

  path.append(s)
  cost.append(0.0)
  path.reverse()
  return [path,cost]    

if __name__=="__main__":
  # input_file = sys.argv[1]
  input_file = '../data/problem6.npz'
  file_name = os.path.splitext(input_file)[0]
  
  # Load data 
  n,s,t,C = load_data(input_file)
  n = int(n)
  s = int(s)
  t = int(t)
  print(s,t)
  
  # Generate results
  path,cost = LC(n,s,t,C)
  print(path)
  print(list(np.around(np.array(cost),1)))
  
  # Visualize (requires: pip install graphviz --user)
  #plot_graph(C,path,file_name)
  
  # Save the results
  save_results(path,cost,file_name+"_results.txt")
  


