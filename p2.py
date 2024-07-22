import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./ToyotaCorolla.csv')
x=df['KM']
y=df['Weight']
z=df['Price']
plt.tricontourf(x,y,z, cmap='jet')
plt.colorbar(label='Price')
plt.xlabel('KM')
plt.ylabel('Weight')
plt.title('Color')
plt.show()


graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C', 'E'],
    'E': ['D']
}
heuristics={
    'A':4,
    'B':2,
    'C':3,
    'D':1,
    'E':0
}
class Node:
    def __init__(self,position, parent=None):
        self.position=position
        self.parent = parent
        self.h=0
        self.g=0
        self.f=0
from queue import PriorityQueue
def astar(graph, heuristics, start, goal):
    openlist = PriorityQueue()
    closedlist = set()
    start_node = Node(start) 
    openlist.put((start_node.f,start_node))
    while not openlist.empty():
        _, curr = openlist.get()
        closedlist.add(curr.position)
        if curr.position == goal:
            path = []
            while curr:
                path.append(curr.position)
                curr=curr.parent
            return path[::-1]
        for nei in graph[curr.position]:
            if nei in closedlist:
                continue
            nei_node = Node(nei,curr)
            nei_node.g=curr.g+1
            nei_node.h=heuristics[nei]
            nei_node.f=nei_node.g+nei_node.h
            # Add the neighbor to the open list if it's not already there or if it has a lower f value
            if nei_node not in [n[1] for n in openlist.queue]:
                openlist.put((nei_node.f, nei_node))
    return []
print(astar(graph,heuristics,'A','E'))
