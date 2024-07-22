import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./ToyotaCorolla.csv')
df.head()
x=df['KM']
y=df['Doors']
z=df['Price']
ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap='jet')
ax.set_title('3d Surface Plot')
ax.set_xlabel('KM')
ax.set_ylabel('Doors')
ax.set_zlabel('Price')
plt.show()

graph = {
    'S':['A','B'],
    'A':['C','D'],
    'B':['E','F'],
    'C':[],
    'D':[],
    'E':['H'],
    'F':['I','G'],
    'H':[],
    'I':[],
    'G':[],
}
heuristics = {
    'S':13,
    'A':12,
    'B':4,
    'C':7,
    'D':3,
    'E':8,
    'F':2,
    'H':4,
    'I':9,
    'G':0,
}
from queue import PriorityQueue
def bfs(graph, heuristics,start,goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristics[start],start))
    while not pq.empty():
        _, node = pq.get()
        if node == goal:
            print("Visiting ",node)
            print('Done Goal Reached')
            return 
        for nei in graph[node]:
            if nei not in visited:
                pq.put((heuristics[nei],nei))
        visited.add(node)
        print('Visiting ',node)
    print('No Connection')
bfs(graph,heuristics,'S','G')
