import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict

class DFS:
    def __init__(self, edges):
        self.graph = defaultdict(list)
        for start, end in edges:
            self.graph[start].append(end)

    def find_path(self, start, end):
        visited = set()
        stack = [[start]]

        while stack:
            path = stack.pop()
            node = path[-1]

            if node == end:
                return path

            if node not in visited:
                for neighbor in self.graph[node]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)

                visited.add(node)

        return []

    def show_path(self, start, end):
        path = self.find_path(start, end)
        if path:
            print("Path derived by Depth-First Search Algorithm:")
            print(" -> ".join(path))
        else:
            print("No valid path found.")

edges_dfs = [('S', 'B'), ('S', 'A'), ('A', 'B'), ('B', 'A'), ('A', 'D'), ('D', 'F'), ('B', 'C'), ('C', 'E'), ('F', 'G')]

dfs = DFS(edges_dfs)
dfs.show_path('S', 'G')

G = nx.DiGraph()

for start, end in edges_dfs:
    G.add_edge(start, end)

nx.draw(G, with_labels=True)
plt.show()