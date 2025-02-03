import networkx as nx # type: ignore
import matplotlib.pyplot as plt
import math
from queue import PriorityQueue
import math
import copy
import heapq

fgjeog = 0
def create_tree():
    n = int(input("Enter the number of nodes: "))
    G = nx.DiGraph()
    node_names = [input(f"Enter the name of node {i+1}: ") for i in range(n)]
    
    heuristics = {}
    for name in node_names:
        heuristic_value = int(input(f"Enter the heuristic value for node {name}: "))
        heuristics[name] = heuristic_value
    
    for i in range(n-1):
        fgjeog = int(input(f"Enter the number of parents for {node_names[i+1]}: "))
        while fgjeog>0:
            parent = input(f"Enter the parent node {fgjeog} for node {node_names[i+1]}: ")
            cost = int(input(f"Enter the cost from {parent} to node {node_names[i+1]}: "))
            G.add_edge(parent, node_names[i+1], cost=cost)
            fgjeog= fgjeog-1
    
    return G, heuristics

def visualize_tree(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_weight="bold", font_color="black")
    edge_labels = nx.get_edge_attributes(G, 'cost')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()


tree, heuristics = create_tree()
visualize_tree(tree)


start_node = str(input("Enter the Start Node: "))
goal_node = str(input("Enter the End Node: "))
end_node = goal_node

###BMS AND ORACLE
print("BMS And Oracle")
print("The BMS paths are as follows")
G = tree

paths = nx.all_simple_paths(G, source=start_node, target=goal_node)
paths_final = []

for path in paths:
    cost = sum(G[path[i]][path[i+1]]["cost"] for i in range(len(path)-1))
    paths_final.append((path, cost))

paths_final.sort(key=lambda x: x[1])

for path, cost in paths_final:
    result_for_this_loop = " -> ".join(path)
    print(f"The Path Followed is {result_for_this_loop} and the cost of it is {cost}")
    path_graph = G.edge_subgraph([(path[i], path[i+1]) for i in range(len(path)-1)])
    
    pos = nx.shell_layout(path_graph)
    nx.draw_networkx_nodes(path_graph, pos)
    nx.draw_networkx_edges(path_graph, pos)
    nx.draw_networkx_edge_labels(path_graph, pos, edge_labels={(path[i], path[i+1]): G[path[i]][path[i+1]]["cost"] for i in range(len(path)-1)})
    nx.draw_networkx_labels(path_graph, pos)
    plt.title(f"Path: {' -> '.join(path)}\nCost: {cost}")
    plt.show()

final_result = " -> ".join(paths_final[0][0])
print(f"The oracle is {final_result} and the cost is {paths_final[0][1]}")

oracle_path = paths_final[0][0]

####DFS####
print("DFS")
def dfs_path(graph, start, end, path=[], visited_nodes=[]):
    path = path + [start]
    visited_nodes.append(start)
    if start == end:
        return path, visited_nodes
    if start not in graph:
        return None, visited_nodes
    shortest_path = None
    for neighbor in graph[start]:
        neighbor_name = neighbor
        if neighbor_name not in path:
            new_path, visited_nodes = dfs_path(graph, neighbor_name, end, path, visited_nodes)
            if new_path:
                if shortest_path is None or (new_path < shortest_path):
                    shortest_path = new_path
    return shortest_path, visited_nodes

end_node = goal_node

shortest_path, visited_nodes = dfs_path(G, start_node, end_node)

if shortest_path:
    print("DFS path from {} to {}:".format(start_node, end_node))
    print(" -> ".join(shortest_path))
else:
    print("No path found from {} to {}.".format(start_node, end_node))

# Create a list of edges in the DFS path for visualization
if shortest_path:
    edges_in_dfs_path = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

# Draw the graph with labels
pos = nx.spring_layout(G)

# Highlight visited nodes
visited_nodes_set = set(visited_nodes)
unvisited_nodes = [node for node in G.nodes() if node not in visited_nodes_set]

# Draw the visited nodes in light green and unvisited nodes in light gray
nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, node_color='lightgreen', node_size=500)
nx.draw_networkx_nodes(G, pos, nodelist=unvisited_nodes, node_color='lightgray', node_size=500)

# Draw the edges in blue dotted lines for non-DFS path edges
for edge in G.edges():
    if edge not in edges_in_dfs_path:
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='blue', width=1, style='dotted')

# Highlight the DFS path in red
nx.draw_networkx_edges(G, pos, edgelist=edges_in_dfs_path, edge_color='red', width=2)

# Draw the labels for nodes
node_labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
plt.show()
####BFS####
print("BFS")
start_node = start_node
bfs_edges = list(nx.bfs_edges(G, source=start_node))


pos = nx.spring_layout(G) 
node_colors = ['g' if node == start_node else 'b' for node in G.nodes()]
edge_colors = ['r' if edge in bfs_edges else 'k' for edge in G.edges()]

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=800, font_size=12, font_color='white')
edge_labels = nx.get_edge_attributes(G, 'cost')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.4)

plt.title('Breadth-First Search (BFS) Visualization')
plt.show()
print("BFS Traversal Order:")
for edge in bfs_edges:
    print( '->'.join(map(str, edge)))

new_graph = nx.DiGraph(bfs_edges)

pos = nx.spring_layout(new_graph)  # Position nodes for a nicer layout

plt.figure(figsize=(8, 6))
nx.draw(new_graph, pos, with_labels=True, node_size=800, font_size=12, font_color='black')

plt.title('New Graph with Specified Edges')
plt.show()
###Hill Climbing###
print("Hill Climbing")
def hill_climbing_with_heuristic(graph, start_node, goal_node, heuristics):
    current_node = start_node
    path = [current_node]
    total_cost = 0
    canceled_nodes = []

    while current_node != goal_node:
        neighbors = list(graph.neighbors(current_node))
        min_cost = float('inf')
        next_node = None

        for neighbor in neighbors:
            edge_cost = graph[current_node][neighbor].get('cost', 0)
            neighbor_heuristic = heuristics.get(neighbor, 0)
            if neighbor not in path and (edge_cost + neighbor_heuristic) < min_cost:
                min_cost = edge_cost + neighbor_heuristic
                next_node = neighbor

        if next_node is None:
            canceled_nodes.append(current_node)
            break

        path.append(next_node)

        current_node = next_node

    return path, canceled_nodes

path, canceled_nodes = hill_climbing_with_heuristic(G, start_node, goal_node, heuristics)
final_path = " -> ".join(path)

new_G = G.copy()
node_colors = ['g' if node in path else 'r' if node in canceled_nodes else 'b' for node in new_G.nodes()]
edge_colors = ['r' if edge in zip(path, path[1:]) else 'k' for edge in new_G.edges()]

pos = nx.spring_layout(new_G)

plt.figure(figsize=(10, 6))
nx.draw(new_G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=800, font_size=12, font_color='white')
edge_labels = nx.get_edge_attributes(new_G, 'cost')
nx.draw_networkx_edge_labels(new_G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.4)

plt.title(f'Hill Climbing Path from {start_node} to {goal_node}')
plt.show()

print(f"Hill Climbing Path from {start_node} to {goal_node}: {final_path}")
###Beam Search###
print("Beam Search")
def beam_search(graph, start, goal, beam_width):
    # Initialize the search with the start node
    initial_path = [start]
    beam = [(initial_path, 0)]  # Each element is a tuple (path, cost)

    while beam:
        # Generate new candidates by extending the paths
        candidates = []
        for path, cost in beam:
            current_node = path[-1]
            for neighbor in graph.neighbors(current_node):
                if neighbor not in path:
                    new_path = path + [neighbor]
                    new_cost = cost + graph[path[-1]][neighbor]['cost']
                    candidates.append((new_path, new_cost))
        
        # Sort the candidates by cost and select the top-k
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_width]
        
        # Check if the goal node is reached in any of the paths
        for path, cost in beam:
            if path[-1] == goal:
                return path, cost
    
    return None, float('inf')  # Return None if no path is found

beam_width = int(input("Enter the Beam Width "))
best_path, min_cost = beam_search(G, start_node, goal_node, beam_width)

# Create a layout for the nodes in the graph
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')

# Draw the best path found by the beam search
if best_path:
    path_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

best_path1 = " -> ".join(best_path)
# Label the nodes with their names
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

# Display the graph, best path, and cost
if best_path:
    print(f"Beam Search from {start_node} to {goal_node} (Min Cost: {min_cost})")
    print(f"Best path from {start_node} to {goal_node}: {best_path1}")
    print(f"Minimum cost: {min_cost}")
else:
    print(f"No path found from {start_node} to {goal_node}")

plt.title(f"Beam Search from {start_node} to {goal_node} (Min Cost: {min_cost})")
plt.axis('off')
plt.show()
###B&B###
print("B&B")

# Initialize the priority queue (min-heap) to store partial paths
priority_queue = PriorityQueue()

# Add the initial path with cost 0 to the queue
priority_queue.put((0, [start_node]))

# Create an empty list to store nodes explored
explored_nodes = []

# Create an empty dictionary to store the best-known costs
best_known_costs = {}

# Initialize the plot
plt.figure(figsize=(12, 8))
fgfg = 0
while not priority_queue.empty():
    fgfg = fgfg + 1
    # Get the path with the lowest cost from the queue
    current_cost, path = priority_queue.get()
    
    # Get the last node in the current path
    current_node = path[-1]
    
    # Mark the current node as explored
    explored_nodes.append(current_node)
    
    # Create a copy of the original graph for visualization
    current_graph = copy.deepcopy(G)
    
    # Draw the graph with explored nodes in green and current node in yellow
    node_colors = ["green" if node in explored_nodes else "yellow" for node in current_graph.nodes()]
    edge_colors = ["red" if u == current_node else "gray" for u, v in current_graph.edges()]
    
    # Create a subgraph containing only the nodes and edges on the minimum cost path
    min_cost_path = path
    subgraph = current_graph.subgraph(min_cost_path)
    
    # Draw the graph
    pos = nx.spring_layout(current_graph, seed=42)  # You can change the layout algorithm
    nx.draw(current_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12, font_color="black", edge_color=edge_colors)
    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color="red", font_size=12, font_color="white", width=2)
    
    # Update the plot
    plt.title(f"Branch and Bound Algorithm Visualization (iteration {fgfg})")
    plt.axis("off")
    plt.show()
    
    # If we have reached the end node, stop the algorithm
    if current_node == goal_node:
        break
    
    # Explore neighboring nodes
    for neighbor in current_graph[current_node]:
        neighbor_cost = current_graph[current_node][neighbor]['cost']
        if neighbor not in path:
            # Calculate the total cost of the new path
            total_cost = current_cost + neighbor_cost
            
            # Check if this path has a lower cost than the best-known cost for the neighbor
            if neighbor not in best_known_costs or total_cost < best_known_costs[neighbor]:
                best_known_costs[neighbor] = total_cost
                # Add the new path to the priority queue
                new_path = path + [neighbor]
                priority_queue.put((total_cost, new_path))

# Oracle: Print the final chosen path and its cost
if end_node in explored_nodes:
    final_path = path
    final_cost = current_cost  # The cost of the final path
    print("Final Chosen Path by Branch And Bound: ", " -> ".join(final_path))
    print("Cost of the Path: ", final_cost)
else:
    print("No path to the destination exists.")

# Oracle: Check if the algorithm's result matches the oracle path
if final_path == oracle_path:
    print("Algorithm matched the oracle path:", " -> ".join(path))
else:
    print("Algorithm did not match the oracle path.")

###B&B Greedy###
print("B&B Greedy")
# Initialize the plot
plt.figure(figsize=(12, 8))

current_node = start_node # Start from node "P"
explored_nodes = [current_node]
total_cost = 0
final_path = [current_node]
vb = 0
while current_node != goal_node:
    vb = vb + 1

    # Get neighboring nodes and their costs
    neighbors = G[current_node]
    min_cost = float('inf')
    min_cost_neighbor = None

    for neighbor, data in neighbors.items():
        neighbor_cost = data['cost']
        if neighbor not in explored_nodes:
            if neighbor_cost < min_cost:
                min_cost = neighbor_cost
                min_cost_neighbor = neighbor

    if min_cost_neighbor is None:
        print("No path to the destination exists.")
        break

    # Move to the neighbor with the lowest cost
    current_node = min_cost_neighbor
    explored_nodes.append(current_node)
    total_cost += min_cost
    final_path.append(current_node)

    # Draw the graph
    current_graph = copy.deepcopy(G)
    node_colors = ["green" if node in explored_nodes else "yellow" for node in current_graph.nodes()]
    edge_colors = ["red" if u == current_node else "gray" for u, v in current_graph.edges()]

    pos = nx.spring_layout(current_graph, seed=42)
    nx.draw(current_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12,
            font_color="black", edge_color=edge_colors)
    plt.title(f"Greedy Branch and Bound Algorithm Visualization (iteration{vb})")
    plt.axis("off")
    plt.show()

# Print the final chosen path and its cost
if current_node == goal_node:
    print("Final Chosen Path by Greedy Branch and Bound Algorithm Visualization:", " -> ".join(final_path))
    print("Cost of the Path:", total_cost)

# Check if the algorithm's result matches the oracle path
if final_path == oracle_path:
    print("Algorithm Matched the oracle path:", " -> ".join(final_path))
else:
    print("Algorithm did not find the oracle path.")
###B&B greedy with extended list###
print("B&B greedy with extended list")

def calculate_heuristic(node, goal, heuristics):
    return heuristics.get(node, 0)


# Initialize the priority queue (min-heap) to store partial paths with heuristic
priority_queue = PriorityQueue()

# Add the initial path with cost 0 and heuristic to the queue
priority_queue.put((0, [start_node], calculate_heuristic(start_node, goal_node, heuristics)))

# Create an empty list to store nodes explored
explored_nodes = []

# Create an empty dictionary to store the best-known costs
best_known_costs = {}
fsdghfsdg = 0
plt.figure(figsize=(12, 8))
while not priority_queue.empty():
    fsdghfsdg = fsdghfsdg + 1
    # Get the path with the lowest cost and heuristic from the queue
    current_cost, path, heuristic = priority_queue.get()
    
    # Get the last node in the current path
    current_node = path[-1]
    
    # Mark the current node as explored
    explored_nodes.append(current_node)
    # Create a copy of the original graph for visualization
    current_graph = copy.deepcopy(G)
    
    # Draw the graph with explored nodes in green and current node in yellow
    node_colors = ["green" if node in explored_nodes else "yellow" for node in current_graph.nodes()]
    edge_colors = ["red" if u == current_node else "gray" for u, v in current_graph.edges()]
    
    # Create a subgraph containing only the nodes and edges on the minimum cost path
    min_cost_path = path
    subgraph = current_graph.subgraph(min_cost_path)
    
    # Draw the graph
    pos = nx.spring_layout(current_graph, seed=42)
    nx.draw(current_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12,
            font_color="black", edge_color=edge_colors)
    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color="red", font_size=12, font_color="white", width=2)
    
    # Update the plot
    plt.title(f"Greedy Branch and Bound Algorithm Visualization with extended list (iteration {fsdghfsdg})")
    plt.axis("off")
    plt.show()
    
    # If we have reached the end node, stop the algorithm
    if current_node == goal_node:
        break
    
    # Explore neighboring nodes
    for neighbor in G.neighbors(current_node):
        neighbor_cost = G[current_node][neighbor]['cost']
        if neighbor not in path:
            # Calculate the total cost of the new path
            total_cost = current_cost + neighbor_cost
            # Calculate the heuristic for the neighbor
            neighbor_heuristic = calculate_heuristic(neighbor, goal_node, heuristics)
            
            # Calculate the total heuristic (greedy strategy: prioritize low heuristic)
            total_heuristic = neighbor_heuristic
            
            # Check if this path has a lower cost than the best-known cost for the neighbor
            if neighbor not in best_known_costs or total_cost < best_known_costs[neighbor]:
                best_known_costs[neighbor] = total_cost
                # Add the new path to the priority queue with heuristic
                new_path = path + [neighbor]
                priority_queue.put((total_cost, new_path, total_heuristic))

# Print the final chosen path and its cost
if current_node == goal_node:
    print("Final Chosen Path by the Greedy Branch and Bound Algorithm Visualization with extended list :", " -> ".join(path))
    print("Cost of the Path:", current_cost)
###B&B with Hueristics and Cost###
print("B&B with Hueristics and Cost")
plt.figure(figsize=(12, 8))


# Create a function to calculate the heuristic estimate
def calculate_heuristic(node, goal, heuristics):
    return heuristics.get(node, 0)


class PathNode:
    def __init__(self, node, path, cost, heuristic):
        self.node = node
        self.path = path
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        # Compare based on total estimated cost (cost + heuristic)
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def branch_and_bound_with_heuristics_and_cost(graph, start, goal, heuristics):
    # Initialize the priority queue
    priority_queue = PriorityQueue()
    
    # Add the initial path node with cost and heuristic
    initial_heuristic = calculate_heuristic(start, goal, heuristics)
    initial_path_node = PathNode(start, [start], 0, initial_heuristic)
    priority_queue.put(initial_path_node)
    
    # Create an empty dictionary to store the best-known costs
    best_known_costs = {}
    
    while not priority_queue.empty():
        # Get the path node with the lowest estimated total cost
        current_path_node = priority_queue.get()
        
        current_node = current_path_node.node
        current_path = current_path_node.path
        current_cost = current_path_node.cost
        
        # If we have reached the goal node, stop the algorithm
        if current_node == goal:
            return current_path, current_cost
        
        # Explore neighboring nodes
        for neighbor in graph[current_node]:
            neighbor_cost = graph[current_node][neighbor].get('cost', 0)
            
            # Calculate the total cost of the new path
            total_cost = current_cost + neighbor_cost
            
            # Calculate the heuristic for the neighbor
            neighbor_heuristic = calculate_heuristic(neighbor, goal, heuristics)
            
            # Calculate the total heuristic (greedy strategy: prioritize low heuristic)
            total_heuristic = neighbor_heuristic
            
            # Check if this path has a lower cost than the best-known cost for the neighbor
            if neighbor not in best_known_costs or total_cost < best_known_costs[neighbor]:
                best_known_costs[neighbor] = total_cost
                
                # Add the new path node to the priority queue with heuristic
                new_path = current_path + [neighbor]
                new_path_node = PathNode(neighbor, new_path, total_cost, neighbor_heuristic)
                priority_queue.put(new_path_node)
    
    # If no path to the destination exists
    return [], math.inf

# Example usage:
optimal_path, optimal_cost = branch_and_bound_with_heuristics_and_cost(G, start_node, goal_node, heuristics)

# Print the optimal path and cost
if not optimal_path:
    print("No path to the destination exists.")
else:
    print("Path Chosend by Branch and Bound with Hueristics and cost is :", " -> ".join(optimal_path))
    print("Least Cost that is found :", optimal_cost)

# Visualize the final path
final_graph = copy.deepcopy(G)
node_colors = ["green" if node in optimal_path else "yellow" for node in final_graph.nodes()]
edge_colors = ["red" if u in optimal_path and v in optimal_path else "gray" for u, v in final_graph.edges()]

pos = nx.spring_layout(final_graph, seed=42)
nx.draw(final_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12,
        font_color="black", edge_color=edge_colors)
plt.title("Branch and Bound with Hueristics and cost Algorithm Visualization")
plt.axis("off")
plt.show()

###B&B with Hueristics and greedy###
print("B&B with Hueristics and greedy")
plt.figure(figsize=(12, 8))
class PathNode:
    def __init__(self, node, path, cost, heuristic):
        self.node = node
        self.path = path
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        # Compare based on the sum of cost and heuristic (greedy strategy)
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def branch_and_bound_greedy_with_heuristics(graph, start, goal, heuristics):
    # Initialize the priority queue (min-heap) to store partial paths
    priority_queue = PriorityQueue()

    # Add the initial path with cost 0 and heuristic to the queue
    initial_path_node = (0, [start], heuristics[start])
    priority_queue.put(initial_path_node)

    while not priority_queue.empty():
        # Get the path with the lowest cost and heuristic from the queue
        current_cost, path, current_heuristic = priority_queue.get()

        current_node = path[-1]

        # If we have reached the goal node, stop the algorithm
        if current_node == goal:
            return path, current_cost

        # Explore neighboring nodes
        for neighbor in graph[current_node]:
            neighbor_cost = graph[current_node][neighbor].get('cost', 0)

            # Calculate the total cost of the new path
            total_cost = current_cost + neighbor_cost

            # Calculate the heuristic for the neighbor
            neighbor_heuristic = heuristics.get(neighbor, 0)

            # Calculate the total heuristic (greedy strategy: prioritize low heuristic)
            total_heuristic = neighbor_heuristic

            # Create a new path with updated cost and heuristic
            new_path = path + [neighbor]

            # Add the new path to the priority queue with cost and heuristic
            priority_queue.put((total_cost, new_path, total_heuristic))

    # If no path to the destination exists
    return [], float('inf')

# Find the optimal path using Branch and Bound with greedy heuristics
optimal_path, optimal_cost = branch_and_bound_greedy_with_heuristics(G, start_node, goal_node, heuristics)

# Print the optimal path and cost
if not optimal_path:
    print("No path to the destination exists.")
else:
    print("Path Chosen by B&B with Hueristics and greedy:", " -> ".join(optimal_path))
    print("Optimal Cost:", optimal_cost)

# Check if the algorithm's result matches the oracle path
if optimal_path == oracle_path:
    print("Algorithm matched the oracle path:", "->".join(optimal_path))
else:
    print("Algorithm did not match the oracle path.")

# Visualize the final path
final_graph = copy.deepcopy(G)
node_colors = ["green" if node in optimal_path else "yellow" for node in final_graph.nodes()]
edge_colors = ["red" if u in optimal_path and v in optimal_path else "gray" for u, v in final_graph.edges()]

pos = nx.spring_layout(final_graph, seed=42)
nx.draw(final_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12,
        font_color="black", edge_color=edge_colors)
plt.title("Branch and Bound with Huersitics and greedy Algorithm Visualization (Optimal Path)")
plt.axis("off")
plt.show()
###A*###
print("A*")
plt.figure(figsize=(12, 8))




def a_star(graph, start, goal, heuristic):
    open_list = []  # Priority queue to store paths
    closed_set = set()  # Set to store explored nodes

    # Add the start node to the priority queue
    heapq.heappush(open_list, (0, [start]))

    while open_list:
        # Get the path with the lowest priority
        current_cost, path = heapq.heappop(open_list)
        current_node = path[-1]

        if current_node == goal:
            return path, current_cost  # Goal reached, return the path and cost

        if current_node in closed_set:
            continue

        closed_set.add(current_node)

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor not in closed_set:
                # Calculate the total cost from start to neighbor
                cost = graph[current_node][neighbor].get('cost', 1)
                total_cost = current_cost + cost

                # Calculate the heuristic estimate from neighbor to goal
                neighbor_heuristic = heuristic(neighbor, goal)

                # Calculate the priority (cost + heuristic)
                priority = total_cost + neighbor_heuristic

                # Create a new path to the neighbor
                new_path = path + [neighbor]

                # Add the new path to the priority queue
                heapq.heappush(open_list, (total_cost, new_path))

    return None, None  # No path to the goal exists

def heuristic(node, goal):
    # Example heuristic: Manhattan distance
    return abs(ord(node) - ord(goal))

optimal_path, optimal_cost = a_star(G, start_node, goal_node, heuristic)

if optimal_path:
    print("Optimal Path:", " -> ".join(optimal_path))
    print("Optimal Cost:", optimal_cost)
else:
    print("No path to the destination exists.")

# Check if the algorithm's result matches the oracle path
if optimal_path == oracle_path:
    print("Algorithm matched the oracle path:", optimal_path)
else:
    print("Algorithm did not match the oracle path.")

# Visualize the final path
final_graph = copy.deepcopy(G)
node_colors = ["green" if node in optimal_path else "yellow" for node in final_graph.nodes()]
edge_colors = ["red" if u in optimal_path and v in optimal_path else "gray" for u, v in final_graph.edges()]

pos = nx.spring_layout(final_graph, seed=42)
nx.draw(final_graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12,
        font_color="black", edge_color=edge_colors)
plt.title("A* Search Algorithm Visualization")
plt.axis("off")
plt.show()