import heapq

def dijkstra(graph, start):
    """
    Dijkstra's algorithm for shortest paths in a weighted graph.

    Parameters:
        graph: dict of dicts {u: {v: weight, ...}, ...}
        start: starting node

    Returns:
        dist: dict of shortest distances from start to each node
        prev: dict of predecessors (for path reconstruction)
    """
    # Initialize distances and predecessors
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0

    # Priority queue: (distance, node)
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)

        # Skip if we've already found a better path
        if current_dist > dist[u]:
            continue

        for v, weight in graph[u].items():
            alt = dist[u] + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    return dist, prev


def reconstruct_path(prev, start, goal):
    """Reconstruct shortest path from start to goal using prev map."""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path if path[0] == start else []

# Graph as adjacency dictionary
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2, 'Z': 6},
    'E': {'C': 10, 'D': 2, 'Z': 3},
    'Z': {'D': 6, 'E': 3}
}

dist, prev = dijkstra(graph, 'A')

print("Shortest distances:", dist)
print("Shortest path A → Z:", reconstruct_path(prev, 'A', 'Z'))