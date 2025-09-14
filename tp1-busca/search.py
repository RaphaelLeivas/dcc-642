import heapq
from math import sqrt

WALL = 'X'
START_STATE = 'S'
GOAL_STATE  = 'G'

def plan(map, algorithm='bfs', heuristic=None):
    """ Loads a level, searches for a path between the given waypoints, and displays the result.

    Args:
        filename: The name of the text file containing the level.
        src_waypoint: The character associated with the initial waypoint.
        dst_waypoint: The character associated with the destination waypoint.

    """
    print(map)
    print("Algorithm:", algorithm)
    print("Heuristic:", heuristic)

    # Load the level from the file
    level = parse_level(map)

    # Retrieve the source and destination coordinates from the level.
    start = level['start']
    goal = level['goal']

    # Search for and display the path from src to dst.
    path = []
    visited = {}

    if algorithm == 'bfs':
        path, visited = bfs(start, goal, level, transition_model)
    elif algorithm == 'dfs':
        path, visited = dfs(start, goal, level, transition_model)
    elif algorithm == 'ucs':
        path, visited = ucs(start, goal, level, transition_model)
    elif algorithm == 'greedy':
        if heuristic == 'euclidian':
            path, visited = greedy_best_first(start, goal, level, transition_model, h_euclidian)
        elif heuristic == 'manhattan':
            path, visited = greedy_best_first(start, goal, level, transition_model, h_manhattan)
    elif algorithm == 'astar':
        if heuristic == 'euclidian':
            path, visited = a_star(start, goal, level, transition_model, h_euclidian)
        elif heuristic == 'manhattan':
            path, visited = a_star(start, goal, level, transition_model, h_manhattan)

    return path, path_cost(path, level), visited

def parse_level(map):
    """ Parses a level from a string.

    Args:
        level_str: A string containing a level.

    Returns:
        The parsed level (dict) containing the locations of walls (set), the locations of spaces 
        (dict), and a mapping of locations to waypoints (dict).
    """
    start = None
    goal = None
    walls = set()
    spaces = {}

    for j, line in enumerate(map.split('\n')):
        for i, char in enumerate(line):
            if char == '\n':
                continue
            elif char == WALL:
                walls.add((i, j))
            elif char == START_STATE:
                start = (i, j)
                spaces[(i, j)] = 1.
            elif char == GOAL_STATE:
                goal = (i, j) 
                spaces[(i, j)] = 1.
            elif char.isnumeric():
                spaces[(i, j)] = float(char)

    level = {'walls': walls, 'spaces': spaces, 'start': start, 'goal': goal}

    return level

def path_cost(path, level):
    """ Returns the cost of the given path.

    Args:
        path: A list of cells from the source to the goal.
        level: A loaded level, containing walls, spaces, and waypoints.

    Returns:
        The cost of the given path.
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += cost_function(level, path[i], path[i + 1], 
                              level['spaces'][path[i]], 
                              level['spaces'][path[i + 1]])

    return cost

# =============================
# Transition Model
# =============================

def cost_function(level, state1, state2, cost1, cost2):
    """ Returns the cost of the edge joining state1 and state2.

    Args:
        state1: A source location.
        state2: A target location.

    Returns:
        The cost of the edge joining state1 and state2.
    """

    ################################
    # 1.1 INSIRA SEU CÓDIGO AQUI
    ################################

    return 0

def transition_model(level, state1):
    """ Provides a list of adjacent states and their respective costs from the given state.

    Args:
        level: A loaded level, containing walls, spaces, and waypoints.
        state: A target location.

    Returns:
        A list of tuples containing an adjacent sates's coordinates and the cost of 
        the edge joining it and the originating state.

        E.g. from (0,0):
            [((0,1), 1),
             ((1,0), 1),
             ((1,1), 1.4142135623730951),
             ... ]
    """
    adj_states = {}

    ################################
    # 1.2 INSIRA SEU CÓDIGO AQUI
    ################################

    return adj_states.items()

# =============================
# Uninformed Search Algorithms
# =============================

def bfs(s, g, level, adj):
    """ Searches for a path from the source to the goal using the Breadth-First Search algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary 
        containing the visited cells and their respective parent cells.
    """
    visited = {s: None}

    ################################
    # 2.1 INSIRA SEU CÓDIGO AQUI
    ################################

    return [], visited

def dfs(s, g, level, adj):
    """ Searches for a path from the source to the goal using the Depth-First Search algorithm.
    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
    
    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}

    ################################
    # 2.2 INSIRA SEU CÓDIGO AQUI
    ################################

    return [], visited

def ucs(s, g, level, adj):
    """ Searches for a path from the source to the goal using the Uniform-Cost Search algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}

    ################################
    # 2.3 INSIRA SEU CÓDIGO AQUI
    ################################

    return [], visited

# ======================================
# Informed (Heuristic) Search Algorithms
# ======================================
def greedy_best_first(s, g, level, adj, h):
    """ Searches for a path from the source to the goal using the Greedy Best-First Search algorithm.
    
    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
        h: A heuristic function that estimates the cost from the current cell to the goal.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}

    ################################
    # 3.2 INSIRA SEU CÓDIGO AQUI
    ################################

    return [], visited

def a_star(s, g, level, adj, h):
    """ Searches for a path from the source to the goal using the A* algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
        h: A heuristic function that estimates the cost from the current cell to the goal.
    
    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}

    ################################
    # 3.3 INSIRA SEU CÓDIGO AQUI
    ################################

    return [], visited

# ======================================
# Heuristic functions
# ======================================
def h_euclidian(s, g):
    """ Estimates the cost from the current cell to the goal using the Euclidian distance.

    Args:
        s: The current location.
        g: The goal location.
    
    Returns:
        The estimated cost from the current cell to the goal.    
    """

    ################################
    # 3.1 INSIRA SEU CÓDIGO AQUI
    ################################

    return 0

def h_manhattan(s, g):
    """ Estimates the cost from the current cell to the goal using the Manhattan distance.
    
    Args:
        s: The current location.
        g: The goal location.
    
    Returns:
        The estimated cost from the current cell to the goal.
    """

    ################################
    # 3.1 INSIRA SEU CÓDIGO AQUI
    ################################

    return 0