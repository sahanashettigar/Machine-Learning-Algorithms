"""
You can create any other helper funtions.
Do not modify the given functions
"""

import heapq


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    if(len(cost) == 0 or len(goals) == 0 or len(heuristic) == 0):
        return path
    frontier = []
    visited = [0 for i in range(len(heuristic))]
    key_value_dict = {}
    heapq.heappush(frontier, (0, start_point, 0, [start_point]))
    heapq.heapify(frontier)
    while(len(frontier) > 0):
        cost_of_node, node1, pathcost, path_taken = heapq.heappop(frontier)
        if(visited[node1] == 1):
            continue
        else:
            visited[node1] = 1
        if(node1 in goals):
            path = path_taken
            return path
        for i in range(len(heuristic)-1, 0, -1):
            new_path = []
            if(cost[node1][i] > 0 and visited[i] == 0):
                totalcost = heuristic[i] + cost[node1][i] + pathcost
                new_path_cost = pathcost+cost[node1][i]
                new_path = path_taken.copy()
                new_path.append(i)
                if (i not in key_value_dict):
                    key_value_dict[i] = totalcost
                    heapq.heappush(
                        frontier, (totalcost, i, new_path_cost, new_path))
                else:
                    if(key_value_dict[i] > totalcost):
                        key_value_dict[i] = totalcost
                        heapq.heappush(
                            frontier, (totalcost, i, new_path_cost, new_path))
        heapq.heapify(frontier)
    return path


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    visited = [0 for i in range(len(cost))]
    frontier = []
    if(len(goals) == 0 or len(cost) == 0):
        return []
    path_taken = [start_point]
    frontier.append((start_point, path_taken))
    while(len(frontier) > 0):
        node1, path_taken = frontier.pop()
        visited[node1] = 1
        haspath = 0
        if (node1 in goals):
            path = path_taken
            return path
        for i in range(len(cost)-1, 0, -1):
            new_path = []
            if (cost[node1][i] > 0 and visited[i] == 0):
                new_path = path_taken.copy()
                new_path.append(i)
                frontier.append((i, new_path))
    return path
