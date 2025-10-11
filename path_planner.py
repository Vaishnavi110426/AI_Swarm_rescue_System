"""
A* path planner for grid-based environment.
Returns a list of grid coordinates (tuples) from start (exclusive) to goal (inclusive).
"""
import heapq

class AStarPlanner:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        x, y = node
        moves = [(1,0), (-1,0), (0,1), (0,-1)]
        neighbors = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def plan(self, start, goal, obstacles):
        # start, goal: tuples (x,y)
        start = tuple(start)
        goal = tuple(goal)
        obstacles_set = set(obstacles) if obstacles is not None else set()

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self.get_neighbors(current):
                if nxt in obstacles_set:
                    continue
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic(goal, nxt)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return []

        # reconstruct path
        path = []
        node = goal
        while node != start:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path
