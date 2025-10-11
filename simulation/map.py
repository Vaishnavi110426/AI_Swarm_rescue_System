class DisasterMap:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.grid = [[0]*grid_size for _ in range(grid_size)]
        self.humans = []

    def add_obstacle(self, x, y):
        self.grid[y][x] = 1

    def add_human(self, x, y):
        self.grid[y][x] = 2
        self.humans.append([x, y])

    def get_humans(self):
        return self.humans
    
    # simple map constants or helper functions if needed
def sample_obstacles(grid_size):
    import random
    obs = set()
    while len(obs) < max(1, grid_size // 2):
        obs.add((random.randrange(0, grid_size), random.randrange(0, grid_size)))
    return obs

