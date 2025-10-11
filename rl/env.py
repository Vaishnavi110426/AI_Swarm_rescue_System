# rl/env.py
import gymnasium as gym
from gym import spaces
import numpy as np
import random

class SwarmRescueEnv(gym.Env):
    """
    Gym-compatible grid-world environment for multi-drone rescue.
    Observation: flattened positions (drones + humans) as floats in [0,1], length = num_drones*2 + num_humans*2
    Action (per drone): discrete {0,1,2,3,4} -> stay, up, down, left, right
    For SB3 we will wrap multi-agent as a single action vector flattened to a discrete vector via MultiDiscrete or Dict.
    Here we implement a vector action as MultiDiscrete.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=20, num_drones=3, num_humans=5):
        super().__init__()
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.num_humans = num_humans

        # action for each drone: 5 discrete choices
        self.action_space = spaces.MultiDiscrete([5] * self.num_drones)

        # observation: normalized positions (drone x,y then human x,y) shape = (num_drones*2+num_humans*2,)
        obs_dim = self.num_drones * 2 + self.num_humans * 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        # drones in corners (or random)
        self.drone_pos = np.zeros((self.num_drones, 2), dtype=int)
        # simple placement: corners then center for next drones
        init_positions = [
            (0, 0),
            (self.grid_size - 1, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, self.grid_size - 1)
        ]
        for i in range(self.num_drones):
            if i < len(init_positions):
                self.drone_pos[i] = np.array(init_positions[i])
            else:
                self.drone_pos[i] = np.array((random.randrange(self.grid_size), random.randrange(self.grid_size)))

        self.human_pos = np.array([[-1, -1]] * self.num_humans, dtype=int)
        self.obstacles = self._generate_obstacles(count=int(self.grid_size * 1.2))
        self.timestep = 0
        self.rescued_count = 0
        return self._get_obs()

    def _generate_obstacles(self, count=20):
        obs = set()
        while len(obs) < count:
            x, y = random.randrange(0, self.grid_size), random.randrange(0, self.grid_size)
            # avoid spawn points approx
            if (x, y) in [(0,0),(self.grid_size-1,0),(0,self.grid_size-1),(self.grid_size-1,self.grid_size-1)]:
                continue
            obs.add((x, y))
        return obs

    def _get_obs(self):
        flat = []
        for p in self.drone_pos:
            flat += [p[0] / (self.grid_size - 1), p[1] / (self.grid_size - 1)]
        for h in self.human_pos:
            if h[0] >= 0:
                flat += [h[0] / (self.grid_size - 1), h[1] / (self.grid_size - 1)]
            else:
                flat += [-1.0, -1.0]
        return np.array(flat, dtype=np.float32)

    def step(self, actions):
        """
        actions: MultiDiscrete array-like length=num_drones
        """
        rewards = np.zeros(self.num_drones, dtype=float)
        for i, a in enumerate(actions):
            x, y = int(self.drone_pos[i][0]), int(self.drone_pos[i][1])
            if a == 1 and y > 0 and (x, y-1) not in self.obstacles:
                y -= 1
            elif a == 2 and y < self.grid_size - 1 and (x, y+1) not in self.obstacles:
                y += 1
            elif a == 3 and x > 0 and (x-1, y) not in self.obstacles:
                x -= 1
            elif a == 4 and x < self.grid_size - 1 and (x+1, y) not in self.obstacles:
                x += 1
            # else 0 = stay or blocked
            self.drone_pos[i] = np.array([x, y])

        # check rescues
        for hi in range(self.num_humans):
            hx, hy = self.human_pos[hi]
            if hx < 0:
                continue
            for di in range(self.num_drones):
                if np.array_equal(self.drone_pos[di], self.human_pos[hi]):
                    rewards[di] += 10.0
                    self.human_pos[hi] = np.array([-1, -1])
                    self.rescued_count += 1

        # small negative step penalty to encourage shorter missions
        rewards -= 0.01

        self.timestep += 1
        done = False
        # optional done criteria
        if self.timestep > 500 or self.rescued_count >= self.num_humans:
            done = True

        obs = self._get_obs()
        # return observation, sum of rewards or vector? SB3 expects a scalar reward per env.
        # We'll return average reward for now
        return obs, float(rewards.sum()), done, {}

    def place_humans(self, human_list):
        """
        Accept a list of (x,y) ints to set human locations, e.g. coming from detections.
        Fill missing humans with (-1,-1).
        """
        arr = np.array([[-1, -1]] * self.num_humans, dtype=int)
        for i, h in enumerate(human_list[: self.num_humans]):
            arr[i] = np.array(h)
        self.human_pos = arr

    def render(self, mode='human'):
        # optional ascii or matplotlib render; we use external visualizer
        pass
