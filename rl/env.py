import gym
import numpy as np

class DroneLocalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

    def reset(self):
        obs = np.zeros(5, dtype=np.float32)
        return obs

    def step(self, action):
        obs = np.zeros(5, dtype=np.float32)
        reward = -np.linalg.norm(obs[:2])
        done = False
        return obs, reward, done, {}
