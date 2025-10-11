# rl/train_rl.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from rl.env import SwarmRescueEnv

def make_env_fn(grid_size=20, num_drones=3, num_humans=5):
    def _init():
        env = SwarmRescueEnv(grid_size=grid_size, num_drones=num_drones, num_humans=num_humans)
        return env
    return _init

def train_ppo(total_timesteps=200_000, save_path="rl/checkpoints/ppo_swarm"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = DummyVecEnv([make_env_fn()])  # single env wrapped for SB3
    env = VecMonitor(env)

    # policy: MlpPolicy; observation shape is flat; action_space MultiDiscrete supported by SB3 (since it's gym)
    # SB3 requires action_space to be Discrete or MultiDiscrete (supported)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/ppo_swarm")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved PPO model to {save_path}")
    return model

def load_ppo(path="rl/checkpoints/ppo_swarm.zip"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = PPO.load(path)
    return model

if __name__ == "__main__":
    train_ppo(50000)
