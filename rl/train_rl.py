from stable_baselines3 import PPO
from rl.env import DroneLocalEnv

if __name__ == '__main__':
    env = DroneLocalEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save('ppo_drone_local')
