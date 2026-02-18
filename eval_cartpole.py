from stable_baselines3 import PPO
from envs.swingup_cartpole import SwingUpCartPole
import numpy as np

env = SwingUpCartPole()   # ❌ no render_mode
model = PPO.load("ppo_swingup_cartpole")

obs, _ = env.reset()

upright_count = 0

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    theta = obs[2]
    if abs(np.cos(theta)) > 0.95:
        upright_count += 1

    if terminated or truncated:
        obs, _ = env.reset()

print("Upright steps:", upright_count)
print("Evaluation completed successfully.")