from stable_baselines3 import PPO
from envs.double_cartpole import DoubleCartPole
import numpy as np
import os
import csv

EXPERIMENT_NAME = "double_v2_reward_shaping"  # change as needed
MODEL_PATH = os.path.join("experiments", EXPERIMENT_NAME, "ppo_model")

env = DoubleCartPole()
model = PPO.load(MODEL_PATH)

obs, _ = env.reset()

upright_log = []
upright_steps = 0

for step in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    t1, t2 = obs[2], obs[4]
    is_upright = int(abs(np.cos(t1)) > 0.95 and abs(np.cos(t2)) > 0.95)

    upright_log.append(is_upright)
    upright_steps += is_upright

    if truncated:
        obs, _ = env.reset()

# Save log
os.makedirs("results/logs", exist_ok=True)
log_file = f"results/logs/{EXPERIMENT_NAME}_eval.csv"

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "upright"])
    for i, v in enumerate(upright_log):
        writer.writerow([i, v])

print("Experiment:", EXPERIMENT_NAME)
print("Both poles upright steps:", upright_steps)
print("Saved evaluation log to:", log_file)