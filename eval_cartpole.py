from stable_baselines3 import PPO
from envs.swingup_cartpole import SwingUpCartPole
import numpy as np
import csv
import os

# =====================
# Configuration
# =====================
MODEL_PATH = "ppo_swingup_cartpole"
LOG_DIR = "results/logs"
LOG_FILE = os.path.join(LOG_DIR, "task1_eval.csv")
TOTAL_STEPS = 2000

# =====================
# Setup
# =====================
os.makedirs(LOG_DIR, exist_ok=True)

env = SwingUpCartPole()
model = PPO.load(MODEL_PATH)

obs, _ = env.reset()

upright_log = []
upright_count = 0

# =====================
# Evaluation Loop
# =====================
for step in range(TOTAL_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    theta = obs[2]
    is_upright = int(abs(np.cos(theta)) > 0.95)

    upright_log.append(is_upright)
    upright_count += is_upright

    if terminated or truncated:
        obs, _ = env.reset()

# =====================
# Saving Log
# =====================
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "upright"])
    for i, v in enumerate(upright_log):
        writer.writerow([i, v])

print("Upright steps:", upright_count)
print("Saved evaluation log to:", LOG_FILE)
