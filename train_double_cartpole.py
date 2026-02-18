from stable_baselines3 import PPO
from envs.double_cartpole import DoubleCartPole
import os

EXPERIMENT_NAME = "double_v2_reward_shaping"
SAVE_DIR = os.path.join("experiments", EXPERIMENT_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)

env = DoubleCartPole()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01
)

model.learn(total_timesteps=1_500_000)

model.save(os.path.join(SAVE_DIR, "ppo_model"))