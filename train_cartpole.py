from stable_baselines3 import PPO
from envs.swingup_cartpole import SwingUpCartPole

env = SwingUpCartPole()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=400_000)

model.save("ppo_swingup_cartpole")