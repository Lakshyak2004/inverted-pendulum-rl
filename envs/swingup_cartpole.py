import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class SwingUpCartPole(CartPoleEnv):
    """
    Custom CartPole environment for swing-up task.
    """

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start pole pointing DOWN (theta = pi)
        self.state = np.array([
            0.0,        # cart position
            0.0,        # cart velocity
            np.pi,      # pole angle (downward)
            0.0         # pole angular velocity
        ], dtype=np.float32)

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        x, x_dot, theta, theta_dot = obs

        # Reward shaping
        reward = (
            np.cos(theta)          # upright = +1
            - 0.01 * (x ** 2)       # penalize cart movement
            - 0.001 * (x_dot ** 2)  # penalize jerky motion
        )

        return obs, reward, terminated, truncated, info