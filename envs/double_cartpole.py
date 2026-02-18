import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DoubleCartPole(gym.Env):
    """
    Double inverted pendulum on a cart.
    State:
    [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Physics constants
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length1 = 0.5
        self.length2 = 0.5
        self.dt = 0.02

        # Action: horizontal force
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(1,), dtype=np.float32
        )

        # Observation space
        high = np.array(
            [np.inf, np.inf, np.pi, np.inf, np.pi, np.inf], dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start both poles downward
        self.state = np.array([
            0.0,            # x
            0.0,            # x_dot
            np.pi,          # theta1
            0.0,            # theta1_dot
            np.pi,          # theta2
            0.0             # theta2_dot
        ], dtype=np.float32)

        self.steps = 0
        return self.state, {}

    def step(self, action):
        x, x_dot, t1, t1_dot, t2, t2_dot = self.state
        force = float(action[0])

        # Very simplified coupled dynamics (sufficient for RL)
        x_ddot = force / self.mass_cart

        t1_ddot = (
            self.gravity * np.sin(t1)
            + x_ddot * np.cos(t1)
        ) / self.length1

        t2_ddot = (
            self.gravity * np.sin(t2)
            + t1_ddot * np.cos(t2)
        ) / self.length2

        # Euler integration
        x_dot += x_ddot * self.dt
        x += x_dot * self.dt

        t1_dot += t1_ddot * self.dt
        t1 += t1_dot * self.dt

        t2_dot += t2_ddot * self.dt
        t2 += t2_dot * self.dt

        self.state = np.array(
            [x, x_dot, t1, t1_dot, t2, t2_dot], dtype=np.float32
        )

        # Reward: both poles upright
        reward = (
    		1.5 * np.cos(t1)
    		+ 1.5 * np.cos(t2)
    		- 0.02 * (x ** 2)
	)

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}