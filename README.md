\# Inverted Pendulum Control using Reinforcement Learning



\## Overview

This project implements a reinforcement learning solution for the classical inverted pendulum control problem, focusing on swing-up and stabilization using Proximal Policy Optimization (PPO).



The implementation is CPU-based and emphasizes algorithmic correctness, reward design, and experimental analysis.



---



\## Tasks Implemented

\### Task 1: Single Inverted Pendulum

\- Swing-up from downward configuration

\- Stabilization around upright equilibrium

\- Reliable convergence using PPO



\### Task 2: Double Inverted Pendulum (Exploratory)

\- Two coupled unstable poles

\- Partial stabilization achieved

\- Demonstrates increased complexity and instability



---



\## Methodology

\- Algorithm: PPO (Stable-Baselines3)

\- Environment: Custom Gymnasium environments

\- Observation Space: Cart position, velocity, pole angle(s), angular velocity

\- Action Space: Continuous horizontal force



---



\## Reward Design

Reward is shaped using:

\- Cosine of pole angle to encourage upright configuration

\- Penalties on cart displacement and velocity for stability



---



\## Results

Key results are available in the `results/plots` directory:

\- Training reward curves

\- Upright probability during evaluation

\- Comparative analysis of experiments



---



\## Simulator Note

High-fidelity simulators (Isaac Sim / Gazebo) were explored but could not be executed due to system-level GPU and GLIBC constraints. A lightweight CPU-based simulation was adopted to ensure reproducibility and focus on RL behavior.



---



\## How to Run

```bash

python train\_cartpole.py

python eval\_cartpole.py

Save and close.



---





\## Author

Lakshya Kochar





