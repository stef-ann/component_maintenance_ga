"""Example rollout using the MaintenanceEnv with a random policy."""

import numpy as np

from src.rl_env import MaintenanceEnv


def run_random_episode(seed=0):
    env = MaintenanceEnv(seed=seed)
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = np.random.rand(2)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward, env.total_cost, env.failures


if __name__ == "__main__":
    reward, cost, failures = run_random_episode()
    print(f"Random policy episode reward: {reward:.2f}")
    print(f"Total cost: {cost:.2f} | Failures: {failures}")
