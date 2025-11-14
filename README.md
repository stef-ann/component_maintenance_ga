# Optimization Entry Points

- `python simulations/single_run.py` — runs the genetic algorithm, plots convergence, and generates a maintenance report.
- `python simulations/cma_run.py` — launches a CMA-ES search over the same action space, saving comparable plots and reports.

## Reinforcement-Learning Environment

- `src/rl_env.py` implements a lightweight Gym-style `MaintenanceEnv` that exposes `reset()` and `step(action)` so you can train action–reward policies directly on the tank dynamics.
- Actions are 2-D vectors `[maintenance_inflow, degradation_control]` in `[0, 1]`; rewards follow the lambda-weighted trade-off already used by the GA fitness.
- Run `python simulations/rl_random_policy.py` to see a single random rollout; plug any RL library (e.g., PPO, SAC) into `MaintenanceEnv` for learning-based optimization.
