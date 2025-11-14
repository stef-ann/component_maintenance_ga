import numpy as np
import torch

from config import LAMBDA_FACTOR, INITIAL_WATER_LEVEL, TIME_STEPS
from config import BETA, ETA
from src.failure_models import failure_probability_from_health
from src.utils import get_torch_device

DEVICE = get_torch_device()


def _rand_uniform(device, low, high):
    return torch.empty((), device=device).uniform_(low, high)


def simulate_component(strategy):
    """Simulate a maintenance strategy using tensor operations."""

    if not isinstance(strategy, torch.Tensor):
        strategy_tensor = torch.as_tensor(
            strategy,
            dtype=torch.float32,
            device=DEVICE,
        )
    else:
        strategy_tensor = strategy.to(dtype=torch.float32)

    device = strategy_tensor.device
    strategy_tensor = torch.clamp(strategy_tensor, 0.0, 1.0)

    H = torch.tensor(float(INITIAL_WATER_LEVEL), device=device)
    prev_health = H.clone()
    total_cost = torch.tensor(0.0, device=device)
    failures = torch.tensor(0.0, device=device)
    horizon = strategy_tensor.shape[0]
    health_history = torch.empty(horizon, device=device)

    for t in range(horizon):
        inflow = strategy_tensor[t, 0].clamp(0.0, 1.0)
        outflow = strategy_tensor[t, 1].clamp(0.0, 1.0)

        maintenance = inflow * _rand_uniform(device, 0.8, 1.2)
        degradation = outflow * _rand_uniform(device, 0.8, 1.2)
        H = H + (maintenance - degradation) * 5 - inflow * 0.5
        H = H.clamp(0.0, 100.0)

        fail_prob = failure_probability_from_health(
            H,
            t,
            beta=BETA,
            eta=ETA,
            previous_health=prev_health,
        )

        if (torch.rand((), device=device) < fail_prob).item():
            failures += 1.0
            repair_cost = 10.0 + _rand_uniform(device, 0.0, 5.0)
            downtime_penalty = torch.tensor(5.0, device=device)
            H = torch.clamp(
                H + _rand_uniform(device, 20.0, 50.0),
                0.0,
                100.0,
            )
            total_cost += repair_cost + downtime_penalty

        health_history[t] = H
        prev_health = H.clone()

    mean_health = health_history.mean()
    stability_penalty = torch.var(health_history, unbiased=False)

    reliability_component = mean_health - 0.1 * stability_penalty
    cost_component = total_cost + 5.0 * failures

    fitness = LAMBDA_FACTOR * reliability_component - (1 - LAMBDA_FACTOR) * cost_component

    return (
        float(fitness.item()),
        float(total_cost.item()),
        float(failures.item()),
    )


def failure_probability(health):
    """
    Failure probability grows exponentially as health decreases.
    When health drops below 30, failure chance rises sharply.
    """

    if isinstance(health, torch.Tensor):
        return torch.exp(-0.05 * health)
    return np.exp(-0.05 * health)  # lower health → higher failure prob
