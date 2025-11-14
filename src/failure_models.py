import numpy as np

try:
    import torch
except ImportError:
    torch = None

def exponential(health):
    return np.exp(-0.05 * health)

def linear(health):
    return np.clip(1 - health / 100, 0, 1)

def weibull(health, shape=2.5, scale=40):
    x = np.clip(100 - health, 0, 100)
    return 1 - np.exp(-((x / scale) ** shape))

def threshold(health, threshold=25):
    return 1.0 if health < threshold else 0.0

def shock(health):
    base = np.exp(-0.05 * health)
    shock = np.random.choice([0, 1], p=[0.98, 0.02])
    return np.clip(base + 0.2 * shock, 0, 1)


def weibull_bathtub(t, beta=1.5, eta=50):
    """
    Weibull 'bathtub curve' failure rate model.
    λ(t) = (β / η) * (t / η)^(β - 1)
    Returns instantaneous failure probability at time t.
    """
    if torch is not None and isinstance(t, torch.Tensor):
        t = torch.clamp(t, min=1e-3)
        hazard_rate = (beta / eta) * torch.pow(t / eta, beta - 1)
        return torch.clamp(hazard_rate, 0.0, 1.0)

    t = np.maximum(t, 0.001)  # avoid division by zero
    hazard_rate = (beta / eta) * ((t / eta) ** (beta - 1))
    return np.clip(hazard_rate, 0, 1)

def failure_probability_from_health(health, step, beta=1.5, eta=50, previous_health=None):
    """Blend age-based wear-out with health stress and sudden degradation spikes."""

    if torch is not None and (
        isinstance(health, torch.Tensor)
        or isinstance(previous_health, torch.Tensor)
    ):
        device = None
        if isinstance(health, torch.Tensor):
            device = health.device
        elif isinstance(previous_health, torch.Tensor):
            device = previous_health.device

        def _to_tensor(value, default=0.0):
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=torch.float32)
            return torch.tensor(float(value if value is not None else default), device=device)

        health_t = _to_tensor(health)
        prev_t = _to_tensor(previous_health) if previous_health is not None else None
        step_t = _to_tensor(step)

        normalized_health = torch.clamp(1 - health_t / 100.0, 0.0, 1.0)
        age_equivalent = step_t + normalized_health * eta
        wear_out = weibull_bathtub(age_equivalent, beta, eta)
        stress_component = torch.clamp(torch.pow(normalized_health, 1.3), 0.0, 1.0)

        trend_penalty = torch.tensor(0.0, device=device)
        if prev_t is not None:
            delta = torch.clamp((prev_t - health_t) / 100.0, 0.0, 1.0)
            trend_penalty = torch.sqrt(delta)

        combined = 1 - (1 - wear_out) * (1 - stress_component)
        fail_prob = combined + 0.4 * trend_penalty
        return torch.clamp(fail_prob, 0.0, 1.0)

    normalized_health = np.clip(1 - health / 100, 0, 1)
    age_equivalent = step + normalized_health * eta
    wear_out = weibull_bathtub(age_equivalent, beta, eta)

    # Higher stress as health drifts from 100, even if chronological age is low
    stress_component = np.clip(normalized_health ** 1.3, 0, 1)

    trend_penalty = 0.0
    if previous_health is not None:
        delta = np.clip((previous_health - health) / 100, 0, 1)
        trend_penalty = np.sqrt(delta)  # emphasize sharp drops more than slow drift

    combined = 1 - (1 - wear_out) * (1 - stress_component)
    fail_prob = combined + 0.4 * trend_penalty
    return np.clip(fail_prob, 0, 1)
