import numpy as np
from config import LAMBDA_FACTOR, INITIAL_WATER_LEVEL, TIME_STEPS
from src.failure_models import failure_probability_from_health
from config import BETA, ETA



"H represents health (water level height)"
"inflow represents maintenance actions, rate (water inflow)"
"outflow represents degradation/failure rate (water outflow)"

def simulate_component(strategy):
    H = INITIAL_WATER_LEVEL
    prev_health = H
    total_cost = 0
    failures = 0
    health_history = []

    for t, (inflow, outflow) in enumerate(strategy):
        inflow = np.clip(inflow, 0, 1)
        outflow = np.clip(outflow, 0, 1)

        # Natural degradation + maintenance
        degradation = outflow * np.random.uniform(0.8, 1.2)
        maintenance = inflow * np.random.uniform(0.8, 1.2)
        H += (maintenance - degradation) * 5 - inflow * 0.5
        H = np.clip(H, 0, 100)

        # --- Failure mechanics ---
        fail_prob = failure_probability_from_health(
            H,
            t,
            beta=BETA,
            eta=ETA,
            previous_health=prev_health,
        )
        if np.random.rand() < fail_prob:
            failures += 1
            repair_cost = 10 + np.random.uniform(0, 5)
            downtime_penalty = 5
            H = np.clip(H + np.random.uniform(20, 50), 0, 100)
            total_cost += repair_cost + downtime_penalty

        health_history.append(H)
        prev_health = H

    # --- Compute overall fitness with lambda factor ---
    mean_health = np.mean(health_history)
    stability_penalty = np.var(health_history)

    reliability_component = mean_health - 0.1 * stability_penalty
    cost_component = total_cost + 5 * failures

    fitness = (
        LAMBDA_FACTOR * reliability_component
        - (1 - LAMBDA_FACTOR) * cost_component
    )

    # ✅ Return all three metrics
    return fitness, total_cost, failures



def failure_probability(health):
    """
    Failure probability grows exponentially as health decreases.
    When health drops below 30, failure chance rises sharply.
    """

    return np.exp(-0.05 * health)  # lower health → higher failure prob
