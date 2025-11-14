"""Minimal Gym-style environment for maintenance control policies."""

from __future__ import annotations

import numpy as np

from config import (
    INITIAL_WATER_LEVEL,
    TIME_STEPS,
    LAMBDA_FACTOR,
    PERFORMANCE_TARGET,
    PERFORMANCE_TOLERANCE,
    RELIABILITY_REQUIREMENT,
    REQUIREMENT_PENALTY_SCALE,
    BETA,
    ETA,
)
from src.failure_models import failure_probability_from_health


class MaintenanceEnv:
    """Simulates tank health dynamics for reinforcement-learning agents."""

    def __init__(
        self,
        time_steps: int = TIME_STEPS,
        lambda_factor: float = LAMBDA_FACTOR,
        performance_target: float = PERFORMANCE_TARGET,
        performance_tolerance: float = PERFORMANCE_TOLERANCE,
        reliability_requirement: float = RELIABILITY_REQUIREMENT,
        requirement_penalty_scale: float = REQUIREMENT_PENALTY_SCALE,
        seed: int | None = None,
    ):
        self.time_steps = time_steps
        self.lambda_factor = np.clip(lambda_factor, 0.0, 1.0)
        self.performance_target = float(np.clip(performance_target, 0.0, 1.0))
        self.performance_tolerance = float(np.clip(performance_tolerance, 0.0, 0.5))
        self.reliability_requirement = float(np.clip(reliability_requirement, 0.0, 1.0))
        self.requirement_penalty_scale = float(max(requirement_penalty_scale, 0.0))
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        self.health = INITIAL_WATER_LEVEL
        self.prev_health = self.health
        self.total_cost = 0.0
        self.failures = 0
        self.total_operating_time = 0.0
        self.time_since_last_failure = 0.0
        self.cumulative_time_between_failures = 0.0
        self.step_idx = 0
        self.last_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.base_seed is not None:
            self.rng = np.random.default_rng(self.base_seed)

        self.health = INITIAL_WATER_LEVEL
        self.prev_health = self.health
        self.total_cost = 0.0
        self.failures = 0
        self.total_operating_time = 0.0
        self.time_since_last_failure = 0.0
        self.cumulative_time_between_failures = 0.0
        self.step_idx = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action):
        """Apply [maintenance_inflow, degradation_control] action."""

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        inflow, outflow = action

        degradation = outflow * self.rng.uniform(0.8, 1.2)
        maintenance = inflow * self.rng.uniform(0.8, 1.2)
        maintenance_cost = inflow * 0.5

        self.health += (maintenance - degradation) * 5 - maintenance_cost
        self.health = float(np.clip(self.health, 0.0, 100.0))

        fail_prob = failure_probability_from_health(
            self.health,
            self.step_idx,
            beta=BETA,
            eta=ETA,
            previous_health=self.prev_health,
        )

        repair_cost = 0.0
        downtime_penalty = 0.0
        failure_happened = False
        self.total_operating_time += 1.0
        self.time_since_last_failure += 1.0
        if self.rng.random() < fail_prob:
            failure_happened = True
            self.failures += 1
            repair_cost = 10 + self.rng.uniform(0, 5)
            downtime_penalty = 5
            self.health = float(np.clip(self.health + self.rng.uniform(20, 50), 0, 100))
            self.cumulative_time_between_failures += self.time_since_last_failure
            self.time_since_last_failure = 0.0

        step_cost = maintenance_cost + repair_cost + downtime_penalty
        compliance_score, requirement_penalty, perf_gap, reliability_gap = self._performance_requirement_model(fail_prob)
        self.total_cost += step_cost

        reliability_component = self.health - 0.05 * abs(self.health - self.prev_health)
        adjusted_reliability = reliability_component * compliance_score
        cost_component = step_cost + (5 if failure_happened else 0) + requirement_penalty
        reward = self.lambda_factor * adjusted_reliability - (1 - self.lambda_factor) * cost_component

        self.prev_health = self.health
        self.last_action = action
        self.step_idx += 1
        done = self.step_idx >= self.time_steps

        info = {
            "fail_probability": fail_prob,
            "failure": failure_happened,
            "total_cost": self.total_cost,
            "failures": self.failures,
            "requirement_compliance": compliance_score,
            "performance_gap": perf_gap,
            "reliability_gap": reliability_gap,
            "requirement_penalty": requirement_penalty,
            "mean_time_between_failures": self._mtbf(),
            "time_since_last_failure": self.time_since_last_failure,
        }

        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        normalized_health = self.health / 100.0
        time_fraction = self.step_idx / max(self.time_steps - 1, 1)
        return np.array(
            [
                normalized_health,
                time_fraction,
                self.last_action[0],
                self.last_action[1],
                self.failures / max(self.step_idx, 1),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def _performance_requirement_model(self, fail_prob: float):
        """Implements a simple performance-requirement model penalty."""

        normalized_health = self.health / 100.0
        performance_gap = max(
            0.0,
            self.performance_target - normalized_health - self.performance_tolerance,
        )

        reliability = 1.0 - fail_prob
        reliability_gap = max(0.0, self.reliability_requirement - reliability)

        total_gap = performance_gap + reliability_gap
        compliance_score = 1.0 - np.clip(total_gap, 0.0, 1.0)
        penalty = self.requirement_penalty_scale * total_gap
        return compliance_score, penalty, performance_gap, reliability_gap

    # ------------------------------------------------------------------
    def _mtbf(self):
        """Return the empirical MTBF accumulated so far."""

        if self.failures == 0:
            return self.total_operating_time
        return self.cumulative_time_between_failures / self.failures


__all__ = ["MaintenanceEnv"]
