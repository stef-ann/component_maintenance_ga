"""CMA-ES optimizer for maintenance schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import cma
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'cma' package is required for CMA-ES optimization. Install with 'pip install cma'."
    ) from exc

from config import (
    TIME_STEPS,
    CMA_SIGMA,
    CMA_POPULATION,
    CMA_GENERATIONS,
    CMA_EVAL_REPEATS,
    SEED,
)
from src.tank_model import simulate_component


@dataclass
class CMAResult:
    best_strategy: np.ndarray
    best_fitness: float
    best_cost: float
    best_failures: float
    fitness_history: List[float]
    cost_history: List[float]
    failure_history: List[float]


class CMAOptimizer:
    """Wraps pycma to optimize the maintenance strategy."""

    def __init__(
        self,
        sigma: float = CMA_SIGMA,
        popsize: int = CMA_POPULATION,
        generations: int = CMA_GENERATIONS,
        eval_repeats: int = CMA_EVAL_REPEATS,
        seed: int | None = SEED,
    ) -> None:
        self.dim = TIME_STEPS * 2
        self.initial_mean = np.full(self.dim, 0.5, dtype=np.float64)
        self.sigma = sigma
        self.popsize = popsize
        self.generations = generations
        self.eval_repeats = max(1, eval_repeats)
        self.seed = seed

        self.best_strategy = None
        self.best_fitness = -np.inf
        self.best_cost = np.inf
        self.best_failures = np.inf

        self.fitness_history: List[float] = []
        self.cost_history: List[float] = []
        self.failure_history: List[float] = []

    # ------------------------------------------------------------------
    def run(self) -> CMAResult:
        options = {
            "popsize": self.popsize,
            "bounds": [0.0, 1.0],
            "seed": self.seed,
            "verb_log": 0,
            "verbose": -9,
        }

        es = cma.CMAEvolutionStrategy(self.initial_mean, self.sigma, options)

        for _ in range(self.generations):
            solutions = es.ask()
            values = []
            gen_best_fitness = -np.inf
            gen_best_cost = np.inf
            gen_best_failures = np.inf
            gen_best_vec = None

            for vector in solutions:
                fitness, cost, failures = self._evaluate(vector)
                values.append(-fitness)  # CMA-ES minimizes

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_cost = cost
                    gen_best_failures = failures
                    gen_best_vec = vector.copy()

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_cost = cost
                    self.best_failures = failures
                    self.best_strategy = vector.copy()

            es.tell(solutions, values)

            if gen_best_vec is not None:
                self.fitness_history.append(gen_best_fitness)
                self.cost_history.append(gen_best_cost)
                self.failure_history.append(gen_best_failures)

            if es.stop():
                break

        if self.best_strategy is None:
            self.best_strategy = self.initial_mean.copy()
            self.best_fitness = float("-inf")
            self.best_cost = float("inf")
            self.best_failures = float("inf")

        strategy = self._reshape_strategy(self.best_strategy)
        return CMAResult(
            best_strategy=strategy,
            best_fitness=self.best_fitness,
            best_cost=self.best_cost,
            best_failures=self.best_failures,
            fitness_history=self.fitness_history,
            cost_history=self.cost_history,
            failure_history=self.failure_history,
        )

    # ------------------------------------------------------------------
    def _evaluate(self, vector: np.ndarray) -> Tuple[float, float, float]:
        strategy = self._reshape_strategy(vector)
        fitnesses = []
        costs = []
        failures = []

        for _ in range(self.eval_repeats):
            fitness, cost, failure = simulate_component(strategy)
            fitnesses.append(fitness)
            costs.append(cost)
            failures.append(failure)

        return (
            float(np.mean(fitnesses)),
            float(np.mean(costs)),
            float(np.mean(failures)),
        )

    @staticmethod
    def _reshape_strategy(vector: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0)
        return clipped.reshape(TIME_STEPS, 2)


__all__ = ["CMAOptimizer", "CMAResult"]
