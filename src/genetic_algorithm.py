import numpy as np
from src.tank_model import simulate_component
from config import (
    POPULATION_SIZE,
    TIME_STEPS,
    NUM_GENERATIONS,
    MUTATION_RATE,
    CROSSOVER_RATE,
)
from config import BETA, ETA


class ComponentGA:
    def __init__(self):
        self.population = np.random.rand(POPULATION_SIZE, TIME_STEPS, 2)
        self.best_history = []
        self.best_so_far = -np.inf
        self.stagnant_generations = 0
        self.base_mutation_rate = MUTATION_RATE
        self.base_crossover_rate = CROSSOVER_RATE
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE

    def evolve(self):
        self.best_history = []
        cost_history = []
        failure_history = []

        for gen in range(NUM_GENERATIONS):
            fitness, costs, failures = [], [], []

            # --- Run simulation for each chromosome ---
            for ind in self.population:
                result = simulate_component(ind)

                # Some versions of simulate_component may return (fitness, cost, failures)
                if isinstance(result, tuple):
                    f, c, fl = result
                else:
                    f, c, fl = result, 0, 0

                fitness.append(f)
                costs.append(c)
                failures.append(fl)

            # --- Collect generation stats ---
            fitness = np.array(fitness)
            best = np.max(fitness)
            self.best_history.append(best)
            cost_history.append(np.mean(costs))
            failure_history.append(np.mean(failures))

            # --- Self-optimize GA hyperparameters ---
            self._self_optimize(best)

            # --- Selection + reproduction (same as before) ---
            survivors = self.population[np.argsort(fitness)[::-1][:POPULATION_SIZE // 2]]
            children = self._crossover(survivors)
            children = self._mutate(children)
            self.population = np.vstack((survivors, children))

            print(f"Gen {gen+1}/{NUM_GENERATIONS} | β={BETA:.2f}, η={ETA} | "
                f"Best Fitness: {best:.2f} | Avg Cost: {np.mean(costs):.2f} | Avg Failures: {np.mean(failures):.2f} | "
                f"Mut: {self.mutation_rate:.3f} | Cross: {self.crossover_rate:.2f}")


        # Return multiple histories so you can plot them later
        return self.best_history, cost_history, failure_history


    def _crossover(self, parents):
        offspring = []
        for _ in range(len(parents)):
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            if np.random.rand() < self.crossover_rate:
                cut = np.random.randint(1, TIME_STEPS - 1)
                child = np.vstack((p1[:cut], p2[cut:]))
            else:
                child = p1.copy()
            offspring.append(child)
        return np.array(offspring)

    def _mutate(self, offspring):
        mask = np.random.rand(*offspring.shape) < self.mutation_rate
        offspring[mask] += np.random.randn(*offspring[mask].shape) * 0.1
        np.clip(offspring, 0, 1, out=offspring)
        return offspring

    def _self_optimize(self, current_best):
        improvement_threshold = 1e-3
        if current_best > self.best_so_far + improvement_threshold:
            self.best_so_far = current_best
            self.stagnant_generations = 0
            # Cool mutation/crossover back toward their baselines
            self.mutation_rate = max(
                self.base_mutation_rate,
                self.mutation_rate * 0.85,
            )
            self.crossover_rate = max(
                self.base_crossover_rate,
                self.crossover_rate * 0.95,
            )
            return

        self.stagnant_generations += 1
        pressure = min(self.stagnant_generations / 10.0, 2.0)
        self.mutation_rate = np.clip(
            self.base_mutation_rate * (1 + 0.5 * pressure),
            self.base_mutation_rate,
            0.35,
        )
        self.crossover_rate = np.clip(
            self.base_crossover_rate + 0.15 * pressure,
            0.4,
            0.95,
        )
