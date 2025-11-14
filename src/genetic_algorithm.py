import numpy as np
import torch

from config import (
    POPULATION_SIZE,
    TIME_STEPS,
    NUM_GENERATIONS,
    MUTATION_RATE,
    CROSSOVER_RATE,
    SEED,
)
from config import BETA, ETA
from src.tank_model import simulate_component
from src.utils import get_torch_device


class ComponentGA:
    def __init__(self):
        self.device = get_torch_device()
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        self.population = torch.rand(
            POPULATION_SIZE,
            TIME_STEPS,
            2,
            device=self.device,
        )
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
            fitness_values, costs, failures = [], [], []

            # --- Run simulation for each chromosome ---
            for ind in self.population:
                result = simulate_component(ind)

                # Some versions of simulate_component may return (fitness, cost, failures)
                if isinstance(result, tuple):
                    f, c, fl = result
                else:
                    f, c, fl = result, 0, 0

                fitness_values.append(f)
                costs.append(c)
                failures.append(fl)

            # --- Collect generation stats ---
            fitness_tensor = torch.tensor(fitness_values, device=self.device)
            best = float(torch.max(fitness_tensor).item())
            self.best_history.append(best)
            cost_history.append(float(np.mean(costs)))
            failure_history.append(float(np.mean(failures)))

            # --- Self-optimize GA hyperparameters ---
            self._self_optimize(best)

            # --- Selection + reproduction (same as before) ---
            ranked_indices = torch.argsort(fitness_tensor, descending=True)
            survivors = self.population.index_select(0, ranked_indices[: POPULATION_SIZE // 2])
            children = self._crossover(survivors)
            children = self._mutate(children)
            self.population = torch.cat((survivors, children), dim=0)

            print(
                f"Gen {gen+1}/{NUM_GENERATIONS} | β={BETA:.2f}, η={ETA} | "
                f"Best Fitness: {best:.2f} | Avg Cost: {np.mean(costs):.2f} | Avg Failures: {np.mean(failures):.2f} | "
                f"Mut: {self.mutation_rate:.3f} | Cross: {self.crossover_rate:.2f}"
            )


        # Return multiple histories so you can plot them later
        return self.best_history, cost_history, failure_history


    def _crossover(self, parents):
        num_parents = parents.shape[0]
        if num_parents == 0:
            return parents

        pair_indices = torch.randint(
            0,
            num_parents,
            (num_parents, 2),
            device=self.device,
        )
        offspring = parents[pair_indices[:, 0]].clone()

        crossover_mask = torch.rand(num_parents, device=self.device) < self.crossover_rate
        if TIME_STEPS <= 2:
            return offspring

        cut_points = torch.randint(
            1,
            TIME_STEPS - 1,
            (num_parents,),
            device=self.device,
        )
        crossover_indices = torch.nonzero(crossover_mask, as_tuple=False).flatten()
        for idx in crossover_indices.tolist():
            cut = int(cut_points[idx].item())
            donor = parents[pair_indices[idx, 1]]
            offspring[idx, cut:] = donor[cut:]

        return offspring

    def _mutate(self, offspring):
        if offspring.numel() == 0:
            return offspring

        noise = torch.randn_like(offspring) * 0.1
        mask = (torch.rand_like(offspring) < self.mutation_rate).to(offspring.dtype)
        mutated = torch.clamp(offspring + noise * mask, 0.0, 1.0)
        return mutated

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
