"""Run CMA-ES optimization on the maintenance simulator."""

import os
from datetime import datetime

import matplotlib.pyplot as plt

from src.cma_optimizer import CMAOptimizer
from src.analysis_utils import generate_maintenance_report


def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    optimizer = CMAOptimizer()
    result = optimizer.run()

    fitness_hist = result.fitness_history or [result.best_fitness]
    cost_hist = result.cost_history or [result.best_cost]
    fail_hist = result.failure_history or [result.best_failures]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(8, 5))
    plt.plot(fitness_hist, label="Best Fitness (CMA)")
    plt.plot(cost_hist, label="Avg Cost (CMA)")
    plt.plot(fail_hist, label="Avg Failures (CMA)")
    plt.title("CMA-ES Maintenance Optimization")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plot_path = f"results/maintenance_cma_plot_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()

    report_path = generate_maintenance_report(
        fitness_hist=fitness_hist,
        cost_hist=cost_hist,
        fail_hist=fail_hist,
        best_strategy=result.best_strategy,
        max_generations=optimizer.generations,
    )

    print("CMA-ES optimization finished.")
    print(f"Best fitness: {result.best_fitness:.2f} | cost: {result.best_cost:.2f} | failures: {result.best_failures:.2f}")
    print(f"Plot saved to: {plot_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
