import os
import sys
from datetime import datetime
from pathlib import Path

if "MPLBACKEND" not in os.environ:
    # Use a non-interactive backend so the script runs cleanly in headless environments
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt

# Ensure the repository root (where `src` lives) is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.genetic_algorithm import ComponentGA
from src.analysis_utils import generate_maintenance_report

def main():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Run the genetic algorithm
    ga = ComponentGA()
    fitness_hist, cost_hist, fail_hist = ga.evolve()

    # --- Plot results ---
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_hist, label="Best Fitness (Health)")
    plt.plot(cost_hist, label="Avg Cost")
    plt.plot(fail_hist, label="Avg Failures")
    plt.title("Maintenance Optimization with Failure Dynamics")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/maintenance_plot_{timestamp}.png')
    plt.close()

    # Generate and save analysis report
    best_strategy = ga.population[0].detach().cpu().numpy()
    report_path = generate_maintenance_report(
        fitness_hist=fitness_hist,
        cost_hist=cost_hist,
        fail_hist=fail_hist,
        best_strategy=best_strategy,  # Best strategy from final population
        final_mutation_rate=ga.mutation_rate,
        final_crossover_rate=ga.crossover_rate,
    )
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to:")
    print(f"- Plot: results/maintenance_plot_{timestamp}.png")
    print(f"- Report: {report_path}")
    print(
        f"Adaptive GA parameters → mutation: {ga.mutation_rate:.3f}, crossover: {ga.crossover_rate:.2f}"
    )

if __name__ == "__main__":
    main()
