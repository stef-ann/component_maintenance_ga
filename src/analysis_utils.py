import numpy as np
from datetime import datetime
import os
from config import (
    BETA,
    ETA,
    POPULATION_SIZE,
    NUM_GENERATIONS,
    MUTATION_RATE,
    LAMBDA_FACTOR,
    CROSSOVER_RATE,
)

def generate_maintenance_report(
    fitness_hist,
    cost_hist,
    fail_hist,
    best_strategy,
    final_mutation_rate=None,
    final_crossover_rate=None,
    max_generations=None,
):
    """Generate a detailed maintenance strategy analysis report"""
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Calculate key metrics
    final_fitness = fitness_hist[-1]
    avg_cost = np.mean(cost_hist[-20:])  # Average of last 20 generations
    avg_failures = np.mean(fail_hist[-20:])  # Average of last 20 generations
    
    if max_generations is None:
        max_generations = NUM_GENERATIONS

    # Analyze convergence
    fitness_improvement = fitness_hist[-1] - fitness_hist[0]
    convergence_gen = len(fitness_hist)
    for i in range(len(fitness_hist)-20):
        if abs(fitness_hist[i] - fitness_hist[-1]) < 0.01:
            convergence_gen = i
            break

    cost_reduction = cost_hist[0] - cost_hist[-1]
    failure_reduction = fail_hist[0] - fail_hist[-1]
    stability_index = np.std(fitness_hist[-20:])
            
    # Analyze maintenance strategy patterns
    early_life = np.mean(best_strategy[:20])
    mid_life = np.mean(best_strategy[20:70])
    late_life = np.mean(best_strategy[70:])
    phase_levels = {
        "early-life": early_life,
        "mid-life": mid_life,
        "late-life": late_life
    }
    dominant_phase = max(phase_levels, key=phase_levels.get)

    def _trend(delta, positive_word, negative_word):
        if abs(delta) < 1e-3:
            return "held steady", 0.0
        return (positive_word if delta > 0 else negative_word), abs(delta)

    cost_trend_desc, cost_delta = _trend(cost_reduction, "decreased", "increased")
    failure_trend_desc, failure_delta = _trend(failure_reduction, "decreased", "increased")

    def _format_rate(value):
        return "n/a" if value is None else f"{value:.3f}"

    if stability_index < 2:
        stability_label = "very stable"
    elif stability_index < 5:
        stability_label = "stable"
    elif stability_index < 10:
        stability_label = "moderately volatile"
    else:
        stability_label = "highly volatile"
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'results/maintenance_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write("=== Component Maintenance Strategy Analysis ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. Configuration Parameters\n")
        f.write("--------------------------\n")
        f.write(f"β (Beta): {BETA:.2f} - ")
        if BETA < 1:
            f.write("Early-life failure mode\n")
        elif BETA == 1:
            f.write("Random failure mode\n")
        else:
            f.write("Wear-out failure mode\n")
        f.write(f"η (Eta): {ETA} - Characteristic life parameter\n")
        f.write(f"Population Size: {POPULATION_SIZE}\n")
        f.write(f"Number of Generations: {NUM_GENERATIONS}\n")
        f.write(f"Mutation Rate: {MUTATION_RATE}\n")
        f.write(f"Lambda Factor: {LAMBDA_FACTOR} (reliability weight vs. cost penalties)\n\n")
        
        f.write("2. Performance Metrics\n")
        f.write("--------------------\n")
        f.write(f"Final Fitness Score: {final_fitness:.2f}\n")
        f.write(f"Average Maintenance Cost: {avg_cost:.2f}\n")
        f.write(f"Average Failure Rate: {avg_failures:.4f}\n")
        f.write(f"Total Fitness Improvement: {fitness_improvement:.2f}\n")
        f.write(f"Convergence Generation: {convergence_gen}\n\n")
        f.write(f"Cost Trend: {cost_trend_desc} by {cost_delta:.2f} (from {cost_hist[0]:.2f} to {cost_hist[-1]:.2f})\n")
        f.write(f"Failure Trend: {failure_trend_desc} by {failure_delta:.3f} (from {fail_hist[0]:.3f} to {fail_hist[-1]:.3f})\n")
        f.write(f"Fitness Stability (σ last 20 gens): {stability_index:.2f} — {stability_label}\n\n")
        
        f.write("3. Maintenance Strategy Analysis\n")
        f.write("------------------------------\n")
        f.write(f"Early-life Maintenance Level (0-20%): {early_life:.3f}\n")
        f.write(f"Mid-life Maintenance Level (20-70%): {mid_life:.3f}\n")
        f.write(f"Late-life Maintenance Level (70-100%): {late_life:.3f}\n\n")
        
        f.write("4. Recommendations\n")
        f.write("-----------------\n")
        
        # Generate recommendations based on metrics
        if avg_failures > 0.3:
            f.write("⚠️ High failure rate detected. Consider:\n")
            f.write("- Increasing maintenance intensity in high-risk periods\n")
            f.write("- Reducing intervals between maintenance actions\n")
        
        if avg_cost > 40:
            f.write("⚠️ High maintenance costs detected. Consider:\n")
            f.write("- Optimizing maintenance scheduling\n")
            f.write("- Reviewing cost-effectiveness of preventive maintenance\n")
        
        if early_life < 0.4 and BETA < 1:
            f.write("⚠️ Insufficient early-life maintenance for component with early-life failures. Consider:\n")
            f.write("- Increasing maintenance intensity in early life stages\n")
            f.write("- Implementing better break-in procedures\n")
        
        if convergence_gen > max_generations * 0.8:
            f.write("⚠️ Slow convergence detected. Consider:\n")
            f.write("- Increasing population size\n")
            f.write("- Adjusting mutation rate\n")
        
        if final_fitness < 20:
            f.write("⚠️ Low overall fitness score. Consider:\n")
            f.write("- Reviewing maintenance strategy fundamentals\n")
            f.write("- Adjusting cost-reliability trade-off parameters\n")
        
        f.write("\n5. Summary\n")
        f.write("---------\n")
        if final_fitness > 50 and avg_failures < 0.2 and avg_cost < 30:
            f.write("✅ OPTIMAL: Strategy achieves good balance of reliability and cost\n")
        elif final_fitness > 30:
            f.write("🟨 ACCEPTABLE: Strategy works but has room for improvement\n")
        else:
            f.write("❌ NEEDS IMPROVEMENT: Strategy requires significant optimization\n")

        f.write("\n6. Conclusion\n")
        f.write("------------\n")
        f.write(
            f"Reliability {failure_trend_desc} by {failure_delta:.3f} while costs {cost_trend_desc} by {cost_delta:.2f}, "
            f"indicating a {stability_label} balance between risk and spend.\n"
        )
        f.write(
            f"The strategy leans on {dominant_phase} interventions (avg intensity {phase_levels[dominant_phase]:.3f}); "
            "consider redistributing effort if other life stages show lagging health.\n"
        )
        if stability_index > 5:
            f.write("Volatility in late generations suggests tuning mutation/population settings for smoother convergence.\n")
        else:
            f.write("Convergence behavior is consistent, so further gains likely require parameter sweeps rather than GA tuning.\n")

        if final_mutation_rate is not None or final_crossover_rate is not None:
            f.write("\n7. GA Self-Optimization\n")
            f.write("---------------------\n")
            f.write(
                f"Adaptive Mutation Rate: {_format_rate(final_mutation_rate)} (base {MUTATION_RATE})\n"
            )
            f.write(
                f"Adaptive Crossover Rate: {_format_rate(final_crossover_rate)} (base {CROSSOVER_RATE})\n"
            )
            f.write(
                "Rates expand when fitness plateaus and cool once improvements resume, providing self-optimization without manual tuning.\n"
            )

    return report_path
