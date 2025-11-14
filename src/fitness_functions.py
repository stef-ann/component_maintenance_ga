def failure_aware_fitness(strategy):
    from src.tank_model import simulate_component
    result = simulate_component(strategy)
    return result  # now includes failure penalties internally
