# --- Tank system parameters ---
INITIAL_WATER_LEVEL = 25
TIME_STEPS = 100
LAMBDA_FACTOR = 0.7  # weight for reliability gain vs cost

# --- Performance requirement model ---
PERFORMANCE_TARGET = 0.65  # desired normalized health level
PERFORMANCE_TOLERANCE = 0.05  # slack before the requirement triggers
RELIABILITY_REQUIREMENT = 0.8  # minimum (1 - fail probability)
REQUIREMENT_PENALTY_SCALE = 12.0  # converts requirement gap into extra penalty

# --- Genetic Algorithm parameters ---
POPULATION_SIZE = 1000  # Increased for better diversity
NUM_GENERATIONS = 300  # More generations for better convergence
MUTATION_RATE = 0.02  # Reduced for more stable evolution
CROSSOVER_RATE = 0.8  # Increased to generate more diverse offspring

# --- Random seed for reproducibility ---
SEED = 42

# --- CMA-ES parameters ---
CMA_SIGMA = 0.2
CMA_POPULATION = 80  # lambda (population size)
CMA_GENERATIONS = 200
CMA_EVAL_REPEATS = 3  # Monte Carlo samples per individual

# --- Weibull Reliability Model Parameters ---
# β (shape): <1 early-life, =1 random, >1 wear-out
BETA = 1.0    # try 0.7 (early), 1.0 (random), 2.5 (wear-out)
# η (scale): characteristic life (larger η = longer component lifespan)
ETA = 50
