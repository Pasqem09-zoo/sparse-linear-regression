

RANDOM_SEED = 42


# --------------------------------------------------
# Dataset parameters
# --------------------------------------------------

N_SAMPLES = 50          # number of data points (n)
N_FEATURES = 20         # number of features (p)
SPARSITY_LEVEL = 3      # number of nonzero coefficients (k)

NOISE_STD = 1        # noise level in synthetic data


# --------------------------------------------------
# IHT algorithm parameters
# --------------------------------------------------

MAX_ITER = 100
TOL = 1e-6


# # --------------------------------------------------
# # MIQP solver parameters
# # --------------------------------------------------

# GUROBI_VERBOSE = False
# GUROBI_TIME_LIMIT = None


# --------------------------------------------------
# Experiment parameters
# --------------------------------------------------

USE_WANDB = False


# --------------------------------------------------
# Wandb configuration
# --------------------------------------------------

WANDB_PROJECT = "sparse-linear-regression"