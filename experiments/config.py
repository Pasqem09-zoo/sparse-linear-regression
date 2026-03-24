

RANDOM_SEED = 42


# --------------------------------------------------
# Dataset parameters
# --------------------------------------------------

N_SAMPLES = 100          # number of data points (n)
N_FEATURES = [10, 20, 50, 100]         # number of features (p)
SPARSITY_RATIO = [0.1, 0.2]      # ratio of nonzero coefficients (k)

NOISE_STD = 1        # noise level in synthetic data

DATASET_TYPE = 3


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

USE_WANDB = True


# --------------------------------------------------
# Wandb configuration
# --------------------------------------------------

WANDB_PROJECT = "sparse-linear-regression"