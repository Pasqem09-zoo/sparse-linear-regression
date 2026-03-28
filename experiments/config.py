

RANDOM_SEED = 42


# --------------------------------------------------
# Dataset parameters
# --------------------------------------------------

N_SAMPLES = 100          # number of data points (n)
N_FEATURES = [300]         # number of features (p)
SPARSITY_RATIO = [0.05]      # ratio of nonzero coefficients (k)

NOISE_STD = 1        # noise level in synthetic data

DATASET_TYPE = 1


# --------------------------------------------------
# IHT algorithm parameters
# --------------------------------------------------

MAX_ITER = 100000
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