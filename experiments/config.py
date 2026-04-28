

RANDOM_SEED = 42


# --------------------------------------------------
# Dataset parameters
# --------------------------------------------------

N_SAMPLES = 100          # number of data points (n) 100
N_FEATURES = [50, 100, 200, 400]         # number of features (p) [50, 100, 200, 400]
SPARSITY_RATIO = [0.02, 0.05, 0.10]      # ratio of nonzero coefficients (k) [0.02, 0.05, 0.10] 

NOISE_STD = 1        # noise level in synthetic data

DATASET_TYPE = 1


# --------------------------------------------------
# IHT algorithm parameters
# --------------------------------------------------

MAX_ITER = 100000
TOL = 1e-8
IHT_N_RUNS = 500
EPSILON = 1e-7


# # --------------------------------------------------
# # MIQP solver parameters
# # --------------------------------------------------

# GUROBI_VERBOSE = False
# GUROBI_TIME_LIMIT = None
GUROBI_OUTPUT_FLAG = 0    # 0 = silenzioso, 1 = log attivo
GUROBI_TIME_LIMIT = 180   # in secondi
MIQP_IMPROVEMENT_THRESHOLD = 0.01  # soglia per considerare un miglioramento significativo nella soluzione MIQP



# --------------------------------------------------
# Experiment parameters
# --------------------------------------------------

USE_WANDB = True # True False


# --------------------------------------------------
# Wandb configuration
# --------------------------------------------------

WANDB_PROJECT = "sparse-linear-regression"