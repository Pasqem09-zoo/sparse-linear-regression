import time
import numpy as np
import wandb

from src.least_squares import LeastSquaresProblem
from src.iht import iht
from src.miqp import MIQPSolver
from src.synthetic_dataset import generate_sparse_regression_dataset
from experiments.config import *  #import experiment parameters


# ----------------------------------------------------
# Run IHT algorithm
# ----------------------------------------------------
def run_iht(problem, k):

    start_time = time.time() #start timer to measure runtime

    beta_solution, loss_history = iht(problem, k) #run IHT algorithm to get solution and loss history

    end_time = time.time() #end timer

    runtime = end_time - start_time

    loss_value = problem.loss(beta_solution) #final loss

    return runtime, loss_value


# ----------------------------------------------------
# Run MIQP solver
# ----------------------------------------------------
def run_miqp(problem, k):

    start_time = time.time() #start timer to measure runtime

    solver = MIQPSolver(problem, k) 
    solver.solve() #run MIQP solver to get solution

    beta_solution = solver.get_solution() #get solution from solver

    if beta_solution is None:
        print("MIQP solver failed to find a solution.")
        return None, None

    end_time = time.time()

    runtime = end_time - start_time

    loss_value = problem.loss(beta_solution)

    return runtime, loss_value


# ----------------------------------------------------
# Run a single experiment
# ----------------------------------------------------
def run_experiment(n, p, k, experiment_id):

    print("\nRunning experiment with p =", p, "and k =", k)

    X, y, beta_true = generate_sparse_regression_dataset(n, p, k, NOISE_STD)

    # least squares problem instance
    problem = LeastSquaresProblem(X, y)

    # iht instance
    runtime_iht, loss_iht = run_iht(problem, k)

    print("IHT runtime:", runtime_iht)
    print("IHT loss:", loss_iht)

    if USE_WANDB:
        wandb.log({
            "experiment_id": experiment_id,
            "method": "IHT",
            "p": p,
            "k": k,
            "runtime": runtime_iht,
            "loss": loss_iht
        })

    # miqp instance
    runtime_miqp, loss_miqp = run_miqp(problem, k)

    print("MIQP runtime:", runtime_miqp)
    print("MIQP loss:", loss_miqp)

    if USE_WANDB:
        wandb.log({
            "experiment_id": experiment_id,
            "method": "MIQP",
            "p": p,
            "k": k,
            "runtime": runtime_miqp,
            "loss": loss_miqp
        })


# ----------------------------------------------------
# Main function
# ----------------------------------------------------
def main():

    np.random.seed(RANDOM_SEED)

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT)

        wandb.config = {
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "sparsity_level": SPARSITY_LEVEL,
            "noise_std": NOISE_STD
        }

    n = N_SAMPLES
    p = N_FEATURES
    k = SPARSITY_LEVEL
    experiment_id = 1

    run_experiment(n, p, k, experiment_id)


# ----------------------------------------------------
# Run the program
# ----------------------------------------------------
if __name__ == "__main__":
    main()