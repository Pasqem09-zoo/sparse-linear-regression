import time
import numpy as np
import wandb

from src.least_squares import LeastSquaresProblem
from src.iht import iht
from src.miqp import MIQPSolver
from src.synthetic_dataset import generate_dataset
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

    #print(f"Running experiment {experiment_id}: n={n}, p={p}, k={k}")

    X, y, beta_true = generate_dataset(DATASET_TYPE, n, p, k, NOISE_STD)

    # least squares problem instance
    problem = LeastSquaresProblem(X, y)

    runtime_iht, loss_iht = run_iht(problem, k)
    # print("IHT runtime:", runtime_iht)
    # print("IHT loss:", loss_iht)

    runtime_miqp, loss_miqp = run_miqp(problem, k)
    # print("MIQP runtime:", runtime_miqp)
    # print("MIQP loss:", loss_miqp)


    #log results to wandb 
    if USE_WANDB:
        wandb.log({
            "p": p,
            "k": k,
            "runtime_iht": runtime_iht,
            "runtime_miqp": runtime_miqp,
            "loss_iht": loss_iht,
            "loss_miqp": loss_miqp
        })

    print(f"{experiment_id:3d} | {p:3d} | {k:3d} | {runtime_iht:8.4f} | {loss_iht:9.2f} | "
          f"{runtime_miqp if runtime_miqp is not None else '  None':>9} | "
          f"{loss_miqp if loss_miqp is not None else '  None':>9}")


# ----------------------------------------------------
# Main function
# ----------------------------------------------------
def main():

    np.random.seed(RANDOM_SEED)

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT)
        # track "p" as the x-axis for plotting results in wandb
        wandb.define_metric("p")
        wandb.define_metric("*", step_metric="p")
        print("\nexp |   p |   k | IHT_time | IHT_loss | MIQP_time | MIQP_loss")
        print("-" * 75)
        
        #
        wandb.config = {
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "sparsity_ratio": SPARSITY_RATIO,
            "noise_std": NOISE_STD
        }

    n = N_SAMPLES
    p_values = N_FEATURES

    experiment_id = 0

    for p in p_values:
        for ratio in SPARSITY_RATIO:

            k = int(ratio * p)

            experiment_id += 1
            run_experiment(n, p, k, experiment_id)


# ----------------------------------------------------
# Run the program
# ----------------------------------------------------
if __name__ == "__main__":
    main()