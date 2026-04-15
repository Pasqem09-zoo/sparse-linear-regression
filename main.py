import time
import numpy as np
from src.least_squares import LeastSquaresProblem
from src.iht import iht
from src.miqp import MIQPSolver
from src.synthetic_dataset import generate_dataset
from experiments.config import *  #import experiment parameters
from src.iht import hard_thresholding


# ----------------------------------------------------
# Run IHT algorithm
# ----------------------------------------------------
def run_iht(problem, k):

    start_time = time.time()

    first_losses = []
    final_losses = []
    runtime = []
    iter_used = []

    best_loss = float("inf")
    best_beta = None

    for r in range(IHT_N_RUNS):

        runiht_start_time = time.time() #t singola run

        ### partire con "0.01 * np.random.randn(problem.p)" come sol iniziale non è corretto (perche è densa quindi p>k). IHt funziona lo stesso perche a quella dopo la proietta sull'insieme ammissibile (quindi torni dentro) 
        ### però stai facendo un'iterazione sporca. è meglio proiettare subito e partire già con un beta ammissibile cosi ogni tierazione è coerente col metodo.
        ### la proiezione iniziale cambia il supporto iniziale, cioè quali variabili sono attive all’inizio!
        beta0 = 0.01 * np.random.randn(problem.p)
        beta0 = hard_thresholding(beta0, k)

        beta_solution, loss_history, n_iters = iht(problem, k, beta_init=beta0)

        runiht_end_time = time.time()
        runtime.append(runiht_end_time - runiht_start_time)
        iter_used.append(n_iters)

        n_iter = len(loss_history)
        first_loss = loss_history[0]
        final_loss = loss_history[-1]

        first_losses.append(first_loss)
        final_losses.append(final_loss)

        if final_loss < best_loss:
            best_loss = final_loss
            best_beta = beta_solution

    # ---- stampa tabellina ----
    # print("IHT runs:")
    # header = "run   | " + " | ".join([f"{i+1:>8}" for i in range(IHT_N_RUNS)])
    # print(header)
    # #first_row = "first | " + " | ".join([f"{l:8.2f}" for l in first_losses])
    # final_row = "final | " + " | ".join([f"{l:8.2f}" for l in final_losses])
    # time_row = "time  | " + " | ".join([f"{t:8.4f}" for t in runtime])
    # iter_row = "iter  | " + " | ".join([f"{it:8d}" for it in iter_used])
    # # print(first_row)
    # print(final_row)
    # print(time_row)
    # print(iter_row)

    end_time = time.time()
    total_runtime = end_time - start_time

    loss_value = best_loss  # migliore tra le run

    avg_loss = np.mean(final_losses)
    std_loss = np.std(final_losses)

    avg_iter = np.mean(iter_used)
    std_iter = np.std(iter_used)

    return total_runtime, loss_value, avg_loss, std_loss, avg_iter, std_iter


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

    X, y, beta_true = generate_dataset(DATASET_TYPE, n, p, k, NOISE_STD)

    # least squares problem instance
    problem = LeastSquaresProblem(X, y)

    runtime_iht, loss_iht, avg_loss_iht, std_loss_iht, avg_iter_iht, std_iter_iht = run_iht(problem, k)
    runtime_miqp, loss_miqp = run_miqp(problem, k)

    ### quando stampi una sola riga, quindi fai un solo esperimento, attiva questo print e disattiva quello nella funzione main
    print("\nexp |   p |   k | IHT_tot_time | IHT_best_loss | IHT_avg_loss ± std | IHT_avg_iter ± std | MIQP_time | MIQP_loss")
    print("-" * 130)

    #log results to wandb 
    if USE_WANDB:
        wandb.log({
            "p": p,
            "k": k,
            "runtime_iht": runtime_iht,
            "runtime_miqp": runtime_miqp,
            "loss_iht": loss_iht,
            "loss_miqp": loss_miqp,
            "avg_loss_iht": avg_loss_iht,
            "std_loss_iht": std_loss_iht,
            "avg_iter_iht": avg_iter_iht,
            "std_iter_iht": std_iter_iht
        })

    print(f"{experiment_id:3d} | {p:3d} | {k:3d} | {runtime_iht:12.4f} | {loss_iht:13.4f} | {avg_loss_iht:8.4f} ± {std_loss_iht:7.4f} | {avg_iter_iht:8.4f} ± {std_iter_iht:7.4f} | {runtime_miqp:9.4f} | {loss_miqp:9.4f}")


# ----------------------------------------------------
# Main function
# ----------------------------------------------------
def main():

    np.random.seed(RANDOM_SEED)

    if USE_WANDB:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(project=WANDB_PROJECT)
        # track "p" as the x-axis for plotting results in wandb
        wandb.define_metric("p")
        wandb.define_metric("*", step_metric="p")
        wandb.config = {
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "sparsity_ratio": SPARSITY_RATIO,
            "noise_std": NOISE_STD
        }

    n = N_SAMPLES
    p_values = N_FEATURES

    experiment_id = 0
    
    # print("\nexp |   p |   k | IHT_tot_time | IHT_best_loss | IHT_avg_loss ± std | IHT_avg_iter ± std | MIQP_time | MIQP_loss")
    # print("-" * 130)

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