import time
import wandb
import numpy as np

from src.least_squares import LeastSquaresProblem
from src.iht import iht
from src.miqp import MIQPSolver
from src.synthetic_dataset import generate_dataset
from experiments.config import *
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

        runiht_start_time = time.time() #### time single run

        #### partire con "0.01 * np.random.randn(problem.p)" come sol iniziale non è corretto (perche è densa quindi p>k). IHt funziona lo stesso perche a quella dopo la proietta sull'insieme ammissibile (quindi torni dentro) 
        #### però stai facendo un'iterazione sporca. è meglio proiettare subito e partire già con un beta ammissibile cosi ogni iterazione è coerente col metodo.
        #### la proiezione iniziale cambia il supporto iniziale, cioè quali variabili sono attive all’inizio!
        beta0 = 0.5 * np.random.randn(problem.p) #### sono indeciso tra 0.1 e 0.5
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

    end_time = time.time()
    total_runtime = end_time - start_time

    loss_value = best_loss  #### migliore tra le run

    avg_loss = np.mean(final_losses)
    std_loss = np.std(final_losses)

    avg_iter = np.mean(iter_used)
    std_iter = np.std(iter_used)

    return total_runtime, loss_value, avg_loss, std_loss, avg_iter, std_iter, best_beta


# ----------------------------------------------------
# Run MIQP solver
# ----------------------------------------------------
def run_miqp(problem, k):

    solver = MIQPSolver(problem, k) 
    solver.solve() #run MIQP solver to get solution

    summary = solver.get_last_significant_solution()
    final_info = solver.get_final_info()
    beta_solution = solver.get_solution() #get solution from solver

    if beta_solution is None:
        print("MIQP solver failed to find a solution.")
        return None, None, None, None

    loss_value = problem.loss(beta_solution)

    if summary is None:
        miqp_time = None
    else:
        miqp_time = summary["time"] ### tempo dall'inizio fino a quando quella sol significativa è stata trovata
    if final_info is None:
        miqp_gap = None
        miqp_tot_time = None
    else:
        miqp_gap = final_info["gap"]
        miqp_tot_time = final_info["total_time_info"] ### tempo totale fino alla fine della risoluzione, che può essere time limit o tempo se trova la sol
    
    return miqp_time, loss_value, miqp_gap, miqp_tot_time



# ----------------------------------------------------
# log wandb
# ----------------------------------------------------
def log_experiment_to_wandb(p, k,
                            runtime_iht, loss_iht,
                            runtime_miqp, loss_miqp, gap_miqp):
    if not USE_WANDB:
        return

    wandb.log({
        "p": p,

        f"runtime_iht_k={k}": runtime_iht,
        f"runtime_miqp_k={k}": runtime_miqp if runtime_miqp is not None else np.nan,

        f"loss_iht_k={k}": loss_iht,
        f"loss_miqp_k={k}": loss_miqp if loss_miqp is not None else np.nan,

        f"gap_miqp_k={k}": gap_miqp if gap_miqp is not None else np.nan,
    })



# ----------------------------------------------------
# Run a single experiment
# ----------------------------------------------------
def run_experiment(n, p, k, experiment_id):

    X, y, beta_true = generate_dataset(DATASET_TYPE, n, p, k, NOISE_STD)

    # least squares problem instance
    problem = LeastSquaresProblem(X, y)

    runtime_iht, loss_iht, avg_loss_iht, std_loss_iht, avg_iter_iht, std_iter_iht, beta_iht = run_iht(problem, k)
    solver_debug = MIQPSolver(problem, k)
    print("\nDEBUG Big-M")
    print("p:", p, "k:", k)
    print("M:", solver_debug.M)
    print("max |beta_IHT|:", np.max(np.abs(beta_iht)))
    print("ratio max|beta_IHT| / M:", np.max(np.abs(beta_iht)) / solver_debug.M)
    runtime_miqp, loss_miqp, gap_miqp, miqp_tot_time = run_miqp(problem, k)

    # log results to wandb
    log_experiment_to_wandb(
        p=p,
        k=k,
        runtime_iht=runtime_iht,
        loss_iht=loss_iht,
        runtime_miqp=runtime_miqp,
        loss_miqp=loss_miqp,
        gap_miqp=gap_miqp
    )

    ### quando stampi una sola riga, quindi fai un solo esperimento, attiva questo print e disattiva quello nella funzione main
    # print("\nexp |   p |   k | IHT_tot_time | IHT_best_loss | IHT_avg_loss ± std | IHT_avg_iter ± std | MIQP_time | MIQP_loss | MIQP_gap (%) | MIQP_tot_time")
    # print("-" * 170)

    ### preparo la stringa perche può essere un numero o time limit
    if miqp_tot_time is None:
        total_time_str = "None"
    elif isinstance(miqp_tot_time, str):
        total_time_str = miqp_tot_time
    else:
        total_time_str = f"{miqp_tot_time:.2f}"

    print(f"{experiment_id:3d} | {p:3d} | {k:3d} | "
          f"{runtime_iht:12.4f} | {loss_iht:13.4f} | "
          f"{avg_loss_iht:8.4f} ± {std_loss_iht:7.4f} | "
          f"{avg_iter_iht:8.4f} ± {std_iter_iht:7.4f} | "
          f"{runtime_miqp:9.4f} | {loss_miqp:9.4f} | "
          f"{gap_miqp:11.4f} | {total_time_str:>13}")


# ----------------------------------------------------
# Main function
# ----------------------------------------------------
def main():

    np.random.seed(RANDOM_SEED)

    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            config={
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "sparsity_ratio": SPARSITY_RATIO,
                "noise_std": NOISE_STD,
                "dataset_type": DATASET_TYPE
            }
        )
        wandb.define_metric("p") # usa p come asse x
        wandb.define_metric("*", step_metric="p")

    n = N_SAMPLES
    p_values = N_FEATURES
    experiment_id = 0
    
    print("\nexp |   p |   k | IHT_tot_time | IHT_best_loss | IHT_avg_loss ± std | IHT_avg_iter ± std | MIQP_time | MIQP_loss | MIQP_gap (%) | MIQP_tot_time")
    print("-" * 170)

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