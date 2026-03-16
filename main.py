import time
import numpy as np
import wandb

from src.least_squares import LeastSquaresProblem
from src.iht import iht
from src.miqp import MIQPSolver


# ----------------------------------------------------
# Generate a synthetic sparse regression dataset
# ----------------------------------------------------
def generate_dataset(n, p, k_true):

    # Generate random feature matrix
    X = np.random.randn(n, p)

    # Create sparse true beta
    beta_true = np.zeros(p)

    # Choose k_true random indices
    indices = np.random.choice(p, k_true, replace=False)

    # Assign random values to those indices
    beta_true[indices] = np.random.randn(k_true)

    # Generate small noise
    noise = 0.01 * np.random.randn(n)

    # Generate target vector
    y = X @ beta_true + noise

    return X, y, beta_true


# ----------------------------------------------------
# Run IHT algorithm
# ----------------------------------------------------
def run_iht(problem, k):

    # Start timer
    start_time = time.time()

    # Run algorithm
    beta_solution, loss_history = iht(problem, k)

    # Stop timer
    end_time = time.time()

    runtime = end_time - start_time

    # Compute final loss
    loss_value = problem.loss(beta_solution)

    return runtime, loss_value


# ----------------------------------------------------
# Run MIQP solver
# ----------------------------------------------------
def run_miqp(problem, k):

    # Start timer
    start_time = time.time()

    # Create solver
    solver = MIQPSolver(problem, k)

    # Solve optimization problem
    solver.solve()

    # Get solution
    beta_solution = solver.get_solution()

    # Stop timer
    end_time = time.time()

    runtime = end_time - start_time

    # Compute final loss
    loss_value = problem.loss(beta_solution)

    return runtime, loss_value


# ----------------------------------------------------
# Run a single experiment
# ----------------------------------------------------
def run_experiment(n, p, k):

    print("Running experiment with p =", p, "and k =", k)

    # Generate dataset
    X, y, beta_true = generate_dataset(n, p, k)

    # Create optimization problem
    problem = LeastSquaresProblem(X, y)

    # -----------------------
    # Run IHT
    # -----------------------
    runtime_iht, loss_iht = run_iht(problem, k)

    print("IHT runtime:", runtime_iht)
    print("IHT loss:", loss_iht)

    # Log results to wandb
    wandb.log({
        "method": "IHT",
        "p": p,
        "k": k,
        "runtime": runtime_iht,
        "loss": loss_iht
    })

    # -----------------------
    # Run MIQP
    # -----------------------
    runtime_miqp, loss_miqp = run_miqp(problem, k)

    print("MIQP runtime:", runtime_miqp)
    print("MIQP loss:", loss_miqp)

    # Log results to wandb
    wandb.log({
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

    # Initialize wandb
    wandb.init(project="sparse-linear-regression")

    # Number of samples
    n = 100

    # Different numbers of features
    p_values = [50, 100, 200, 400]

    # Different sparsity levels
    k_values = [5, 10]

    # Run all experiments
    for p in p_values:

        for k in k_values:

            run_experiment(n, p, k)


# ----------------------------------------------------
# Run the program
# ----------------------------------------------------
if __name__ == "__main__":
    main()