"""
In this module, we implement the Iterative Hard Thresholding (IHT) algorithm
for solving the cardinality-constrained sparse linear regression problem.

The algorithm performs projected gradient steps combined with
hard thresholding to enforce sparsity constraints.

This module relies on the least squares objective defined in
least_squares.py and provides an efficient approximate solver.
"""


import numpy as np

import wandb
from experiments.config import USE_WANDB


def hard_thresholding(beta, k):
    """
    Keep only the k largest (in absolute value) entries of beta.
    Set the others to zero.
    """

    beta_new = beta.copy()

    #if k is larger than the number of features, keep all coefficients
    if k == 0:
        return np.zeros_like(beta_new)

    if k >= len(beta):
        return beta_new

    #### azzeri le n-k varaibili più piccole di beta in valore assoluto
    indices = np.argsort(np.abs(beta_new)) # argosrt da gli indici ordinati dal valore assoluto più piccolo al più grande
    indices_zero = indices[0:-k]
    beta_new[indices_zero] = 0

    return beta_new


def iht(problem, k, max_iter=100000, epsilon=1e-6):
    """
    Parameters
    ----------
    problem : LeastSquaresProblem
    k : (int) Maximum number of nonzero coefficients
    max_iter : (int) Maximum number of iterations
    epsilon : (float) Tolerance for stopping criterion
    """

    #initialize beta at zero
    beta = np.zeros(problem.p) #Crea un vettore beta con lunghezza uguale al numero di feature

    L = problem.lipschitz_constant()
    if L <= 0:
        raise ValueError("Lipschitz constant must be positive")
    
    loss_history = []

    for i in range(max_iter):

        loss_history.append(problem.loss(beta)) #per tenere traccia della loss ad ogni iterazione

        #for plot loss history
        if USE_WANDB:
            wandb.log({
                "iteration": i,
                "loss_iter": problem.loss(beta),
                "method": "IHT"
            })

        #step gradiente
        grad = problem.gradient(beta)
        beta_t = beta - (1.0 / L) * grad

        #hard thresholding: projection onto C_k
        beta_new = hard_thresholding(beta_t, k)

        #stopping criterion: if the change in beta is small, we can stop
        if np.linalg.norm(beta_new - beta) < epsilon:
            break

        beta = beta_new

    return beta, loss_history

