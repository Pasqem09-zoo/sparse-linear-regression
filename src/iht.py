import numpy as np

from experiments.config import MAX_ITER, EPSILON


def hard_thresholding(beta, k):
    """
    Keep only the k largest (in absolute value) entries of beta.
    Set the others to zero.
    """

    beta_new = beta.copy()

    # if k is larger than the number of features, keep all coefficients
    if k == 0:
        return np.zeros_like(beta_new)

    if k >= len(beta):
        return beta_new

    #### azzeri le n-k varaibili più piccole di beta in valore assoluto
    indices = np.argsort(np.abs(beta_new)) #### argosrt da gli indici ordinati dal valore assoluto più piccolo al più grande
    indices_zero = indices[0:-k]
    beta_new[indices_zero] = 0

    return beta_new


def iht(problem, k, beta_init=None, max_iter=MAX_ITER, epsilon=EPSILON):
    """
    solve the sparse linear regression problem using the Iterative Hard Thresholding (IHT) algorithm.
    Prameters:
    problem : LeastSquaresProblem
    k : (int) number of nonzero coefficients
    beta_init : initial guess for beta
    max_iter : (int) maximum number of iterations
    epsilon : (float) tolerance for stopping criterion
    """

    # initialize beta
    if beta_init is None: #### senza inizializzazione, partiamo da zero
        beta = np.zeros(problem.p)
    else:
        beta = beta_init.copy()

    L = problem.lipschitz_constant()
    if L <= 0:
        raise ValueError("Lipschitz constant must be positive")
    
    loss_history = []

    for i in range(max_iter): 

        loss_history.append(problem.loss(beta))

        # gradient step
        grad = problem.gradient(beta)
        beta_t = beta - (1.0 / L) * grad

        # hard thresholding: projection onto C_k
        beta_new = hard_thresholding(beta_t, k)

        #### se due iterazioni successive non cambiano molto beta, allora abbiamo raggiunto un punto di stallo e possiamo fermarci
        if np.linalg.norm(beta_new - beta) < epsilon:
            break
        beta = beta_new

    return beta, loss_history, len(loss_history)

