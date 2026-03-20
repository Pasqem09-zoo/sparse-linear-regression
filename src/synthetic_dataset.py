import numpy as np


def generate_sparse_regression_dataset(n, p, k, noise_std):
    """
    Generate a synthetic sparse linear regression dataset.

    Parameters
    n : Number of samples
    p : Number of features
    k : Number of nonzero coefficients
    noise_std : Standard deviation of the noise
    """

    #feature matrix
    X = np.random.randn(n, p)

    #sparse true coefficients
    beta_true = np.zeros(p)
    indices = np.random.choice(p, k, replace=False) #randomly select k indices to be nonzero
    beta_true[indices] = np.random.randn(k)

    #noise
    noise = noise_std * np.random.randn(n)

    #target variable
    y = X @ beta_true + noise

    return X, y, beta_true