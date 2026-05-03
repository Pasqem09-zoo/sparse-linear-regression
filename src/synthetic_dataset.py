import numpy as np

"""
    Parameters
    n : Number of samples
    p : Number of features
    k : Number of nonzero coefficients
    noise_std : Standard deviation of the noise
"""



# --------- dataset 1: standard Gaussian (baseline)
def generate_dataset_1(n, p, k, noise_std):
    """
    X ~ N(0, I) uncorrelated features with same scale
    """

    #feature matrix (same scale for all features)
    X = np.random.randn(n, p)

    #sparse true coefficients
    beta_true = np.zeros(p)
    indices = np.random.choice(p, k, replace=False) #randomly select k indices to be nonzero
    beta_true[indices] = np.random.randn(k)

    noise = noise_std * np.random.randn(n)
    y = X @ beta_true + noise

    return X, y, beta_true


# ----------- Dataset 2: different variances for each feature
def generate_dataset_2(n, p, k, noise_std):

    """
    - different means
    - different variances
    - uncorrelated features
    """

    # mean vector (mu)
    mu = np.random.randn(p)

    # variances (small range)
    variances = np.random.uniform(0.5, 1.5, size=p)

    # covariance matrix (Sigma)
    Sigma = np.diag(variances) # uncorrelated features => all zeros out of diagonal

    # sample X from multivariate normal
    X = np.random.multivariate_normal(mu, Sigma, size=n)

    # sparse true coefficients
    beta_true = np.zeros(p)
    indices = np.random.choice(p, k, replace=False) #randomly select k indices to be nonzero
    beta_true[indices] = np.random.randn(k)

    noise = noise_std * np.random.randn(n)
    y = X @ beta_true + noise

    return X, y, beta_true




# ----------- dataset 3: correlated features
def generate_dataset_3(n, p, k, noise_std):
    
    """
    - different means
    - different variances
    - correlated features
    """

    # mean vector (mu)
    mu = np.random.randn(p)

    # variances (small range)
    variances = np.random.uniform(0.5, 1.5, size=p)
    std_devs = np.sqrt(variances)

    # random matrix used to build a valid covariance structure
    A = np.random.randn(p, p) #values from standard normal distribution
    C = A @ A.T   #positive semidefinite matrix

    # convert C to correlation matrix
    diag_C = np.sqrt(np.diag(C))
    Corr = C / np.outer(diag_C, diag_C)

    # covariance matrix (Sigma)
    Sigma = np.outer(std_devs, std_devs) * Corr
    X = np.random.multivariate_normal(mu, Sigma, size=n)

    # sparse true coefficients
    beta_true = np.zeros(p)
    indices = np.random.choice(p, k, replace=False) #randomly select k indices to be nonzero
    beta_true[indices] = np.random.randn(k)

    noise = noise_std * np.random.randn(n)
    y = X @ beta_true + noise

    return X, y, beta_true


# ----------------------------------------------------
# General wrapper to select dataset type
# ----------------------------------------------------
def generate_dataset(dataset_type, n, p, k, noise_std):

    if dataset_type == 1:
        return generate_dataset_1(n, p, k, noise_std)

    elif dataset_type == 2:
        return generate_dataset_2(n, p, k, noise_std)

    elif dataset_type == 3:
        return generate_dataset_3(n, p, k, noise_std)

    else:
        raise ValueError("Invalid dataset type")