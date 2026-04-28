"""
In this module, we define the least squares objective function
for linear regression and implement its main components:
- loss function
- gradient
- Lipschitz constant of the gradient

This module provides the continuous optimization core
used by both IHT and MIQP approaches.
"""

import numpy as np


class LeastSquaresProblem:
    """
    This class represents a least squares linear regression problem.
    We want to minimize:
        min(beta) ||y - X beta||^2
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

        # save dimensions
        self.n = X.shape[0]   # number of samples
        self.p = X.shape[1]   # number of features



    def loss(self, beta):
        """
        Compute the least squares loss function value at beta.
        """
        prediction = self.X @ beta

        residual = self.y - prediction

        # compute squared L2 norm of the residual: ||y - X beta||^2
        value = np.linalg.norm(residual) ** 2

        return value
    


    def gradient(self, beta):
        """
        Compute the gradient of the least squares loss function at beta.
        """
        prediction = self.X @ beta

        residual = self.y - prediction

        # gradient in beta: -2 X^T (y - X beta)
        grad = -2 * self.X.T @ residual

        return grad
    
    
    def lipschitz_constant(self):
        """
        Compute the Lipschitz constant of the gradient of the least squares loss function.
        The Lipschitz constant L is given by:
            L = 2 * lambda_max(X^T X)
        where lambda_max is the largest eigenvalue of X^T X.
        """

        XtX = self.X.T @ self.X
        eigenvalues = np.linalg.eig(XtX)[0]

        # take the largest eigenvalue (real part for numerical stability)
        lambda_max = np.max(eigenvalues.real)

        # Lipschitz constant
        L = 2 * lambda_max 

        return L

    



