"""
In this module, we implement the Iterative Hard Thresholding (IHT) algorithm
for solving the cardinality-constrained sparse linear regression problem.

The algorithm performs projected gradient steps combined with
hard thresholding to enforce sparsity constraints.

This module relies on the least squares objective defined in
least_squares.py and provides an efficient approximate solver.
"""
