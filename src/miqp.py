"""
In this module, we implement the Mixed-Integer Quadratic Programming (MIQP)
formulation of the sparse linear regression problem.

The cardinality constraint is modeled using binary variables and
Big-M constraints, and the resulting problem is solved using Gurobi.

This module provides an exact baseline solution for comparison
with the IHT algorithm.
"""
