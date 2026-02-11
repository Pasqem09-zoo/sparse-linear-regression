# Sparse Linear Regression with Cardinality Constraint

This project studies the **cardinality-constrained sparse linear regression problem**, where the goal is to fit a linear model using at most *k* nonzero coefficients.

Formally, given:
- Feature matrix X ∈ R^(n × p)
- Target vector y ∈ R^n
- Sparsity level k

we solve:

min_{β ∈ R^p} ||y − Xβ||²  
subject to ||β||₀ ≤ k

The problem is NP-hard due to the combinatorial nature of the ℓ₀ constraint.

---

## Objectives

The project implements and compares two approaches:

- **Iterative Hard Thresholding (IHT)**  
  A first-order projected gradient method that produces locally optimal sparse solutions.

- **Mixed-Integer Quadratic Programming (MIQP)**  
  An exact reformulation solved using the Gurobi solver, providing globally optimal solutions for small instances.

The methods are evaluated in terms of:
- Runtime
- Final objective value
- Scalability with respect to problem size

---

## Project Structure

- `src/` – Core algorithm implementations  
- `experiments/` – Jupyter notebooks for experiments  
- `report/` – LaTeX report
