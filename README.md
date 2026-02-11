# Sparse Linear Regression with Cardinality Constraint

This project studies the **cardinality-constrained sparse linear regression problem**, where the goal is to fit a linear model using at most *k* nonzero coefficients.

Formally, given a feature matrix \( X \in \mathbb{R}^{n \times p} \), a target vector \( y \in \mathbb{R}^n \), and a sparsity level \( k \), we solve:

\[
\min_{\beta \in \mathbb{R}^p} \|y - X\beta\|_2^2
\quad \text{s.t.} \quad \|\beta\|_0 \le k.
\]

The problem is NP-hard due to the combinatorial nature of the ℓ₀ constraint.

---

## Objectives

The project implements and compares two approaches:

- **Iterative Hard Thresholding (IHT)**  
  A first-order projected gradient method that efficiently produces locally optimal sparse solutions.

- **Mixed-Integer Quadratic Programming (MIQP)**  
  An exact reformulation solved using the Gurobi solver, providing globally optimal solutions for small to medium-sized instances.

The methods are evaluated in terms of:
- Runtime
- Final objective value
- Scalability with respect to problem size

---

## Project Structure

