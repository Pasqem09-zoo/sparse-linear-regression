# Sparse Linear Regression with Cardinality Constraint

This project studies the **sparse linear regression problem**, where the goal is to learn a linear model that explains the data while using only a limited number of features.
In many applications, sparse models are desirable because they improve interpretability and perform automatic feature selection.

The optimization problem considered in this project imposes a **cardinality constraint** on the regression coefficients, meaning that at most \(k\) coefficients can be nonzero.
More formally, given a feature matrix $X \in \mathbb{R}^{n \times p}$, a target vector $y \in \mathbb{R}^{n}$, and a sparsity level $k$, the goal is to solve the following optimization problem:

$$
\min_{\beta \in \mathbb{R}^p} \|y - X\beta\|^2
\quad
\text{subject to}
\quad
\|\beta\|_0 \le k
$$

Here, $\beta$ represents the vector of regression coefficients. The term $\|y - X\beta\|^2$
is the squared least squares loss, while $
\|\beta\|_0
$ denotes the number of nonzero entries in $\beta$. The constraint $
\|\beta\|_0 \le k
$ enforces sparsity by allowing at most $k$ nonzero coefficients in the model.

This problem is known to be **NP-hard**, since the cardinality constraint introduces a combinatorial search over subsets of features.

To solve this problem, we implement and compare two different approaches:

- **Iterative Hard Thresholding (IHT)**, a fast heuristic algorithm based on gradient steps and hard thresholding.
- **Mixed Integer Quadratic Programming (MIQP)**, an exact formulation that can be solved using the Gurobi optimizer.

The goal of the project is to experimentally compare these two methods in terms of:

- computational runtime
- quality of the obtained solution (**objective value** at the obtained solution)

as the size of the regression problem increases (different numbers of features, different values for the cardinality threshold $k$).


<details>
<summary><strong>How to run the project?</strong></summary>

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure you have:
- Gurobi installed and properly licensed
- Weights & Biases account (logged in)

Let's run the experiments:

```bash
python main.py
```

The script will:
- Generate synthetic sparse regression datasets
- Run the IHT algorithm
- Solve the MIQP formulation using Gurobi
- Log runtime and objective values to Weights & Biases

</details>

<details>
<summary><strong>Project structure</strong></summary>

```text

.
├── src/
│   ├── least_squares.py
│   │   └── loss, gradient, Lipschitz constant
│   │
│   ├── iht.py
│   │   └── Implementation of the Iterative Hard Thresholding algorithm
│   │
│   ├── miqp.py
│   │   └── MIQP formulation of sparse regression solved with Gurobi
│   │
│   └── utils.py
│
├── experiments/
│   ├── exp.ipynb
│   │   └── Notebook for exploratory experiments and testing
│   │
│   └── config.py
│       └── Experiment configuration parameters
│
├── repo.tex
│
├── main.py
│   └── Entry point that runs experiments and results
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
```
</details>


## Implemented Algorithms

This project implements two different approaches to solve the sparse linear regression problem. The goal is to compare a fast heuristic algorithm with an exact optimization method.

### Iterative Hard Thresholding (IHT)

Iterative Hard Thresholding is an iterative algorithm designed for optimization problems with sparsity constraints. At each iteration, the algorithm performs a gradient descent step on the least squares objective and then applies a hard thresholding operator that keeps only the largest coefficients in absolute value.

More precisely, the update rule alternates between:

- a gradient step that reduces the least squares loss
- a projection step that enforces the sparsity constraint by keeping only $k$ nonzero coefficients

This approach is computationally efficient and scales well to larger problems, but it only guarantees convergence to a **local solution**.

### Mixed Integer Quadratic Programming (MIQP)

The sparse regression problem can also be reformulated as a Mixed Integer Quadratic Program. In this formulation, binary variables are introduced to indicate whether a coefficient is active or not. 

Using these binary variables together with M constraints, the cardinality constraint can be modeled explicitly. The resulting optimization problem can then be solved using the **Gurobi solver**. 
This approach provides a **globally optimal solution**, but it becomes computationally expensive as the number of features increases.




## Experiments

### Plots (wandb)

In this project we use **Weights & Biases (wandb)** to log and visualize the results of the experiments.

Each experiment corresponds to a synthetic regression problem defined by the number of features $p$ and sparsity level $k$. For every instance, we run both algorithms (IHT and MIQP) and log their performance.
The following quantities are recorded:
- **runtime**: computational time required to obtain a solution
- **loss**: value of the least squares objective at the computed solution

The main plots generated are:
- **Runtime vs number of features ($p$)**  
  This plot shows how the computational cost of each method scales with the problem size.
- **Loss vs number of features ($p$)**  
  This plot compares the quality of the solutions returned by the two methods.  

### Experiment 1 — Synthetic Sparse Regression

#### Dataset description

All datasets follow the same model:

$y = X \beta^\star + \varepsilon$

where $\beta^\star \in \mathbb{R}^p$ is $k$-sparse and $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$.

---

#### Dataset 1 — Standard Gaussian (baseline)

Features are sampled independently from a standard normal distribution:

$X \sim \mathcal{N}(0, I_p)$

This corresponds to:
- independent features
- identical variance
- well-conditioned problem

---

#### Dataset 2 — Different scales (diagonal covariance)

Each feature has its own mean and variance:

$X \sim \mathcal{N}(\mu, \Sigma)$

with:

$\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$

This implies:
- independent features
- different means
- different variances

---

#### Dataset 3 — Correlated features

Features are sampled from a multivariate normal distribution with non-diagonal covariance:

$X \sim \mathcal{N}(\mu, \Sigma)$

where:

$\Sigma_{ij} =
\begin{cases}
\sigma_i^2 & \text{if } i = j \\
\rho \, \sigma_i \sigma_j & \text{if } i \neq j
\end{cases}$

This implies:
- correlated features
- different variances

---


For each generated dataset we solve the sparse regression problem using both methods and compare their performance.
The comparison focuses on two metrics:

- **runtime**, i.e. the computational time required by the algorithm
- **objective value**, corresponding to the least squares loss at the obtained solution

By varying the number of features $p$ and the sparsity level $k$, we analyze how the two approaches behave as the problem size increases.


#### Experimental setup

- number of samples: $n = 100$
- number of features: $p \in \{10,20,50,100\}$
- sparsity ratio: $\{0.1,0.2\}$

For each pair $(p, k)$ a synthetic dataset is generated and the sparse regression problem is solved using both methods.

For each experiment, the results are printed in a compact tabular format:


Each column has the following meaning:

| Column        | Description |
|--------------|------------|
| `exp`        | Index of the experiment |
| `p`          | Number of features |
| `k`          | Sparsity level (number of nonzero coefficients) |
| `IHT_time`   | Runtime of the IHT algorithm |
| `IHT_loss`   | Objective value $\|y - X\beta\|^2$ obtained by IHT |
| `MIQP_time`  | Runtime of the MIQP solver |
| `MIQP_loss`  | Objective value $\|y - X\beta\|^2$ obtained by MIQP |

This output allows a direct comparison between the two methods in terms of both computational efficiency and solution quality.

In particular:
- the runtime highlights the scalability of the algorithms
- the objective value measures how close the solution is to the optimal least squares fit

The objective value reported in the experiments corresponds to the least squares loss:

$f(\beta) = \|y - X\beta\|_2^2$

Given a solution $\beta$ returned by an algorithm, the loss is computed by directly evaluating this objective function.

Importantly, the loss depends only on the quality of the fit and does not directly measure sparsity or the correctness of the selected variables.
The optimal value of the problem is defined as:

$f^\star = \min_{\|\beta\|_0 \le k} \|y - X\beta\|_2^2$

When the MIQP solver reaches optimality, it provides a solution that achieves (or is extremely close to) this value. For this reason, the MIQP loss can be used as a benchmark.

The IHT algorithm, on the other hand, produces an approximate solution. Its loss should therefore be interpreted relative to the MIQP result:

- if $\text{IHT\_loss} \approx \text{MIQP\_loss}$, IHT is close to optimal
- if $\text{IHT\_loss} > \text{MIQP\_loss}$, IHT is suboptimal

In larger problems, the MIQP solver may be stopped early (e.g. due to a time limit). In this case, the reported MIQP loss corresponds to the best solution found so far and may not be globally optimal.
Therefore, the comparison should be interpreted with care when MIQP does not certify optimality.

The loss evaluates the quality of the fit but not the structure of the solution. In particular, two different sparse vectors can achieve similar loss values while selecting different sets of variables.
