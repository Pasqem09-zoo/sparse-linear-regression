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
- quality of the obtained solution

as the size of the regression problem increases.


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

### Experiment 1 — Synthetic Sparse Regression

In this experiment we compare the performance of the IHT algorithm and the MIQP formulation on synthetically generated regression problems.

Synthetic datasets are generated in order to control the sparsity structure of the true regression coefficients. In particular, we generate:

- a feature matrix $X \in \mathbb{R}^{n \times p}$ with entries sampled from a standard normal distribution
- a sparse vector of true coefficients $\beta_{\text{true}}$ with at most $k$ nonzero entries
- a target vector

$
y = X \beta_{\text{true}} + \varepsilon
$

where $\varepsilon$ is Gaussian noise with sd = .....

For each generated dataset we solve the sparse regression problem using both methods and compare their performance.
The comparison focuses on two metrics:

- **runtime**
- **objective value**

By varying the number of features $p$ and the sparsity level $k$, we analyze how the two approaches behave as the problem size increases.


#### Experimental setup

In this experiment we generate synthetic regression datasets with a fixed number of samples and varying number of features.

The main parameters of the experiment are:

- number of samples: $n = 100$
- number of features: $p \in \{50, 100, 200, 400\}$
- sparsity level: $k \in \{5, 10\}$

For each pair $(p, k)$ a synthetic dataset is generated and the sparse regression problem is solved using both methods:

- Iterative Hard Thresholding (IHT)
- Mixed Integer Quadratic Programming (MIQP)

For every run we measure the following metrics:

- **runtime**, i.e. the computational time required by the algorithm
- **objective value**, corresponding to the least squares loss at the obtained solution

All experiment results are logged using **Weights & Biases (wandb)**, which allows us to easily compare the performance of the two approaches.