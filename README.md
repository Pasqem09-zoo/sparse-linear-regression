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

In this project we use **Weights & Biases (wandb)** to log and visualize the results of the experiments.

Each experiment corresponds to a synthetic regression problem defined by the number of features $p$ and sparsity level $k$. For every instance, we run both algorithms (IHT and MIQP) and log their performance.
The following quantities are recorded:
- **runtime**: computational time required to obtain a solution
- **loss**: value of the least squares objective at the computed solution

The main plots generated are:
- **Runtime vs number of features**  
  This plot shows how the computational cost of each method scales with the problem size.
- **Loss vs number of features**  
  This plot compares the quality of the solutions returned by the two methods.  

By varying the number of features $p$ and the sparsity level $k$, we analyze how the two approaches behave as the problem size increases.

---

#### Dataset description

All datasets follow the same model:

$y = X \beta^\star + \varepsilon$

where $\beta^\star \in \mathbb{R}^p$ is $k$-sparse and $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$.


#### Dataset 1 — Standard Gaussian (baseline)

Features are sampled independently from a standard normal distribution:

$X \sim \mathcal{N}(0, I_p)$

This corresponds to:
- independent features
- identical variance
- well-conditioned problem


#### Dataset 2 — Different scales (diagonal covariance)

Each feature has its own mean and variance:

$X \sim \mathcal{N}(\mu, \Sigma)$

with:

$\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$

This implies:
- independent features
- different means
- different variances


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

#### Setup

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
| `IHT_tot_time`   | Total runtime of the IHT procedure (over multiple runs) |
| `IHT_best_loss`   | Best objective value $\|y - X\beta\|^2$ obtained by IHT runs |
| `IHT_avg_loss ± std`   | measurement of robustness of IHT to random initializations |
| `IHT_avg_iter ± std`   | Mean and standard deviation of the number of iterations required for convergence across IHT runs |
| `MIQP_time`  | Runtime of the MIQP solver to reach the reported objective value |
| `MIQP_loss`  | Objective value $\|y - X\beta\|^2$ of the best feasible solution found by MIQP (last significant improvement) |
| `MIQP_gap`   | Final optimality gap at the end of the solver execution |
| `MIQP_tot_time` | Total runtime of Gurobi to certify the optimum. `time limit` indicates that optimality is not certified |



#### About LOSS
The objective value reported in the experiments corresponds to the least squares loss:

$f(\beta) = \|y - X\beta\|_2^2$

Given a solution $\beta$ returned by an algorithm, the loss is computed by directly evaluating this objective function.

The loss measures the quality of the fit, but not the structure of the solution. In particular, two different sparse vectors can achieve similar loss values while selecting different sets of variables. Importantly, the loss depends only on how well the model fits the data and does not directly measure sparsity or the correctness of the selected variables.

The optimal value of the problem is defined as:

$f^\star = \min_{\|\beta\|_0 \le k} \|y - X\beta\|_2^2$

When the MIQP solver reaches optimality, it provides a solution that achieves (or is extremely close to) this value. For this reason, the MIQP loss can be used as a benchmark.

The IHT algorithm, on the other hand, produces an approximate solution. Its loss should therefore be interpreted relative to the MIQP result:

- if $\text{IHT\_loss} \approx \text{MIQP\_loss}$, IHT is close to optimal  
- if $\text{IHT\_loss} > \text{MIQP\_loss}$, IHT is suboptimal  
- if $\text{IHT\_loss} < \text{MIQP\_loss}$, IHT is better than MIQP

The third case happens when the MIQP solver did not reach global optimality (typically, due to a time limit) and returned the best feasible solution found so far. In this case, the MIQP loss corresponds to the best solution found and it's only an upper bound on the true optimal value.



#### About IHT

The performance of IHT depends on the initialization because the problem is nonconvex.  
Different initial values of the parameter vector $\beta$ (starting points) may lead the algorithm to converge to different local minima.  
To mitigate this issue, we adopt a **multi-start strategy**.

IHT is executed multiple times (e.g., $R$ runs), each starting from a different random initialization (Gaussian initialization in our implementation).  
Formally, if $\beta^{(1)}, \dots, \beta^{(R)}$ are the solutions obtained from $R$ independent runs of IHT, we report:

$$
\text{IHT\_loss} = \min_{r=1,\dots,R} \|y - X\beta^{(r)}\|^2
$$

The corresponding runtime is the **total time required to perform all runs**.

In addition to the best solution, we also report statistics over all runs to assess the robustness and convergence behavior of the algorithm.
- The **average loss and its standard deviation** are computed as:
$$
\text{IHT\_avg\_loss} = \frac{1}{R} \sum_{r=1}^R \|y - X\beta^{(r)}\|^2
$$
This quantity captures the typical performance of IHT, while the standard deviation measures its sensitivity to the initialization.  
A small standard deviation indicates stable behavior, whereas a large one suggests the presence of multiple local minima and high variability across runs.
- The **average number of iterations and its standard deviation** are also reported:
$$
\text{IHT\_avg\_iter} = \frac{1}{R} \sum_{r=1}^R T^{(r)}
$$
where $T^{(r)}$ is the number of iterations required for convergence in the $r$-th run.  
This provides insight into the convergence speed of the algorithm and how it varies depending on the starting point.

Overall, the multi-start strategy improves the reliability of IHT by reducing the impact of poor initializations, while the additional statistics provide a more complete characterization of its performance.  
We also tested the effect of the scale of the random initialization used in the multi-start IHT procedure. Even after projecting the initial point onto the feasible set, the initialization scale still affects the final performance. In our experiments, very small initializations (e.g. scale 0.01) led to worse best-case and average performance, suggesting that the starting points remained too close to the origin and did not provide enough diversity across runs. Larger scales such as 0.1 and 1 produced significantly better results, confirming that this choice has a non-negligible practical impact.  




#### About MIQP

The MIQP solver provides an exact formulation of the sparse regression problem, but its behavior is more nuanced than a simple runtime/loss trade-off.

In particular, we distinguish between:
- the time required to find a high-quality feasible solution (`MIQP_time`)
- the total time spent by the solver (`MIQP_tot_time`), which may include additional effort to improve the optimality bound

The final optimality gap (`MIQP_gap`) measures how far the best feasible solution is from the best lower bound found by the solver. When the solver reaches optimality, the gap is zero. Otherwise, a positive gap indicates that optimality has not been certified, typically due to the time limit.

This distinction is important, as in many cases Gurobi finds good solutions quickly, but requires significantly more time to certify optimality.