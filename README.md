# Sparse Linear Regression with Cardinality Constraint

This project studies the **sparse linear regression problem**, where the goal is to learn a linear model using only a limited number of features.

We consider the following optimization problem:

$$
\min_{\beta \in \mathbb{R}^p} \|y - X\beta\|^2
\quad
\text{subject to}
\quad
\|\beta\|_0 \le k
$$

This problem is **NP-hard** due to the combinatorial nature of the sparsity constraint.




## ⚙️ Methods

We compare two approaches:

- **Iterative Hard Thresholding (IHT)**  
  A fast heuristic based on gradient steps and hard thresholding.

- **Mixed Integer Quadratic Programming (MIQP)**  
  An exact formulation solved using Gurobi.


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
│   │   
│   ├── iht.py  
│   │
│   └── miqp.py
│   
│   
├── experiments/
│   └── config.py
│
├── main.py
│
├── requirements.txt  
│
└── README.md
```
</details>




## 📊Experiments

In this project we use **Weights & Biases (wandb)** to log and visualize the results of the experiments.

Experiments are conducted on synthetic datasets generated according to:

$
y = X \beta^\star + \varepsilon
$

where $\beta^\star$ is a sparse vector and $\varepsilon$ is Gaussian noise.

We consider different settings by varying:

- the number of features $p$
- the sparsity ratio $k/p$
- the structure of the dataset




## 📈 Datasets

All datasets follow the same model:

$y = X \beta^\star + \varepsilon$

where $\beta^\star \in \mathbb{R}^p$ is $k$-sparse and $\varepsilon \sim \mathcal{N}(0,1)$.


#### Dataset 1 — Standard Gaussian (baseline)

Features are sampled independently from a standard normal distribution:

$X \sim \mathcal{N}(0, I_p)$

This corresponds to:
- identical variance
- well-conditioned problem


#### Dataset 2 — Different scales

$X \sim \mathcal{N}(\mu, \Sigma)$

where:

$\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$

This implies:
- different means
- different variances


#### Dataset 3 — Correlated features

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



## 🧪 Experimental Setup

- number of samples: $n = 100$
- number of features: $p \in \{50, 100, 200, 400\}$
- sparsity ratio: $\{0.02, 0.05, 0.10\}$
- IHT multi-start runs: $R = 500$
- MIQP time limit: 180 seconds
- 

For each pair $(p, k/p)$ a synthetic dataset is generated and the sparse regression problem is solved using both methods.

For each experiment, we report:

- **IHT\_tot\_time**: total runtime across all IHT runs  
- **IHT\_best\_loss**: best loss obtained over multiple runs  
- **IHT\_avg\_loss ± std**: average and standard deviation of the loss  
- **IHT\_avg\_iter ± std**: statistics on convergence iterations  

- **MIQP\_time**: time to reach the best reported feasible solution  
- **MIQP\_loss**: objective value of the best feasible solution  
- **MIQP\_gap**: final optimality gap  
- **MIQP\_tot\_time**: total solver runtime (or `time limit` if not optimal)  



## 📌 Conclusions

The experimental results reveal a clear trade-off between scalability and optimality.

- **IHT** scales efficiently to high-dimensional settings and often achieves competitive solutions, but lacks guarantees of global optimality.
- **MIQP**, on the other hand, provides exact solutions when the optimality gap is zero, but becomes computationally expensive as the problem size increases.

These findings suggest that IHT is a practical choice for large-scale problems, while MIQP is better suited for smaller instances where optimality certification is required.
