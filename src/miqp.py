"""
In this module, we implement the Mixed-Integer Quadratic Programming (MIQP)
formulation of the sparse linear regression problem.

The cardinality sparsity constraint is modeled using binary variables and
Big-M constraints, and the resulting problem is solved using Gurobi.

This module provides an exact baseline solution for comparison
with the IHT algorithm.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np


class MIQPSolver:

    def __init__(self, problem, k, M=None):
        """
        Parameters
        ----------
        problem : LeastSquaresProblem
            Instance containing X and y
        k : int
            Maximum number of nonzero coefficients
        M : float
            Big-M constant
        """

        self.problem = problem
        self.k = k

        self.X = problem.X
        self.y = problem.y
        self.p = problem.p

        if M is None:
            self.M = self._choose_M()
        else:
            self.M = M

        self.model = None
        self.beta = None
        self.z = None


    #idea per trovare M ottimale in modo da non tagliare certi valori di beta, andando a tagliare potenziali soluzioni ottimali
    #NOTA: funziona solo se X'X è invertibile e se n>p e se X ha rango pieno
    def _choose_M(self):
        # Ordinary Least Squares solution
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        beta_ls = np.linalg.solve(XtX, Xty) #soluzione beta che minimizza ||y - X beta||^2 senza vincoli di sparsità

        # Safety factor
        M = 2.0 * np.max(np.abs(beta_ls)) #prendo il massimo valore assoluto tra i coefficienti di beta_ls e lo moltiplico per 2

        # Degenerate safeguard
        if M == 0:
            M = 1.0

        return M


    def build_model(self):

        # Create Gurobi model
        self.model = gp.Model("SparseRegressionMIQP")

        # Create continuous variables beta_i
        self.beta = self.model.addVars(self.p, lb=-GRB.INFINITY, name="beta")

        # Create binary variables z_i
        self.z = self.model.addVars(self.p, vtype=GRB.BINARY, name="z")

        # Set objective: minimize ||y - X beta||^2
        obj = 0
        n = self.X.shape[0]

        for i in range(n): #per ogni riga i di X faccio X*beta e lo confronto con y[i]
            expr = 0
            for j in range(self.p):
                expr += self.X[i, j] * self.beta[j]
            obj += (self.y[i] - expr) * (self.y[i] - expr)

        self.model.setObjective(obj, GRB.MINIMIZE)

        # Add sparsity constraints: Big-M constraints to link beta and z
        for j in range(self.p):
            self.model.addConstr(self.beta[j] <= self.M * self.z[j], name=f"upper_{j}") #se z[j] = 0 allora beta[j] <- 0, se z[j] = 1 allora beta[j] <- (M)
            self.model.addConstr(self.beta[j] >= -self.M * self.z[j], name=f"lower_{j}") #se z[j] = 0 allora beta[j] <- 0, se z[j] = 1 allora beta[j] <- (-M)

        # Add cardinality constraint: sum of z_i <= k
        self.model.addConstr(
            gp.quicksum(self.z[j] for j in range(self.p)) <= self.k,
            name="cardinality"
        ) #quicksum è una funzione di Gurobi che somma, in questo caso somma tutti i z[j] e impone che la somma sia minore o uguale a k


    def solve(self):
        if self.model is None:
            self.build_model()

        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            print("Warning: optimal solution not found.")


    def get_solution(self):
        if self.model.status != GRB.OPTIMAL:
            return None

        beta_sol = np.zeros(self.p)
        for j in range(self.p):
            beta_sol[j] = self.beta[j].X

        return beta_sol