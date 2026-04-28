"""
In this module, we implement the Mixed-Integer Quadratic Programming (MIQP)
formulation of the sparse linear regression problem.

The cardinality sparsity constraint is modeled using binary variables and
M constraints, and the resulting problem is solved using Gurobi.

This module provides an exact baseline solution for comparison
with the IHT algorithm.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from experiments.config import GUROBI_OUTPUT_FLAG, GUROBI_TIME_LIMIT, MIQP_IMPROVEMENT_THRESHOLD


class MIQPSolver:

    def __init__(self, problem, k, M=None):

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
        self.history = [] ### salviamo la storia di gurobi


    #### idea: per trovare M ottimale in modo da non tagliare certi valori di beta, andando a tagliare potenziali soluzioni ottimali
    #### qui costruiamo 
    #NOTA: funziona anche se X'X è non-invertibile e se n>p e se X ha rango pieno
    def _choose_M(self):
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        lambda_reg = 1e-6
        #### aggiunta di termine di regolarizzazione per garantire invertibilità di X'X. 
        XtX_reg = XtX + lambda_reg * np.eye(self.p)  # X'X + lambda * I(pxp)
        
        beta_ls = np.linalg.solve(XtX_reg, Xty) #### soluzione beta che minimizza ||y - X beta||^2 senza vincoli di sparsità

        # Safety factor
        M = 2.0 * np.max(np.abs(beta_ls)) #### prendo il massimo valore assoluto tra i coefficienti di beta_ls e lo moltiplico per 2
        if M == 0: # safeguard
            M = 1.0

        return M
    
    #### TODO: M=1 POTREBBE MIGLIORARE I RISULTATI????



    # costruisce il modello MIQP
    def build_model(self):


        # gurobipy.Model è la classe che rappresenta un modello di ottimizzazione
        self.model = gp.Model("SparseRegressionMIQP")
        self.model.setParam('OutputFlag', GUROBI_OUTPUT_FLAG)
        self.model.setParam('TimeLimit', GUROBI_TIME_LIMIT)

        self.beta = self.model.addVars(self.p, lb=-GRB.INFINITY, name="beta")
        self.z = self.model.addVars(self.p, vtype=GRB.BINARY, name="z") #binary variables z_i

        #objective: minimize ||y - X beta||^2
        obj = 0
        n = self.X.shape[0]
        for i in range(n): #per ogni riga i di X faccio X*beta e lo confronto con y[i]
            expr = 0
            for j in range(self.p):
                expr += self.X[i, j] * self.beta[j]
            obj += (self.y[i] - expr) * (self.y[i] - expr)

        self.model.setObjective(obj, GRB.MINIMIZE) #GRB.MINIMIZE indica che vogliamo minimizzare l'obiettivo


        #sparsity constraints M to link beta and z
        for j in range(self.p):
            self.model.addConstr(self.beta[j] <= self.M * self.z[j], name=f"upper_{j}") #se z[j] = 0 allora beta[j] <- 0, se z[j] = 1 allora beta[j] <- (M)
            self.model.addConstr(self.beta[j] >= -self.M * self.z[j], name=f"lower_{j}") #se z[j] = 0 allora beta[j] <- 0, se z[j] = 1 allora beta[j] <- (-M)


        #cardinality constraint: sum of z_i <= k
        self.model.addConstr(
            gp.quicksum(self.z[j] for j in range(self.p)) <= self.k,
            name="cardinality"
        ) #quicksum è una funzione di Gurobi che somma, in questo caso somma tutti i z[j] e impone che la somma sia minore o uguale a k


    ### callback che salva ogni nuova soluzione migliore trovata da Gurobi durante il processo di ottimizzazione
    def _callback(self, model, where):

        if where == GRB.Callback.MIPSOL:  ### entra qui solo quando gurobi trova una nuova soluzione ammissibile migliore di quella precedente (quindi aggiorna l'UB)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)        # best feasible solution found so far (UB)
            bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)   # best bound available at that moment (LB)
            runtime = model.cbGet(GRB.Callback.RUNTIME)       # elapsed solver time

            if obj > 1e-12:  ### per evitare di dividere per zero
                gap = 100.0 * abs(obj - bound) / abs(obj)
            else:
                gap = 0.0

            ### diz per tenere traccia delle callback
            record = {
                "time": round(runtime, 4),
                "obj": round(obj, 4),
                "bound": round(bound, 4),
                "gap": round(gap, 2)
            }

            if len(self.history) == 0:  ### se è la prima soluzione trovata, inizializza la storia con questa soluzione
                self.history.append(record)
            else:
                last_obj = self.history[-1]["obj"]
                if last_obj - obj > MIQP_IMPROVEMENT_THRESHOLD:  ### se la nuova soluzione è significativamente migliore di quella precedente, aggiungila alla storia
                    self.history.append(record)



    def solve(self):
        if self.model is None:
            self.build_model()
        self.model.optimize(self._callback)  ###passo la callback a gurobi in modo che venga chiamata ogni volta che gurobi trova una nuova soluzione ammissibile migliore di quella precedente


        #check if an optimal solution was found
        if self.model.status != GRB.OPTIMAL:
            print("Status:", self.model.status)


    ### ultima sol significativamente diversa dalla precedente
    def get_last_significant_solution(self):        
        if len(self.history) == 0:
            return None

        return self.history[-1]
    

    ### ultimo gap fatto da gurobi, che verosimilmente è associato a loss circa uguale a quella in get summary
    ### MIPGap è un attributo di gurobi tra 0 e 1 che indica il gap tra UB e il miglior bound al momento LB

    def get_final_info(self):
        if self.model is None:
            return None

        info = {
            "gap": round(self.model.MIPGap * 100.0, 2),
            "runtime": round(self.model.Runtime, 2),
            "status": self.model.Status
        }

        if self.model.Status == GRB.TIME_LIMIT:
            info["total_time_info"] = "time limit"
        else:
            info["total_time_info"] = round(self.model.Runtime, 2)

        return info


    # confronta soluzioni (UB) e limiti inferiori (LB)_ gurobi ottiene un LB con z in [0,1] e un UB con z in {0,1}. poi divide
    # in sottoproblemi per cercare qualcosa di meglio e aggiorna LB e UB fino a trovare la soluzione ottimale cioè quando LB=UB, lo stato ottimale. 
    def get_solution(self):
        # check if at least one feasible solution exists
        if self.model.SolCount == 0:
            print("No feasible solution found.")
            return None

        # check optimality
        if self.model.status != GRB.OPTIMAL: #status: attributo di gurobi, risultato logico basato su UB e LB; GRB.OPTIMAL indica la soluzione ottimale
            print("Non optimal solution, using best found.")

        beta_sol = np.zeros(self.p) 
        for j in range(self.p):
            beta_sol[j] = self.beta[j].X #.X è l'attributo di Gurobi che contiene il valore ottimale della variabile beta[j]

        return beta_sol