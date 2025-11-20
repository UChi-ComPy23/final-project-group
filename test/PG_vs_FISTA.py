import numpy as np
import matplotlib.pyplot as plt
import time

#Lasso Problem Definition
class LassoProblem:
    def __init__(self, A, b, lam):
        self.A = A
        self.b = b
        self.lam = lam

    def f(self, x):
        return 0.5*np.linalg.norm(self.A.dot(x)-self.b)**2 + self.lam*np.linalg.norm(x,1)

    def grad(self, x):
        return self.A.T.dot(self.A.dot(x) - self.b)

    def prox_g(self, x, alpha):
        return np.sign(x) * np.maximum(np.abs(x) - self.lam * alpha, 0)


#Generate Lasso instance
np.random.seed(0)
m, n = 200, 1000
A = np.random.randn(m,n) / np.sqrt(m)

x_true = np.zeros(n)
x_true[np.random.choice(n, 20, replace=False)] = np.random.randn(20)

b = A.dot(x_true) + 0.01*np.random.randn(m)
lam = 0.1

# Estimate Lipschitz constant L ≈ ||A^T A||
v = np.random.randn(n)
for _ in range(30):
    v = A.T.dot(A.dot(v))
    v /= np.linalg.norm(v)
L = np.linalg.norm(A.dot(v))**2

problem = LassoProblem(A, b, lam)
x0 = np.zeros(n)


#Import PG & FISTA class
from src.Method.prox_gradient import ProxGradient
from src.Method.fista import FISTA


#Run PG
pg_solver = ProxGradient(problem, x0, alpha=1.0/L)

t0 = time.time()
for _ in range(800):
    pg_solver.step()
time_pg = time.time() - t0

x_pg = pg_solver.x
obj_pg = np.array(pg_solver.objs)


#Run FISTA
fista_solver = FISTA(problem, x0, alpha=1.0/L)

t0 = time.time()
for _ in range(800):
    fista_solver.step()
time_fista = time.time() - t0

x_fista = fista_solver.x
obj_fista = np.array(fista_solver.objs)


#Plot
plt.figure(figsize=(7,4))
best_obj = min(obj_pg[-1], obj_fista[-1])
plt.semilogy(obj_pg - best_obj, label="PG")
plt.semilogy(obj_fista - best_obj, label="FISTA")
plt.xlabel("Iteration")
plt.ylabel("Objective - best")
plt.legend()
plt.title("Lasso: PG vs FISTA (class version)")
plt.tight_layout()
plt.show()


print("PG time:   ", time_pg)
print("FISTA time:", time_fista)
