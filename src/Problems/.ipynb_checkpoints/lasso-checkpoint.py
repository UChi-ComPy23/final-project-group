from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1
import numpy as np

"""
Concrete problem.
LASSO: f(x) = 1/2 ||A x - y||^2, g(x) = lambda * ||x||_1
"""

class LassoProblem(ProblemBase):
    """
    Unified LASSO problem that supports both
    - Prox-gradient / FISTA (smooth + L1)
    - ADMM / ADLPMM (splitting: x = z)
    """

    def __init__(self, A, y, lam, mode="pg"):
        self.Amat = A
        self.y = y
        self.lam = lam
        self.mode = mode

        # operator norm
        if mode == "admm":
            self.normA = 1.0      # A = I for ADMM splitting
        else:
            self.normA = np.linalg.norm(A, 2)

    # NONSMOOTH PART g
    def g(self, x):
        return self.lam * np.sum(np.abs(x))

    def prox_g(self, x, alpha):
        return np.sign(x) * np.maximum(np.abs(x) - alpha * self.lam, 0)

    # SMOOTH PART f
    def f(self, x):
        r = self.Amat @ x - self.y
        return 0.5 * (r @ r)

    def grad(self, x):
        return self.Amat.T @ (self.Amat @ x - self.y)

    # LINEAR OPERATORS FOR ADMM / ADLPMM
    def A(self, x):
        if self.mode == "admm":
            return x      # identity for x=z
        return self.Amat @ x

    def AT(self, u):
        if self.mode == "admm":
            return u
        return self.Amat.T @ u

    # PROX-F (used by ADMM and AL)
    def prox_f(self, v, tau):
        # Solve: (I + τ AᵀA)x = v + τAᵀy
        ATA = self.Amat.T @ self.Amat
        rhs = v + tau * (self.Amat.T @ self.y)
        M = np.eye(len(v)) + tau * ATA
        return np.linalg.solve(M, rhs)
