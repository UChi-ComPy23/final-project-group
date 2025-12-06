import numpy as np
from src.Core.problems import ProblemBase

"""
Concrete problem.
Group Lasso: f(x) = 1/2 ||A x - b||^2, g(x) = lam * sum_g ||x_g||_2
"""

class GroupLassoProblem(ProblemBase):
    """
    Group Lasso problem.
    """
    def __init__(self, A, b, groups, lam):
        self.A = A
        self.b = b
        self.groups = groups
        self.lam = lam

    def f(self, x):
        """
        Smooth part f(x)
        """
        r = self.A @ x - self.b
        return 0.5 * (r @ r)

    def grad(self, x):
        """
        Gradient of f(x)
        """
        return self.A.T @ (self.A @ x - self.b)

    def prox_g(self, x, alpha):
        """
        Proximal operator of g(x) = lam * sum_g ||x_g||_2
        """
        out = x.copy()
        for g in self.groups:
            v = out[g]
            nrm = np.linalg.norm(v)
            if nrm <= alpha * self.lam:
                out[g] = 0
            else:
                out[g] = (1 - alpha * self.lam / nrm) * v
        return out
