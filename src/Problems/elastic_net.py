import numpy as np
from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1

"""
Concrete problem.
Elastic Net: f(x) = 1/2 ||A x - b||^2 + (lam2/2)||x||^2, g(x) = lam1 ||x||_1
"""

class ElasticNetProblem(ProblemBase):
    """
    Elastic Net regression problem.
    """
    def __init__(self, A, b, lam1, lam2):
        self.A = A
        self.b = b
        self.lam1 = lam1
        self.lam2 = lam2

    def f(self, x):
        """
        Smooth part f(x)
        """
        r = self.A @ x - self.b
        return 0.5 * (r @ r) + 0.5 * self.lam2 * (x @ x)

    def grad(self, x):
        """
        Gradient of f(x)
        """
        return self.A.T @ (self.A @ x - self.b) + self.lam2 * x

    def prox_g(self, x, alpha):
        """
        Proximal operator of g(x) = lam1 * ||x||_1
        """
        return prox_l1(x, alpha * self.lam1)
