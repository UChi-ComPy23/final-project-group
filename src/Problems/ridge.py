import numpy as np
from src.Core.problems import ProblemBase

"""
Concrete problem.
Ridge regression: f(x) = 1/2 ||A x - b||^2 + (lam/2) ||x||^2
"""

class RidgeRegressionProblem(ProblemBase):
    """
    Ridge regression problem.
    """
    def __init__(self, A, b, lam):
        self.A = A
        self.b = b
        self.lam = lam

    def f(self, x):
        """
        Smooth part f(x)
        """
        r = self.A @ x - self.b
        return 0.5 * (r @ r) + 0.5 * self.lam * (x @ x)

    def grad(self, x):
        """
        Gradient of f(x)
        """
        return self.A.T @ (self.A @ x - self.b) + self.lam * x

    def subgrad(self, x):
        """
        Smooth function, so subgradient = gradient
        """
        return self.grad(x)
