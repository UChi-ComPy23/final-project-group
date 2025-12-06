import numpy as np
from src.Core.problems import ProblemBase

"""
Concrete problem.
Huber regression: huber(Ax - b) + (lam/2)||x||^2
"""

class HuberRegressionProblem(ProblemBase):
    """
    Huber regression problem.
    """
    def __init__(self, A, b, delta, lam=0.0):
        self.A = A
        self.b = b
        self.delta = delta
        self.lam = lam

    def f(self, x):
        """
        Smooth part f(x) using Huber penalty
        """
        r = self.A @ x - self.b
        absr = np.abs(r)
        quad = absr <= self.delta

        val = np.sum(0.5 * r[quad]**2)
        val += np.sum(self.delta * (absr[~quad] - 0.5 * self.delta))
        val += 0.5 * self.lam * (x @ x)
        return val

    def grad(self, x):
        """
        Gradient of f(x)
        """
        r = self.A @ x - self.b
        absr = np.abs(r)
        sign_r = np.sign(r)

        grad_vec = np.where(absr <= self.delta, r, self.delta * sign_r)
        return self.A.T @ grad_vec + self.lam * x

    def subgrad(self, x):
        """
        Smooth function, so subgradient = gradient
        """
        return self.grad(x)
