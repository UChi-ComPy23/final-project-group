import numpy as np
from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1

"""
Huber composite model: minimize Huber(Ax - b) + lam ||x||_1.
"""

class HuberCompositeProblem(ProblemBase):
    """Huber composite problem
    """

    def __init__(self, A, b, delta, lam):
        self.Amat = A
        self.b = b
        self.delta = delta
        self.lam = lam

    def f(self, x):
        """Smooth part f(x)
        """
        r = self.Amat @ x - self.b
        absr = np.abs(r)
        quad = absr <= self.delta
        val = np.sum(0.5 * r[quad]**2)
        val += np.sum(self.delta * (absr[~quad] - 0.5 * self.delta))
        return val

    def grad(self, x):
        """Gradient of f(x)
        """
        r = self.Amat @ x - self.b
        absr = np.abs(r)
        g = np.where(absr <= self.delta, r, self.delta * np.sign(r))
        return self.Amat.T @ g

    def phi(self, u):
        """Outer function phi(u)
        """
        return u

    def g(self, x):
        """Non-smooth part g(x)
        """
        return self.lam * np.sum(np.abs(x))

    def prox_g(self, x, alpha):
        """Proximal operator of g
        """
        return prox_l1(x, alpha * self.lam)

    def A(self, x):
        """Apply A(x)
        """
        return self.Amat @ x

    def AT(self, y):
        """Apply A^T(y)
        """
        return self.Amat.T @ y
