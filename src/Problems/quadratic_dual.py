import numpy as np
from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1
"""
Quadratic problem with explicit conjugate:
    f(x) = 0.5 * ||x||^2
    g(z) = lam * ||z||_1
Used to test FDPG, smoothed fista.
"""

class QuadraticDualProblem(ProblemBase):
    """Quadratic problem with known conjugate."""

    def __init__(self, lam):
        self.lam = lam

    def f(self, x):
        """Smooth part f(x)."""
        return 0.5 * (x @ x)

    def grad(self, x):
        """Gradient of f(x)."""
        return x

    def grad_conjugate(self, s):
        """Gradient of conjugate f*(s)."""
        # f*(s) = 0.5 ||s||^2, so grad f* = s
        return s

    def g(self, z):
        """Non-smooth part g(z)."""
        return self.lam * np.sum(np.abs(z))

    def prox_g(self, z, alpha):
        """Proximal operator of g(z)."""
        return prox_l1(z, alpha * self.lam)

